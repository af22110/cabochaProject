import os
import sys
import pickle
import re
import math
import time
import glob
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sudachipy import dictionary as sudachi_dictionary
from sudachipy import tokenizer as sudachi_tokenizer
import CaboCha # 事前にインストールされている想定
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pykakasi
import Levenshtein
import gc # ガベージコレクション

# --- グローバル設定 ---
INPUT_EXCEL_PATH = '/home/ubuntu/cabochaProject/input_clustering.xlsx'
JAPANESE_WORD_LIST_PATH = '/home/ubuntu/cabochaProject/japanese_word_list2.txt'
NGRAM_DATA_BASE_PATH = '/home/ubuntu/cabochaProject/nwc2010-ngrams/word/over9'
PICKLE_CHUNK_DIR = '/home/ubuntu/cabochaProject/pickle_chunks'
OUTPUT_EXCEL_PATH = '/home/ubuntu/cabochaProject/clustering_results.xlsx'
OUTPUT_GRAPH_PATH = '/home/ubuntu/cabochaProject/clustering_visualization.png'

HF_MODEL_NAME = "rinna/japanese-gpt2-xsmall"
EDIT_DISTANCE_THRESHOLD_LOG_PROB = -10.8799
WORD_3GRAM_THRESHOLD_LOG_PROB = -8.8
WORD_3GRAM_ALPHA = 0.001

# メモリ管理用設定
NGRAM_FILES_LIMIT_FOR_VOCAB = None  # ★★★ V計算時のファイル制限をデフォルトで無効化 ★★★
CACHE_CLEAR_INTERVAL = 100       # 何件のテキスト処理ごとにキャッシュをクリアするか (適宜調整)

# --- グローバル変数 (初期化はmain関数内で行う) ---
sudachi_tokenizer_dict_core = None
sudachi_dict_obj_core_for_lookup = None
sudachi_tokenizer_for_ngram = None
cabocha_parser = None
hf_tokenizer_for_edit_dist = None
hf_model_for_edit_dist = None
kakasi_converter_for_edit_dist = None
all_dictionary_data_for_edit_dist = []

_ngram_cache = {}
_chunk_cache = {}
_chunk_cache_lru = []
_CHUNK_CACHE_SIZE = 5

# --- 1. 特徴量計算関数群 ---
print("--- 1. 特徴量計算関数の定義開始 ---")

def check_word_in_sudachi_dictionary_revised(word, dict_obj):
    morphemes = dict_obj.lookup(word)
    return len(morphemes) > 0

def calculate_dict_feature(text, tokenizer_obj, dict_obj_for_lookup_func):
    if not text or not isinstance(text, str): return 1
    try:
        mode_b = sudachi_tokenizer.Tokenizer.SplitMode.B
        morphemes = tokenizer_obj.tokenize(text, mode_b)
        found_undefined_noun = False
        noun_count = 0
        for m in morphemes:
            pos = m.part_of_speech()
            if pos and pos[0] == '名詞':
                noun_count += 1
                base_form = m.normalized_form()
                if not check_word_in_sudachi_dictionary_revised(base_form, dict_obj_for_lookup_func):
                    found_undefined_noun = True
                    break
        if noun_count == 0: return 0
        del morphemes # メモリ解放
        gc.collect()
        return 1 if found_undefined_noun else 0
    except Exception as e:
        print(f"辞書照合エラー: {e} - テキスト: {text[:50]}...")
        return 1

def calculate_cabocha_features(text, parser):
    if not text or not isinstance(text, str): return 0.0, 0.0
    if parser is None:
        # print("警告: CaboChaパーサーが利用できません。CaboCha特徴量は0になります。") # 頻繁に出力される可能性があるのでコメントアウト
        return 0.0, 0.0
    try:
        tree = parser.parse(text)
        chunks = [tree.chunk(i) for i in range(tree.chunk_size())]
        if not chunks:
            del tree # メモリ解放
            return 0.0, 0.0

        token_to_chunk = [None] * tree.token_size()
        for ci, ch in enumerate(chunks):
            start = ch.token_pos
            size = ch.token_size
            for ti in range(start, start + size):
                if ti < len(token_to_chunk):
                    token_to_chunk[ti] = ci

        token_distances = []
        for ti in range(tree.token_size()):
            if ti >= len(token_to_chunk) or token_to_chunk[ti] is None: continue
            ci = token_to_chunk[ti]
            if ci >= len(chunks): continue
            head_ci = chunks[ci].link
            if head_ci >= 0 and head_ci < len(chunks):
                head_token_pos = chunks[head_ci].token_pos
                token_distances.append(abs(ti - head_token_pos))

        var_td = 0.0
        if token_distances:
            mean_td = sum(token_distances) / len(token_distances)
            var_td = sum((d - mean_td) ** 2 for d in token_distances) / len(token_distances)
            del mean_td
        del token_distances # メモリ解放

        scores = [ch.score for ch in chunks if ch.score is not None]
        var_sc = 0.0
        if scores:
            mean_sc = sum(scores) / len(scores)
            var_sc = sum((s - mean_sc) ** 2 for s in scores) / len(scores)
            del mean_sc
        del scores # メモリ解放

        del tree, chunks, token_to_chunk # メモリ解放
        gc.collect()
        return var_td, var_sc
    except Exception as e:
        print(f"CaboCha特徴量計算エラー: {e} - テキスト: {text[:50]}...")
        return 0.0, 0.0


def is_jukugo_for_edit_dist(word):
    return bool(re.match(r'^[\u4E00-\u9FFF]{2,}$', word))

def katakana_to_romaji_for_edit_dist(katakana_text, kakasi_cv):
    if not katakana_text: return ""
    try:
        return "".join([item.get('hepburn', item.get('orig', '')) for item in kakasi_cv.convert(katakana_text)]).lower()
    except Exception: return "___romaji_error___"

def estimate_segment_reading_for_edit_dist(segment_kanji, dictionary_data):
    min_dist = sys.maxsize
    est_reading = None
    if not dictionary_data: return None
    for kanji, reading in dictionary_data:
        if not kanji: continue
        dist = Levenshtein.distance(segment_kanji, kanji)
        if dist < min_dist:
            min_dist = dist
            est_reading = reading
            if min_dist == 0: break
    return est_reading

def find_kanji_candidates_by_distance_for_edit_dist(segment_kanji, dictionary_data, max_dist=2, jukugo_only=True):
    candidates = []
    dict_to_search = dictionary_data
    if jukugo_only:
        jukugo_dict = [(k, r) for k, r in dictionary_data if is_jukugo_for_edit_dist(k)]
        if jukugo_dict: dict_to_search = jukugo_dict
        # del jukugo_dict # 不要になったら解放

    min_found_dist = sys.maxsize
    temp_candidates = []
    for kanji, reading in dict_to_search:
        if not kanji: continue
        dist = Levenshtein.distance(segment_kanji, kanji)
        if dist <= max_dist:
             temp_candidates.append({'kanji': kanji, 'reading': reading, 'dist': dist})
        if dist < min_found_dist:
            min_found_dist = dist

    if not temp_candidates and min_found_dist != sys.maxsize:
         for kanji, reading in dict_to_search:
            if not kanji: continue
            dist = Levenshtein.distance(segment_kanji, kanji)
            if dist == min_found_dist:
                temp_candidates.append({'kanji': kanji, 'reading': reading, 'dist': dist})

    seen_kanji = set()
    for cand_info in sorted(temp_candidates, key=lambda x: x['dist']):
        if cand_info['kanji'] not in seen_kanji:
            candidates.append((cand_info['kanji'], cand_info['reading'], cand_info['dist']))
            seen_kanji.add(cand_info['kanji'])
    del temp_candidates, seen_kanji, dict_to_search # メモリ解放 (jukugo_dictもここで解放される)
    gc.collect()
    return candidates


def find_best_candidate_romaji_rerank_for_edit_dist(est_seg_reading_romaji, initial_candidates, kakasi_cv):
    min_romaji_dist = sys.maxsize
    best_cand_kanji, best_cand_reading, best_kanji_dist_orig = None, None, sys.maxsize
    if est_seg_reading_romaji in ["", "___romaji_error___"] or not initial_candidates:
        return None, sys.maxsize

    for kanji, reading, kanji_dist in initial_candidates:
        cand_reading_romaji = katakana_to_romaji_for_edit_dist(reading, kakasi_cv)
        if cand_reading_romaji == "___romaji_error___": continue

        romaji_dist = Levenshtein.distance(est_seg_reading_romaji, cand_reading_romaji)
        if romaji_dist < min_romaji_dist:
            min_romaji_dist = romaji_dist
            best_cand_kanji = kanji
            best_cand_reading = reading
            best_kanji_dist_orig = kanji_dist

    if best_cand_kanji is None: return None, sys.maxsize
    return best_cand_kanji, best_kanji_dist_orig

def calculate_edit_distance_feature(text, tokenizer, model, dictionary_data, kakasi_cv, threshold_log_p):
    if not text or not isinstance(text, str): return 0.0
    try:
        # モデルがどのデバイスにあるか確認
        device = next(model.parameters()).device

        inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device) # ★★★ input_ids をモデルと同じデバイスに移動 ★★★
        offset_mapping = inputs["offset_mapping"].squeeze().tolist()
        # del inputs # input_ids を後で使うのでここでは削除しない

        kanji_segments_info = []
        for match in re.finditer(r'[\u4E00-\u9FFF]{2,}', text):
            kanji_segments_info.append({'segment': match.group(0), 'start_char': match.start(), 'end_char': match.end()})

        if not kanji_segments_info:
            del inputs, input_ids, offset_mapping # メモリ解放
            gc.collect()
            return 0.0

        with torch.no_grad():
            outputs = model(input_ids) # これで model と input_ids が同じデバイスになる
            logits = outputs.logits
        log_probs_full_sequence = torch.log_softmax(logits, dim=-1).squeeze(0)
        del outputs, logits # メモリ解放

        segment_scores = []
        for seg_info in kanji_segments_info:
            segment_str = seg_info['segment']
            seg_start_char_idx = seg_info['start_char']
            hf_indices_for_segment = []
            for hf_token_idx, (tok_char_start, tok_char_end) in enumerate(offset_mapping):
                if tok_char_start >= seg_start_char_idx and tok_char_end <= seg_info['end_char']:
                    if hf_token_idx > 0 and hf_token_idx < input_ids.shape[1]:
                        hf_indices_for_segment.append(hf_token_idx)

            avg_log_prob = -float('inf')
            if hf_indices_for_segment:
                log_prob_sum = 0
                valid_tokens = 0
                for hf_idx in hf_indices_for_segment:
                    token_id = input_ids[0, hf_idx].item()
                    # log_probs_full_sequence も GPU にあるはずなので、.item() でCPUに持ってくる
                    if hf_idx -1 < log_probs_full_sequence.shape[0]:
                        log_prob_sum += log_probs_full_sequence[hf_idx-1, token_id].item()
                        valid_tokens +=1
                avg_log_prob = log_prob_sum / valid_tokens if valid_tokens > 0 else -float('inf')
            segment_scores.append({'segment': segment_str, 'avg_log_prob': avg_log_prob})

        del inputs, input_ids, offset_mapping, log_probs_full_sequence, kanji_segments_info # メモリ解放
        gc.collect()

        if not segment_scores: return 0.0

        min_score = float('inf')
        segment_with_min_score_str = None
        valid_scores = [s for s in segment_scores if s['avg_log_prob'] > -float('inf')]
        if not valid_scores:
            del segment_scores
            gc.collect()
            return 0.0

        segment_with_min_score = min(valid_scores, key=lambda x: x['avg_log_prob'])
        min_score = segment_with_min_score['avg_log_prob']
        segment_with_min_score_str = segment_with_min_score['segment']
        del segment_scores, valid_scores
        gc.collect()

        edit_distance_val = 0.0
        if min_score < threshold_log_p and segment_with_min_score_str:
            target_segment = segment_with_min_score_str
            est_reading = estimate_segment_reading_for_edit_dist(target_segment, dictionary_data)
            if not est_reading:
                edit_distance_val = float(len(target_segment))
            else:
                est_reading_romaji = katakana_to_romaji_for_edit_dist(est_reading, kakasi_cv)
                if est_reading_romaji == "___romaji_error___":
                    edit_distance_val = float(len(target_segment))
                else:
                    initial_candidates = find_kanji_candidates_by_distance_for_edit_dist(target_segment, dictionary_data)
                    if not initial_candidates:
                        edit_distance_val = float(len(target_segment))
                    else:
                        best_cand_kanji, _ = find_best_candidate_romaji_rerank_for_edit_dist(est_reading_romaji, initial_candidates, kakasi_cv)
                        if best_cand_kanji:
                            edit_distance_val = float(Levenshtein.distance(target_segment, best_cand_kanji))
                        else:
                            edit_distance_val = float(len(target_segment))
                        del initial_candidates
            del est_reading
        gc.collect()
        return edit_distance_val
    except Exception as e:
        print(f"編集距離計算エラー: {e} - テキスト: {text[:50]}...")
        # エラー時にも確保した可能性のあるメモリを解放
        if 'inputs' in locals(): del inputs
        if 'input_ids' in locals(): del input_ids
        if 'offset_mapping' in locals(): del offset_mapping
        if 'outputs' in locals(): del outputs
        if 'logits' in locals(): del log_probs_full_sequence
        if 'segment_scores' in locals(): del segment_scores
        if 'valid_scores' in locals(): del valid_scores
        gc.collect()
        return 0.0


# 特徴量5: 単語3-gram
def load_ngrams_from_file_for_pre(filepath, n):
    ngram_counts = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    words = parts[0].split(' ')
                    if len(words) == n:
                        try:
                            ngram_counts[tuple(words)] = int(parts[1])
                        except ValueError:
                            pass # 数値変換できない行はスキップ
    except FileNotFoundError:
        print(f"エラー: N-gramファイルが見つかりません {filepath}")
    except Exception as e:
        print(f"N-gramファイル読み込みエラー {filepath}: {e}")
    return ngram_counts

def calculate_total_vocabulary_size_for_pre(ngram_data_dirs, files_limit_per_type=None):
    all_words = set()
    print("N-gramから語彙サイズ(V)を計算中...")
    for n_type, dir_path in ngram_data_dirs.items():
        if not os.path.isdir(dir_path):
            print(f"警告: ディレクトリが見つかりません {dir_path} ({n_type}-grams)")
            continue
        file_paths = sorted(glob.glob(os.path.join(dir_path, f"{n_type}gm-*")))
        if files_limit_per_type is not None:
            file_paths = file_paths[:files_limit_per_type]
            print(f"  {n_type}-grams: 最初の {len(file_paths)} ファイルのみ処理します。")

        for filepath in tqdm(file_paths, desc=f"V計算 ({n_type}-grams)", leave=False):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) == 2:
                            for word in parts[0].split(' '):
                                if word: # 空文字列でないことを確認
                                    all_words.add(word)
            except Exception as e:
                print(f"警告: ファイル読み込みエラー {filepath}: {e}")
    V = len(all_words)
    print(f"計算された総語彙サイズ (V): {V}")
    del all_words # メモリ解放
    gc.collect()
    return V

def process_and_chunk_ngrams_for_pre(directory_path, n, pickle_out_dir, total_vocab_size, files_limit=None):
    if not os.path.isdir(directory_path):
        print(f"エラー: ディレクトリが見つかりません {directory_path}")
        return False
    os.makedirs(pickle_out_dir, exist_ok=True)
    file_paths = sorted(glob.glob(os.path.join(directory_path, f"{n}gm-*")))
    if files_limit is not None:
        file_paths = file_paths[:files_limit]
    if not file_paths:
        print(f"警告: {n}-gramファイルが見つかりません ({directory_path})")
        return False

    print(f"{n}-gramデータをチャンク化中 ({len(file_paths)}ファイル対象)...")
    ngram_index = {}  # ここを ngram_first_word_index から ngram_index に変更
    for filepath in tqdm(file_paths, desc=f"チャンク化 ({n}-grams)", leave=False):
        ngrams_in_file = load_ngrams_from_file_for_pre(filepath, n)
        if not ngrams_in_file:
            continue

        base_filename = os.path.basename(filepath).replace('.txt', '')
        output_chunk_filename = f"{n}gram_{base_filename}.pkl"
        data_to_save = {'n': n, 'counts': ngrams_in_file, 'total_vocab_size': total_vocab_size}

        try:
            with open(os.path.join(pickle_out_dir, output_chunk_filename), "wb") as f_out:
                pickle.dump(data_to_save, f_out, pickle.HIGHEST_PROTOCOL)
            # インデックスにトリグラム（またはバイグラム）ごとにカウント値を直接保存
            for ngram_tuple, count in ngrams_in_file.items():
                ngram_index[ngram_tuple] = count
        except Exception as e:
            print(f"警告: チャンク保存エラー {output_chunk_filename}: {e}")
        finally:
            del ngrams_in_file
            del data_to_save
            gc.collect()

    index_filename = f"{n}gram_first_word_index.pkl"
    try:
        print(f"保存する{n}-gramインデックスのエントリ数: {len(ngram_index)}") # ★追加★
        if n == 3 and len(ngram_index) > 0: # 3-gramの場合、いくつかキーを表示してみる
            print(f"  3-gramインデックスのサンプルキー: {list(ngram_index.keys())[:5]}")
        # ngram_index をそのままpickle化
        with open(os.path.join(pickle_out_dir, index_filename), "wb") as f_idx:
            pickle.dump(ngram_index, f_idx, pickle.HIGHEST_PROTOCOL)
        print(f"{n}-gramのインデックスを保存: {index_filename}")
    except Exception as e:
        print(f"エラー: {n}-gramのインデックス保存失敗: {index_filename}: {e}")
        return False
    del ngram_index # メモリ解放
    gc.collect()
    return True

def run_ngram_preprocessing(base_ngram_path, out_pickle_dir, files_limit=None):
    print("--- N-gramデータの前処理開始 ---")
    if not os.path.exists(base_ngram_path):
        print(f"エラー: N-gramデータパス '{base_ngram_path}' が見つかりません。")
        return 0, {}, {}, [], []

    os.makedirs(out_pickle_dir, exist_ok=True)

    idx_2gram_path = os.path.join(out_pickle_dir, '2gram_first_word_index.pkl')
    idx_3gram_path = os.path.join(out_pickle_dir, '3gram_first_word_index.pkl')
    # チャンクファイルの存在も確認
    glob_2gram_chunks = glob.glob(os.path.join(out_pickle_dir, "2gram_*.pkl"))
    glob_3gram_chunks = glob.glob(os.path.join(out_pickle_dir, "3gram_*.pkl"))

    required_files_exist = (os.path.exists(idx_2gram_path) and
                            os.path.exists(idx_3gram_path) and
                            bool(glob_2gram_chunks) and # 少なくとも1つはチャンクファイルがある
                            bool(glob_3gram_chunks))

    if required_files_exist:
        print("N-gramの前処理済みファイルが見つかりました。ロードします。")
        v_loaded = 0
        try:
            first_chunk_path = None
            if glob_2gram_chunks:
                first_chunk_path = glob_2gram_chunks[0]
            elif glob_3gram_chunks:
                first_chunk_path = glob_3gram_chunks[0]

            if first_chunk_path and os.path.exists(first_chunk_path):
                with open(first_chunk_path, 'rb') as f_chk:
                    sample_chunk = pickle.load(f_chk)
                    v_loaded = sample_chunk.get('total_vocab_size', 0)
                print(f"ロードされた総語彙サイズ (V): {v_loaded}")
                del sample_chunk
            else:
                 print(f"警告: Vをロードするためのチャンクファイルが見つかりません。V=0で続行します。")
        except Exception as e_load_v:
            print(f"警告: チャンクからのVのロードに失敗: {e_load_v}。V=0で続行します。")
            v_loaded = 0

        bigram_idx_loaded, trigram_idx_loaded = {}, {}
        try:
            if os.path.exists(idx_2gram_path):
                with open(idx_2gram_path, 'rb') as f: bigram_idx_loaded = pickle.load(f)
                print(f"ロードされた2-gramインデックスのエントリ数: {len(bigram_idx_loaded)}") # ★追加★
            else: print(f"警告: 2-gramインデックスファイルが見つかりません: {idx_2gram_path}")
            if os.path.exists(idx_3gram_path):
                with open(idx_3gram_path, 'rb') as f: trigram_idx_loaded = pickle.load(f)
                print(f"ロードされた3-gramインデックスのエントリ数: {len(trigram_idx_loaded)}") # ★追加★
            else: print(f"警告: 3-gramインデックスファイルが見つかりません: {idx_3gram_path}")
        except Exception as e_load_idx:
             print(f"警告: N-gramインデックスのロードに失敗: {e_load_idx}。空のインデックスで続行します。")

        print("--- N-gramデータの前処理スキップ (ロード完了) ---")
        gc.collect()
        return v_loaded, bigram_idx_loaded, trigram_idx_loaded, glob_2gram_chunks, glob_3gram_chunks

    ngram_dirs = {2: base_ngram_path, 3: base_ngram_path}
    total_v = calculate_total_vocabulary_size_for_pre(ngram_dirs, files_limit_per_type=files_limit)
    if total_v == 0:
        print("警告: 総語彙サイズ(V)が0です。N-gram特徴量は正しく計算されない可能性があります。")

    process_and_chunk_ngrams_for_pre(base_ngram_path, 2, out_pickle_dir, total_v, files_limit=files_limit)
    process_and_chunk_ngrams_for_pre(base_ngram_path, 3, out_pickle_dir, total_v, files_limit=files_limit)

    print("N-gramデータをロード中...")
    bigram_idx, trigram_idx = {}, {}
    try:
        if os.path.exists(idx_2gram_path):
            with open(idx_2gram_path, 'rb') as f: bigram_idx = pickle.load(f)
            print(f"新規作成後ロードされた2-gramインデックスのエントリ数: {len(bigram_idx)}") # ★追加★
        if os.path.exists(idx_3gram_path):
            with open(idx_3gram_path, 'rb') as f: trigram_idx = pickle.load(f)
            print(f"新規作成後ロードされた3-gramインデックスのエントリ数: {len(trigram_idx)}") # ★追加★
    except Exception as e:
        print(f"警告: 前処理後のN-gramインデックスのロードに失敗しました: {e}")

    bigram_paths_final = sorted(glob.glob(os.path.join(out_pickle_dir, "2gram_*.pkl")))
    trigram_paths_final = sorted(glob.glob(os.path.join(out_pickle_dir, "3gram_*.pkl")))
    print("--- N-gramデータの前処理完了 ---")
    gc.collect()
    return total_v, bigram_idx, trigram_idx, glob_2gram_chunks, glob_3gram_chunks


def get_ngram_count_from_pickle(ngram_tuple, n_type, chunk_dir, index_data, all_chunk_paths_for_type):
    if not ngram_tuple:
        return 0

    cache_key = (ngram_tuple, n_type)
    if cache_key in _ngram_cache:
        return _ngram_cache[cache_key]

    # index_dataは now {ngram_tuple: count} 形式
    if ngram_tuple in index_data:
        count = index_data[ngram_tuple]
        _ngram_cache[cache_key] = count
        return count

    # インデックスに無い場合は0
    _ngram_cache[cache_key] = 0
    return 0

def get_ngram_count_from_full_index(ngram_tuple, index_data):
    if not ngram_tuple:
        return 0
    return index_data.get(ngram_tuple, 0)

def calculate_score_per_ngram_for_feature(text_words, V, alpha_smooth,
                                           ngram_pickle_dir, trigram_idx_data, bigram_idx_data,
                                           all_trigram_paths, all_bigram_paths):
    if len(text_words) < 3 or V == 0: return []
    log_probs = []
    for i in range(len(text_words) - 2):
        w1, w2, w3 = text_words[i], text_words[i+1], text_words[i+2]

        count_w1_w2_w3 = get_ngram_count_from_pickle((w1, w2, w3), 3, ngram_pickle_dir, trigram_idx_data, all_trigram_paths)
        count_w1_w2 = get_ngram_count_from_pickle((w1, w2), 2, ngram_pickle_dir, bigram_idx_data, all_bigram_paths)

        numerator = count_w1_w2_w3 + alpha_smooth
        denominator = count_w1_w2 + alpha_smooth * V

        log_prob = -float('inf') if denominator == 0 else np.log(numerator) - np.log(denominator)
        log_probs.append(log_prob)
    return log_probs

def calculate_3gram_feature(text, tokenizer_obj_ngram, V_ngram, alpha_3gram, threshold_3gram,
                             ngram_pk_dir, trigram_idx_map, bigram_idx_map,
                             trigram_cpaths, bigram_cpaths): # trigram_cpaths, bigram_cpaths は現在のロジックでは不要
    if not text or not isinstance(text, str) or V_ngram == 0:
        # print(f"Debug 3gram: 初期条件で0を返します。Text: {text[:10]}, V_ngram: {V_ngram}")
        return 0
    try:
        morphemes = [m.surface() for m in tokenizer_obj_ngram.tokenize(text, sudachi_tokenizer.Tokenizer.SplitMode.C)]
        if not morphemes:
            # print(f"Debug 3gram: 形態素解析結果が空です。Text: {text[:10]}")
            return 0

        # print(f"Debug 3gram: Text='{text[:30]}...', Morphemes={morphemes[:10]}") # デバッグ用

        sentence_end_punctuations = {'。', '！', '？', '．', '!'}

        all_log_probs = []
        current_segment = []
        for m_idx, m in enumerate(morphemes):
            if m in sentence_end_punctuations:
                if len(current_segment) >= 3:
                    # print(f"  Debug 3gram: Segment to process: {current_segment}") # デバッグ用
                    segment_probs = calculate_score_per_ngram_for_feature(
                        current_segment, V_ngram, alpha_3gram, ngram_pk_dir,
                        trigram_idx_map, bigram_idx_map, trigram_cpaths, bigram_cpaths
                    )
                    all_log_probs.extend(segment_probs)
                    # if segment_probs: print(f"    Segment probs: {segment_probs[:5]}") # デバッグ用
                current_segment = []
            else:
                current_segment.append(m)

        if len(current_segment) >= 3:
            # print(f"  Debug 3gram: Final segment to process: {current_segment}") # デバッグ用
            segment_probs = calculate_score_per_ngram_for_feature(
                current_segment, V_ngram, alpha_3gram, ngram_pk_dir,
                trigram_idx_map, bigram_idx_map, trigram_cpaths, bigram_cpaths
            )
            all_log_probs.extend(segment_probs)
            # if segment_probs: print(f"    Final segment probs: {segment_probs[:5]}") # デバッグ用
        del morphemes, current_segment
        gc.collect()

        if not all_log_probs:
            # print(f"Debug 3gram: No log_probs generated. Text: {text[:30]}...")
            return 0

        # print(f"Debug 3gram: All log_probs for '{text[:30]}...': {all_log_probs}") # デバッグ用
        is_unnatural = any(prob < threshold_3gram for prob in all_log_probs if prob != -float('inf'))
        # if is_unnatural:
        #     print(f"Debug 3gram: Unnatural 3-gram found. Min prob: {min(p for p in all_log_probs if p != -float('inf'))}")
        del all_log_probs
        gc.collect()
        return 3 if is_unnatural else 0
    except Exception as e:
        print(f"3-gram特徴量計算エラー: {e} - テキスト: {text[:50]}...")
        return 0

def calculate_3gram_feature_with_full_index(text, tokenizer_obj_ngram, alpha_3gram, threshold_3gram, trigram_full_index, bigram_full_index):
    # V is the number of unique words in the index
    V_ngram = 0
    all_words = set()
    for ngram in trigram_full_index.keys():
        all_words.update(ngram)
    for ngram in bigram_full_index.keys():
        all_words.update(ngram)
    V_ngram = len(all_words)
    if not text or not isinstance(text, str) or V_ngram == 0:
        return 0
    try:
        morphemes = [m.surface() for m in tokenizer_obj_ngram.tokenize(text, sudachi_tokenizer.Tokenizer.SplitMode.C)]
        if not morphemes:
            return 0
        sentence_end_punctuations = {'。', '！', '？', '．', '!'}
        all_log_probs = []
        current_segment = []
        for m in morphemes:
            if m in sentence_end_punctuations:
                if len(current_segment) >= 3:
                    segment_probs = []
                    for i in range(len(current_segment) - 2):
                        w1, w2, w3 = current_segment[i], current_segment[i+1], current_segment[i+2]
                        count_w1_w2_w3 = get_ngram_count_from_full_index((w1, w2, w3), trigram_full_index)
                        count_w1_w2 = get_ngram_count_from_full_index((w1, w2), bigram_full_index)
                        numerator = count_w1_w2_w3 + alpha_3gram
                        denominator = count_w1_w2 + alpha_3gram * V_ngram
                        log_prob = -float('inf') if denominator == 0 else np.log(numerator) - np.log(denominator)
                        segment_probs.append(log_prob)
                    all_log_probs.extend(segment_probs)
                current_segment = []
            else:
                current_segment.append(m)
        if len(current_segment) >= 3:
            segment_probs = []
            for i in range(len(current_segment) - 2):
                w1, w2, w3 = current_segment[i], current_segment[i+1], current_segment[i+2]
                count_w1_w2_w3 = get_ngram_count_from_full_index((w1, w2, w3), trigram_full_index)
                count_w1_w2 = get_ngram_count_from_full_index((w1, w2), bigram_full_index)
                numerator = count_w1_w2_w3 + alpha_3gram
                denominator = count_w1_w2 + alpha_3gram * V_ngram
                log_prob = -float('inf') if denominator == 0 else np.log(numerator) - np.log(denominator)
                segment_probs.append(log_prob)
            all_log_probs.extend(segment_probs)
        if not all_log_probs:
            return 0
        is_unnatural = any(prob < threshold_3gram for prob in all_log_probs if prob != -float('inf'))
        return 3 if is_unnatural else 0
    except Exception as e:
        print(f"3-gram特徴量計算エラー: {e} - テキスト: {text[:50]}...")
        return 0

# --- N-gramピックル分割ファイルを逐次処理してn-gramカウントを集計する関数 ---
def collect_ngram_counts_from_pickles(pickle_paths, texts, n=2):
    """
    pickle_paths: 2-gramまたは3-gramのpklファイルパスリスト
    texts: エクセルから読み込んだ全テキスト（リスト）
    n: 2ならbi-gram, 3ならtri-gram
    戻り値: {(w1, w2): count, ...} または {(w1, w2, w3): count, ...}
    """
    from sudachipy import tokenizer as sudachi_tokenizer
    from sudachipy import dictionary as sudachi_dictionary
    import gc
    
    # SudachiPyのtokenizerを使う（グローバル変数が初期化済みならそれを使う）
    global sudachi_tokenizer_for_ngram
    if sudachi_tokenizer_for_ngram is None:
        sudachi_tokenizer_for_ngram = sudachi_dictionary.Dictionary(dict_type="core").create()
    mode = sudachi_tokenizer.Tokenizer.SplitMode.C

    # 全テキストの全n-gramを事前に列挙
    ngram_set = set()
    for text in texts:
        try:
            morphemes = [m.surface() for m in sudachi_tokenizer_for_ngram.tokenize(text, mode)]
            for i in range(len(morphemes) - n + 1):
                ngram = tuple(morphemes[i:i+n])
                ngram_set.add(ngram)
        except Exception as e:
            print(f"形態素解析エラー: {e} - テキスト: {text[:30]}...")
            continue
    print(f"全テキストから抽出されたユニーク{n}-gram数: {len(ngram_set)}")

    ngram_count_dict = dict()
    for pkl_path in tqdm(pickle_paths, desc=f"{n}-gramピックル逐次処理"): 
        try:
            with open(pkl_path, 'rb') as f:
                chunk = pickle.load(f)
                counts = chunk.get('counts', {})
                for ngram in ngram_set:
                    if ngram in counts:
                        ngram_count_dict[ngram] = counts[ngram]
            del chunk, counts
            gc.collect()
        except Exception as e:
            print(f"ピックルファイル読み込みエラー: {e} - {pkl_path}")
            continue
    print(f"集計された{n}-gram数: {len(ngram_count_dict)}")
    return ngram_count_dict

# --- メイン処理 ---
def main():
    global sudachi_tokenizer_dict_core, sudachi_dict_obj_core_for_lookup, sudachi_tokenizer_for_ngram
    global cabocha_parser, hf_tokenizer_for_edit_dist, hf_model_for_edit_dist
    global kakasi_converter_for_edit_dist, all_dictionary_data_for_edit_dist
    global _ngram_cache, _chunk_cache, _chunk_cache_lru

    print("--- メイン処理開始 ---")

    # --- 0. 初期化処理 ---
    print("--- 0. 初期化処理開始 ---")

    print("SudachiPy 初期化中...")
    try:
        sudachi_tokenizer_dict_core = sudachi_dictionary.Dictionary(dict_type="core").create()
        sudachi_dict_obj_core_for_lookup = sudachi_dictionary.Dictionary(dict_type="core")
        sudachi_tokenizer_for_ngram = sudachi_dictionary.Dictionary(dict_type="core").create()
        print("SudachiPy 初期化完了。")
    except Exception as e:
        print(f"SudachiPyの初期化に失敗しました: {e}")
        sys.exit(1)

    print("CaboCha 初期化中...")
    try:
        cabocha_parser = CaboCha.Parser()
        print("CaboCha 初期化完了。")
    except Exception as e:
        print(f"CaboChaの初期化に失敗しました: {e}。CaboCha関連の特徴量は0になります。")
        cabocha_parser = None

    print(f"Hugging Faceモデル {HF_MODEL_NAME} をロード中...")
    try:
        hf_tokenizer_for_edit_dist = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        hf_model_for_edit_dist = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME)
        hf_model_for_edit_dist.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hf_model_for_edit_dist.to(device)
        print(f"Hugging Faceモデルを {device} にロード完了。")
    except Exception as e:
        print(f"Hugging FaceモデルまたはTokenizerのロード中にエラーが発生しました: {e}")
        sys.exit(1)

    print("pykakasi 初期化中...")
    try:
        kakasi_converter_for_edit_dist = pykakasi.kakasi()
        kakasi_converter_for_edit_dist.setMode("K", "E")
        kakasi_converter_for_edit_dist.setMode("r", "Hepburn")
        print("pykakasi 初期化完了。")
    except Exception as e:
        print(f"pykakasiの初期化に失敗しました: {e}")
        sys.exit(1)

    print(f"編集距離用辞書データ '{JAPANESE_WORD_LIST_PATH}' をロード中...")
    if os.path.exists(JAPANESE_WORD_LIST_PATH):
        try:
            with open(JAPANESE_WORD_LIST_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    parts = line.split('\t')
                    if len(parts) >= 2 and parts[0] and parts[1]:
                        all_dictionary_data_for_edit_dist.append((parts[0], parts[1]))
            if all_dictionary_data_for_edit_dist:
                print(f"編集距離用辞書データから {len(all_dictionary_data_for_edit_dist)} 件ロードしました。")
            else:
                print(f"警告: 編集距離用辞書ファイル '{JAPANESE_WORD_LIST_PATH}' は空または有効なデータがありません。")
        except Exception as e:
            print(f"警告: 編集距離用辞書ファイル '{JAPANESE_WORD_LIST_PATH}' の読み込みエラー: {e}")
    else:
        print(f"警告: 編集距離用辞書ファイル '{JAPANESE_WORD_LIST_PATH}' が見つかりませんでした。")
    print("--- 0. 初期化処理完了 ---")

    # N-gram前処理の実行とデータロード
    _ngram_cache, _chunk_cache, _chunk_cache_lru = {}, {}, []
    # full indexファイルは使わず、分割pickleファイルのみを使う
    total_vocab_size_ngram, bigram_index_ngram, trigram_index_ngram, \
    all_bigram_chunk_paths, all_trigram_chunk_paths = run_ngram_preprocessing(
        NGRAM_DATA_BASE_PATH, PICKLE_CHUNK_DIR, files_limit=NGRAM_FILES_LIMIT_FOR_VOCAB
    )
    if total_vocab_size_ngram == 0:
        print("警告: N-gramの総語彙サイズが0です。単語3-gram特徴量は常に0になります。")

    # 入力Excelデータの読み込み
    print(f"入力ファイル '{INPUT_EXCEL_PATH}' を読み込み中...")
    try:
        df_input = pd.read_excel(INPUT_EXCEL_PATH, header=None, usecols=[0])
        texts_to_cluster = df_input.iloc[:, 0].astype(str).tolist()
        print(f"{len(texts_to_cluster)} 件のテキストを読み込みました。")
        if not texts_to_cluster:
            print("エラー: 入力ファイルにクラスタリング対象のテキストが含まれていません。")
            return
        del df_input
        gc.collect()
    except FileNotFoundError:
        print(f"エラー: 入力ファイル '{INPUT_EXCEL_PATH}' が見つかりません。")
        return
    except Exception as e:
        print(f"エラー: 入力ファイルの読み込み中にエラーが発生しました: {e}")
        return

    # 特徴量抽出
    print("特徴量抽出を開始します...")
    all_features = []
    feature_names = ["熟語編集距離", "単語3-gram異常度", "Token依存距離分散", "係り受けスコア分散", "辞書照合"]

    # 特徴量抽出前に、全テキストの全3gram/2gramカウントを分割pickleから構築
    print("全テキストの3-gramカウントを分割pickleから集計中...")
    trigram_counts = collect_ngram_counts_from_pickles(all_trigram_chunk_paths, texts_to_cluster, n=3)
    print("全テキストの2-gramカウントを分割pickleから集計中...")
    bigram_counts = collect_ngram_counts_from_pickles(all_bigram_chunk_paths, texts_to_cluster, n=2)

    for text_idx, text_content in tqdm(enumerate(texts_to_cluster), total=len(texts_to_cluster), desc="特徴量抽出中"):
        if text_idx > 0 and text_idx % CACHE_CLEAR_INTERVAL == 0:
            print(f"\n{text_idx}件処理後、キャッシュをクリアします。")
            _ngram_cache.clear()
            _chunk_cache.clear()
            _chunk_cache_lru.clear()
            gc.collect()

        f_edit_dist = calculate_edit_distance_feature(
            text_content, hf_tokenizer_for_edit_dist, hf_model_for_edit_dist,
            all_dictionary_data_for_edit_dist, kakasi_converter_for_edit_dist,
            EDIT_DISTANCE_THRESHOLD_LOG_PROB
        )

        # --- 3-gram異常度特徴量をtrigram_counts/bigram_countsで計算 ---
        try:
            morphemes = [m.surface() for m in sudachi_tokenizer_for_ngram.tokenize(text_content, sudachi_tokenizer.Tokenizer.SplitMode.C)]
            sentence_end_punctuations = {'。', '！', '？', '．', '!'}
            all_log_probs = []
            current_segment = []
            for m in morphemes:
                if m in sentence_end_punctuations:
                    if len(current_segment) >= 3:
                        for i in range(len(current_segment) - 2):
                            w1, w2, w3 = current_segment[i], current_segment[i+1], current_segment[i+2]
                            count_w1_w2_w3 = trigram_counts.get((w1, w2, w3), 0)
                            count_w1_w2 = bigram_counts.get((w1, w2), 0)
                            numerator = count_w1_w2_w3 + WORD_3GRAM_ALPHA
                            denominator = count_w1_w2 + WORD_3GRAM_ALPHA * total_vocab_size_ngram
                            log_prob = -float('inf') if denominator == 0 else np.log(numerator) - np.log(denominator)
                            all_log_probs.append(log_prob)
                    current_segment = []
                else:
                    current_segment.append(m)
            if len(current_segment) >= 3:
                for i in range(len(current_segment) - 2):
                    w1, w2, w3 = current_segment[i], current_segment[i+1], current_segment[i+2]
                    count_w1_w2_w3 = trigram_counts.get((w1, w2, w3), 0)
                    count_w1_w2 = bigram_counts.get((w1, w2), 0)
                    numerator = count_w1_w2_w3 + WORD_3GRAM_ALPHA
                    denominator = count_w1_w2 + WORD_3GRAM_ALPHA * total_vocab_size_ngram
                    log_prob = -float('inf') if denominator == 0 else np.log(numerator) - np.log(denominator)
                    all_log_probs.append(log_prob)
            is_unnatural = any(prob < WORD_3GRAM_THRESHOLD_LOG_PROB for prob in all_log_probs if prob != -float('inf'))
            f_3gram = 3 if is_unnatural else 0
            del morphemes, current_segment, all_log_probs
            gc.collect()
        except Exception as e:
            print(f"3-gram特徴量計算エラー: {e} - テキスト: {text_content[:50]}...")
            f_3gram = 0

        f_token_dist, f_dep_score = calculate_cabocha_features(text_content, cabocha_parser)

        f_dict = calculate_dict_feature(text_content, sudachi_tokenizer_dict_core, sudachi_dict_obj_core_for_lookup)

        all_features.append([f_edit_dist, f_3gram, f_token_dist, f_dep_score, f_dict])

        del f_edit_dist, f_3gram, f_token_dist, f_dep_score, f_dict
        if text_idx % (CACHE_CLEAR_INTERVAL * 2) == 0 and text_idx > 0: # 少し頻度を落としてGC
             gc.collect()

    features_matrix = np.array(all_features, dtype=np.float32)
    del all_features
    gc.collect()

    imputer = SimpleImputer(strategy='mean')
    features_matrix_imputed = imputer.fit_transform(features_matrix)
    del features_matrix
    gc.collect()

    print("特徴量を標準化中...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_matrix_imputed)

    print("エルボー法で最適なクラスタ数 (k) を探索中...")
    sse = []
    max_k = min(10, len(texts_to_cluster) - 1 if len(texts_to_cluster) > 1 else 1)
    k_range = range(1, max_k + 1)

    if not k_range or features_scaled.shape[0] == 0: # k_rangeが空、またはデータがない場合
        print("クラスタリング対象のデータが1件以下、または特徴量データが空のため、エルボー法とクラスタリングをスキップします。")
        optimal_k_val = 1 # または適切な処理
        if features_scaled.shape[0] > 0: # データがあるがクラスタ数が1の場合
            cluster_labels = np.zeros(features_scaled.shape[0], dtype=int) # 全てクラスタ0とする
        else:
            cluster_labels = np.array([])
    else:
        for k_val in tqdm(k_range, desc="エルボー法"):
            if k_val > features_scaled.shape[0]:
                print(f"k={k_val} はサンプル数({features_scaled.shape[0]})を超えるためスキップします。")
                break
            kmeans_elbow = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
            kmeans_elbow.fit(features_scaled)
            sse.append(kmeans_elbow.inertia_)
            del kmeans_elbow
            gc.collect()

        if not sse:
            print("エルボー法の計算ができませんでした。k=3で続行します。")
            optimal_k_val = 3
        else:
            plt.figure(figsize=(8, 5))
            plt.plot(k_range, sse, marker='o')
            plt.xlabel("クラスタ数 (k)")
            plt.ylabel("SSE (クラスタ内誤差平方和)")
            plt.title("エルボー法による最適なkの探索")
            plt.xticks(list(k_range))
            plt.grid(True)
            elbow_plot_path = os.path.join(os.path.dirname(OUTPUT_EXCEL_PATH), "elbow_plot.png")
            plt.savefig(elbow_plot_path)
            print(f"エルボー法のグラフを {elbow_plot_path} に保存しました。")
            plt.close()
            gc.collect()

            optimal_k_val = 3 # デフォルト
            try:
                from kneed import KneeLocator
                if len(list(k_range)) >= 3 and len(sse) >=3:
                    kl = KneeLocator(list(k_range), sse, curve='convex', direction='decreasing', S=1.0)
                    optimal_k_val = kl.elbow if kl.elbow else 3
                    print(f"KneeLocatorによる推奨k: {optimal_k_val}")
                else:
                    print("データポイントが不足しているため、KneeLocatorは使用できません。デフォルトのk=3を使用します。")
            except ImportError:
                print("kneedライブラリが見つかりません。`pip install kneed`でインストールしてください。エルボー点の自動検出はスキップします。")
                if len(sse) > 1:
                    diff1 = np.diff(sse)
                    if len(diff1) > 1:
                        diff2 = np.diff(diff1)
                        try: optimal_k_val = np.argmax(diff2) + 2
                        except ValueError: optimal_k_val = 3
                    else: optimal_k_val = 3
                else: optimal_k_val = 1

        if optimal_k_val > len(texts_to_cluster):
            optimal_k_val = len(texts_to_cluster)
        if optimal_k_val == 0 and features_scaled.shape[0] > 0:
            optimal_k_val = 1

        print(f"k={optimal_k_val} でk-meansクラスタリングを実行中...")
        kmeans = KMeans(n_clusters=optimal_k_val, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(features_scaled)
        del kmeans # メモリ解放
        gc.collect()


    print(f"クラスタリング結果を '{OUTPUT_EXCEL_PATH}' に出力中...")
    # texts_to_cluster は既にリストなので、そのまま使用
    df_results = pd.DataFrame({'元のテキスト': texts_to_cluster})
    # features_matrix_imputed は標準化前・欠損値補完済みの値
    df_features = pd.DataFrame(features_matrix_imputed, columns=feature_names)
    df_results = pd.concat([df_results, df_features], axis=1)
    if len(cluster_labels) == len(df_results): # ラベル数とデータ数が一致するか確認
        df_results['クラスタラベル'] = cluster_labels
    else:
        print(f"警告: クラスタラベル数({len(cluster_labels)})とデータ数({len(df_results)})が一致しません。クラスタラベルは追加されません。")

    try:
        df_results.to_excel(OUTPUT_EXCEL_PATH, index=False)
        print("Excelファイルへの出力完了。")
    except Exception as e:
        print(f"エラー: Excelファイルへの出力中にエラーが発生しました: {e}")
    del df_results, df_features
    gc.collect()

    if features_scaled.shape[1] >= 2 and features_scaled.shape[0] > 0:
        print("PCAで2次元に削減し、クラスタリング結果を可視化中...")
        pca = PCA(n_components=2, random_state=42)
        features_pca = pca.fit_transform(features_scaled)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)

        unique_labels_viz = np.unique(cluster_labels)
        # scatter.legend_elements() は cmap を使ったプロットの場合に有効
        # ここでは、ユニークなラベルに基づいて凡例を生成する方が確実
        handles_viz = [plt.scatter([],[], marker='o', color=scatter.cmap(scatter.norm(label)), label=f'クラスタ {label}') for label in unique_labels_viz]

        if handles_viz: # ハンドルが生成された場合のみ凡例表示
             plt.legend(handles=handles_viz, title="クラスタ")
        else:
            print("警告: 可視化グラフの凡例が生成できませんでした。")


        plt.title(f'k-means クラスタリング結果 (k={optimal_k_val}, PCAで2次元化)')
        plt.xlabel("PCA 第1主成分")
        plt.ylabel("PCA 第2主成分")
        plt.grid(True)
        try:
            plt.savefig(OUTPUT_GRAPH_PATH)
            print(f"可視化グラフを '{OUTPUT_GRAPH_PATH}' に保存しました。")
        except Exception as e:
            print(f"エラー: 可視化グラフの保存中にエラーが発生しました: {e}")
        plt.close()
        del features_pca, scatter, handles_viz, unique_labels_viz
        gc.collect()
    else:
        print("特徴量が2次元未満またはデータがないため、PCAによる可視化はスキップします。")

    del features_scaled, texts_to_cluster, cluster_labels
    gc.collect()
    print("--- メイン処理完了 ---")

if __name__ == "__main__":
    # ローカル環境でCaboChaとCRF++がシステムにインストール済み、
    # または適切なパスが通っていることを前提とします。
    # もしソースからビルド・インストールする場合は、前回の回答の
    # check_and_install_cabocha_dependencies() 関数を呼び出すか、
    # 事前に手動でビルド・インストールを行ってください。
    main()