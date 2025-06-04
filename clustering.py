import os
import sys
import pickle
import re
import math
import time
import glob
from tqdm.auto import tqdm  # .auto は notebook と script 両対応

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib #日本語文字化け対策

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# SudachiPy
from sudachipy import dictionary as sudachi_dictionary
from sudachipy import tokenizer as sudachi_tokenizer

# CaboCha
import CaboCha

# Hugging Face Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# pykakasi
import pykakasi

# Levenshtein
import Levenshtein

# --- グローバル設定 ---
# ファイルパス (Kaggle環境に合わせて調整してください)
INPUT_EXCEL_PATH = '/kaggle/input/input-for-clustering/input_clustering.xlsx'
JAPANESE_WORD_LIST_PATH = '/kaggle/input/clustering-data/japanese_word_list.txt'
NGRAM_DATA_BASE_PATH = '/kaggle/input/hindo10' # hindo10データセットのルート
PICKLE_CHUNK_DIR = '/kaggle/working/pickle_chunks/'
OUTPUT_EXCEL_PATH = '/kaggle/working/clustering_results.xlsx'
OUTPUT_GRAPH_PATH = '/kaggle/working/clustering_visualization.png'

# HuggingFaceモデル名
HF_MODEL_NAME = "rinna/japanese-gpt2-xsmall"

# 特徴量計算の閾値など
EDIT_DISTANCE_THRESHOLD_LOG_PROB = -10.8799
WORD_3GRAM_THRESHOLD_LOG_PROB = -8.8
WORD_3GRAM_ALPHA = 0.001 # スムージング用

# --- 0. 初期化処理 ---
print("--- 0. 初期化処理開始 ---")

# SudachiPyの初期化
print("SudachiPy 初期化中...")
try:
    sudachi_tokenizer_dict_core = sudachi_dictionary.Dictionary(dict_type="core").create()
    sudachi_dict_obj_core_for_lookup = sudachi_dictionary.Dictionary(dict_type="core")
    # 単語N-gram用 (SudachiDict-full を推奨するコードがあったため、ここではcoreで統一するか検討)
    # 今回はN-gramのSudachiもcoreで統一してみる (元コードはfullだったが、整合性のため)
    sudachi_tokenizer_for_ngram = sudachi_dictionary.Dictionary(dict_type="core").create()
    print("SudachiPy 初期化完了。")
except Exception as e:
    print(f"SudachiPyの初期化に失敗しました: {e}")
    sys.exit(1)

# CaboChaの初期化
print("CaboCha 初期化中...")
try:
    cabocha_parser = CaboCha.Parser() # デフォルト設定で初期化
    print("CaboCha 初期化完了。")
except Exception as e:
    print(f"CaboChaの初期化に失敗しました: {e}")
    sys.exit(1)

# Hugging FaceモデルとTokenizerのロード (編集距離用)
print(f"Hugging Faceモデル {HF_MODEL_NAME} をロード中...")
try:
    hf_tokenizer_for_edit_dist = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    hf_model_for_edit_dist = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME)
    hf_model_for_edit_dist.eval() # 推論モード
    print("Hugging Faceモデルのロード完了。")
except Exception as e:
    print(f"Hugging FaceモデルまたはTokenizerのロード中にエラーが発生しました: {e}")
    sys.exit(1)

# pykakasiの初期化 (編集距離用)
print("pykakasi 初期化中...")
try:
    kakasi_converter_for_edit_dist = pykakasi.kakasi()
    kakasi_converter_for_edit_dist.setMode("K", "E") # カタカナを英語（ローマ字）に変換
    kakasi_converter_for_edit_dist.setMode("r", "Hepburn") # ヘボン式ローマ字
    print("pykakasi 初期化完了。")
except Exception as e:
    print(f"pykakasiの初期化に失敗しました: {e}")
    sys.exit(1)

# 編集距離計算用の辞書データロード
print(f"編集距離用辞書データ '{JAPANESE_WORD_LIST_PATH}' をロード中...")
all_dictionary_data_for_edit_dist = []
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


# --- 1. 特徴量計算関数群 ---
print("--- 1. 特徴量計算関数の定義開始 ---")

# 特徴量1: 辞書照合
def check_word_in_sudachi_dictionary_revised(word, dict_obj):
    morphemes = dict_obj.lookup(word)
    return len(morphemes) > 0

def calculate_dict_feature(text, tokenizer_obj, dict_obj_for_lookup_func):
    if not text or not isinstance(text, str): return 1 # 空や不正な入力は「辞書にない」扱い
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
        if noun_count == 0:
            return 0
        return 1 if found_undefined_noun else 0
    except Exception:
        return 1 # エラー時も「辞書にない」扱い

# 特徴量2 & 3: Token依存距離の分散, 係り受けスコアの分散 (CaboCha)
def calculate_cabocha_features(text, parser):
    if not text or not isinstance(text, str): return 0.0, 0.0
    try:
        tree = parser.parse(text)
        chunks = [tree.chunk(i) for i in range(tree.chunk_size())]
        if not chunks: return 0.0, 0.0

        token_to_chunk = [None] * tree.token_size() # tree.size() から tree.token_size() に変更
        for ci, ch in enumerate(chunks):
            start = ch.token_pos
            size = ch.token_size
            for ti in range(start, start + size):
                if ti < len(token_to_chunk): # 配列範囲チェック
                    token_to_chunk[ti] = ci

        token_distances = []
        for ti in range(tree.token_size()): # tree.size() から tree.token_size() に変更
            if ti >= len(token_to_chunk) or token_to_chunk[ti] is None: continue # 不正なインデックスやマッピングなしはスキップ
            ci = token_to_chunk[ti]
            if ci >= len(chunks): continue # ciがchunksの範囲外ならスキップ

            head_ci = chunks[ci].link
            if head_ci >= 0 and head_ci < len(chunks): # head_ciも範囲チェック
                head_token_pos = chunks[head_ci].token_pos
                token_distances.append(abs(ti - head_token_pos))

        var_td = 0.0
        if token_distances:
            mean_td = sum(token_distances) / len(token_distances)
            var_td = sum((d - mean_td) ** 2 for d in token_distances) / len(token_distances)

        scores = [ch.score for ch in chunks if ch.score is not None] # スコアがNoneの場合を除外
        var_sc = 0.0
        if scores:
            mean_sc = sum(scores) / len(scores)
            var_sc = sum((s - mean_sc) ** 2 for s in scores) / len(scores)
        
        return var_td, var_sc
    except Exception:
        return 0.0, 0.0


# 特徴量4: 熟語の編集距離 (Hugging Face)
# (提供された長いコードから必要な部分を抜粋・再構成)
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
    
    min_found_dist = sys.maxsize
    temp_candidates = []
    for kanji, reading in dict_to_search:
        if not kanji: continue
        dist = Levenshtein.distance(segment_kanji, kanji)
        if dist <= max_dist:
             temp_candidates.append({'kanji': kanji, 'reading': reading, 'dist': dist})
        if dist < min_found_dist:
            min_found_dist = dist

    # max_distで見つからなければ、最小距離のものを採用
    if not temp_candidates and min_found_dist != sys.maxsize:
         for kanji, reading in dict_to_search: # 再度全探索
            if not kanji: continue
            dist = Levenshtein.distance(segment_kanji, kanji)
            if dist == min_found_dist:
                temp_candidates.append({'kanji': kanji, 'reading': reading, 'dist': dist})
    
    # 重複除去とソート
    seen_kanji = set()
    for cand_info in sorted(temp_candidates, key=lambda x: x['dist']):
        if cand_info['kanji'] not in seen_kanji:
            candidates.append((cand_info['kanji'], cand_info['reading'], cand_info['dist']))
            seen_kanji.add(cand_info['kanji'])
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
            best_cand_reading = reading # 今回は使わないが保持
            best_kanji_dist_orig = kanji_dist # 元の漢字編集距離
    
    if best_cand_kanji is None: return None, sys.maxsize
    return best_cand_kanji, best_kanji_dist_orig # 修正候補漢字と、元の入力との漢字編集距離

def calculate_edit_distance_feature(text, tokenizer, model, dictionary_data, kakasi_cv, threshold_log_p):
    if not text or not isinstance(text, str): return 0.0
    try:
        inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        input_ids = inputs["input_ids"]
        offset_mapping = inputs["offset_mapping"].squeeze().tolist()

        kanji_segments_info = []
        for match in re.finditer(r'[\u4E00-\u9FFF]{2,}', text):
            kanji_segments_info.append({'segment': match.group(0), 'start_char': match.start(), 'end_char': match.end()})

        if not kanji_segments_info: return 0.0

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
        log_probs_full_sequence = torch.log_softmax(logits, dim=-1).squeeze()

        segment_scores = []
        for seg_info in kanji_segments_info:
            segment_str = seg_info['segment']
            seg_start_char_idx = seg_info['start_char']
            hf_indices_for_segment = []
            for hf_token_idx, (tok_char_start, tok_char_end) in enumerate(offset_mapping):
                if tok_char_start >= seg_start_char_idx and tok_char_end <= seg_info['end_char']:
                    if hf_token_idx > 0: hf_indices_for_segment.append(hf_token_idx)
            
            if not hf_indices_for_segment:
                avg_log_prob = -float('inf')
            else:
                log_prob_sum = 0
                valid_tokens = 0
                for hf_idx in hf_indices_for_segment:
                    token_id = input_ids[0, hf_idx].item()
                    if hf_idx -1 < log_probs_full_sequence.shape[0]:
                        log_prob_sum += log_probs_full_sequence[hf_idx-1, token_id].item()
                        valid_tokens +=1
                avg_log_prob = log_prob_sum / valid_tokens if valid_tokens > 0 else -float('inf')
            segment_scores.append({'segment': segment_str, 'avg_log_prob': avg_log_prob})
        
        if not segment_scores: return 0.0

        min_score = float('inf')
        segment_with_min_score_str = None
        valid_scores = [s for s in segment_scores if s['avg_log_prob'] > -float('inf')]
        if not valid_scores : # 全て-infの場合、適当なものを一つ選ぶか、0を返すか。ここでは0を返す。
            # もし、-inf でも処理を続けたいなら、最初のセグメントを選ぶなど。
            # segment_with_min_score_str = segment_scores[0]['segment']
            # min_score = -float('inf')
            return 0.0
        
        segment_with_min_score = min(valid_scores, key=lambda x: x['avg_log_prob'])
        min_score = segment_with_min_score['avg_log_prob']
        segment_with_min_score_str = segment_with_min_score['segment']


        if min_score < threshold_log_p and segment_with_min_score_str:
            target_segment = segment_with_min_score_str
            
            est_reading = estimate_segment_reading_for_edit_dist(target_segment, dictionary_data)
            if not est_reading: return Levenshtein.distance(target_segment, "") # 読み推定失敗時は空文字列との距離

            est_reading_romaji = katakana_to_romaji_for_edit_dist(est_reading, kakasi_cv)
            if est_reading_romaji == "___romaji_error___": return Levenshtein.distance(target_segment, "")

            initial_candidates = find_kanji_candidates_by_distance_for_edit_dist(target_segment, dictionary_data)
            if not initial_candidates: return Levenshtein.distance(target_segment, "")

            best_cand_kanji, _ = find_best_candidate_romaji_rerank_for_edit_dist(est_reading_romaji, initial_candidates, kakasi_cv)
            
            if best_cand_kanji:
                return float(Levenshtein.distance(target_segment, best_cand_kanji))
            else: # 最適候補なし
                return float(Levenshtein.distance(target_segment, ""))
        return 0.0 # 閾値以上、または不自然箇所なし
    except Exception:
        return 0.0 # エラー時は0


# 特徴量5: 単語3-gram
# (N-gram前処理と計算部分の抜粋・再構成)

# N-gram前処理 (pre_word3gram.py相当)
def load_ngrams_from_file_for_pre(filepath, n):
    ngram_counts = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    words = parts[0].split(' ')
                    if len(words) == n:
                        try: ngram_counts[tuple(words)] = int(parts[1])
                        except ValueError: pass
    except Exception: pass # エラー時は空辞書
    return ngram_counts

def calculate_total_vocabulary_size_for_pre(ngram_data_dirs, files_limit_per_type=None):
    all_words = set()
    print("N-gramから語彙サイズ(V)を計算中...")
    for n_type, dir_path in ngram_data_dirs.items():
        if not os.path.isdir(dir_path): continue
        file_paths = sorted(glob.glob(os.path.join(dir_path, f"{n_type}gm-*")))
        if files_limit_per_type: file_paths = file_paths[:files_limit_per_type]
        
        for filepath in tqdm(file_paths, desc=f"V計算 ({n_type}-grams)", leave=False):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) == 2:
                            for word in parts[0].split(' '):
                                if word: all_words.add(word)
            except Exception: pass
    V = len(all_words)
    print(f"計算された総語彙サイズ (V): {V}")
    return V

def process_and_chunk_ngrams_for_pre(directory_path, n, pickle_out_dir, total_vocab_size, files_limit=None):
    if not os.path.isdir(directory_path): return False
    os.makedirs(pickle_out_dir, exist_ok=True)
    file_paths = sorted(glob.glob(os.path.join(directory_path, f"{n}gm-*")))
    if files_limit: file_paths = file_paths[:files_limit]
    if not file_paths: return False

    print(f"{n}-gramデータをチャンク化中 ({len(file_paths)}ファイル対象)...")
    ngram_first_word_index = {}
    for filepath in tqdm(file_paths, desc=f"チャンク化 ({n}-grams)", leave=False):
        ngrams_in_file = load_ngrams_from_file_for_pre(filepath, n)
        if not ngrams_in_file: continue

        base_filename = os.path.basename(filepath).replace('.txt', '')
        output_chunk_filename = f"{n}gram_{base_filename}.pkl"
        data_to_save = {'n': n, 'counts': ngrams_in_file, 'total_vocab_size': total_vocab_size}
        
        try:
            with open(os.path.join(pickle_out_dir, output_chunk_filename), "wb") as f_out:
                pickle.dump(data_to_save, f_out, pickle.HIGHEST_PROTOCOL)
            for ngram_tuple in ngrams_in_file.keys():
                first_word = ngram_tuple[0]
                ngram_first_word_index.setdefault(first_word, set()).add(output_chunk_filename)
        except Exception: pass # 保存エラーはスキップ

    index_filename = f"{n}gram_first_word_index.pkl"
    try:
        with open(os.path.join(pickle_out_dir, index_filename), "wb") as f_idx:
            pickle.dump({word: list(files) for word, files in ngram_first_word_index.items()}, f_idx, pickle.HIGHEST_PROTOCOL)
        print(f"{n}-gramのインデックスを保存: {index_filename}")
    except Exception:
        print(f"エラー: {n}-gramのインデックス保存失敗: {index_filename}")
        return False
    return True

def run_ngram_preprocessing(base_ngram_path, out_pickle_dir, files_limit=None):
    print("--- N-gramデータの前処理開始 ---")
    if not os.path.exists(base_ngram_path):
        print(f"エラー: N-gramデータパス '{base_ngram_path}' が見つかりません。")
        return None, {}, {}, [], [] # V, bi_idx, tri_idx, bi_paths, tri_paths

    # 既に前処理済みかチェック (簡易的)
    required_files_exist = True
    for n_val in [2,3]:
        if not os.path.exists(os.path.join(out_pickle_dir, f"{n_val}gram_first_word_index.pkl")):
            required_files_exist = False
            break
        if not glob.glob(os.path.join(out_pickle_dir, f"{n_val}gram_*.pkl")): # チャンクファイルも確認
            required_files_exist = False
            break
            
    if required_files_exist:
        print("N-gramの前処理済みファイルが見つかりました。ロードします。")
        # Vのロード (最初のチャンクファイルから)
        v_loaded = 0
        try:
            # 2gramか3gramのどちらかのチャンクからVをロード
            first_chunk_path = None
            pkl_files = glob.glob(os.path.join(out_pickle_dir, "2gram_*.pkl"))
            if pkl_files: first_chunk_path = pkl_files[0]
            else:
                pkl_files = glob.glob(os.path.join(out_pickle_dir, "3gram_*.pkl"))
                if pkl_files: first_chunk_path = pkl_files[0]

            if first_chunk_path:
                with open(first_chunk_path, 'rb') as f_chk:
                    sample_chunk = pickle.load(f_chk)
                    v_loaded = sample_chunk.get('total_vocab_size', 0)
                print(f"ロードされた総語彙サイズ (V): {v_loaded}")
            else:
                 print("警告: Vをロードするためのチャンクファイルが見つかりません。V=0で続行します。")


        except Exception as e_load_v:
            print(f"警告: チャンクからのVのロードに失敗: {e_load_v}。V=0で続行します。")
            v_loaded = 0 # フォールバック

        bigram_idx_loaded, trigram_idx_loaded = {}, {}
        try:
            with open(os.path.join(out_pickle_dir, '2gram_first_word_index.pkl'), 'rb') as f: bigram_idx_loaded = pickle.load(f)
            with open(os.path.join(out_pickle_dir, '3gram_first_word_index.pkl'), 'rb') as f: trigram_idx_loaded = pickle.load(f)
        except Exception as e_load_idx:
             print(f"警告: N-gramインデックスのロードに失敗: {e_load_idx}。空のインデックスで続行します。")
        
        bigram_chunk_paths_loaded = sorted(glob.glob(os.path.join(out_pickle_dir, "2gram_*.pkl")))
        trigram_chunk_paths_loaded = sorted(glob.glob(os.path.join(out_pickle_dir, "3gram_*.pkl")))
        print("--- N-gramデータの前処理スキップ (ロード完了) ---")
        return v_loaded, bigram_idx_loaded, trigram_idx_loaded, bigram_chunk_paths_loaded, trigram_chunk_paths_loaded

    # 前処理実行
    ngram_dirs = {2: base_ngram_path, 3: base_ngram_path}
    total_v = calculate_total_vocabulary_size_for_pre(ngram_dirs, files_limit_per_type=files_limit)
    if total_v == 0:
        print("警告: 総語彙サイズ(V)が0です。N-gram特徴量は正しく計算されない可能性があります。")

    process_and_chunk_ngrams_for_pre(base_ngram_path, 2, out_pickle_dir, total_v, files_limit=files_limit)
    process_and_chunk_ngrams_for_pre(base_ngram_path, 3, out_pickle_dir, total_v, files_limit=files_limit)
    
    print("N-gramデータをロード中...")
    bigram_idx, trigram_idx = {}, {}
    try:
        with open(os.path.join(out_pickle_dir, '2gram_first_word_index.pkl'), 'rb') as f: bigram_idx = pickle.load(f)
        with open(os.path.join(out_pickle_dir, '3gram_first_word_index.pkl'), 'rb') as f: trigram_idx = pickle.load(f)
    except Exception:
        print("警告: 前処理後のN-gramインデックスのロードに失敗しました。")

    bigram_paths = sorted(glob.glob(os.path.join(out_pickle_dir, "2gram_*.pkl")))
    trigram_paths = sorted(glob.glob(os.path.join(out_pickle_dir, "3gram_*.pkl")))
    print("--- N-gramデータの前処理完了 ---")
    return total_v, bigram_idx, trigram_idx, bigram_paths, trigram_paths

# N-gramカウント取得 (word3gram.py相当)
_ngram_cache = {} # メモリリークを避けるため、適宜クリアするかサイズ制限が必要だが、今回は簡易的に
_CHUNK_CACHE_SIZE = 10 # 例: 直近10チャンクをキャッシュ
_chunk_cache = {} 
_chunk_cache_lru = []

def get_ngram_count_from_pickle(ngram_tuple, n_type, chunk_dir, index_data, all_chunk_paths_for_type):
    if not ngram_tuple: return 0
    
    # キャッシュから取得
    cache_key = (ngram_tuple, n_type)
    if cache_key in _ngram_cache:
        return _ngram_cache[cache_key]

    first_word = ngram_tuple[0]
    relevant_chunk_filenames = index_data.get(first_word)
    
    paths_to_search = all_chunk_paths_for_type
    if relevant_chunk_filenames:
        paths_to_search = [os.path.join(chunk_dir, fname) for fname in sorted(list(relevant_chunk_filenames)) if os.path.exists(os.path.join(chunk_dir, fname))]

    for chunk_path in paths_to_search:
        # チャンクキャッシュ
        if chunk_path in _chunk_cache:
            chunk_data = _chunk_cache[chunk_path]
            # LRU更新
            if chunk_path in _chunk_cache_lru: _chunk_cache_lru.remove(chunk_path)
            _chunk_cache_lru.append(chunk_path)
        else:
            try:
                with open(chunk_path, 'rb') as f_chk_data:
                    chunk_data = pickle.load(f_chk_data)
                _chunk_cache[chunk_path] = chunk_data
                _chunk_cache_lru.append(chunk_path)
                if len(_chunk_cache_lru) > _CHUNK_CACHE_SIZE:
                    oldest_chunk = _chunk_cache_lru.pop(0)
                    if oldest_chunk in _chunk_cache:
                        del _chunk_cache[oldest_chunk]

            except Exception: continue
        
        if 'counts' in chunk_data and ngram_tuple in chunk_data['counts']:
            count = chunk_data['counts'][ngram_tuple]
            _ngram_cache[cache_key] = count # N-gramキャッシュに保存
            return count
    
    _ngram_cache[cache_key] = 0 # 見つからなかった場合もキャッシュ（再検索を防ぐ）
    return 0

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
                             trigram_cpaths, bigram_cpaths):
    if not text or not isinstance(text, str) or V_ngram == 0: return 0
    try:
        # SudachiPy Cモードで形態素解析
        morphemes = [m.surface() for m in tokenizer_obj_ngram.tokenize(text, sudachi_tokenizer.Tokenizer.SplitMode.C)]
        if not morphemes: return 0

        # 句読点などで区切らず、そのまま3-gram確率を計算（元コードの挙動に合わせる）
        # 元のword3gram.pyでは句読点処理があったが、ここではシンプル化のため一旦全体で計算
        # 必要であれば句読点除去ロジックをここに追加
        
        # 句読点除去 (提供コードの挙動に合わせる)
        sentence_end_punctuations = {'。', '！', '？', '．', '！', '？'} # 半角も追加
        
        clean_morphemes = []
        current_segment = []
        all_log_probs = []

        for m in morphemes:
            if m in sentence_end_punctuations:
                if len(current_segment) >= 3:
                    segment_probs = calculate_score_per_ngram_for_feature(
                        current_segment, V_ngram, alpha_3gram, ngram_pk_dir,
                        trigram_idx_map, bigram_idx_map, trigram_cpaths, bigram_cpaths
                    )
                    all_log_probs.extend(segment_probs)
                current_segment = []
            else:
                current_segment.append(m)
        
        if len(current_segment) >=3: # 最後のセグメント
             segment_probs = calculate_score_per_ngram_for_feature(
                        current_segment, V_ngram, alpha_3gram, ngram_pk_dir,
                        trigram_idx_map, bigram_idx_map, trigram_cpaths, bigram_cpaths
                    )
             all_log_probs.extend(segment_probs)


        if not all_log_probs: return 0 # 3-gramが一つも生成されなかった

        if any(prob < threshold_3gram for prob in all_log_probs if prob != -float('inf')):
            return 3
        return 0
    except Exception:
        return 0 # エラー時は0

print("--- 1. 特徴量計算関数の定義完了 ---")


# --- メイン処理 ---
# def main():
#     print("--- メイン処理開始 ---")

#     # N-gram前処理の実行とデータロード
#     # Kaggleではファイル数が多いと時間がかかるため、制限をかけることも可能 (例: files_limit=10)
#     # Noneで全ファイル。初回実行時は時間がかかります。
#     # 2回目以降は pickle_chunks ディレクトリにファイルがあればスキップされます。
#     global _ngram_cache, _chunk_cache, _chunk_cache_lru # グローバルキャッシュをクリア
#     _ngram_cache, _chunk_cache, _chunk_cache_lru = {}, {}, []

#     total_vocab_size_ngram, bigram_index_ngram, trigram_index_ngram, \
#     all_bigram_chunk_paths, all_trigram_chunk_paths = run_ngram_preprocessing(
#         NGRAM_DATA_BASE_PATH, PICKLE_CHUNK_DIR, files_limit=None # 必要なら制限:例 10
#     )
#     if total_vocab_size_ngram == 0:
#         print("警告: N-gramの総語彙サイズが0です。単語3-gram特徴量は常に0になります。")


#     # 入力Excelデータの読み込み
#     print(f"入力ファイル '{INPUT_EXCEL_PATH}' を読み込み中...")
#     try:
#         df_input = pd.read_excel(INPUT_EXCEL_PATH, header=None, usecols=[0])
#         texts_to_cluster = df_input.iloc[:, 0].astype(str).tolist()
#         print(f"{len(texts_to_cluster)} 件のテキストを読み込みました。")
#         if not texts_to_cluster:
#             print("エラー: 入力ファイルにクラスタリング対象のテキストが含まれていません。")
#             return
#     except FileNotFoundError:
#         print(f"エラー: 入力ファイル '{INPUT_EXCEL_PATH}' が見つかりません。")
#         return
#     except Exception as e:
#         print(f"エラー: 入力ファイルの読み込み中にエラーが発生しました: {e}")
#         return

#     # 特徴量抽出
#     print("特徴量抽出を開始します...")
#     all_features = []
#     feature_names = ["熟語編集距離", "単語3-gram異常度", "Token依存距離分散", "係り受けスコア分散", "辞書照合"]
    
#     # tqdmで進捗表示
#     for text_idx, text_content in tqdm(enumerate(texts_to_cluster), total=len(texts_to_cluster), desc="特徴量抽出中"):
#         # キャッシュクリア (各テキストごとに行うか、N回ごとに行うか検討)
#         # ここでは簡易的に毎回クリアはしないが、大規模データでは必要になる可能性
#         if text_idx % 100 == 0 and text_idx > 0 : # 100テキストごとにN-gramキャッシュをクリア
#             _ngram_cache.clear()
#             _chunk_cache.clear()
#             _chunk_cache_lru.clear()


#         f_edit_dist = calculate_edit_distance_feature(
#             text_content, hf_tokenizer_for_edit_dist, hf_model_for_edit_dist,
#             all_dictionary_data_for_edit_dist, kakasi_converter_for_edit_dist,
#             EDIT_DISTANCE_THRESHOLD_LOG_PROB
#         )
        
#         f_3gram = calculate_3gram_feature(
#             text_content, sudachi_tokenizer_for_ngram, total_vocab_size_ngram, WORD_3GRAM_ALPHA,
#             WORD_3GRAM_THRESHOLD_LOG_PROB, PICKLE_CHUNK_DIR, trigram_index_ngram, bigram_index_ngram,
#             all_trigram_chunk_paths, all_bigram_chunk_paths
#         )
        
#         f_token_dist, f_dep_score = calculate_cabocha_features(text_content, cabocha_parser)
        
#         f_dict = calculate_dict_feature(text_content, sudachi_tokenizer_dict_core, sudachi_dict_obj_core_for_lookup)
        
#         all_features.append([f_edit_dist, f_3gram, f_token_dist, f_dep_score, f_dict])

#     features_matrix = np.array(all_features)
    
#     # NaN値の補完 (エラー時に特定の値でなくNaNを返すようにした場合)
#     # 今回はエラー時に0などを返しているので、imputerは不要かもしれないが、念のため
#     imputer = SimpleImputer(strategy='mean') # または 'median', 'most_frequent', 'constant'
#     features_matrix_imputed = imputer.fit_transform(features_matrix)

#     # 標準化
#     print("特徴量を標準化中...")
#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(features_matrix_imputed)

#     # エルボー法で最適なkを決定
#     print("エルボー法で最適なクラスタ数 (k) を探索中...")
#     sse = []
#     k_range = range(1, 11) # 1から10クラスタまで試す (データ数に応じて調整)
#     if len(texts_to_cluster) < 10: # データ数が少ない場合はkの上限を調整
#         k_range = range(1, max(2, len(texts_to_cluster)))


#     for k_val in tqdm(k_range, desc="エルボー法"):
#         if k_val == 0: continue # k=0はkmeansでエラー
#         if k_val > features_scaled.shape[0]: # kがサンプル数を超える場合
#             print(f"k={k_val} はサンプル数({features_scaled.shape[0]})を超えるためスキップします。")
#             break
#         kmeans_elbow = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
#         kmeans_elbow.fit(features_scaled)
#         sse.append(kmeans_elbow.inertia_)

#     # エルボープロット (ファイル保存はしない、コンソール表示用)
#     plt.figure(figsize=(8, 5))
#     plt.plot(k_range, sse, marker='o')
#     plt.xlabel("クラスタ数 (k)")
#     plt.ylabel("SSE (クラスタ内誤差平方和)")
#     plt.title("エルボー法による最適なkの探索")
#     plt.xticks(list(k_range))
#     plt.grid(True)
#     # plt.show() # Jupyter Notebookなどでは表示される
#     elbow_plot_path = "/kaggle/working/elbow_plot.png"
#     plt.savefig(elbow_plot_path)
#     print(f"エルボー法のグラフを {elbow_plot_path} に保存しました。")
#     plt.close()


#     # 最適なkの選択 (ここではエルボーが明確でない場合もあるため、経験的にまたは自動検出ロジックが必要)
#     # 簡単な自動検出: 変化率が最も大きい点の次、など。今回は手動で選択するか、固定値を使う。
#     # 例として、k=3 を仮定 (エルボープロットを見て調整してください)
#     # 簡易的なエルボー点検出 (KneeLocatorなどを使うのがより堅牢)
#     if len(sse) > 2:
#         # 変化量の変化率を計算 (2階差分のようなもの)
#         # sse_diff = np.diff(sse, 2)
#         # optimal_k_val = np.argmax(sse_diff) + 2 # 差分のインデックスに注意
#         # もっと単純に、SSEの減少が緩やかになる点を探す
#         # 簡易的に、SSEリストからエルボー点を探す (ここでは固定値を使用)
#         optimal_k_val = 3 # デフォルト値。エルボープロットを見て調整してください。
#         if len(texts_to_cluster) <=3 : optimal_k_val = max(1, len(texts_to_cluster) //2) if len(texts_to_cluster) > 1 else 1
#         if optimal_k_val == 0 and len(texts_to_cluster) > 0: optimal_k_val = 1
#         print(f"エルボープロットに基づき、最適なkとして {optimal_k_val} を選択 (必要に応じて調整してください)。")
#     elif len(texts_to_cluster) > 0:
#         optimal_k_val = max(1, len(texts_to_cluster) // 2) if len(texts_to_cluster) > 1 else 1
#         print(f"データ数が少ないため、kとして {optimal_k_val} を選択。")
#     else: # texts_to_cluster が空の場合
#         print("クラスタリング対象のデータがありません。処理を終了します。")
#         return

#     if optimal_k_val == 0 and features_scaled.shape[0] > 0: # もし0になってしまった場合のフォールバック
#         optimal_k_val = 1
#         print(f"kが0になりましたが、データが存在するためk=1で実行します。")


#     # k-meansクラスタリング実行
#     print(f"k={optimal_k_val} でk-meansクラスタリングを実行中...")
#     kmeans = KMeans(n_clusters=optimal_k_val, random_state=42, n_init='auto')
#     cluster_labels = kmeans.fit_predict(features_scaled)

#     # 結果をExcelに出力
#     print(f"クラスタリング結果を '{OUTPUT_EXCEL_PATH}' に出力中...")
#     df_results = pd.DataFrame(texts_to_cluster, columns=['元のテキスト'])
#     df_features = pd.DataFrame(features_matrix_imputed, columns=feature_names) # 標準化前の値（補完後）
#     df_results = pd.concat([df_results, df_features], axis=1)
#     df_results['クラスタラベル'] = cluster_labels
#     try:
#         df_results.to_excel(OUTPUT_EXCEL_PATH, index=False)
#         print("Excelファイルへの出力完了。")
#     except Exception as e:
#         print(f"エラー: Excelファイルへの出力中にエラーが発生しました: {e}")

#     # 結果をグラフで可視化 (PCAで2次元に削減)
#     if features_scaled.shape[1] >= 2: # 特徴量が2次元以上の場合のみPCA
#         print("PCAで2次元に削減し、クラスタリング結果を可視化中...")
#         pca = PCA(n_components=2, random_state=42)
#         features_pca = pca.fit_transform(features_scaled)

#         plt.figure(figsize=(10, 8))
#         scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
        
#         # 各クラスタの中心をプロット (オプション)
#         # centroids_pca = pca.transform(kmeans.cluster_centers_)
#         # plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, c='red', label='Centroids')

#         plt.title(f'k-means クラスタリング結果 (k={optimal_k_val}, PCAで2次元化)')
#         plt.xlabel("PCA 第1主成分")
#         plt.ylabel("PCA 第2主成分")
        
#         # 凡例の作成
#         handles, labels = scatter.legend_elements()
#         unique_labels = np.unique(cluster_labels)
#         legend_labels = [f'クラスタ {l}' for l in unique_labels]

#         if len(handles) == len(legend_labels): # ハンドルとラベルの数が一致する場合のみ凡例表示
#              plt.legend(handles, legend_labels, title="クラスタ")
#         else:
#             print(f"警告: 凡例のハンドル数({len(handles)})とラベル数({len(legend_labels)})が一致しません。凡例は表示されません。")


#         plt.grid(True)
#         try:
#             plt.savefig(OUTPUT_GRAPH_PATH)
#             print(f"可視化グラフを '{OUTPUT_GRAPH_PATH}' に保存しました。")
#         except Exception as e:
#             print(f"エラー: 可視化グラフの保存中にエラーが発生しました: {e}")
#         plt.close()
#     else:
#         print("特徴量が2次元未満のため、PCAによる可視化はスキップします。")

#     print("--- メイン処理完了 ---")

# if __name__ == "__main__":
#     # Kaggle Notebooksでは直接 main() を呼び出す
#     # ローカル環境でスクリプトとして実行する場合もこのままでOK
#     main()