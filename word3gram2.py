import os
import pickle
import numpy as np
from sudachipy import tokenizer
from sudachipy import dictionary
import time
import glob
import sys
from tqdm.notebook import tqdm

# SudachiPyの初期化
SUD_DICT_TYPE = 'full'
tokenizer_obj = dictionary.Dictionary(dict_type=SUD_DICT_TYPE).create()
mode = tokenizer.Tokenizer.SplitMode.C

# N-gramデータチャンクのパス
pickle_chunk_dir = os.path.expanduser('/kaggle/working/pickle_chunks/')

# --- N-gramインデックスをロード ---
bigram_index = {}
trigram_index = {}
total_vocab_size = 0

print(f"Loading N-gram indexes from: {pickle_chunk_dir}")

try:
    with open(os.path.join(pickle_chunk_dir, '2gram_first_word_index.pkl'), 'rb') as f:
        bigram_index = pickle.load(f)
    print(f"Loaded 2-gram index with {len(bigram_index)} entries.")
except FileNotFoundError:
    print("Warning: 2-gram index file (2gram_first_word_index.pkl) not found. Falling back to full scan for 2-grams. This will be slow.")
except Exception as e:
    print(f"Error loading 2-gram index: {e}. Falling back to full scan for 2-grams. This will be slow.")

try:
    with open(os.path.join(pickle_chunk_dir, '3gram_first_word_index.pkl'), 'rb') as f:
        trigram_index = pickle.load(f)
    print(f"Loaded 3-gram index with {len(trigram_index)} entries.")
except FileNotFoundError:
    print("Warning: 3-gram index file (3gram_first_word_index.pkl) not found. Falling back to full scan for 3-grams. This will be slow.")
except Exception as e:
    print(f"Error loading 3-gram index: {e}. Falling back to full scan for 3-grams. This will be slow.")

bigram_chunk_paths = sorted(glob.glob(os.path.join(pickle_chunk_dir, "2gram_*.pkl")))
trigram_chunk_paths = sorted(glob.glob(os.path.join(pickle_chunk_dir, "3gram_*.pkl")))

if not bigram_chunk_paths and not trigram_chunk_paths:
    print(f"Error: No N-gram chunk files found in {pickle_chunk_dir}. Please run pre_word3gram.py first.")
    sys.exit(1)

try:
    if bigram_chunk_paths:
        with open(bigram_chunk_paths[0], 'rb') as f:
            sample_chunk_data = pickle.load(f)
            if 'total_vocab_size' in sample_chunk_data:
                total_vocab_size = sample_chunk_data['total_vocab_size']
    elif trigram_chunk_paths:
        with open(trigram_chunk_paths[0], 'rb') as f:
            sample_chunk_data = pickle.load(f)
            if 'total_vocab_size' in sample_chunk_data:
                total_vocab_size = sample_chunk_data['total_vocab_size']
    
    if total_vocab_size == 0:
        print("Warning: Could not determine total vocabulary size (V) from chunk files. Ensure V is saved in chunks.")
    else:
        print(f"Total vocabulary size (V) obtained from chunks: {total_vocab_size}")

except Exception as e:
    print(f"Error obtaining total vocabulary size (V) from chunk files: {e}. V will remain 0 or an invalid value.")

# --- N-gramカウント取得関数 ---
def get_ngram_count(ngram_tuple, n_type, chunk_dir, current_index, all_chunk_paths):
    if not ngram_tuple:
        return 0

    first_word = ngram_tuple[0]
    relevant_chunk_filenames = current_index.get(first_word)
    
    if relevant_chunk_filenames:
        relevant_chunk_paths = [os.path.join(chunk_dir, fname) for fname in sorted(list(relevant_chunk_filenames))]
    else:
        relevant_chunk_paths = all_chunk_paths

    for chunk_path in relevant_chunk_paths:
        try:
            with open(chunk_path, 'rb') as f:
                chunk_data = pickle.load(f)
            
            if 'counts' in chunk_data and isinstance(chunk_data['counts'], dict):
                if ngram_tuple in chunk_data['counts']: 
                    count = chunk_data['counts'][ngram_tuple]
                    del chunk_data
                    return count
            del chunk_data
        except Exception as e:
            continue
    
    return 0

# --- 予測スコア計算関数 (3-gramごとの確率リストを返すように変更) ---
# この関数は、句点等の区切り文字を含まない形態素リストを受け取ることを想定
def calculate_score_per_ngram(text_words, V, alpha):
    """
    与えられた単語列の各3-gramに対する条件付き対数確率のリストを返す。
    """
    if len(text_words) < 3:
        return []

    log_probs_per_ngram = []

    for i in range(len(text_words) - 2):
        w1, w2, w3 = text_words[i], text_words[i+1], text_words[i+2]
        
        raw_count_w1_w2_w3 = get_ngram_count((w1, w2, w3), 3, pickle_chunk_dir, trigram_index, trigram_chunk_paths)
        raw_count_w1_w2 = get_ngram_count((w1, w2), 2, pickle_chunk_dir, bigram_index, bigram_chunk_paths)

        numerator = raw_count_w1_w2_w3 + alpha
        denominator = raw_count_w1_w2 + alpha * V

        if denominator == 0:
            log_prob = -float('inf')
        else:
            log_prob = np.log(numerator) - np.log(denominator)
        
        log_probs_per_ngram.append(log_prob)
        
    return log_probs_per_ngram

# --- 単一文章の折れ線グラフ描画関数 (アスキーアート) ---
def plot_single_text_line_graph(input_text, ngram_log_probs, morphemes_for_display, alpha, graph_width=70, graph_height=20):
    """
    一つの入力文章における各3-gramの条件付き対数確率をアスキーアートの折れ線グラフで表示する。
    横軸は3-gramの開始形態素インデックス、縦軸は対数確率。
    グラフの下に対応する3-gramの単語を表示する。
    """
    if not ngram_log_probs:
        print(f"\nInput: \"{input_text}\"")
        print("  No 3-grams to plot for this text.")
        return

    print(f"\nInput: \"{input_text}\"")
    print(f"3-gram Conditional Log Probability (Add-alpha Smoothed, alpha={alpha})")

    # 有効なスコアのみを抽出
    valid_probs = [p for p in ngram_log_probs if p != -float('inf')]
    if not valid_probs:
        print("  No valid 3-gram probabilities to plot.")
        return

    min_prob = min(valid_probs)
    max_prob = max(valid_probs)

    # グラフのY軸ラベルの準備
    y_labels = []
    if max_prob == min_prob:
        y_labels = [f"{max_prob:.2f}"] * graph_height
    else:
        for i in range(graph_height):
            val = min_prob + (max_prob - min_prob) * (i / (graph_height - 1))
            y_labels.append(f"{val:.2f}")
        y_labels.reverse() # 最上段が最大値になるように反転

    # グラフ描画エリアの初期化
    graph_chars = [[' ' for _ in range(graph_width)] for _ in range(graph_height)]

    # 各3-gramの確率をグラフにプロット
    # plot_single_text_line_graph には、元の形態素リストと、それに対応する3-gramの確率リストが渡される
    # X軸のインデックスは、元のテキスト全体での形態素インデックスに合わせる
    
    # 3-gramとそれに対応する元のテキスト全体での形態素インデックス、プロット位置を記録
    ngram_details_for_display = [] # [(元の形態素インデックス, (w1, w2, w3))]
    
    num_plottable_ngrams = len(ngram_log_probs)
    
    # 各プロット点の座標を保存
    plot_points = [] # [(x_graph_pos, y_idx, log_prob)]

    for i, log_prob in enumerate(ngram_log_probs):
        w1, w2, w3 = morphemes_for_display[i], morphemes_for_display[i+1], morphemes_for_display[i+2]
        ngram_details_for_display.append((i, (w1, w2, w3)))

        if log_prob == -float('inf'):
            continue

        if max_prob == min_prob:
            y_pos = graph_height // 2
        else:
            y_pos = int((log_prob - min_prob) / (max_prob - min_prob) * (graph_height - 1))
        
        y_idx = (graph_height - 1) - y_pos

        if num_plottable_ngrams > 1:
            x_graph_pos = int(i / (num_plottable_ngrams - 1) * (graph_width - 1))
        else:
            x_graph_pos = graph_width // 2
        
        graph_chars[y_idx][x_graph_pos] = '*'
        plot_points.append((x_graph_pos, y_idx, log_prob)) # プロット点を記録

    # 点と点の間を線で結ぶ
    plot_points.sort(key=lambda p: p[0]) # X座標でソート
    for j in range(len(plot_points) - 1):
        x1, y1, prob1 = plot_points[j]
        x2, y2, prob2 = plot_points[j+1]

        # X座標が隣接していない場合はスキップ（飛び飛びの点では線は引かない）
        if x2 - x1 < 1: # x2 == x1 も含む（同じX位置は線で結ばない）
            continue

        # Y座標の変化に応じて適切な線を描画
        if y1 == y2: # 水平線
            for x_coord in range(x1 + 1, x2):
                if graph_chars[y1][x_coord] == ' ':
                    graph_chars[y1][x_coord] = '-'
        elif y1 > y2: # 右肩上がり
            # y = ax + b で傾きを計算し、途中の点を埋める
            # 簡略化のため、斜め線記号を試行
            for x_coord in range(x1 + 1, x2):
                # 比例配分でy座標を計算
                y_interp = y1 - (y1 - y2) * ((x_coord - x1) / (x2 - x1))
                y_interp_round = int(round(y_interp))
                if 0 <= y_interp_round < graph_height and graph_chars[y_interp_round][x_coord] == ' ':
                     # 基本的には '/' で良いが、より細かく制御するなら傾きに応じて '-' や '|' も検討
                    graph_chars[y_interp_round][x_coord] = '/'
        else: # y1 < y2: # 右肩下がり
            for x_coord in range(x1 + 1, x2):
                y_interp = y1 + (y2 - y1) * ((x_coord - x1) / (x2 - x1))
                y_interp_round = int(round(y_interp))
                if 0 <= y_interp_round < graph_height and graph_chars[y_interp_round][x_coord] == ' ':
                    graph_chars[y_interp_round][x_coord] = '\\'


    # グラフの出力
    print("-" * (graph_width + 10))
    for r in range(graph_height):
        print(f"{y_labels[r]: >8} |{''.join(graph_chars[r])}|")
    
    # X軸の描画とラベル（3-gramの開始形態素インデックス）
    print(f"{' ' * 8} +{'=' * graph_width}+")
    
    # グラフの横幅にX軸の目盛りを均等に配置
    tick_indices = []
    if num_plottable_ngrams > 0:
        num_desired_ticks = min(num_plottable_ngrams, 10) # 最大10個の目盛り
        
        if num_desired_ticks > 1:
            step = (num_plottable_ngrams - 1) / (num_desired_ticks - 1)
            for k in range(num_desired_ticks):
                idx_on_log_probs_list = int(k * step)
                tick_indices.append(idx_on_log_probs_list)
        else:
            tick_indices.append(0) # 1つしかプロットがない場合

    x_axis_line1 = [' ' for _ in range(graph_width)] # 数字のラベル

    for idx_on_list in tick_indices:
        if num_plottable_ngrams > 1:
            x_graph_pos = int(idx_on_list / (num_plottable_ngrams - 1) * (graph_width - 1))
        else:
            x_graph_pos = graph_width // 2

        str_label = str(idx_on_list) # これは ngram_log_probs リスト内でのインデックス
        for k, char in enumerate(str_label):
            if x_graph_pos + k < graph_width:
                x_axis_line1[x_graph_pos + k] = char

    print(f"{' ' * 8}  {''.join(x_axis_line1)}")
    print(f"{' ' * 8}  {'3-gram Start Index (0-indexed)'}") 

    # --- 3-gram とそのインデックスのリスト ---
    print("\n--- 3-gram Details ---")
    for idx, ngram_tuple in ngram_details_for_display:
        print(f"  [{idx}]: {' '.join(ngram_tuple)}")



#main


    alpha = 0.001 

    test_texts = [
        "今日は、お天気が良いですね。", # 句点が含まれる例
        "私は犬が好きです。新しいパソコンが欲しいです。", # 複数の句読点が含まれる例
        "今日は、彼が来てくえたらしい。"
    ]

    # 解析から除外する句読点
    SENTENCE_END_PUNCTUATIONS = {'。', '！', '？'}

    print(f"\n--- N-gram Processing with SudachiPy and Indexing ---")
    print(f"SudachiPy Dictionary Type: {SUD_DICT_TYPE}, Split Mode: {mode}")

    start_overall_time = time.time()

    for i, text in tqdm(enumerate(test_texts), total=len(test_texts), desc="Processing texts"):
        start_text_time = time.time()

        # 1. 形態素解析
        try:
            full_morphemes = [m.surface() for m in tokenizer_obj.tokenize(text, mode)]
            # print(f"  Full Morphemes: {full_morphemes}") # デバッグ用
        except Exception as e:
            print(f"  Error during SudachiPy tokenization for '{text}': {e}")
            full_morphemes = []
            
        if not full_morphemes:
            print(f"\nInput: \"{text}\"")
            print("  Skipping processing: Empty morpheme list.")
            continue

        # 句読点によるセグメンテーションとN-gram確率の収集
        all_ngram_log_probs_for_text = [] # この文章全体の3-gram確率を格納
        morphemes_for_display = [] # グラフ表示用の、句読点を除去した形態素リスト

        current_segment_morphemes = []
        
        for morpheme in full_morphemes:
            if morpheme in SENTENCE_END_PUNCTUATIONS:
                # 句点に到達した場合
                if len(current_segment_morphemes) >= 3:
                    # 現在のセグメントで3-gramを計算
                    segment_probs = calculate_score_per_ngram(current_segment_morphemes, total_vocab_size, alpha)
                    all_ngram_log_probs_for_text.extend(segment_probs)
                    morphemes_for_display.extend(current_segment_morphemes)
                # セグメントをリセット（句点は解析対象外）
                current_segment_morphemes = []
            else:
                current_segment_morphemes.append(morpheme)
        
        # 最後のセグメントの処理（句点で終わらない場合、または最後の句点の後）
        if len(current_segment_morphemes) >= 3:
            segment_probs = calculate_score_per_ngram(current_segment_morphemes, total_vocab_size, alpha)
            all_ngram_log_probs_for_text.extend(segment_probs)
            morphemes_for_display.extend(current_segment_morphemes)
        elif 0 < len(current_segment_morphemes) < 3:
            # 句読点を除いた結果、3単語未満になった場合
            morphemes_for_display.extend(current_segment_morphemes)


        if not all_ngram_log_probs_for_text:
            print(f"\nInput: \"{text}\"")
            print("  Skipping graph generation: Not enough valid 3-grams after punctuation removal.")
            continue
            
        # グラフ描画
        # morphemes_for_display は、句読点を除去した単語の連なりなので、
        # plot_single_text_line_graph の `ngram_details_for_display` の再構築で利用可能
        plot_single_text_line_graph(text, all_ngram_log_probs_for_text, morphemes_for_display, alpha)
        
        end_text_time = time.time()
        time_taken_text = end_text_time - start_text_time
        
        total_score_sum = sum(p for p in all_ngram_log_probs_for_text if p != -float('inf'))
        print(f"  Total score for this text (sum of log probs): {total_score_sum:.4f}")
        print(f"  Time taken for this text: {time_taken_text:.4f}s")


    end_overall_time = time.time()
    print(f"\n--- Processing Finished ---")
    print(f"Total processing time for all texts: {end_overall_time - start_overall_time:.2f}s")
