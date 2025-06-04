import os
import pickle
import numpy as np
import time
from tqdm.notebook import tqdm
import sys
import matplotlib.pyplot as plt

# --- 警告と注意 ---
# 以下の変数は、word3gram.pyの内容を貼り付けた「前のセル」で定義されている必要があります。
# したがって、このセルを直接実行する前に、word3gram.pyの内容が書かれたセルを
# 実行していることを確認してください。

# このセルで利用する、word3gram.pyから来る想定の変数・関数のリスト:
# tokenizer_obj
# mode
# pickle_chunk_dir
# bigram_index
# trigram_index
# total_vocab_size
# get_ngram_count
# calculate_score_per_ngram
# SENTENCE_END_PUNCTUATIONS
# alpha


# --- 閾値評価関数 ---
def evaluate_threshold(threshold, data_min_log_probs, true_labels):
    """
    指定された閾値で分類を行い、精度指標を計算する。
    「文章中の3-gramの最小対数確率」が閾値を下回る場合に「不自然」と判定する。

    Args:
        threshold (float): 判定に使用する対数確率の閾値。
        data_min_log_probs (list): 各文章の「最小対数確率」のリスト。
        true_labels (list): 各文章の正解ラベル (True: 正しい, False: 不正)。
    
    Returns:
        dict: TP, TN, FP, FN, Accuracyを含む辞書。
    """
    tp = 0  # True Positive: 正しい文章を正しいと判定
    tn = 0  # True Negative: 不正な文章を不正と判定
    fp = 0  # False Positive: 不正な文章を正しいと判定
    fn = 0  # False Negative: 正しい文章を不正と判定

    for min_log_prob, true_label in zip(data_min_log_probs, true_labels):
        # 判定ロジック:
        # 最小確率が閾値を下回る（より負の値が大きい）場合、不自然と判定
        # 例: 閾値が-10.0のとき、最小確率が-12.0なら不自然
        predicted_is_unnatural = min_log_prob < threshold 
        
        # モデルの判定 (True: 自然, False: 不自然)
        predicted_label = not predicted_is_unnatural

        if true_label is True: # 正しい文章
            if predicted_label is True:
                tp += 1
            else: # 正しい文章を不自然と判定してしまった
                fn += 1
        else: # 不正な文章 (true_label is False)
            if predicted_label is True: # 不正な文章を自然と判定してしまった
                fp += 1
            else: # 不正な文章を不自然と判定
                tn += 1
    
    total = tp + tn + fp + fn
    if total == 0:
        return {
            'threshold': threshold, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'accuracy': 0.0
        }

    accuracy = (tp + tn) / total

    return {
        'threshold': threshold, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'accuracy': accuracy
    }


# --- メイン処理 ---
if __name__ == "__main__":
    print(f"\n--- Starting Threshold Determination ---")
    
    # --- 閾値決定のための評価データセット ---
    evaluation_dataset = [
        {"text": "私は猫が好きです。", "label": True},
        {"text": "今日の天気は晴れです。", "label": True},
        {"text": "新しいコンピュータを買いました。", "label": True},
        {"text": "これは本です。面白いです。", "label": True},
        {"text": "彼はりんごを食べました。", "label": True},
        {"text": "東京タワーは高い。", "label": True},
        {"text": "電車に乗って学校へ行った。", "label": True},
        {"text": "昨日映画を見た。", "label": True},
        {"text": "あの店のラーメンは美味しい。", "label": True},
        {"text": "公園で遊んだ。", "label": True},
        
        {"text": "今日は、彼が来てくえたらしい。", "label": False}, # 順序の誤り
        {"text": "天気は晴れれです。", "label": False}, # 順序の誤り
        {"text": "そこが住んでいます。", "label": False}, # 順序の誤り
        {"text": "電車を逃してしまいまた。", "label": False}, # 助詞の欠落
        {"text": "これは誤りの文章だす", "label": False}, # 順序の誤り
        {"text": "私は道が歩いています。", "label": False}, # 順序の誤り
        {"text": "これを美味しい。", "label": False}, # 意味的な不自然さ
        {"text": "とても時間がかかってしまいまいました。", "label": False}, # 助詞の欠落
        {"text": "また明日お願いしたいと存ずる。", "label": False}, # 不自然な修飾順序
        {"text": "この問題をに解きます。", "label": False}, # 順序の誤り

    ]
    
    print(f"Loaded {len(evaluation_dataset)} evaluation samples.")

    start_overall_time = time.time()

    # --- 評価データセットのスコア計算 (最小確率を記録) ---
    evaluated_scores_and_labels = [] # [(min_log_prob, label), ...]

    print("\n--- Calculating min log probabilities for evaluation dataset ---")
    for item in tqdm(evaluation_dataset, desc="Calculating min log probs"):
        text = item["text"]
        label = item["label"]

        try:
            full_morphemes = [m.surface() for m in tokenizer_obj.tokenize(text, mode)]
        except Exception as e:
            print(f"  Error during SudachiPy tokenization for '{text}': {e}")
            full_morphemes = []
            
        if not full_morphemes:
            print(f"  Skipping '{text}': Empty morpheme list or tokenization failed.")
            evaluated_scores_and_labels.append((-float('inf'), label)) # 解析失敗は不自然として扱う
            continue

        all_ngram_log_probs_for_text = []
        current_segment_morphemes = []
        
        for morpheme in full_morphemes:
            # 句読点を除去する処理 (必要であれば追加)
            # if morpheme in {'、', '。', '！', '？'}:
            #    continue # 句読点をスキップ

            if morpheme in SENTENCE_END_PUNCTUATIONS: # 文末句読点は依然としてセグメント区切りとして機能
                if len(current_segment_morphemes) >= 3:
                    segment_probs = calculate_score_per_ngram(current_segment_morphemes, total_vocab_size, alpha)
                    all_ngram_log_probs_for_text.extend(segment_probs)
                current_segment_morphemes = []
            else:
                current_segment_morphemes.append(morpheme)
        
        if len(current_segment_morphemes) >= 3:
            segment_probs = calculate_score_per_ngram(current_segment_morphemes, total_vocab_size, alpha)
            all_ngram_log_probs_for_text.extend(segment_probs)

        min_log_prob_for_text = min(p for p in all_ngram_log_probs_for_text if p != -float('inf')) if any(p != -float('inf') for p in all_ngram_log_probs_for_text) else -float('inf')
        
        if not all_ngram_log_probs_for_text:
            min_log_prob_for_text = -float('inf') 

        evaluated_scores_and_labels.append((min_log_prob_for_text, label))

    end_score_calc_time = time.time()
    print(f"Time for score calculation: {end_score_calc_time - start_overall_time:.2f}s")


    # --- 閾値の探索と最適閾値の決定 ---
    # 探索方向を逆にする: 小さい負の値から大きい負の値へ
    threshold_start = -15.0 # 開始をより小さい負の値に
    threshold_end = -2.5    # 終了をより大きい負の値に
    threshold_step = 0.1    # 増分を正の値に

    best_accuracy = -1.0
    best_threshold_by_accuracy = None
    all_results = []

    # グラフ描画用のデータを保存するリスト
    threshold_values = []
    accuracy_values = []

    print("\n--- Searching for optimal threshold ---")
    current_threshold = threshold_start
    while current_threshold <= threshold_end + 0.001: # 浮動小数点数の比較誤差を考慮
        scores = [item[0] for item in evaluated_scores_and_labels]
        labels = [item[1] for item in evaluated_scores_and_labels]
        
        results = evaluate_threshold(current_threshold, scores, labels)
        all_results.append(results)
        
        # グラフ描画用にデータを追加
        threshold_values.append(current_threshold)
        accuracy_values.append(results['accuracy'])

        if results['accuracy'] > best_accuracy:
            best_accuracy = results['accuracy']
            best_threshold_by_accuracy = results['threshold']
        
        # 探索方向が逆になったため、同じAccuracyの場合は自動的に最も左端（小さい負の値）が記録される
        # (最初にそのAccuracyに到達する閾値が、この方向では最も左端になるため)

        current_threshold += threshold_step

    print("\n--- Threshold Evaluation Results ---")
    print(f"{'Threshold':<10} {'Accuracy':<10} {'TP':<5} {'TN':<5} {'FP':<5} {'FN':<5}")
    print("-" * 45)
    for r in all_results:
        print(f"{r['threshold']:<10.2f} {r['accuracy']:<10.4f} {r['TP']:<5} {r['TN']:<5} {r['FP']:<5} {r['FN']:<5}")

    print(f"\nOptimal Threshold (by Accuracy): {best_threshold_by_accuracy:.2f} with Accuracy: {best_accuracy:.4f}")

    end_overall_time = time.time()
    print(f"\n--- Overall Processing Finished ---")
    print(f"Total overall time: {end_overall_time - start_overall_time:.2f}s")

    # --- グラフの可視化 ---
    plt.figure(figsize=(10, 6)) # グラフのサイズを設定
    plt.plot(threshold_values, accuracy_values, marker='o', linestyle='-', color='blue', label='Accuracy')
    
    if best_threshold_by_accuracy is not None:
        plt.axvline(best_threshold_by_accuracy, color='red', linestyle=':', 
                    label=f'Optimal Threshold ({best_threshold_by_accuracy:.4f}) (Accuracy: {best_accuracy:.4f})')
        # 最適な閾値の点を赤丸でプロット
        plt.plot(best_threshold_by_accuracy, best_accuracy, 'ro', markersize=10, label='Optimal Threshold Point')

    plt.title('Accuracy vs. Threshold for Sentence Classification')
    plt.xlabel('Threshold (Minimum Log Probability)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()