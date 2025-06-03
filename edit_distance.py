# Hugging FaceモデルとTokenizerのロード
import sys # sysモジュールをtryブロックの外に移動

model_name = "rinna/japanese-gpt2-xsmall"
print(f"\nHugging Faceモデル {model_name} をロード中...")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import re # 正規表現モジュールをインポート
    import os # ファイル存在チェックに使用
    import pykakasi # ローマ字変換のために再度導入
    import Levenshtein # Levenshteinをここでインポート

    # 日本語フォント設定（文字化け対策）
    plt.rcParams['font.family'] = 'IPAexGothic' # またはお使いの環境にある日本語フォント名 (例: 'Hiragino Sans GB', 'Yu Gothic')
    plt.rcParams['axes.unicode_minus'] = False # 負の符号の表示

    tokenizer_hf = AutoTokenizer.from_pretrained(model_name)
    model_hf = AutoModelForCausalLM.from_pretrained(model_name)
    print("モデルのロードが完了しました。")

    # pykakasi 初期化
    # ローマ字変換（カタカナ -> 英語/ローマ字）のみを設定
    kakasi = pykakasi.kakasi()
    kakasi.setMode("K", "E") # カタカナを英語（ローマ字）に変換
    kakasi.setMode("r", "Hepburn") # ヘボン式ローマ字を使用
    # setMode("J", "E") や "H", "A", "N", "s" などは読みの変換には不要

except ImportError:
    print(f"エラー: 必要なライブラリが見つかりません。")
    print("pip install transformers torch sentencepiece numpy matplotlib regex pykakasi python-Levenshtein で必要なライブラリをインストールしてください。") # Levenshteinのインストール方法も追記
    sys.exit()
except Exception as e:
    print(f"Hugging FaceモデルまたはTokenizerのロード中にエラーが発生しました。エラー: {e}")
    print(f"詳細エラー: {e}")
    sys.exit()

# --- 辞書データ、編集距離関数、ローマ字変換関数、熟語判定関数の準備 ---

# デモンストレーション用のフォールバック辞書データ (読み込みファイルが見つからない場合に使用)
# フォーマット: (漢字, 読み - カタカナ想定)
sample_dictionary_data = [
    ("天気", "テンキ"), ("特別", "トクベツ"), ("料金", "リョウキン"), ("形態素", "ケイタイソ"),
    ("解析", "カイセキ"), ("結果", "ケッカ"), ("表示", "ヒョウジ"), ("今日", "キョウ"),
    ("明日", "アシタ"), ("明後日", "アサッテ"), ("昨日", "キノウ"), ("一昨日", "オトトイ"),
    ("機械学習", "キカイガクシュウ"), ("深層学習", "シンソウガクシュウ"), ("自然言語処理", "シゼンゲンゴショリ"),
    ("面白い", "オモシロイ"), ("分野", "ブンヤ"), ("技術", "ギジュツ"), ("形態", "ケイタイ"),
    ("要素", "ヨウソ"), ("解釈", "カイシャク"), ("分析", "ブンセキ"), ("単語", "タンゴ"),
    ("辞書", "ジショ"), ("編集", "ヘンシュウ"), ("距離", "キョリ"), ("候補", "コウホ"),
    ("烏龍茶", "ウーロンチャ"), ("お茶", "オチャ"), ("紅茶", "コウチャ"), ("緑茶", "リョクチャ"),
    ("指摘", "シテキ"), ("不適", "フテキ"), ("適", "テキ"), ("指", "シ"), ("摘", "テキ"), # 単漢字の読みは複数ありうるが代表的なものを設定
    ("重複", "ジュウフク"), ("修復", "シュウフク"), ("重", "ジュウ"), ("複", "フク"), ("修", "シュウ"),
    ("確認", "カクニン"), ("操作", "ソウサ"), ("処理", "ショリ"), ("実行", "ジッコウ"),
    ("日本語", "ニホンゴ"), ("単語", "タンゴ"), ("文字", "モジ"), ("文章", "ブンショウ"),
    ("指し", "サシ"), ("重一", "ジュウイチ") # 重一の読みを追加（実際にあるか不明だがデモ用）
]


# 公開されている単語リストファイル（漢字 タブ 読み のフォーマットを想定）をロード
dictionary_file = 'japanese_word_list.txt' # 使用する単語リストファイル名
all_dictionary_data = [] # (漢字, 読み) のタプルのリストとして格納

print(f"\n単語リストファイル '{dictionary_file}' をロードを試みます...")
if os.path.exists(dictionary_file):
    try:
        # japanese_word_list.txt は通常 UTF-8 エンコーディングを想定
        with open(dictionary_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): # 空行またはコメント行はスキップ
                    continue
                parts = line.split('\t') # タブ区切りを想定
                # 少なくとも漢字と読みの2つ以上の部分があるか確認
                if len(parts) >= 2:
                    kanji = parts[0]
                    reading = parts[1] # 2番目の要素を読み（カタカナ想定）とする
                    # 漢字も読みも空でない場合のみ追加
                    if kanji and reading:
                        all_dictionary_data.append((kanji, reading))
                # else: 行のフォーマットが不正な場合はスキップ（警告も出せる）

        if all_dictionary_data:
            print(f"単語リストファイル '{dictionary_file}' から {len(all_dictionary_data)} 件の (漢字, 読み) データをロードしました。")
        else:
            print(f"警告: 単語リストファイル '{dictionary_file}' は空であるか、有効な (漢字, 読み) ペアを含んでいません。組み込みのサンプル辞書データを使用します。")
            all_dictionary_data = sample_dictionary_data # ロード失敗時はサンプルを使用
    except Exception as e:
        print(f"警告: 単語リストファイル '{dictionary_file}' の読み込み中にエラーが発生しました: {e}。組み込みのサンプル辞書データを使用します。")
        all_dictionary_data = sample_dictionary_data # ロードエラー時もサンプルを使用
else:
    print(f"警告: 単語リストファイル '{dictionary_file}' が見つかりませんでした。組み込みのサンプル辞書データを使用します。")
    all_dictionary_data = sample_dictionary_data # ファイルがない時もサンプルを使用


# 単語が「2文字以上の漢字のみで構成される熟語」であるかを判定する関数
def is_jukugo(word):
    # 文字列が完全に2文字以上の漢字のみで構成されているか正規表現でチェック
    kanji_pattern = re.compile(r'^[\u4E00-\u9FFF]{2,}$')
    return bool(kanji_pattern.match(word))


# Levenshtein編集距離を計算する関数 (文字列の種類に関わらず使用可能)
def edit_distance(s1, s2):
    if s1 is None or s2 is None: # どちらかがNoneの場合は無限大距離とする
        return sys.maxsize
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1]), distances[i1 + 1], distances_[-1]))
        distances = distances_
    return distances[-1]

# カタカナ文字列をローマ字に変換する関数
def katakana_to_romaji(katakana_text, kakasi_converter):
    if not katakana_text:
        return "" # 空の入力には空文字列を返す

    try:
        result_list_of_dicts = kakasi_converter.convert(katakana_text)

        romaji_parts = []
        for item in result_list_of_dicts:
            romaji_parts.append(item.get('hepburn', item.get('orig', '')))

        return "".join(romaji_parts).lower()

    except Exception as e:
        print(f"警告: ローマ字変換中にエラー発生（入力: '{katakana_text}'）: {e}")
        return "___romaji_conversion_error___"


# 不自然な文字列の推定読みを辞書データから取得する関数
# 漢字編集距離が最も近い辞書単語の読みを返す
def estimate_segment_reading(segment_kanji, dictionary_data):
    min_kanji_distance = sys.maxsize
    estimated_reading = None

    if not dictionary_data:
        return None

    for kanji, reading in dictionary_data:
        if not kanji:
            continue
        dist = edit_distance(segment_kanji, kanji)

        if dist < min_kanji_distance:
            min_kanji_distance = dist
            estimated_reading = reading
            if min_kanji_distance == 0:
                break

    return estimated_reading


# 辞書データから、漢字編集距離が近い初期候補を探す関数 (複数候補を返す)
# 熟語限定検索の機能も持つ
# 戻り値: 漢字編集距離が max_distance 以内 (または最小距離) の (漢字, 読み, 漢字距離) のタプルのリスト
def find_kanji_candidates_by_distance(segment_kanji, dictionary_data, max_distance=sys.maxsize, jukugo_only=False):
    candidates = [] # Store (kanji, reading, kanji_dist) tuples
    dictionary_to_search = dictionary_data

    # 熟語限定フラグがTrueの場合、事前に辞書データをフィルタリング
    if jukugo_only:
        jukugo_dictionary_data = [(k, r) for k, r in dictionary_data if is_jukugo(k)]
        if jukugo_dictionary_data:
            dictionary_to_search = jukugo_dictionary_data
            # print(f"     漢字検索対象: {len(dictionary_to_search)} 件の熟語") # Moved to main
        else:
            # print("     警告: 辞書データ内に2文字以上の漢字のみで構成される熟語が見つかりませんでした。熟語以外の単語も含めて漢字検索を行います。") # Moved to main
            dictionary_to_search = dictionary_data
            jukugo_only = False # Flag back to False for consistent message

    if not dictionary_to_search:
        return []

    max_distance_to_collect = max_distance

    if max_distance == sys.maxsize:
        min_distance = sys.maxsize
        for kanji, reading in dictionary_to_search:
            if not kanji: continue
            dist = edit_distance(segment_kanji, kanji)
            if dist < min_distance:
                min_distance = dist
        max_distance_to_collect = min_distance

    for kanji, reading in dictionary_to_search:
        if not kanji: continue
        dist = edit_distance(segment_kanji, kanji)
        if dist <= max_distance_to_collect:
            if not any(c[0] == kanji for c in candidates):
                candidates.append((kanji, reading, dist))

    candidates.sort(key=lambda item: item[2])

    return candidates


# 初期候補リストを、推定読み（ローマ字）との距離で再ランキングし、最適な候補を探す関数
# 戻り値: (最適な候補漢字, 元漢字距離, 候補単語の読み, 読みローマ字距離, 候補単語の読みローマ字)
def find_best_candidate_romaji_rerank(estimated_segment_reading_romaji, initial_candidates_kanji, kakasi_converter):
    min_romaji_distance = sys.maxsize
    best_candidate_kanji = None
    best_candidate_reading = None
    best_kanji_dist_orig = sys.maxsize
    best_candidate_reading_romaji = None

    if estimated_segment_reading_romaji in ["", "___romaji_conversion_error___"] or not initial_candidates_kanji:
        return None, sys.maxsize, None, sys.maxsize, None

    for kanji, reading, kanji_dist_orig_loop in initial_candidates_kanji:
        candidate_reading_romaji = katakana_to_romaji(reading, kakasi_converter)

        if candidate_reading_romaji == "___romaji_conversion_error___":
            romaji_dist = sys.maxsize
        else:
            romaji_dist = edit_distance(estimated_segment_reading_romaji, candidate_reading_romaji)

        if romaji_dist < min_romaji_distance:
            min_romaji_distance = romaji_dist
            best_candidate_kanji = kanji
            best_candidate_reading = reading
            best_kanji_dist_orig = kanji_dist_orig_loop
            best_candidate_reading_romaji = candidate_reading_romaji

    if best_candidate_kanji is None:
        return None, sys.maxsize, None, sys.maxsize, None

    return best_candidate_kanji, best_kanji_dist_orig, best_candidate_reading, min_romaji_distance, best_candidate_reading_romaji


# --- メイン処理 ---

# 入力文章
text = "私は今日、そこを指適されました。"
# text = "今日は良い天気ですね。明後日は特別料金です。形態素解析の結果を表示します。" # 元のテスト用文章
print(f"\n入力文章: {text}")

# 1. Hugging Face Tokenizerでのトークン分割とoffset_mappingの取得
inputs = tokenizer_hf(text, return_tensors="pt", return_offsets_mapping=True)
input_ids = inputs["input_ids"]
offset_mapping = inputs["offset_mapping"].squeeze().tolist()

# 2. 2文字以上の漢字連続部分を抽出
kanji_segments_info = []
for match in re.finditer(r'[\u4E00-\u9FFF]{2,}', text):
    kanji_segments_info.append({
        'segment': match.group(0),
        'start_char': match.start(),
        'end_char': match.end()
    })

if not kanji_segments_info:
    print("\n解析対象となる2文字以上の漢字連続部分が見つかりませんでした。")
    # グラフ生成はスキップし、後続の処理も行わない
    sys.exit()

print(f"\n抽出された漢字連続部分: {kanji_segments_info}")

# 3. モデルによる確率計算
with torch.no_grad():
    outputs = model_hf(input_ids)
    logits = outputs.logits
log_probs_full_sequence = torch.log_softmax(logits, dim=-1).squeeze()

# 4. 各漢字セグメントのスコア計算
segment_scores = []
threshold_log_prob = -10.8799 # ★変更: 閾値を-10.8799に設定
threshold_prob = np.exp(threshold_log_prob) # 閾値の確率表現

print("\n漢字連続部分ごとの確率スコア:")
for seg_info in kanji_segments_info:
    segment_str = seg_info['segment']
    seg_start_char_idx = seg_info['start_char']
    # seg_end_char_idx = seg_start_char_idx + len(segment_str) # 今回は使わない

    hf_indices_for_this_segment = []
    for hf_token_idx, (tok_char_start, tok_char_end) in enumerate(offset_mapping):
        # トークンが漢字セグメント内に完全に含まれるか、部分的にでも重なる場合
        # HFトークンがセグメント内に含まれるかどうかの判定をより厳密に
        if tok_char_start >= seg_start_char_idx and tok_char_end <= seg_info['end_char']:
            if hf_token_idx > 0: # 先頭の特殊トークンは除外
                hf_indices_for_this_segment.append(hf_token_idx)

    if not hf_indices_for_this_segment:
        # print(f"     警告: セグメント '{segment_str}' に対応するスコア計算可能なHFトークンが見つかりませんでした。スキップします。")
        # HFトークンが見つからなくても、そのセグメントはスコア0でプロット対象にする
        avg_segment_log_prob = -float('inf') # または非常に低い値
        num_valid_tokens_in_segment = 0 # トークンがないので0
    else:
        segment_log_prob_sum = 0
        num_valid_tokens_in_segment = 0

        for hf_idx in hf_indices_for_this_segment:
            token_id = input_ids[0, hf_idx].item()
            # 確率テンソルのインデックスは input_ids のインデックス-1 に対応
            if hf_idx - 1 < log_probs_full_sequence.shape[0]:
                log_prob_this_token = log_probs_full_sequence[hf_idx-1, token_id].item()
                segment_log_prob_sum += log_prob_this_token
                num_valid_tokens_in_segment += 1
            else:
                print(f"     警告: トークンインデックス {hf_idx} (分析対象) が確率テンソルの範囲外です。スキップします。")


        if num_valid_tokens_in_segment > 0:
            avg_segment_log_prob = segment_log_prob_sum / num_valid_tokens_in_segment
        else:
            avg_segment_log_prob = -float('inf')


    segment_scores.append({
        'segment': segment_str,
        'start_char_idx': seg_start_char_idx,
        'avg_log_prob': avg_segment_log_prob,
        'hf_indices_for_this_segment': hf_indices_for_this_segment,
        'num_hf_tokens': len(hf_indices_for_this_segment),
        'num_scored_hf_tokens': num_valid_tokens_in_segment
    })
    print(f"     セグメント: '{segment_str}' (開始文字位置: {seg_start_char_idx}), "
          f"Avg Log Prob: {avg_segment_log_prob:.4f}, "
          f"構成HFトークン数: {len(hf_indices_for_this_segment)} (うちスコア計算対象: {num_valid_tokens_in_segment})")

# --- グラフ描画 (一つのグラフに集約) ---
print("\n--- 漢字連続部分の平均確率グラフ ---")

if segment_scores:
    # グラフ用のデータ準備
    plot_x_labels = [] # X軸のラベル (例: "1:天気", "2:特別"など)
    plot_y_values = [] # Y軸の値 (平均対数確率)
    
    # ここを修正: `num_scored_hf_tokens > 0` のフィルタリングを削除
    segments_to_plot = segment_scores # すべてのセグメントを対象とする
    
    if segments_to_plot:
        for i, entry in enumerate(segments_to_plot):
            # 横軸ラベルに「開始文字位置」を追加
            plot_x_labels.append(f"{entry['segment']} ({entry['start_char_idx']})")
            
            # `avg_log_prob`が`-inf`の場合、プロット用に非常に低い有限値に置き換える（オプション）
            # `kanji1.png`の表示崩れを防ぐため。今回はそのままプロットします。
            plot_y_values.append(entry['avg_log_prob'])

        plt.figure(figsize=(max(8, len(plot_x_labels) * 0.8), 6)) # X軸ラベル数に応じて横幅を調整

        # ★変更箇所: linestyle='-' を linestyle='' に変更して線なし、マーカーのみにする
        plt.plot(np.arange(len(plot_y_values)), plot_y_values, marker='o', linestyle='', label='Average Log Probability')
        plt.axhline(y=threshold_log_prob, color='r', linestyle='--', label=f'Threshold ({threshold_log_prob:.4f})') # 画像の凡例に合わせる

        plt.xlabel("Kanji Segment (Text and Start Char Index)") # 横軸ラベルを「開始文字位置」を含むように変更
        plt.ylabel("Average Log Probability")
        plt.title("Word Confidence Score based on Hugging Face LM (Overall Text)") # 画像のタイトルに合わせる
        plt.suptitle(f'Input: "{text}"', fontsize=12, y=1.03) # グラフ全体の上に入力文字列を表示

        plt.xticks(np.arange(len(plot_x_labels)), plot_x_labels, rotation=45, ha='right')
        plt.grid(True, linestyle=':', alpha=0.7) # グリッド線を表示
        plt.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # suptitleのスペースを確保
        graph_filename = 'kanji1.png' # ★変更: ファイル名をkanji1.pngに
        plt.savefig(graph_filename)
        print(f"漢字信頼度スコアグラフを '{graph_filename}' として保存しました。")
    else:
        print("\n警告: グラフ描画のための有効な漢字セグメントスコアが見つかりませんでした。")
else:
    print("\n警告: グラフ描画のための有効な漢字セグメントスコアが見つかりませんでした。")


# 5. 最小確率スコアのセグメントを見つける
min_score = float('inf')
segment_with_min_score = None

if segment_scores:
    # フィルタリングされた後の `segments_to_plot` ではなく、元の `segment_scores` から最小スコアを探す
    # ただし、`-inf` のスコアは最小値として扱いたくない場合は、ここでもフィルターが必要です。
    # 現在のロジックでは `-inf` が最も低い値として扱われるため、「確認」が選択される可能性があります。
    # 「確認」が選択された場合でも、その後の処理は適切に行われるように設計されています。
    valid_score_entries = [entry for entry in segment_scores if entry['num_scored_hf_tokens'] > 0]

    if valid_score_entries:
        finite_score_entries = [entry for entry in valid_score_entries if entry['avg_log_prob'] != -float('inf')]
        if finite_score_entries:
            segment_with_min_score = min(finite_score_entries, key=lambda x: x['avg_log_prob'])
            min_score = segment_with_min_score['avg_log_prob']
        else: # 全てが -inf の場合
            segment_with_min_score = min(valid_score_entries, key=lambda x: x['avg_log_prob'])
            min_score = segment_with_min_score['avg_log_prob']

# 6. 最小確率スコアが閾値を下回っているか判定し、辞書検索・結果出力
print("\n--- 辞書検索と編集距離 ---")

if segment_with_min_score and segment_with_min_score['avg_log_prob'] < threshold_log_prob:
    target_segment = segment_with_min_score['segment']
    seg_start_char_idx = segment_with_min_score['start_char_idx']

    print(f"\n**不自然な文字列として特定されました:** '{target_segment}' (Avg Log Prob: {min_score:.4f} < 閾値: {threshold_log_prob:.4f})")

    # --- 不自然な文字列内の「文字の落ち込み」を分析（詳細をコンソール出力と kanji2.png グラフ） ---
    scored_hf_indices_in_segment = []
    target_segment_entry = None
    for entry in segment_scores:
        if entry['segment'] == target_segment and entry['start_char_idx'] == seg_start_char_idx:
            target_segment_entry = entry
            break

    problematic_chars = "分析不能"
    problematic_span_in_segment = (-1, -1)
    min_token_log_prob_in_segment = float('inf') # セグメント内の最小トークン確率
    min_prob_hf_token_idx_in_segment = -1

    # kanji2.png 用のデータ収集
    plot2_x_labels = []
    plot2_y_values = []
    
    if target_segment_entry and target_segment_entry.get('hf_indices_for_this_segment'):
        all_hf_indices_in_segment = target_segment_entry['hf_indices_for_this_segment']
        scored_hf_indices_in_segment = [hf_idx for hf_idx in all_hf_indices_in_segment if hf_idx > 0]

        if scored_hf_indices_in_segment:
            print(f"     '{target_segment}' 内のトークンごとの詳細な対数確率:")
            for hf_idx in scored_hf_indices_in_segment:
                token_id = input_ids[0, hf_idx].item()
                token_text = tokenizer_hf.convert_ids_to_tokens([token_id])[0]
                if hf_idx - 1 < log_probs_full_sequence.shape[0]:
                    log_prob_this_token = log_probs_full_sequence[hf_idx-1, token_id].item()
                    char_span_orig_text = offset_mapping[hf_idx]
                    
                    # トークンが元のテキストのどの部分に対応するか
                    original_span_text = text[char_span_orig_text[0]:char_span_orig_text[1]] if char_span_orig_text[0] < char_span_orig_text[1] else token_text
                    
                    print(f"         トークン: '{token_text}' (元テキスト: '{original_span_text}', 開始インデックス: {char_span_orig_text[0]}), Log Prob: {log_prob_this_token:.4f}")

                    # kanji2.png 用データに追加 (元の文章内での開始文字位置を使用)
                    plot2_x_labels.append(f"'{original_span_text}' ({char_span_orig_text[0]})")
                    plot2_y_values.append(log_prob_this_token)

                    if log_prob_this_token < min_token_log_prob_in_segment:
                        min_token_log_prob_in_segment = log_prob_this_token
                        min_prob_hf_token_idx_in_segment = hf_idx
                        min_prob_token_char_span_orig_text = char_span_orig_text
            
            # --- kanji2.png グラフ生成 ---
            if plot2_y_values:
                plt.figure(figsize=(max(8, len(plot2_x_labels) * 0.8), 6))
                plt.plot(np.arange(len(plot2_y_values)), plot2_y_values, marker='o', linestyle='-', label='Token Log Probability')
                plt.axhline(y=threshold_log_prob, color='r', linestyle='--', label=f'Threshold ({threshold_log_prob:.4f})')
                
                plt.xlabel("Token (Text and Start Char Index in Original Text)") # 横軸ラベルを「開始文字位置」を含むように変更
                plt.ylabel("Log Probability")
                plt.title(f"Token Confidence Score in Detected Segment: '{target_segment}'")
                plt.suptitle(f'Input: "{text}"', fontsize=12, y=1.03)
                
                plt.xticks(np.arange(len(plot2_x_labels)), plot2_x_labels, rotation=45, ha='right')
                plt.grid(True, linestyle=':', alpha=0.7)
                plt.legend()
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                graph_filename_2 = 'kanji2.png' # ★変更: ファイル名をkanji2.pngに
                plt.savefig(graph_filename_2)
                print(f"特定された文字列内のトークン信頼度スコアグラフを '{graph_filename_2}' として保存しました。")
            else:
                print("     警告: kanji2.png グラフ描画のための有効なトークンスコアが見つかりませんでした。")

        if min_prob_hf_token_idx_in_segment != -1:
            span_start_in_segment = min_prob_token_char_span_orig_text[0] - seg_start_char_idx
            span_end_in_segment = min_prob_token_char_span_orig_text[1] - seg_start_char_idx
            span_start_in_segment = max(0, span_start_in_segment)
            span_end_in_segment = min(len(target_segment), span_end_in_segment)
            problematic_chars = target_segment[span_start_in_segment:span_end_in_segment]
            problematic_span_in_segment = (span_start_in_segment, span_end_in_segment)
    else:
        print("     警告: 特定された文字列に対応するスコア計算可能なHFトークンが見つかりませんでした。文字の落ち込み分析はスキップします。")


    print(f"**「文字の落ち込み」の可能性が高い部分:** '{problematic_chars}'")

    # --- 辞書検索と編集距離計算 ---
    if all_dictionary_data:
        estimated_segment_reading = estimate_segment_reading(target_segment, all_dictionary_data)

        if estimated_segment_reading is None:
            print(f"警告: 対象セグメント '{target_segment}' の推定読みを取得できませんでした。辞書検索をスキップします。")
        else:
            estimated_segment_reading_romaji = katakana_to_romaji(estimated_segment_reading, kakasi)

            if estimated_segment_reading_romaji in ["", "___romaji_conversion_error___"]:
                print(f"警告: 推定読み '{estimated_segment_reading}' のローマ字変換に失敗しました。辞書検索をスキップします。")
            else:
                kanji_search_max_distance = 2 # 漢字編集距離の閾値 (ご希望に応じて1に変更可能)
                # Phase 1: 漢字編集距離で初期候補を絞り込む
                is_jukugo_limited_phase1 = True # Phase 1 で熟語に絞り込むか
                if is_jukugo_limited_phase1:
                    print("     (熟語に絞って漢字検索を行います)")
                else:
                    print("     (辞書データ全体で漢字検索を行います)")

                initial_candidates_kanji = find_kanji_candidates_by_distance(
                    target_segment, all_dictionary_data, max_distance=kanji_search_max_distance, jukugo_only=is_jukugo_limited_phase1
                )

                print(f"**初期辞書候補数:** {len(initial_candidates_kanji)} 件")

                if initial_candidates_kanji:
                    # Phase 2: 初期候補を読み（ローマ字）編集距離で再ランキングし、最終候補を決定
                    best_candidate_kanji, kanji_dist_orig, candidate_reading, romaji_dist_final, candidate_reading_romaji = find_best_candidate_romaji_rerank(
                        estimated_segment_reading_romaji, initial_candidates_kanji, kakasi
                    )

                    if best_candidate_kanji is not None:
                        print(f"**推定読み (ローマ字):** '{estimated_segment_reading_romaji}'")
                        print(f"**修正候補の漢字:** '{best_candidate_kanji}'")
                        print(f"**修正候補のローマ字読み:** '{candidate_reading_romaji}'")
                        final_levenshtein_distance = Levenshtein.distance(target_segment, best_candidate_kanji)
                        print(f"**最終的な漢字編集距離:** {final_levenshtein_distance}")
                        
                    else:
                        print("警告: 辞書候補の中から最適な修正が見つかりませんでした。")
                else:
                    print("警告: 漢字編集距離が近い初期候補が見つかりませんでした。")
    else:
        print("\n警告: 利用できる辞書データがないため、辞書検索はスキップされました。")
else:
    if not segment_scores or (segment_with_min_score and segment_with_min_score['avg_log_prob'] == -float('inf')):
        print("\n有効なスコアを持つ漢字連続部分が見つからなかったため、辞書検索は行いません。")
    else:
        print(f"\n最小確率スコア ({min_score:.4f}) は閾値 ({threshold_log_prob:.4f}) 以上です。不自然な文字列は見つかりませんでした。")