# Hugging FaceモデルとTokenizerのロード
model_name = "rinna/japanese-gpt2-xsmall"
print(f"\nHugging Faceモデル {model_name} をロード中...")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import re # 正規表現モジュールをインポート
    import sys # 最小値判定のために使用
    import os # ファイル存在チェックに使用
    import pykakasi # ローマ字変換のために再度導入
    import Levenshtein # <<<--- Levenshteinをここでインポート

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
# この関数は Levenshtein ライブラリの `Levenshtein.distance` と機能が重複するため、
# ライブラリを使用する場合はこの自作関数は不要になる可能性があります。
# ただし、None を許容するなどの細かい挙動が異なる場合があるため、どちらを使うか明確にする必要があります。
# ここでは自作の edit_distance も残しておきます。
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
        # pykakasi.convert() メソッドを使用。カタカナ -> ローマ字設定済みのはず。
        # 出力は通常、[{'orig': 'カタカナ', 'hepburn': 'katakana'}] のような辞書のリスト
        result_list_of_dicts = kakasi_converter.convert(katakana_text)

        romaji_parts = []
        for item in result_list_of_dicts:
             # 'hepburn' の値を優先して取得。なければ 'orig'、それでもなければ空文字列
             romaji_parts.append(item.get('hepburn', item.get('orig', '')))

        # リストの要素を結合し、小文字に変換して返す
        return "".join(romaji_parts).lower()

    except Exception as e:
        # 変換中にエラーが発生した場合
        print(f"警告: ローマ字変換中にエラー発生（入力: '{katakana_text}'）: {e}")
        # エラーを示す特殊な文字列を返す。編集距離計算で大きな値になるようにする。
        return "___romaji_conversion_error___"


# 不自然な文字列の推定読みを辞書データから取得する関数
# 漢字編集距離が最も近い辞書単語の読みを返す
def estimate_segment_reading(segment_kanji, dictionary_data):
    min_kanji_distance = sys.maxsize
    estimated_reading = None
    # closest_word_kanji = None # 推定に使った漢字単語も必要なら保持

    if not dictionary_data:
        return None # 辞書データがない場合は推定不可

    # 対象セグメントに最も漢字が近い辞書エントリを探す
    for kanji, reading in dictionary_data:
        # 漢字が空の場合はスキップ（編集距離計算できないため）
        if not kanji:
            continue
        dist = edit_distance(segment_kanji, kanji) # 自作のedit_distanceを使用

        # 距離が更新されたら、その単語の読みを推定読み候補とする
        if dist < min_kanji_distance:
            min_kanji_distance = dist
            estimated_reading = reading
            # closest_word_kanji = kanji
            if min_kanji_distance == 0: # 漢字が完全に一致する単語が見つかったら、その読みが推定読み
                break

    # 漢字編集距離が非常に大きい場合、推定読みの信頼性は低い可能性があるが、今回は得られた読みを返す
    # print(f"  推定読み計算に使用した最小漢字距離: {min_kanji_distance}") # デバッグ用

    return estimated_reading # 漢字が最も近かった単語の読みを推定読みとして返す


# 辞書データから、漢字編集距離が近い初期候補を探す関数 (複数候補を返す)
# 熟語限定検索の機能も持つ
# 戻り値: 漢字編集距離が max_distance 以内 (または最小距離) の (漢字, 読み, 漢字距離) のタプルのリスト
def find_kanji_candidates_by_distance(segment_kanji, dictionary_data, max_distance=sys.maxsize, jukugo_only=True):
    candidates = [] # Store (kanji, reading, kanji_dist) tuples
    dictionary_to_search = dictionary_data

    # 熟語限定フラグがTrueの場合、事前に辞書データをフィルタリング
    if jukugo_only:
         jukugo_dictionary_data = [(k, r) for k, r in dictionary_data if is_jukugo(k)]
         if jukugo_dictionary_data:
              dictionary_to_search = jukugo_dictionary_data
              print(f"  漢字検索対象: {len(dictionary_to_search)} 件の熟語")
         else:
              print("  警告: 辞書データ内に2文字以上の漢字のみで構成される熟語が見つかりませんでした。熟語以外の単語も含めて漢字検索を行います。")
              # 熟語が見つからなかった場合は全件検索
              dictionary_to_search = dictionary_data
              jukugo_only = False # フラグをFalseに戻す（出力メッセージ用）
     # 熟語限定フラグがFalseの場合は、dictionary_to_search は dictionary_data のまま

    if not dictionary_to_search:
         return [] # 検索対象が空の場合は空リストを返す


    # 漢字編集距離が max_distance 以内の候補を集める、あるいは最小距離の候補だけを集める
    max_distance_to_collect = max_distance # 初期設定

    if max_distance == sys.maxsize: # max_distance が指定されていない場合は最小距離を探す
        min_distance = sys.maxsize
        # 最初のパスで最小距離を見つける
        for kanji, reading in dictionary_to_search:
            if not kanji: continue
            dist = edit_distance(segment_kanji, kanji) # 自作のedit_distanceを使用
            if dist < min_distance:
                min_distance = dist
        max_distance_to_collect = min_distance # 最小距離の候補を全て集める

    # 2回目のパスで、決定した距離閾値内の候補を収集
    for kanji, reading in dictionary_to_search:
        if not kanji: continue
        dist = edit_distance(segment_kanji, kanji) # 自作のedit_distanceを使用
        if dist <= max_distance_to_collect:
             # 既にリストに同じ漢字の候補がないか確認（複数読みがある場合など）
             if not any(c[0] == kanji for c in candidates):
                 candidates.append((kanji, reading, dist))

    # 収集した候補を漢字距離でソート（任意だが、後続処理で見やすくなる）
    candidates.sort(key=lambda item: item[2])

    return candidates # (漢字, 読み, 漢字距離) のタプルのリスト


# 初期候補リストを、推定読み（ローマ字）との距離で再ランキングし、最適な候補を探す関数
# 戻り値: (最適な候補漢字, 元漢字距離, 候補単語の読み, 読みローマ字距離, 候補単語の読みローマ字)
def find_best_candidate_romaji_rerank(estimated_segment_reading_romaji, initial_candidates_kanji, kakasi_converter):
    min_romaji_distance = sys.maxsize
    best_candidate_kanji = None
    best_candidate_reading = None
    best_kanji_dist_orig = sys.maxsize # 元の漢字距離
    best_candidate_reading_romaji = None

    # 推定読みのローマ字が得られていない、または初期候補リストが空の場合は再ランキング不可
    if estimated_segment_reading_romaji in ["", "___romaji_conversion_error___"] or not initial_candidates_kanji:
        return None, sys.maxsize, None, sys.maxsize, None # 候補なし

    # 漢字編集距離で絞り込まれた初期候補をループ
    for kanji, reading, kanji_dist_orig_loop in initial_candidates_kanji:
        # 候補単語の読みをローマ字に変換
        candidate_reading_romaji = katakana_to_romaji(reading, kakasi_converter)

        # ローマ字編集距離を計算
        # 候補単語の読みのローマ字変換が失敗した場合も考慮
        if candidate_reading_romaji == "___romaji_conversion_error___":
            romaji_dist = sys.maxsize # 変換エラーは無限大距離とみなす
        else:
             # 推定読みのローマ字と候補単語の読みのローマ字間で編集距離を計算
             romaji_dist = edit_distance(estimated_segment_reading_romaji, candidate_reading_romaji) # 自作のedit_distanceを使用

        # 最小ローマ字距離を持つ候補を見つける
        if romaji_dist < min_romaji_distance:
            min_romaji_distance = romaji_dist
            best_candidate_kanji = kanji
            best_candidate_reading = reading
            best_kanji_dist_orig = kanji_dist_orig_loop # この候補の元の漢字距離を記録
            best_candidate_reading_romaji = candidate_reading_romaji

    # 候補が見つからなかった場合 (初期候補リストが空、または全候補のローマ字変換失敗など)
    if best_candidate_kanji is None:
        return None, sys.maxsize, None, sys.maxsize, None # 候補なし

    # 見つかった最適候補と関連情報を返す
    return best_candidate_kanji, best_kanji_dist_orig, best_candidate_reading, min_romaji_distance, best_candidate_reading_romaji


# --- メイン処理 ---

# 入力文章
text = "天気、特別、鳥龍茶。指適、重復も確認します。"
# text = "今日は良い天気ですね。明後日は特別料金です。形態素解析の結果を表示します。" # 元のテスト用文章
print(f"\n入力文章: {text}")

# 1. Hugging Face Tokenizerでのトークン分割とoffset_mappingの取得
inputs = tokenizer_hf(text, return_tensors="pt", return_offsets_mapping=True)
input_ids = inputs["input_ids"]
offset_mapping = inputs["offset_mapping"].squeeze().tolist() # typo は修正済

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
    sys.exit()

print(f"\n抽出された漢字連続部分: {kanji_segments_info}")

# 3. モデルによる確率計算
with torch.no_grad():
    outputs = model_hf(input_ids)
    logits = outputs.logits
# log_probs_full_sequence: テキスト全体の各HFトークン位置における、次のトークンの対数確率分布
log_probs_full_sequence = torch.log_softmax(logits, dim=-1).squeeze()

# 4. 各漢字セグメントのスコア計算
segment_scores = []
threshold_log_prob = -10.8799 # 例として設定した閾値 (この値自体は調整が必要)

print("\n漢字連続部分ごとの確率スコア:")
for seg_info in kanji_segments_info:
    segment_str = seg_info['segment']
    seg_start_char_idx = seg_info['start_char']
    # 終了位置を正しく計算 (修正済)
    seg_end_char_idx = seg_start_char_idx + len(segment_str)


    hf_indices_for_this_segment = []
    for hf_token_idx, (tok_char_start, tok_char_end) in enumerate(offset_mapping):
        overlap_start = max(tok_char_start, seg_start_char_idx)
        overlap_end = min(tok_char_end, seg_end_char_idx)

        if overlap_start < overlap_end:
             # 先頭の特殊トークン(hf_token_idx=0)は除外 (その位置の確率計算は通常行わないため)
             if hf_token_idx > 0:
                 hf_indices_for_this_segment.append(hf_token_idx)

    if not hf_indices_for_this_segment:
        print(f"  警告: セグメント '{segment_str}' に対応するスコア計算可能なHFトークンが見つかりませんでした。スキップします。")
        continue

    segment_log_prob_sum = 0
    num_valid_tokens_in_segment = 0

    for hf_idx in hf_indices_for_this_segment:
        # hf_idx番目のトークンのIDを取得
        token_id = input_ids[0, hf_idx].item()
        # (hf_idx-1)番目のトークンまでの文脈が与えられたときの、hf_idx番目のトークンの対数確率
        # log_probs_full_sequence の最初の次元は入力トークン数-1 です。hf_idxは入力トークンインデックスなので、-1が必要です。
        if hf_idx - 1 < log_probs_full_sequence.shape[0]:
             log_prob_this_token = log_probs_full_sequence[hf_idx-1, token_id].item()
             segment_log_prob_sum += log_prob_this_token
             num_valid_tokens_in_segment += 1
        else:
             # インデックス範囲外の場合は警告 (理論上は起きにくい)
             print(f"  警告: トークンインデックス {hf_idx} (分析対象) が確率テンソルの範囲外です。スキップします。")


    if num_valid_tokens_in_segment > 0:
        avg_segment_log_prob = segment_log_prob_sum / num_valid_tokens_in_segment
    else:
        avg_segment_log_prob = -float('inf') # スコア計算できなかった場合は最低値扱い

    segment_scores.append({
        'segment': segment_str,
        'start_char_idx': seg_start_char_idx,
        'avg_log_prob': avg_segment_log_prob,
        'hf_indices_for_this_segment': hf_indices_for_this_segment, # このリスト自体を保存
        'num_hf_tokens': len(hf_indices_for_this_segment),
        'num_scored_hf_tokens': num_valid_tokens_in_segment
    })
    print(f"  セグメント: '{segment_str}' (開始文字位置: {seg_start_char_idx}), "
          f"Avg Log Prob: {avg_segment_log_prob:.4f}, "
          f"構成HFトークン数: {len(hf_indices_for_this_segment)} (うちスコア計算対象: {num_valid_tokens_in_segment})")


# 5. 最小確率スコアのセグメントを見つける
min_score = float('inf')
segment_with_min_score = None

if segment_scores:
    valid_score_entries = [entry for entry in segment_scores if entry['num_scored_hf_tokens'] > 0]

    if valid_score_entries:
        # スコアがInfの場合は最小値候補から除外する
        finite_score_entries = [entry for entry in valid_score_entries if entry['avg_log_prob'] != -float('inf')]
        if finite_score_entries:
             segment_with_min_score = min(finite_score_entries, key=lambda x: x['avg_log_prob'])
             min_score = segment_with_min_score['avg_log_prob']
        # finite_score_entriesが空の場合はsegment_with_min_scoreはNoneのまま

else:
    print("\nスコア計算が完了した漢字連続部分が見つかりませんでした。")


# 6. 最小確率スコアが閾値を下回っているか判定し、辞書検索・結果出力
print("\n--- 辞書検索と編集距離 ---")

# スコア計算可能なセグメントが見つかったかチェック
if segment_with_min_score and segment_with_min_score['avg_log_prob'] != -float('inf'):
    print(f"最小確率スコアのセグメント: '{segment_with_min_score['segment']}' "
          f"(Avg Log Prob: {segment_with_min_score['avg_log_prob']:.4f})")

    # 最小確率が閾値を下回っているか？
    if segment_with_min_score['avg_log_prob'] < threshold_log_prob:
        target_segment = segment_with_min_score['segment']
        seg_start_char_idx = segment_with_min_score['start_char_idx'] # 元の文章での開始文字位置


        print(f"\n最小確率スコア ({min_score:.4f}) が閾値 ({threshold_log_prob:.4f}) を下回っています。不自然な文字列として特定されました。")
        print(f"特定された文字列: '{target_segment}' ({seg_start_char_idx}文字目から)")

        # --- 不自然な文字列内の「文字の落ち込み」を分析 ---
        # 特定された文字列内のトークンごとの確率を分析し、最も低い部分を特定
        print("\n--- 特定された文字列内のトークン確率分析 ---")

        # 対象セグメントに対応するHFトークンのインデックスリストを、segment_scoresから取得
        target_segment_entry = None
        for entry in segment_scores:
            if entry['segment'] == target_segment and entry['start_char_idx'] == seg_start_char_idx:
                 target_segment_entry = entry
                 break

        scored_hf_indices_in_segment = [] # スコア計算対象となったHFトークンの元の文章全体でのインデックス

        if target_segment_entry and target_segment_entry.get('hf_indices_for_this_segment'): # キーが存在し、かつリストが空でないかチェック
             # セクション4で保存したリストを取得し、スコア計算に有効なトークン（hf_idx > 0 のもの）をフィルタ
             all_hf_indices_in_segment = target_segment_entry['hf_indices_for_this_segment']
             scored_hf_indices_in_segment = [hf_idx for hf_idx in all_hf_indices_in_segment if hf_idx > 0]
             # スコア計算に使われたトークン数は segment_scores の num_scored_hf_tokens を使う

        if not scored_hf_indices_in_segment:
            print("  警告: 特定された文字列に対応するスコア計算可能なHFトークンが見つかりませんでした。文字の落ち込み分析はスキップします。")
            problematic_chars = "分析不能" # 落ち込み特定できなかった場合
            problematic_span_in_segment = (-1, -1) # 落ち込み位置も特定不能
            min_token_log_prob = float('inf') # 分析できなかった場合はログ確率を無限大にしておく
        else:
            min_token_log_prob = float('inf')
            min_prob_hf_token_idx = -1 # 最も確率が低いトークンの、元の文章全体でのHFトークンインデックス
            min_prob_token_text = "" # 最も確率が低いトークンのテキスト
            min_prob_token_char_span_orig_text = (-1, -1) # 最も確率が低いトークンの、元の文章での文字範囲

            # --- 第二のグラフ描画のためのデータを収集 ---
            segment_token_positions = [] # 文字列内でのトークンの連番 (1から開始)
            segment_token_log_probs = [] # 各トークンのログ確率
            segment_token_labels = [] # 各トークンのテキスト
            token_seq_in_segment = 1 # 文字列内でのトークン連番カウンター

            print("  文字列内のスコア計算対象トークンごとの確率:")
            # スコア計算対象トークンを順番に処理 (元の文章での出現順)
            # scored_hf_indices_in_segment は既に元の文章でのhf_idx順になっているはず
            for hf_idx in scored_hf_indices_in_segment:
                 token_id = input_ids[0, hf_idx].item()
                 token_text = tokenizer_hf.convert_ids_to_tokens([token_id])[0] # IDをトークンテキストに変換
                 # このトークンのログ確率 (hf_idx-1番目位置の出力における token_id の確率)
                 if hf_idx - 1 < log_probs_full_sequence.shape[0]:
                      log_prob_this_token = log_probs_full_sequence[hf_idx-1, token_id].item()
                      prob_this_token = np.exp(log_prob_this_token) # 確率に変換

                      # このトークンが元の文章のどの文字範囲に対応するか
                      char_span_orig_text = offset_mapping[hf_idx]

                      print(f"    トークン {hf_idx} ('{token_text}'): Prob = {prob_this_token:.6f} (LogProb = {log_prob_this_token:.4f}) @ 元文章chars {char_span_orig_text}")

                      # ログ確率が最も低いトークンを見つける
                      if log_prob_this_token < min_token_log_prob:
                           min_token_log_prob = log_prob_this_token
                           min_prob_hf_token_idx = hf_idx
                           min_prob_token_text = token_text
                           min_prob_token_char_span_orig_text = char_span_orig_text

                      # 第二グラフ用のデータを収集
                      segment_token_positions.append(token_seq_in_segment)
                      segment_token_log_probs.append(log_prob_this_token)
                      segment_token_labels.append(token_text)
                      token_seq_in_segment += 1
                 else:
                      # インデックス範囲外の場合は警告 (理論上は起きにくい)
                      print(f"  警告: トークンインデックス {hf_idx} (分析対象) が確率テンソルの範囲外です。スキップします。")


            if min_prob_hf_token_idx != -1: # 最も確率が低いトークンが見つかった場合
                 # 最も確率が低いトークンに対応する文字範囲を、特定された文字列内の位置に変換
                 # 元の文章での開始位置からの相対位置を計算
                 span_start_in_segment = min_prob_token_char_span_orig_text[0] - seg_start_char_idx
                 span_end_in_segment = min_prob_token_char_span_orig_text[1] - seg_start_char_idx

                 # 範囲が文字列内に収まっているか確認 (念のため)
                 # Tokenizerのoffset_mappingが文字列境界をまたぐ場合があるので、max/minで調整
                 span_start_in_segment = max(0, span_start_in_segment)
                 span_end_in_segment = min(len(target_segment), span_end_in_segment)

                 # 特定された文字列から該当する部分文字列を抽出
                 problematic_chars = target_segment[span_start_in_segment:span_end_in_segment]
                 problematic_span_in_segment = (span_start_in_segment, span_end_in_segment) # 特定できた位置

                 print(f"\n最も確率が低いトークン (全体中のID): {min_prob_hf_token_idx} ('{min_prob_token_text}')")
                 print(f"  対応する文字位置 (元の文章内): {min_prob_token_char_span_orig_text[0]}文字目から{min_prob_token_char_span_orig_text[1]}文字目")
                 print(f"  対応する文字位置 (特定された文字列内): {problematic_span_in_segment[0]}文字目から{problematic_span_in_segment[1]}文字目")
                 print(f"  **「文字の落ち込み」の可能性が高い部分:** '{problematic_chars}'")

            else: # 最も確率が低いトークンの特定に失敗した場合
                 print("  警告: 最も確率が低いトークンの特定に失敗しました。文字の落ち込み分析は部分的です。")
                 problematic_chars = "分析不全"
                 problematic_span_in_segment = (-1, -1) # 位置も特定不能


            # --- 第二のグラフを描画 ---
            if segment_token_positions: # グラフ描画データがあるか確認
                 plt.figure(figsize=(max(6, len(segment_token_positions) * 0.5), 6)) # Adjust figure size
                 plt.plot(segment_token_positions, segment_token_log_probs, marker='o', linestyle='-', label=f"Log Probability per Token in '{target_segment}'")

                 # Annotate points with token text
                 for i, token_label in enumerate(segment_token_labels):
                      plt.annotate(token_label,
                                   (segment_token_positions[i], segment_token_log_probs[i]),
                                   textcoords="offset points",
                                   xytext=(5, 5),
                                   ha='left',
                                   fontsize=9)

                 # Highlight the minimum probability point if found and corresponds to a plotted token
                 if min_prob_hf_token_idx != -1 and min_prob_hf_token_idx in scored_hf_indices_in_segment:
                      try:
                          # scored_hf_indices_in_segmentリストでのmin_prob_hf_token_idxの位置を探す
                          min_prob_index_in_scored_list = scored_hf_indices_in_segment.index(min_prob_hf_token_idx)
                          # グラフデータリスト (segment_token_positions/log_probs) は scored_hf_indices_in_segment の順序に対応しているので、そのインデックスを使う
                          plt.plot(segment_token_positions[min_prob_index_in_scored_list], segment_token_log_probs[min_prob_index_in_scored_list],
                                   marker='*', markersize=10, color='red', label="Minimum Log Probability Token")
                      except ValueError:
                          # 万が一リストに見つからない場合（理論上起きないはず）
                          print(f"  警告: 最小トークンID {min_prob_hf_token_idx} がグラフ対象リストに見つかりません。最小点のハイライトをスキップします。")


                 plt.xlabel(f"Token Sequence Number within '{target_segment}'")
                 plt.ylabel("Log Probability")
                 plt.title(f"Token Probabilities within '{target_segment}'")
                 # X軸の目盛りをトークン連番に設定
                 if segment_token_positions:
                    plt.xticks(segment_token_positions)
                 plt.grid(True, linestyle=':', alpha=0.7)
                 plt.legend()
                 plt.tight_layout()

                 graph_filename_edit = 'kanji_for_edit.png' # 新しいファイル名
                 plt.savefig(graph_filename_edit)
                 print(f"\n文字列内トークン確率グラフを '{graph_filename_edit}' として保存しました。")
            else:
                 print("  警告: 文字列内のトークン確率グラフ描画データがありません。")

        # --- 文字の落ち込み分析 ここまで ---


        # --- 辞書検索と編集距離計算（漢字距離で初期候補を絞り、読みローマ字距離で再ランキング） ---
        if all_dictionary_data: # 辞書データがロードされているか確認
             print(f"\n特定された文字列 '{target_segment}' に対して、辞書候補を検索します...")

             # 1. 対象セグメントの推定読みを取得 (漢字距離が最も近い辞書単語の読みを利用)
             estimated_segment_reading = estimate_segment_reading(target_segment, all_dictionary_data)

             if estimated_segment_reading is None:
                  print(f"警告: 対象セグメント '{target_segment}' の推定読みを取得できませんでした。辞書検索をスキップします。")
             else:
                  print(f"  対象セグメント '{target_segment}' の推定読み: '{estimated_segment_reading}'")

                  # 2. 推定読みをローマ字に変換
                  estimated_segment_reading_romaji = katakana_to_romaji(estimated_segment_reading, kakasi)

                  if estimated_segment_reading_romaji in ["", "___romaji_conversion_error___"]:
                       print(f"警告: 推定読み '{estimated_segment_reading}' のローマ字変換に失敗しました。辞書検索をスキップします。")
                  else:
                       print(f"  対象セグメント '{target_segment}' の推定読み (ローマ字): '{estimated_segment_reading_romaji}'")

                       # --- Phase 1: 漢字編集距離で初期候補を絞り込む ---
                       # 辞書データ全体（熟語以外も含む）を対象に、漢字編集距離が近い候補を探す
                       # 例えば、漢字編集距離が2以内の候補を全て集める（距離閾値は調整可能）
                       kanji_search_max_distance = 2
                       print(f"\n  段階1: 辞書データ全体を対象に、漢字編集距離が {kanji_search_max_distance} 以内の候補を絞り込みます。")
                       # is_jukugo 判定を find_kanji_candidates_by_distance 関数内でオプションとして行うように変更
                       # この呼び出しでは jukugo_only=True とすることで熟語限定検索になります
                       # 今回は辞書データ全体で検索するため、jukugo_only をデフォルトの False のままにするか、関数を分けます。
                       # シンプルにするため、一旦 find_kanji_candidates_by_distance はそのまま使い、絞り込みはその内部で行うか、外で行うか調整します。
                       # 今回は Phase 1 で熟語に絞り込む機能を復活させましょう。
                       is_jukugo_limited_phase1 = True # Phase 1 で熟語に絞り込むか
                       if is_jukugo_limited_phase1:
                            print("  (熟語に絞って漢字検索を行います)")
                       else:
                            print("  (辞書データ全体で漢字検索を行います)")


                       initial_candidates_kanji = find_kanji_candidates_by_distance(
                           target_segment, all_dictionary_data, max_distance=kanji_search_max_distance, jukugo_only=is_jukugo_limited_phase1 # jukugo_only オプションを追加
                       )


                       if not initial_candidates_kanji:
                           print("警告: 漢字編集距離が近い初期候補が見つかりませんでした。辞書検索をスキップします。")
                       else:
                           print(f"\n  漢字編集距離で絞り込まれた初期候補 ({len(initial_candidates_kanji)} 件):")
                           # 初期候補リストを表示
                           for k, r, d in initial_candidates_kanji:
                               print(f"    - '{k}' (読み: '{r}', 漢字距離: {d})")


                           # --- Phase 2: 初期候補を読み（ローマ字）編集距離で再ランキングし、最終候補を決定 ---
                           print("\n  段階2: 初期候補を読み（ローマ字）編集距離で再ランキングします...")
                           best_candidate_kanji, kanji_dist_orig, candidate_reading, romaji_dist_final, candidate_reading_romaji = find_best_candidate_romaji_rerank(
                               estimated_segment_reading_romaji, initial_candidates_kanji, kakasi
                           )

                           if best_candidate_kanji is not None:
                                print("\n--- 編集候補結果 ---")
                                # print(f"元の単語の並び: {target_segment}") # この行は下の "元の単語" と重複するのでコメントアウトまたは削除
                                print(f"元の単語: {target_segment}") # 修正: original_word ではなく target_segment を使用
                                print(f"辞書の中の候補 (漢字): {best_candidate_kanji}") # 修正: dictionary_candidate ではなく best_candidate_kanji を使用

                                # Levenshtein距離を計算
                                # 修正: original_word を target_segment に、dictionary_candidate を best_candidate_kanji に変更
                                distance = Levenshtein.distance(target_segment, best_candidate_kanji)

                                # 結果を出力 (重複していた部分を整理)
                                if candidate_reading:
                                    print(f"辞書の中の候補 (読み): {candidate_reading}")
                                if candidate_reading_romaji:
                                    print(f"辞書の中の候補 (読みローマ字): {candidate_reading_romaji}")
                                print(f"(初期漢字編集距離: {kanji_dist_orig})") # これは自作の edit_distance によるもの
                                print(f"類似度 (読みローマ字編集距離: {romaji_dist_final})") # これも自作の edit_distance によるもの
                                print(f"最終的な漢字編集距離 (Levenshteinライブラリ): {distance}") # Levenshteinライブラリによる距離
                                print("-" * 20) # 区切り線など

                           else:
                                print("警告: 初期候補の中から読み（ローマ字）編集距離で最適な候補が見つかりませんでした（再ランキング失敗）。")

        else: # Dictionary not loaded or empty
             print("\n警告: 利用できる辞書データがないため、辞書検索はスキップされました。")
else: # segment_with_min_score が None または avg_log_prob が -inf の場合
    if not segment_scores: # スコア計算されたセグメントが一つもなかった場合
        pass # メッセージは既に上で出力されている
    elif not segment_with_min_score : # スコア計算されたセグメントはあったが、有効なものがなかった場合
        print("\n有効なスコアを持つ漢字連続部分が見つからなかったため、辞書検索は行いません。")
    else: # 最小スコアが閾値を下回らなかった場合
        print(f"\n最小確率スコア ({min_score:.4f}) は閾値 ({threshold_log_prob:.4f}) 以上です。不自然な文字列は見つかりませんでした。")


# Update notes
print("\n・辞書ファイルからは、単語（漢字）と読み（カタカナ想定）のペアを読み込みます。")
print("・不自然な文字列の読みは、漢字編集距離が最も近い辞書単語の読みを推定読みとして使用します。")
print("・ローマ字変換には pykakasi ライブラリを使用します（読み（カタカナ）のみを対象）。")
print("・**辞書候補の検索は2段階で行われます。**")
print("・**段階1: 漢字編集距離で初期候補を絞り込みます。** デフォルトでは2文字以上の漢字のみの単語（熟語とみなす）の中から、元の文字列と漢字編集距離が近い候補（例: 距離2以内）を探します。該当する熟語がない場合は辞書データ全体から探します。この距離計算には自作の `edit_distance` 関数が使用されます。")
print("・**段階2: 読み（ローマ字）編集距離で最終候補を選びます。** 段階1で絞り込まれた初期候補の中から、不自然な文字列の推定読みをローマ字にしたものと、候補単語の読みをローマ字にしたものとの間の、ローマ字編集距離が最も低いものが最終的な編集候補となります。この距離計算にも自作の `edit_distance` 関数が使用されます。")
print("・最終的な編集候補と元の文字列との間の漢字編集距離は、`python-Levenshtein` ライブラリを用いて別途計算・表示されます。")