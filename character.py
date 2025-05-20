import os
import time # 処理時間計測のため（pickle関連の高速化コードは一旦削除してシンプルに）

# SudachiPyのインポート (インストールされている場合)
try:
    from sudachipy import tokenizer
    from sudachipy import dictionary
    # SudachiPyのTokenizerインスタンスを作成
    # split_modeはA, B, Cから選択 (Cが最も細かい分割)
    sudachi_tokenizer = dictionary.Dictionary().create()
    sudachi_mode = tokenizer.Tokenizer.SplitMode.C # または A, B
    print("SudachiPy tokenizer loaded.")
except ImportError:
    print("Error: SudachiPy or sudachidict_core not found.")
    print("Please install them: pip install sudachipy sudachidict_core")
    exit()
except Exception as e:
    print(f"Error loading SudachiPy tokenizer: {e}")
    exit()


def load_ngrams_from_file(filepath, n):
    """
    N-gramファイルを読み込み、N-gramと頻度の辞書を返す。
    ファイル形式の想定: "単語1 単語2 ... 単語N<タブ>頻度"
    """
    ngram_counts = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f: # エンコーディング注意
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) != 2:
                    # print(f"Skipping malformed line (tab split): {line} in {filepath}")
                    continue
                ngram_string = parts[0]
                try:
                    frequency = int(parts[1])
                except ValueError:
                    # print(f"Skipping malformed line (freq not int): {line} in {filepath}")
                    continue
                words = ngram_string.split(' ')
                if len(words) != n:
                    # print(f"Skipping malformed line (word count mismatch): {ngram_string} from {line} in {filepath}")
                    continue
                ngram_counts[tuple(words)] = frequency
    except FileNotFoundError:
        print(f"Error: File not found {filepath}")
    except Exception as e:
        print(f"Error reading file {filepath} at line {line_number if 'line_number' in locals() else 'unknown'}: {e}")
    return ngram_counts


def load_all_ngrams_from_directory(directory_path, n, files_to_load_limit=None):
    """
    指定されたディレクトリ内のN-gramファイルを読み込み、一つの辞書にまとめる。
    files_to_load_limit: 読み込むファイル数の上限 (Noneなら全て)
    """
    all_ngram_counts = {}
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found {directory_path}")
        return all_ngram_counts
            
    filenames = sorted(os.listdir(directory_path))
    expected_prefix = f"{n}gm-"
    loaded_count = 0
    
    print(f"Loading N-grams from {directory_path} (n={n}, limit={files_to_load_limit if files_to_load_limit is not None else 'all files'})...")
    for filename in filenames:
        if filename.startswith(expected_prefix):
            if files_to_load_limit is not None and loaded_count >= files_to_load_limit:
                print(f"Reached file load limit ({files_to_load_limit}). Stopping.")
                break 
            
            filepath = os.path.join(directory_path, filename)
            print(f"  Loading from: {filepath} ({loaded_count + 1}/{files_to_load_limit if files_to_load_limit is not None else len(filenames)})")
            ngrams_in_file = load_ngrams_from_file(filepath, n)
            all_ngram_counts.update(ngrams_in_file)
            loaded_count += 1
            
    return all_ngram_counts

# --- メイン処理 ---
if __name__ == "__main__":
    start_overall_time = time.time()

    # !!! 注意: 以下のパスはご自身の環境に合わせて変更してください !!!
    base_data_path = os.path.expanduser('~/cabochaProject/nwc2010-ngrams/word/over9/')
    
    dir_2gms = os.path.join(base_data_path, '2gms')
    dir_3gms = os.path.join(base_data_path, '3gms')

    # メモリ対策: 読み込むファイル数を制限 (最初は1ファイルずつから試すことを推奨)
    FILE_LIMIT_PER_NGRAM_TYPE = 1 # 各N-gramタイプで読み込むファイル数 (例: 2gmsから1ファイル, 3gmsから1ファイル)
                                  # Noneにすると全ファイルを読み込もうとします

    print(f"--- Loading 2-grams (limit: {FILE_LIMIT_PER_NGRAM_TYPE} file(s)) ---")
    start_load_time = time.time()
    bigrams_data = load_all_ngrams_from_directory(dir_2gms, 2, files_to_load_limit=FILE_LIMIT_PER_NGRAM_TYPE)
    if bigrams_data:
        print(f"Loaded {len(bigrams_data)} unique 2-grams. (Time: {time.time() - start_load_time:.2f}s)")
    else:
        print(f"No 2-gram data was loaded. Please check paths, file contents, and limit. (Time: {time.time() - start_load_time:.2f}s)")

    print(f"\n--- Loading 3-grams (limit: {FILE_LIMIT_PER_NGRAM_TYPE} file(s)) ---")
    start_load_time = time.time()
    trigrams_data = load_all_ngrams_from_directory(dir_3gms, 3, files_to_load_limit=FILE_LIMIT_PER_NGRAM_TYPE)
    if trigrams_data:
        print(f"Loaded {len(trigrams_data)} unique 3-grams. (Time: {time.time() - start_load_time:.2f}s)")
    else:
        print(f"No 3-gram data was loaded. Please check paths, file contents, and limit. (Time: {time.time() - start_load_time:.2f}s)")

    print(f"\nTotal N-gram loading time: {time.time() - start_overall_time:.2f}s")

    # --- サンプルデータの表示 (オプション) ---
    # if bigrams_data:
    #     print("\nSample Bigrams (first 5):")
    #     # ... (省略) ...
    # if trigrams_data:
    #     print("\nSample Trigrams (first 5):")
    #     # ... (省略) ...


    # --- 文章解析と不自然箇所特定 ---
    if not trigrams_data or not bigrams_data:
        print("\nN-gram data not sufficiently loaded. Cannot perform analysis.")
        exit()

    print("\n--- Text Analysis ---")
    # 解析対象の文章
    # input_text = "今日は良い天気ですね。明後日は特別料金です。形態素解析の結果を表示します。"
    input_text = "私は猫である。彼の名前はまだ無い。"
    # input_text = "この文は少し不自然な言葉遣いが含まれているかもしれない。"
    print(f"Input text: {input_text}")

    # SudachiPyで形態素解析 (表層形を取得)
    try:
        # mode C は最も細かく分割、mode A は短単位、mode B は中間
        morphemes = [m.surface() for m in sudachi_tokenizer.tokenize(input_text, sudachi_mode)]
        print(f"Tokenized by SudachiPy ({sudachi_mode}): {morphemes}")
    except Exception as e:
        print(f"Error during SudachiPy tokenization: {e}")
        exit()

    if len(morphemes) < 3:
        print("Tokenized text is too short for 3-gram analysis.")
        exit()

    print("\n--- Conditional Probability Analysis (3-grams) ---")
    # m=2 (for P(Xi | Xi-2, Xi-1))
    # 3-gram: (Xi-2, Xi-1, Xi)
    # 2-gram: (Xi-2, Xi-1)
    
    # 不自然さの閾値 (この値は実験的に調整が必要)
    # 確率がこの値より低い、またはゼロの場合に不自然と判定
    UNNATURAL_THRESHOLD_PROB = 0.001 
    # ゼロ頻度の場合に分母に加えるスムージング値 (ラプラススムージング)
    SMOOTHING_ALPHA = 1 # ゼロ割を防ぐため & 未知のN-gramにもわずかな確率を与える

    found_unnatural_sequences = []

    for i in range(len(morphemes) - 2): # 3単語の組を作るため
        w1 = morphemes[i]     # Xi-2
        w2 = morphemes[i+1]   # Xi-1
        w3 = morphemes[i+2]   # Xi
        
        current_trigram_tuple = (w1, w2, w3)
        current_bigram_tuple = (w1, w2)
        
        # P(w3 | w1, w2) = Count(w1, w2, w3) / Count(w1, w2)
        
        # 分子: Count(w1, w2, w3)
        count_w1_w2_w3 = trigrams_data.get(current_trigram_tuple, 0)
        
        # 分母: Count(w1, w2)
        count_w1_w2 = bigrams_data.get(current_bigram_tuple, 0)
        
        # 条件付き確率の計算 (スムージングを考慮)
        # P(Xi | Xi-2, Xi-1) = (Count(Xi-2, Xi-1, Xi) + alpha) / (Count(Xi-2, Xi-1) + alpha * V)
        # ここでは単純なAdd-alpha (alpha=SMOOTHING_ALPHA) を分母・分子に適用する
        # V (語彙数) は非常に大きいため、ここでは分母の count_w1_w2 が0の場合にのみ
        # スムージングが効果的に働くように、分母が0でなければそのまま使い、
        # 0の場合はSMOOTHING_ALPHAを分母とする（厳密なラプラススムージングではないが、ゼロ割回避と低確率付与に）
        
        prob = 0.0
        if count_w1_w2 + SMOOTHING_ALPHA > 0: # ゼロ割回避
            # より一般的なのは、分母が0でなくてもスムージングを適用することだが、
            # 今回はまず単純なケースで
            if count_w1_w2 == 0 and count_w1_w2_w3 == 0:
                # 2-gramも3-gramも未知の場合、非常に低い確率とする (ここでは0とする)
                # (より正確には、未知の単語に対する確率などを考慮する smoothing が必要)
                prob = 0.0 
            elif count_w1_w2 == 0 and count_w1_w2_w3 > 0:
                # このケースは理論上あまりない (3-gramがあれば2-gramもあるはず)
                # が、データセットの特性上ありえるなら考慮
                # 非常に低い確率 (ここでは仮に0)
                prob = 0.0
            elif count_w1_w2 > 0 :
                 prob = (count_w1_w2_w3 + SMOOTHING_ALPHA) / (count_w1_w2 + SMOOTHING_ALPHA * len(trigrams_data)) # ラプラススムージングの分母のVは全トリグラム種類数で近似
                 # もっと単純に、分母にゼロが来ないようにするだけなら:
                 # prob = count_w1_w2_w3 / (count_w1_w2 if count_w1_w2 > 0 else 1)
            
            # もっと単純なスムージング: 分母が0なら確率0、そうでなければそのまま計算
            # ただし、これだと未知のシーケンスへの頑健性が低い
            if count_w1_w2 > 0:
                prob = count_w1_w2_w3 / count_w1_w2
            else:
                prob = 0.0 # 分母がゼロなら確率ゼロ (より良いのはスムージング)


        print(f"  Sequence: ('{w1}', '{w2}', '{w3}')")
        print(f"    P('{w3}' | '{w1}', '{w2}') = {prob:.6f}  "
              f"(Trigram count: {count_w1_w2_w3}, Bigram count: {count_w1_w2})")

        # 不自然箇所の判定
        is_unnatural = False
        reason = ""
        if prob == 0.0 and count_w1_w2_w3 == 0 and count_w1_w2 == 0:
            is_unnatural = True
            reason = "Both 2-gram and 3-gram are unseen (Zero frequency)."
        elif prob == 0.0 and count_w1_w2_w3 == 0 and count_w1_w2 > 0:
            is_unnatural = True
            reason = "This 3-gram continuation is unseen for the existing 2-gram."
        elif prob < UNNATURAL_THRESHOLD_PROB and prob > 0: # ゼロよりは大きいが閾値より低い
            is_unnatural = True
            reason = f"Conditional probability ({prob:.6f}) is below threshold ({UNNATURAL_THRESHOLD_PROB})."
        
        if is_unnatural:
            unnatural_info = {
                "sequence": (w1, w2, w3),
                "probability": prob,
                "trigram_count": count_w1_w2_w3,
                "bigram_count": count_w1_w2,
                "reason": reason,
                "original_text_snippet": f"...{morphemes[max(0,i-2)] if i>1 else ''} {morphemes[max(0,i-1)] if i>0 else ''} >>{w1} {w2} {w3}<< {morphemes[i+3] if i+3<len(morphemes) else ''} {morphemes[i+4] if i+4<len(morphemes) else ''}..."
            }
            found_unnatural_sequences.append(unnatural_info)
            print(f"    >> Unnatural sequence detected! Reason: {reason}")

    if found_unnatural_sequences:
        print("\n--- Detected Unnatural Sequences ---")
        for item in found_unnatural_sequences:
            print(f"  - Sequence: {item['sequence']}")
            print(f"    Probability: {item['probability']:.6f}, Reason: {item['reason']}")
            print(f"    Context: {item['original_text_snippet']}")
    else:
        print("\nNo significantly unnatural sequences detected with the current threshold.")

    print(f"\nTotal execution time: {time.time() - start_overall_time:.2f}s")