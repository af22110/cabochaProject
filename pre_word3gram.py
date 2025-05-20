# pre_word3gram.py
import os
import pickle
import time

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
    
    actual_files_to_process = [f for f in filenames if f.startswith(expected_prefix)]
    total_matching_files = len(actual_files_to_process)

    if total_matching_files == 0:
        print(f"No files matching prefix '{expected_prefix}' found in {directory_path}.")
        return all_ngram_counts

    print(f"Loading N-grams from {directory_path} (n={n}, limit={files_to_load_limit if files_to_load_limit is not None else 'all of '+str(total_matching_files)+' files'})...")
    
    files_to_actually_load = actual_files_to_process
    if files_to_load_limit is not None:
        files_to_actually_load = actual_files_to_process[:files_to_load_limit]

    for filename in files_to_actually_load:
        filepath = os.path.join(directory_path, filename)
        progress_denominator = len(files_to_actually_load)
        progress_str = f"({loaded_count + 1}/{progress_denominator})"
        print(f"  Loading from: {filepath} {progress_str}")
        
        ngrams_in_file = load_ngrams_from_file(filepath, n)
        all_ngram_counts.update(ngrams_in_file)
        loaded_count += 1
            
    return all_ngram_counts

if __name__ == "__main__":
    start_overall_time = time.time()

    # --- 設定項目 ---
    # N-gramデータが格納されているベースディレクトリ
    base_data_path = os.path.expanduser('/home/mitsu/cabochaProject/nwc2010-ngrams/word/over999')
    # Pickleファイルを保存する場所 (base_data_pathと同じ場所にする)
    pickle_output_path = base_data_path 

    # メモリ対策: 各N-gramタイプで読み込むファイル数を制限
    # None にするとディレクトリ内の該当ファイルを全て読み込もうとします
    FILES_TO_LOAD_LIMIT_PER_TYPE = 1 # 例: 各N-gramタイプで1ファイルのみロード
    # --- 設定項目ここまで ---

    # --- 2-gramデータの前処理 ---
    print(f"--- Preprocessing 2-grams (limit: {FILES_TO_LOAD_LIMIT_PER_TYPE if FILES_TO_LOAD_LIMIT_PER_TYPE is not None else 'all'} file(s)) ---")
    dir_2gms = os.path.join(base_data_path, '2gms')
    start_load_time = time.time()
    bigrams_data = load_all_ngrams_from_directory(dir_2gms, 2, files_to_load_limit=FILES_TO_LOAD_LIMIT_PER_TYPE)
    print(f"2-gram loading time: {time.time() - start_load_time:.2f}s. Found {len(bigrams_data)} unique 2-grams.")

    # --- 3-gramデータの前処理 ---
    print(f"\n--- Preprocessing 3-grams (limit: {FILES_TO_LOAD_LIMIT_PER_TYPE if FILES_TO_LOAD_LIMIT_PER_TYPE is not None else 'all'} file(s)) ---")
    dir_3gms = os.path.join(base_data_path, '3gms')
    start_load_time = time.time()
    trigrams_data = load_all_ngrams_from_directory(dir_3gms, 3, files_to_load_limit=FILES_TO_LOAD_LIMIT_PER_TYPE)
    print(f"3-gram loading time: {time.time() - start_load_time:.2f}s. Found {len(trigrams_data)} unique 3-grams.")

    # --- 全体の語彙数を計算 ---
    print("\n--- Calculating total vocabulary size (V) ---")
    all_words = set()
    if bigrams_data:
        for ngram_tuple in bigrams_data.keys():
            for word in ngram_tuple:
                all_words.add(word)
    if trigrams_data: # 3-gramからも語彙を収集 (通常は2-gramの語彙に含まれるはずだが念のため)
        for ngram_tuple in trigrams_data.keys():
            for word in ngram_tuple:
                all_words.add(word)
    
    total_vocab_size = len(all_words)
    print(f"Calculated total vocabulary size (V) from all loaded N-grams: {total_vocab_size}")
    if total_vocab_size == 0 and (len(bigrams_data) > 0 or len(trigrams_data) > 0) :
        print("Warning: N-gram data was loaded, but the calculated vocabulary size is 0. ")
        print("This might indicate an issue with the N-gram data format or content (e.g., empty words).")
        print("Add-k smoothing might behave unexpectedly with V=0.")


    # --- 2-gramデータのPickle保存 (メタデータ含む) ---
    # ファイル名を 'bigrams_data.pkl' に変更
    pickle_file_2gm = os.path.join(pickle_output_path, 'bigrams_data.pkl')
    if bigrams_data or total_vocab_size > 0 : 
        data_to_save_2gm = {
            'n': 2,
            'counts': bigrams_data,
            'total_vocab_size': total_vocab_size
        }
        try:
            with open(pickle_file_2gm, 'wb') as f:
                pickle.dump(data_to_save_2gm, f, pickle.HIGHEST_PROTOCOL)
            print(f"Saved {len(bigrams_data)} unique 2-grams with metadata (V={total_vocab_size}) to {pickle_file_2gm}")
        except Exception as e:
            print(f"Error saving 2-grams to pickle file {pickle_file_2gm}: {e}")
    else:
        print(f"No 2-gram data was loaded (or vocabulary is empty). Skipping pickle save for {pickle_file_2gm}.")

    # --- 3-gramデータのPickle保存 (メタデータ含む) ---
    # ファイル名を 'trigrams_data.pkl' に変更
    pickle_file_3gm = os.path.join(pickle_output_path, 'trigrams_data.pkl')
    if trigrams_data or total_vocab_size > 0: 
        data_to_save_3gm = {
            'n': 3,
            'counts': trigrams_data,
            'total_vocab_size': total_vocab_size
        }
        try:
            with open(pickle_file_3gm, 'wb') as f:
                pickle.dump(data_to_save_3gm, f, pickle.HIGHEST_PROTOCOL)
            print(f"Saved {len(trigrams_data)} unique 3-grams with metadata (V={total_vocab_size}) to {pickle_file_3gm}")
        except Exception as e:
            print(f"Error saving 3-grams to pickle file {pickle_file_3gm}: {e}")
    else:
        print(f"No 3-gram data was loaded (or vocabulary is empty). Skipping pickle save for {pickle_file_3gm}.")

    print(f"\nTotal preprocessing time: {time.time() - start_overall_time:.2f}s")
    print(f"Preprocessed data (including N-gram counts and total vocabulary size) saved in: {pickle_output_path}")
    print("\n--- How to use the preprocessed data for smoothing ---")
    print("The saved .pkl files (e.g., bigrams_data.pkl, trigrams_data.pkl) contain a dictionary with keys: 'n', 'counts', 'total_vocab_size'.")