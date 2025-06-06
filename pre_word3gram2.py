import os
import pickle
import time
import glob
from tqdm import tqdm # ここを修正しました！

def load_ngrams_from_file(filepath, n):
    """
    N-gramファイルを読み込み、N-gramと頻度の辞書を返す。
    ファイル形式の想定: "単語1 単語2 ... 単語N<タブ>頻度"
    """
    ngram_counts = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) != 2:
                    # print(f"DEBUG: Skipping line {line_number} in {filepath} due to incorrect parts count: '{line}'") # デバッグ用
                    continue
                ngram_string = parts[0]
                try:
                    frequency = int(parts[1])
                except ValueError:
                    # print(f"DEBUG: Skipping line {line_number} in {filepath} due to invalid frequency: '{line}'") # デバッグ用
                    continue
                words = ngram_string.split(' ')
                if len(words) != n:
                    # print(f"DEBUG: Skipping line {line_number} in {filepath} for {n}-gram due to incorrect word count: '{line}'") # デバッグ用
                    continue
                ngram_counts[tuple(words)] = frequency
    except FileNotFoundError:
        print(f"Error: File not found {filepath}")
    except Exception as e:
        print(f"Error reading file {filepath} at line {line_number if 'line_number' in locals() else 'unknown'}: {e}")
    return ngram_counts

def calculate_total_vocabulary_size(ngram_data_dirs, files_to_load_limit_per_type=None):
    """
    指定されたN-gramディレクトリから、全体の語彙サイズ(V)を計算する。
    これは、全てのユニークな単語の数を集計することで行われる。
    """
    all_words = set()
    
    print("\n--- Calculating total vocabulary size (V) across all N-gram files ---")
    
    for n_type, dir_path in ngram_data_dirs.items():
        if not os.path.isdir(dir_path):
            print(f"Warning: Directory not found {dir_path} for {n_type}-grams. Skipping.")
            continue

        expected_prefix = f"{n_type}gm-" # 例: "2gm-" や "3gm-"
        file_paths = sorted(glob.glob(os.path.join(dir_path, f"{expected_prefix}*")))
        
        if files_to_load_limit_per_type is not None:
            file_paths = file_paths[:files_to_load_limit_per_type]

        if not file_paths:
            print(f"No files matching prefix '{expected_prefix}' found in {dir_path}. Skipping V calculation for this type.")
            continue

        print(f"   Scanning {len(file_paths)} {n_type}-gram files in {dir_path} for vocabulary...")
        
        for i, filepath in tqdm(enumerate(file_paths), total=len(file_paths), desc=f"Calculating V for {n_type}-grams"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        parts = line.split('\t')
                        if len(parts) != 2: continue
                        
                        ngram_string = parts[0]
                        try:
                            # 頻度を整数に変換できるかチェック (V計算では使わないが、形式の整合性チェック)
                            _ = int(parts[1]) 
                        except ValueError:
                            continue
                        
                        words = ngram_string.split(' ')
                        if len(words) != n_type: # n_type を使う
                            continue
                        
                        for word in words:
                            if word:
                                all_words.add(word)
            except Exception as e:
                print(f"Warning: Error reading {filepath} for V calculation: {e}. Skipping this file.")
                continue

    V = len(all_words)
    print(f"Calculated total vocabulary size (V): {V}")
    return V

def process_and_chunk_ngrams(directory_path, n, pickle_output_dir, total_vocab_size, files_to_load_limit=None):
    """
    指定されたディレクトリ内のN-gramファイルを読み込み、チャンクに分割してpickle保存する。
    この際、計算済みのtotal_vocab_sizeを各チャンクに含める。
    また、インデックスファイル (ngram_first_word_index.pkl) も同時に作成する。
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found {directory_path}")
        return False
            
    # 出力ディレクトリを用意
    os.makedirs(pickle_output_dir, exist_ok=True)

    expected_prefix = f"{n}gm-"
    file_paths = sorted(glob.glob(os.path.join(directory_path, f"{expected_prefix}*")))
    
    if files_to_load_limit is not None:
        file_paths = file_paths[:files_to_load_limit]

    if not file_paths:
        print(f"No files matching prefix '{expected_prefix}' found in {directory_path}. Skipping chunking for {n}-grams.")
        return False

    print(f"\n--- Processing and chunking {n}-grams from {directory_path} (limit: {len(file_paths)} files) ---")
    
    processed_count = 0
    # インデックスを格納するための辞書
    # キー: N-gramの最初の単語, 値: その単語で始まるN-gramを含むチャンクファイル名のセット
    ngram_first_word_index = {} 

    for i, filepath in tqdm(enumerate(file_paths), total=len(file_paths), desc=f"Chunking {n}-grams"):
        ngrams_in_file = load_ngrams_from_file(filepath, n)

        # 保存（ファイル名ごとに）
        base_filename = os.path.basename(filepath)
        # 元のファイル名が .txt 拡張子を持たない場合も考慮し、replace('.txt', '') を調整
        # 安全のため、単純に .pkl を付加する形に変更
        output_chunk_filename = f"{n}gram_{base_filename}.pkl" 
        
        # 保存するデータに、計算済みの total_vocab_size を含める
        data_to_save = {
            'n': n,
            'counts': ngrams_in_file,
            'total_vocab_size': total_vocab_size, # ここで正確なVをセット
        }
        
        try:
            full_output_path = os.path.join(pickle_output_dir, output_chunk_filename)
            with open(full_output_path, "wb") as f:
                pickle.dump(data_to_save, f, pickle.HIGHEST_PROTOCOL)
            processed_count += len(ngrams_in_file)

            # インデックスを構築
            for ngram_tuple in ngrams_in_file.keys():
                first_word = ngram_tuple[0]
                if first_word not in ngram_first_word_index:
                    ngram_first_word_index[first_word] = set() 
                ngram_first_word_index[first_word].add(output_chunk_filename) 
            
        except Exception as e:
            print(f"Error saving chunk {output_chunk_filename}: {e}")
            continue

    # インデックスをファイルに保存
    index_filename = f"{n}gram_first_word_index.pkl"
    index_full_path = os.path.join(pickle_output_dir, index_filename)
    try:
        serializable_index = {word: list(files) for word, files in ngram_first_word_index.items()}

        with open(index_full_path, "wb") as f:
            pickle.dump(serializable_index, f, pickle.HIGHEST_PROTOCOL)
        print(f"Successfully saved index for {n}-grams to {index_full_path}")
    except Exception as e:
        print(f"Error saving index for {n}-grams to {index_full_path}: {e}")
        return False

    print(f"Finished processing {n}-grams. Total {processed_count} N-grams saved across {len(file_paths)} chunks.")
    return True

if __name__ == "__main__":
    start_overall_time = time.time()

    # --- 設定項目 ---
    # N-gramデータが格納されている具体的なディレクトリを指定
    ngram_data_directories = {
        2: os.path.expanduser('/home/ubuntu/cabochaProject/nwc2010-ngrams/word/over9/2gms'), # 2-gram のパス
        3: os.path.expanduser('/home/ubuntu/cabochaProject/nwc2010-ngrams/word/over9/3gms'), # 3-gram のパス
    }
    
    # Pickleファイルを保存する場所 (プロジェクトルートに 'pickle_chunks' サブディレクトリを作成)
    pickle_output_dir = os.path.expanduser('/home/ubuntu/cabochaProject/pickle_chunks/') 

    # メモリ対策: V計算、および各N-gramタイプで処理（読み込んで保存）するファイル数を制限
    FILES_TO_PROCESS_LIMIT_PER_TYPE = None 
    # --- 設定項目ここまで ---

    # Pickleファイルの出力ディレクトリを作成
    os.makedirs(pickle_output_dir, exist_ok=True)
    print(f"Pickle chunk output directory: {pickle_output_dir}")

    # --- 全体の語彙数を計算 (最初に実行) ---
    total_vocab_size = calculate_total_vocabulary_size(ngram_data_directories, 
                                                       files_to_load_limit_per_type=FILES_TO_PROCESS_LIMIT_PER_TYPE)
    
    if total_vocab_size == 0:
        print("Critical Error: Calculated total vocabulary size (V) is 0. Cannot proceed with N-gram processing.")
        print("Please check your N-gram source files and the `load_ngrams_from_file` / `calculate_total_vocabulary_size` logic.")
        import sys; sys.exit(1) 
    
    print(f"\nFinal calculated total vocabulary size (V): {total_vocab_size}")

    # --- 2-gramデータの前処理とチャンク保存 ---
    process_and_chunk_ngrams(
        directory_path=ngram_data_directories[2],
        n=2,
        pickle_output_dir=pickle_output_dir,
        total_vocab_size=total_vocab_size, 
        files_to_load_limit=FILES_TO_PROCESS_LIMIT_PER_TYPE
    )

    # --- 3-gramデータの前処理とチャンク保存 ---
    process_and_chunk_ngrams(
        directory_path=ngram_data_directories[3],
        n=3,
        pickle_output_dir=pickle_output_dir,
        total_vocab_size=total_vocab_size, 
        files_to_load_limit=FILES_TO_PROCESS_LIMIT_PER_TYPE
    )
    
    print(f"\nTotal preprocessing time: {time.time() - start_overall_time:.2f}s")
    print(f"Preprocessed N-gram chunks (including total vocabulary size V) saved in: {pickle_output_dir}")
    print(f"And index files (e.g., 2gram_first_word_index.pkl, 3gram_first_word_index.pkl) also saved in: {pickle_output_dir}")
    print("\n--- Next Step: Update word3gram.py ---")
    print(f"Now, modify word3gram.py to load these index files ({pickle_output_dir}/2gram_first_word_index.pkl, etc.)")
    print("and use them to efficiently find the relevant N-gram chunk files.")