import os
import pickle
import time
import glob
from tqdm.notebook import tqdm # tqdm を追加

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
                    continue
                ngram_string = parts[0]
                try:
                    frequency = int(parts[1])
                except ValueError:
                    continue
                words = ngram_string.split(' ')
                if len(words) != n:
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

        expected_prefix = f"{n_type}gm-"
        file_paths = sorted(glob.glob(os.path.join(dir_path, f"{expected_prefix}*")))
        
        if files_to_load_limit_per_type is not None:
            file_paths = file_paths[:files_to_load_limit_per_type]

        if not file_paths:
            print(f"No files matching prefix '{expected_prefix}' found in {dir_path}. Skipping V calculation for this type.")
            continue

        print(f"  Scanning {len(file_paths)} {n_type}-gram files in {dir_path} for vocabulary...")
        
        # tqdm を使ってV計算の進行状況を表示
        for i, filepath in tqdm(enumerate(file_paths), total=len(file_paths), desc=f"Calculating V for {n_type}-grams"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        parts = line.split('\t')
                        if len(parts) != 2: continue
                        
                        ngram_string = parts[0]
                        words = ngram_string.split(' ')
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

    # tqdm を使ってチャンク処理の進行状況を表示
    for i, filepath in tqdm(enumerate(file_paths), total=len(file_paths), desc=f"Chunking {n}-grams"):
        ngrams_in_file = load_ngrams_from_file(filepath, n)

        # 保存（ファイル名ごとに）
        # 元のファイル名 (例: 2gm-0000.txt) をそのまま使う
        base_filename = os.path.basename(filepath)
        # 出力チャンクファイル名は、word3gram.pyで参照しやすいように調整
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

            # インデックスを構築: このチャンクに含まれるN-gramの最初の単語を記録
            for ngram_tuple in ngrams_in_file.keys():
                first_word = ngram_tuple[0]
                if first_word not in ngram_first_word_index:
                    ngram_first_word_index[first_word] = set() # 重複を避けるためsetを使用
                ngram_first_word_index[first_word].add(output_chunk_filename) # ファイル名を追加
            
        except Exception as e:
            print(f"Error saving chunk {output_chunk_filename}: {e}")
            continue

    # インデックスをファイルに保存
    index_filename = f"{n}gram_first_word_index.pkl"
    index_full_path = os.path.join(pickle_output_dir, index_filename)
    try:
        # set は pickle できないため、list に変換してから保存
        # または、set のままでも pickle は可能ですが、ここでは念のため
        # dict of lists に変換する例
        serializable_index = {word: list(files) for word, files in ngram_first_word_index.items()}

        with open(index_full_path, "wb") as f:
            pickle.dump(serializable_index, f, pickle.HIGHEST_PROTOCOL)
        print(f"Successfully saved index for {n}-grams to {index_full_path}")
    except Exception as e:
        print(f"Error saving index for {n}-grams to {index_full_path}: {e}")
        return False

    print(f"Finished processing {n}-grams. Total {processed_count} N-grams saved across {len(file_paths)} chunks.")
    return True


#main


import os
import pickle
import time
import glob
from tqdm.notebook import tqdm # tqdm を追加

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
                    continue
                ngram_string = parts[0]
                try:
                    frequency = int(parts[1])
                except ValueError:
                    continue
                words = ngram_string.split(' ')
                if len(words) != n:
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

        expected_prefix = f"{n_type}gm-"
        file_paths = sorted(glob.glob(os.path.join(dir_path, f"{expected_prefix}*")))
        
        if files_to_load_limit_per_type is not None:
            file_paths = file_paths[:files_to_load_limit_per_type]

        if not file_paths:
            print(f"No files matching prefix '{expected_prefix}' found in {dir_path}. Skipping V calculation for this type.")
            continue

        print(f"  Scanning {len(file_paths)} {n_type}-gram files in {dir_path} for vocabulary...")
        
        # tqdm を使ってV計算の進行状況を表示
        for i, filepath in tqdm(enumerate(file_paths), total=len(file_paths), desc=f"Calculating V for {n_type}-grams"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        parts = line.split('\t')
                        if len(parts) != 2: continue
                        
                        ngram_string = parts[0]
                        words = ngram_string.split(' ')
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

    # tqdm を使ってチャンク処理の進行状況を表示
    for i, filepath in tqdm(enumerate(file_paths), total=len(file_paths), desc=f"Chunking {n}-grams"):
        ngrams_in_file = load_ngrams_from_file(filepath, n)

        # 保存（ファイル名ごとに）
        # 元のファイル名 (例: 2gm-0000.txt) をそのまま使う
        base_filename = os.path.basename(filepath)
        # 出力チャンクファイル名は、word3gram.pyで参照しやすいように調整
        # 拡張子が複数ある場合 (例: .txt.pkl) を考慮し、元のファイル名から .txt を除去する
        output_chunk_filename = f"{n}gram_{base_filename.replace('.txt', '')}.pkl" 
        
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

            # インデックスを構築: このチャンクに含まれるN-gramの最初の単語を記録
            for ngram_tuple in ngrams_in_file.keys():
                first_word = ngram_tuple[0]
                if first_word not in ngram_first_word_index:
                    ngram_first_word_index[first_word] = set() # 重複を避けるためsetを使用
                ngram_first_word_index[first_word].add(output_chunk_filename) # ファイル名を追加
            
        except Exception as e:
            print(f"Error saving chunk {output_chunk_filename}: {e}")
            continue

    # インデックスをファイルに保存
    index_filename = f"{n}gram_first_word_index.pkl"
    index_full_path = os.path.join(pickle_output_dir, index_filename)
    try:
        # set はそのままpickle可能ですが、list に変換してから保存する方が、
        # 後で word3gram.py で処理しやすいため推奨
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
    # N-gramデータが格納されているベースディレクトリ
    # Kaggleの場合、'/kaggle/input/your-dataset-name' となります
    # 例: '/kaggle/input/hindo10'
    base_data_path = os.path.expanduser('/kaggle/input/hindo10') 
    
    # Pickleファイルを保存する場所 (word3gram.pyから参照するパス)
    # Kaggle Notebooksでは通常 '/kaggle/working/' を利用します
    # ただし、実行環境によってはこのディレクトリはテンポラリであり、
    # 永続化したい場合はKaggle Datasetsとして別途出力する必要があります
    pickle_output_dir = os.path.expanduser('/kaggle/working/pickle_chunks/') 

    # メモリ対策: V計算、および各N-gramタイプで処理（読み込んで保存）するファイル数を制限
    # None にするとディレクトリ内の該当ファイルを全て処理します
    FILES_TO_PROCESS_LIMIT_PER_TYPE = None # 例: 各N-gramタイプで最初の100ファイルのみ処理
    # --- 設定項目ここまで ---

    # Pickleファイルの出力ディレクトリを作成
    os.makedirs(pickle_output_dir, exist_ok=True)
    print(f"Pickle chunk output directory: {pickle_output_dir}")

    # --- 全体の語彙数を計算 (最初に実行) ---
    # 2-gramと3-gramのデータディレクトリをまとめて指定し、全てのユニーク単語からVを計算
    ngram_data_directories = {
        2: base_data_path,
        3: base_data_path,
    }
    total_vocab_size = calculate_total_vocabulary_size(ngram_data_directories, 
                                                        files_to_load_limit_per_type=FILES_TO_PROCESS_LIMIT_PER_TYPE)
    
    if total_vocab_size == 0:
        print("Critical Error: Calculated total vocabulary size (V) is 0. Cannot proceed with N-gram processing.")
        print("Please check your N-gram source files and the `load_ngrams_from_file` / `calculate_total_vocabulary_size` logic.")
        import sys; sys.exit(1) # exit() は対話型環境で問題を起こす可能性があるので sys.exit() を推奨
    
    print(f"\nFinal calculated total vocabulary size (V): {total_vocab_size}")


    # --- 2-gramデータの前処理とチャンク保存 ---
    process_and_chunk_ngrams(
        directory_path=ngram_data_directories[2],
        n=2,
        pickle_output_dir=pickle_output_dir,
        total_vocab_size=total_vocab_size, # 計算済みのVを渡す
        files_to_load_limit=FILES_TO_PROCESS_LIMIT_PER_TYPE
    )

    # --- 3-gramデータの前処理とチャンク保存 ---
    process_and_chunk_ngrams(
        directory_path=ngram_data_directories[3],
        n=3,
        pickle_output_dir=pickle_output_dir,
        total_vocab_size=total_vocab_size, # 計算済みのVを渡す
        files_to_load_limit=FILES_TO_PROCESS_LIMIT_PER_TYPE
    )
    
    print(f"\nTotal preprocessing time: {time.time() - start_overall_time:.2f}s")
    print(f"Preprocessed N-gram chunks (including total vocabulary size V) saved in: {pickle_output_dir}")
    print(f"And index files (e.g., 2gram_first_word_index.pkl, 3gram_first_word_index.pkl) also saved in: {pickle_output_dir}")
    print("\n--- Next Step: Update word3gram.py ---")
    print(f"Now, modify word3gram.py to load these index files ({pickle_output_dir}/2gram_first_word_index.pkl, etc.)")
    print("and use them to efficiently find the relevant N-gram chunk files.")