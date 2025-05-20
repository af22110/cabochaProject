import sqlite3
import logging
from pathlib import Path
from tqdm import tqdm
import time # 時間計測のため

# ─── 設定 ──────────────────────────────────────────
DB_PATH = Path("ngram_counts_v3.sqlite") # DBファイル名を変更して新規作成を推奨
# BASE_DIR = Path("~/cabochaProject/nwc2010-ngrams/word/over99").expanduser() # テスト用 (頻度100以上)
BASE_DIR = Path("~/cabochaProject/nwc2010-ngrams/word/over9").expanduser()   # 本番用 (頻度10以上)
COMMIT_BATCH_SIZE = 50_000     # executemany に一度に渡すN-gramのレコード数
LOG_INTERVAL_LINES = 2_000_000 # 何行処理ごとに進捗ログを出すか (tqdmがあるので必須ではない)

# SQLite 高速化 PRAGMA
PRAGMAS = [
    ("synchronous", "OFF"),    # 最速だがクラッシュ時のDB破損リスクあり。NORMAL推奨の場合も。
    ("journal_mode", "WAL"),   # 書き込みと読み込みの並行性向上
    ("cache_size", "-200000"), # メモリキャッシュを約200MBに（システムのメモリ量に応じて調整）
    ("temp_store", "MEMORY"),  # 一時ファイルをメモリ上に作成
]

logging.basicConfig(
    level=logging.INFO, # 通常はINFO, デバッグ時はDEBUGに変更
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ─── データベース接続コンテキスト ────────────────────
class SQLiteDB:
    def __init__(self, path: Path):
        self.path = path
        self.conn = None

    def __enter__(self):
        logger.debug("DB接続開始: %s", self.path)
        self.conn = sqlite3.connect(self.path, timeout=30.0) # タイムアウト設定
        cursor = self.conn.cursor()
        for key, val in PRAGMAS:
            logger.debug("PRAGMA %s = %s を設定", key, val)
            cursor.execute(f"PRAGMA {key} = {val};")
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type:
                logger.error("DBエラー発生、ロールバックします: %s", exc_val, exc_info=(exc_type, exc_val, exc_tb))
                self.conn.rollback()
            else:
                logger.debug("DB変更をコミットします (最終コミット)")
                self.conn.commit()
            self.conn.close()
            logger.debug("DB接続終了: %s", self.path)

# ─── テーブル作成 ─────────────────────────────────
def create_ngram_tables(db_path: Path):
    logger.info("テーブル作成/確認処理を開始: %s", db_path)
    with SQLiteDB(db_path) as conn:
        cursor = conn.cursor()
        # WITHOUT ROWID は主キーが INTEGER PRIMARY KEY でない場合に有効
        # また、主キー検索以外の検索が少ない場合に有利
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS three_grams (
                w1 TEXT NOT NULL, w2 TEXT NOT NULL, w3 TEXT NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (w1, w2, w3)
            ) WITHOUT ROWID; 
        """)
        logger.info("テーブル 'three_grams' を作成または確認しました。")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS two_grams (
                w1 TEXT NOT NULL, w2 TEXT NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (w1, w2)
            ) WITHOUT ROWID;
        """)
        logger.info("テーブル 'two_grams' を作成または確認しました。")
    logger.info("テーブル作成/確認処理が完了しました。")

# ─── N-gram ファイル投入 ─────────────────────────────
def _write_batch(cursor: sqlite3.Cursor, table_name_for_log: str, ngram_size: int, batch: list):
    if not batch:
        return 0
    
    # logger.debug("  バッチ書き込み開始 (%s, %d件)", table_name_for_log, len(batch))
    try:
        if ngram_size == 3:
            cursor.executemany(
                "INSERT OR REPLACE INTO three_grams (w1, w2, w3, count) VALUES (?, ?, ?, ?);",
                batch
            )
        elif ngram_size == 2: # 2-gramとそれ以外で分岐
            cursor.executemany(
                "INSERT OR REPLACE INTO two_grams (w1, w2, count) VALUES (?, ?, ?);",
                batch
            )
        else:
            logger.error("未対応のngram_sizeです: %d", ngram_size)
            return 0
        # logger.debug("  バッチ書き込み成功 (%s, %d件)", table_name_for_log, len(batch))
        return len(batch)
    except sqlite3.Error as e:
        logger.error("  バッチ書き込みエラー (%s): %s. バッチ先頭数件: %s", table_name_for_log, e, batch[:3])
        # エラー発生時、問題のあるデータを除外してリトライするなどの処理も考えられる
        return 0 # 失敗したら0件挿入として扱う

def insert_ngrams_from_files(db_path: Path, ngram_size: int, ngrams_folder: Path, file_prefix: str):
    table_name_for_log = f"{ngram_size}-grams" # ログ表示用
    file_pattern = f"{file_prefix}*"
    # .xz で終わらないファイル（展開済みファイル）を対象にする
    files_to_process = sorted([f for f in ngrams_folder.glob(file_pattern) if not f.name.endswith(".xz")])

    if not files_to_process:
        logger.warning("処理対象ファイルが見つかりません: %s in %s", file_pattern, ngrams_folder)
        return

    logger.info("[%s] %d個のファイル処理を開始します: %s", table_name_for_log, len(files_to_process), ngrams_folder)

    total_lines_processed_all_files = 0
    total_ngrams_inserted_all_files = 0
    overall_start_time = time.time()

    with SQLiteDB(db_path) as conn: # DB接続は一度だけ行う
        cursor = conn.cursor()
        current_batch = []

        for file_idx, ngram_file_path in enumerate(files_to_process):
            logger.info("[%s] ファイル %d/%d: '%s' の処理開始",
                        table_name_for_log, file_idx + 1, len(files_to_process), ngram_file_path.name)
            file_start_time = time.time()
            lines_in_file_processed = 0
            ngrams_in_file_added_to_batch = 0

            try:
                with ngram_file_path.open('r', encoding='utf-8') as f:
                    # tqdmでファイル全体の行数を取得できない場合でも進捗は表示される
                    for line_num, line_content in enumerate(tqdm(f, desc=f"  {ngram_file_path.name}", unit="行", leave=False), 1):
                        lines_in_file_processed += 1
                        total_lines_processed_all_files +=1
                        line_content = line_content.rstrip('\n\r') # 改行のみ除去

                        if not line_content:
                            # logger.debug("    L%d: 空行スキップ", line_num)
                            continue

                        # logger.debug("    L%d RAW: '%s'", line_num, line_content)

                        ngram_part_str = ""
                        count_str = ""
                        
                        # 1. タブでN-gram部分と頻度部分を分割試行
                        tab_parts = line_content.split('\t')
                        if len(tab_parts) >= 2 and tab_parts[-1].isdigit():
                            ngram_part_str = "\t".join(tab_parts[:-1])
                            count_str = tab_parts[-1]
                            # logger.debug("    L%d タブ区切り成功: ngram_part='%s', count='%s'", line_num, ngram_part_str, count_str)
                        else:
                            # 2. タブで区切れない場合、行の最後尾から見て最初のスペースを区切りとして試行
                            last_space_idx = line_content.rfind(' ')
                            if last_space_idx != -1:
                                potential_ngram_part = line_content[:last_space_idx]
                                potential_count_str = line_content[last_space_idx+1:]
                                if potential_count_str.isdigit():
                                    ngram_part_str = potential_ngram_part.strip() # N-gram部分の前後の空白除去
                                    count_str = potential_count_str
                                    # logger.debug("    L%d スペース区切り成功: ngram_part='%s', count='%s'", line_num, ngram_part_str, count_str)
                                else:
                                    # logger.warning("    L%d スキップ (最後の要素が数値でない@スペース区切り): '%s'", line_num, line_content)
                                    continue
                            else:
                                # logger.warning("    L%d スキップ (適切な区切り文字なし): '%s'", line_num, line_content)
                                continue
                        
                        try:
                            count = int(count_str)
                        except ValueError:
                            # logger.warning("    L%d スキップ (頻度が数値でない: '%s'): '%s'", line_num, count_str, line_content)
                            continue

                        # 3. N-gram部分をスペースで分割してトークンリストにする
                        #    （元データが "単語A 単語B 単語C" のようになっているため）
                        ngram_tokens = [token for token in ngram_part_str.split(' ') if token] # 空白トークン除去

                        if len(ngram_tokens) != ngram_size:
                            # logger.warning("    L%d スキップ (N-gramサイズ不一致 Exp:%d, Got:%d, Tokens:%s): '%s'",
                            #                line_num, ngram_size, len(ngram_tokens), ngram_tokens, line_content)
                            continue
                        
                        # logger.debug("    L%d 追加候補: N-gram:%s, Count:%d", line_num, ngram_tokens, count)
                        current_batch.append((*ngram_tokens, count))
                        ngrams_in_file_added_to_batch += 1

                        # バッチサイズに達したらDBへ書き込み
                        if len(current_batch) >= COMMIT_BATCH_SIZE:
                            inserted_count = _write_batch(cursor, table_name_for_log, ngram_size, current_batch)
                            total_ngrams_inserted_all_files += inserted_count
                            conn.commit() # 定期的なコミット
                            logger.info("[%s]   %s - L%d まで処理、%d件をDBへコミット (総挿入: %d件)",
                                        table_name_for_log, ngram_file_path.name, line_num, inserted_count, total_ngrams_inserted_all_files)
                            current_batch.clear()
                        
                        # # tqdmがあるため、このログは冗長かもしれない
                        # if lines_in_file_processed % LOG_INTERVAL_LINES == 0:
                        #     logger.info("[%s]   %s - %d行処理完了 (現在のバッチサイズ: %d)",
                        #                 table_name_for_log, ngram_file_path.name, lines_in_file_processed, len(current_batch))

            except FileNotFoundError:
                logger.error("[%s] ファイルが見つかりません: %s", table_name_for_log, ngram_file_path)
                continue # 次のファイルへ
            except Exception as e:
                logger.error("[%s] '%s'処理中に予期せぬエラー: %s", table_name_for_log, ngram_file_path.name, e, exc_info=True)
                # エラー発生時、現在のバッチをどうするか、処理を続けるかなどを検討
                current_batch.clear() # 安全のためバッチをクリア
                conn.rollback() # 現在のトランザクションをロールバック
                continue # 次のファイルへ

            # ファイル処理後、残っているバッチがあれば書き込み
            if current_batch:
                logger.info("[%s]   %s - ファイル末尾の残りバッチ %d件を書き込み",
                            table_name_for_log, ngram_file_path.name, len(current_batch))
                inserted_count = _write_batch(cursor, table_name_for_log, ngram_size, current_batch)
                total_ngrams_inserted_all_files += inserted_count
                current_batch.clear()
            
            conn.commit() # ファイルごとの最終コミット
            file_duration = time.time() - file_start_time
            logger.info("[%s] ファイル %d/%d: '%s' の処理完了。追加N-gram: %d, 所要時間: %.2f秒",
                        table_name_for_log, file_idx + 1, len(files_to_process), ngram_file_path.name,
                        ngrams_in_file_added_to_batch, file_duration)

    overall_duration = time.time() - overall_start_time
    logger.info("[%s] === 全ファイル処理完了 ===", table_name_for_log)
    logger.info("[%s] 総処理行数: %d", table_name_for_log, total_lines_processed_all_files)
    logger.info("[%s] 総挿入N-gram数 (INSERT OR REPLACE): %d", table_name_for_log, total_ngrams_inserted_all_files)
    logger.info("[%s] 総所要時間: %.2f秒 (%.2f分)", table_name_for_log, overall_duration, overall_duration / 60)


# ─── メイン処理 ───────────────────────────────────
if __name__ == "__main__":
    main_start_time = time.time()
    logger.info("===== N-gramデータ SQLite投入処理 開始 =====")
    
    create_ngram_tables(DB_PATH)

    logger.info("\n===== 3-gram データ投入処理 開始 =====")
    insert_ngrams_from_files(DB_PATH, 3, BASE_DIR / "3gms", "3gm-")

    logger.info("\n===== 2-gram データ投入処理 開始 =====")
    insert_ngrams_from_files(DB_PATH, 2, BASE_DIR / "2gms", "2gm-")
            
    main_duration = time.time() - main_start_time
    logger.info("\n===== 全てのN-gramデータ投入処理が完了しました =====")
    logger.info("総所要時間: %.2f秒 (%.2f分)", main_duration, main_duration / 60)

    # VACUUM処理 (任意、DBファイルサイズ削減とパフォーマンス最適化に役立つが時間がかかる)
    # logger.info("\n===== データベースのVACUUM処理を開始します (時間がかかることがあります)... =====")
    # vacuum_start_time = time.time()
    # with SQLiteDB(DB_PATH) as conn_vacuum:
    #     conn_vacuum.execute("VACUUM;")
    # vacuum_duration = time.time() - vacuum_start_time
    # logger.info("===== VACUUM処理が完了しました。所要時間: %.2f秒 (%.2f分) =====" , vacuum_duration, vacuum_duration / 60)