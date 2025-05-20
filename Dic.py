import os
import csv
import sys

# --- 設定 ---
# 修正前 (以前試したパス)
# MECAB_IPADIC_CSV_DIR = r"/home/mitsu/cabochaProject/mecab-ipadic-2.7.0-20070610/csv" # csv が余計だった

# 修正後 (pwd で確認したディレクトリのパス)
MECAB_IPADIC_CSV_DIR = r"/home/mitsu/cabochaProject/mecab-ipadic-2.7.0-20070610" # csv を削除

# 出力する単語リストファイル名
OUTPUT_WORD_LIST_FILE = 'japanese_word_list.txt'


# CSVファイルのエンコーディング (IPADicは通常UTF-8)
# CSV_ENCODING = 'utf-8'  # 修正前
CSV_ENCODING = 'euc-jp'   # 修正後
# 見出し語（基本形）が格納されている列のインデックス (0から開始)
# IPADicの多くのエントリで基本形は11列目 -> インデックス 10
BASE_FORM_COLUMN_INDEX = 10

# --- 抽出処理 ---
word_set = set()
csv_files_processed = 0

print(f"MeCab IPADic CSV ディレクトリ: {MECAB_IPADIC_CSV_DIR} から単語を抽出します...")

if not os.path.isdir(MECAB_IPADIC_CSV_DIR):
    print(f"エラー: 指定されたディレクトリが見つかりません: {MECAB_IPADIC_CSV_DIR}")
    print("MECAB_IPADIC_CSV_DIR のパスが正しいか確認してください。")
    sys.exit(1)

try:
    # 指定ディレクトリ内の全ファイル・フォルダをリスト
    for entry_name in os.listdir(MECAB_IPADIC_CSV_DIR):
        entry_path = os.path.join(MECAB_IPADIC_CSV_DIR, entry_name)

        # ファイルであり、かつ.csvで終わるものだけを対象
        if os.path.isfile(entry_path) and entry_name.lower().endswith('.csv'):
            # NOTE: Matrix.csv, char.csv, pos-id.csv など、単語エントリを含まないCSVもありますが、
            # エラー処理を入れれば全て試行しても問題ありません。ここでは全てのCSVを試します。
            print(f"  Processing {entry_name}...")
            csv_files_processed += 1
            try:
                # CSVファイルを読み込みモードで開く
                with open(entry_path, mode='r', encoding=CSV_ENCODING) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        # 行が最低限の列数を持っているかチェック
                        if len(row) > BASE_FORM_COLUMN_INDEX:
                            base_form = row[BASE_FORM_COLUMN_INDEX]
                            # 基本形が '*' でない（存在する） かつ 空でないかチェック
                            if base_form != '*' and base_form.strip() != '':
                                word_set.add(base_form.strip()) # 前後の空白を除去して追加
            except Exception as e:
                print(f"  警告: ファイル {entry_name} の読み込みまたは処理中にエラーが発生しました: {e}")
                # エラーが発生しても他のファイルを続行

    print(f"合計 {csv_files_processed} 個のCSVファイルを処理しました。")

except Exception as e:
    print(f"致命的なエラー: ディレクトリ '{MECAB_IPADIC_CSV_DIR}' の処理中にエラーが発生しました: {e}")
    sys.exit(1)


# 重複排除された単語をソート
sorted_word_list = sorted(list(word_set))

# 結果をファイルに書き出し
print(f"抽出した単語リストを '{OUTPUT_WORD_LIST_FILE}' に書き出します ({len(sorted_word_list)} 単語)...")
try:
    with open(OUTPUT_WORD_LIST_FILE, mode='w', encoding='utf-8') as f:
        for word in sorted_word_list:
            f.write(word + '\n')
    print("書き出しが完了しました。")

except Exception as e:
    print(f"エラー: 単語リストファイル '{OUTPUT_WORD_LIST_FILE}' の書き出し中にエラーが発生しました: {e}")
    sys.exit(1)

print(f"\n'{OUTPUT_WORD_LIST_FILE}' ファイルが作成されました。")
print("このファイルを、モデル解析スクリプトと同じディレクトリに置いてください。")