import pandas as pd
import numpy as np

# 書き換えるExcelファイル名
input_filename = 'train2_data.xlsx'
output_filename = 'OUTPUT.xlsx' # 上書きを避けるため、別のファイル名で保存します

try:
    # Excelファイルの読み込み
    df = pd.read_excel(input_filename)

    # DataFrameの列名を取得
    columns = df.columns

    # --- 列の位置（インデックス）で処理を実行 ---
    # インデックスは0から始まるため、「N番目」の列は「N-1」で指定します。

    # D列（4番目の列、インデックスは3）の処理: 絶対値に変換
    col_d_index = 4
    if col_d_index < len(columns):
        # .iloc[:, col_d_index] で列の位置を指定してデータを操作します
        df.iloc[:, col_d_index] = pd.to_numeric(df.iloc[:, col_d_index], errors='coerce').abs()
        print(f"4番目の列（{columns[col_d_index]}）の値を絶対値に変換しました。")

    # G列（7番目の列、インデックスは6）の処理: 0と1を逆にする
    col_g_index = 7
    if col_g_index < len(columns):
        # G列の値が0なら1に、1なら0に変換します。それ以外の値はそのままです。
        df.iloc[:, col_g_index] = df.iloc[:, col_g_index].apply(lambda x: 1 if x == 0 else (0 if x == 1 else x))
        print(f"7番目の列（{columns[col_g_index]}）の0と1を逆にしました。")

    # N列（14番目の列、インデックスは13）の処理: 逆数に変換
    col_n_index = 14
    if col_n_index < len(columns):
        # 0で割るエラーを避けるため、0の場合はnumpyのinf(無限大)に置き換えます
        # 文字列などが含まれている可能性を考慮し、数値に変換してから処理します
        numeric_col = pd.to_numeric(df.iloc[:, col_n_index], errors='coerce')
        df.iloc[:, col_n_index] = 1 / numeric_col.replace(0, np.inf)
        print(f"14番目の列（{columns[col_n_index]}）の値を逆数に変換しました。")

    # O列（15番目の列、インデックスは14）の処理: 絶対値に変換
    col_o_index = 15
    if col_o_index < len(columns):
        df.iloc[:, col_o_index] = pd.to_numeric(df.iloc[:, col_o_index], errors='coerce').abs()
        print(f"15番目の列（{columns[col_o_index]}）の値を絶対値に変換しました。")

    # Q列（17番目の列、インデックスは16）の処理: 絶対値に変換
    col_q_index = 17
    if col_q_index < len(columns):
        df.iloc[:, col_q_index] = pd.to_numeric(df.iloc[:, col_q_index], errors='coerce').abs()
        print(f"17番目の列（{columns[col_q_index]}）の値を絶対値に変換しました。")

    # 変更を適用したデータフレームを新しいExcelファイルに書き出す
    df.to_excel(output_filename, index=False)

    print(f"\n処理が完了しました。結果は {output_filename} に保存されました。")

except FileNotFoundError:
    print(f"エラー: {input_filename} が見つかりません。")
except Exception as e:
    print(f"エラーが発生しました: {e}")