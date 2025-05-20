from pykakasi import kakasi
import sys

# kakasiコンバーターを初期化
# 漢字からカタカナへの変換を設定 (J -> K)
kks = kakasi()
kks.setMode("J", "K") # ここを "J", "H" から "J", "K" に変更
converter = kks.getConverter()

input_filename = "japanese_word_list.txt"  # 入力ファイル名
output_filename = "japanese_word_list2.txt" # 出力ファイル名

try:
    with open(input_filename, "r", encoding="utf-8") as infile, \
         open(output_filename, "w", encoding="utf-8") as outfile:

        print(f"Reading words from '{input_filename}' and generating '{output_filename}'...")

        for line in infile:
            word = line.strip() # 前後の空白と改行を取り除く

            if not word: # 空行はスキップ
                continue

            try:
                # 単語をカタカナに変換して読みを取得
                reading = converter.do(word)
                # 単語と読みをタブ区切りで出力ファイルに書き込む
                # pykakasiの"K"モードは通常、全角カタカナを出力します
                outfile.write(f"{word}\t{reading}\n")

            except Exception as e:
                # 変換エラーが発生した場合の処理
                print(f"Warning: Could not convert '{word}': {e}", file=sys.stderr)
                # エラーが発生した単語は、読みを空にして書き込むか、スキップするか選択
                # outfile.write(f"{word}\tERROR\n") # 例: エラー表示付きで書き込む

        print(f"Successfully generated '{output_filename}'.")
        # カタカナ変換の確認を促すメッセージを追加
        print("Please review the output file for accuracy and confirm readings are in Katakana, especially for complex words.")


except FileNotFoundError:
    print(f"Error: Input file '{input_filename}' not found.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}", file=sys.stderr)
    sys.exit(1)