import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm  # tqdmをインポート

# --- 1. 設定項目 ---
# 使用するBERTモデル
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
# 入力・出力ファイル名
EXCEL_FILE = 'INPUT.xlsx'
# 誤文が記載されている列のインデックス (A列 = 0)
TEXT_COLUMN_INDEX = 0
# 処理を開始する行のインデックス (2行目 = 1)
START_ROW_INDEX = 1
# 結果を出力する開始列のインデックス (O列 = 14)
OUTPUT_COLUMN_START_INDEX = 14

# --- 2. モデルとトークナイザの準備 ---
print("BERTモデルとトークナイザをロードしています...")
# GPUが利用可能であればGPUを、そうでなければCPUを使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()  # モデルを評価モードに設定
except Exception as e:
    print(f"モデルのロード中にエラーが発生しました: {e}")
    print("インターネット接続を確認するか、モデル名が正しいか確認してください。")
    exit()

# --- 3. 擬似尤度スコアと特徴量を計算する関数 ---
def get_pseudo_likelihood_features(text, model, tokenizer, device):
    """
    文章の擬似対数尤度を単語ごとに計算し、その統計的特徴量を返す。
    """
    # セルが空、または文字列でない場合はNoneのリストを返す
    if not isinstance(text, str) or not text.strip():
        return [None, None, None, None]

    try:
        # 文章をトークンIDに変換
        token_ids = tokenizer.encode(text, return_tensors='pt').to(device)
        
        # 文章の本体部分のトークンID ([CLS]と[SEP]を除く)
        input_ids = token_ids[0][1:-1]

        # トークンが存在しない場合はNoneを返す
        if len(input_ids) == 0:
            return [None, None, None, None]

        log_likelihoods = []

        with torch.no_grad():
            for i in range(len(input_ids)):
                # マスクするトークンの元のIDを保存
                original_token_id = input_ids[i].item()
                
                # 1トークンを[MASK]トークンIDで置き換える
                masked_input_ids = input_ids.clone()
                masked_input_ids[i] = tokenizer.mask_token_id
                
                # [CLS]と[SEP]を再度結合してモデルへの入力形式にする
                full_masked_ids = torch.cat([token_ids[0][:1], masked_input_ids, token_ids[0][-1:]]).unsqueeze(0)

                # モデルで予測を実行
                outputs = model(full_masked_ids)
                logits = outputs.logits
                
                # マスクした位置のロジットを取得 (先頭[CLS]の分オフセット+1)
                mask_logits = logits[0, i + 1, :]
                
                # 数値的に安定した対数ソフトマックス関数で対数確率を計算
                log_probs = torch.nn.functional.log_softmax(mask_logits, dim=0)
                
                # 元のトークンの対数確率を取得してリストに追加
                token_log_prob = log_probs[original_token_id].item()
                log_likelihoods.append(token_log_prob)

        # 特徴量を計算
        sum_val = np.sum(log_likelihoods)
        mean_val = np.mean(log_likelihoods)
        # トークンが1つの場合、分散は0とする
        var_val = np.var(log_likelihoods) if len(log_likelihoods) > 1 else 0.0
        min_val = np.min(log_likelihoods)

        return [sum_val, mean_val, var_val, min_val]

    except Exception as e:
        print(f"\nエラーが発生した文章: '{text}'") # tqdm使用時に見やすいように改行を追加
        print(f"エラー詳細: {e}")
        return ["ERROR", "ERROR", "ERROR", "ERROR"]


# --- 4. メイン処理 ---
try:
    print(f"'{EXCEL_FILE}' を読み込んでいます...")
    # 1行目をヘッダーとして扱わない設定
    df = pd.read_excel(EXCEL_FILE, header=None)
except FileNotFoundError:
    print(f"エラー: '{EXCEL_FILE}' が見つかりません。コードと同じフォルダに配置してください。")
    exit()

# 出力用の列ヘッダーを1行目に設定
output_headers = ['合計(対数尤度)', '平均(対数尤度)', '分散(対数尤度)', '最小値(対数尤度)']
for i, header in enumerate(output_headers):
    # .locを使用して確実に値を設定
    df.loc[0, OUTPUT_COLUMN_START_INDEX + i] = header

# 2行目から順に処理
print("各文章の特徴量抽出を開始します...")
# tqdmを使用してループの進捗バーを表示
for index in tqdm(range(START_ROW_INDEX, len(df)), desc="Processing sentences"):
    sentence = df.iloc[index, TEXT_COLUMN_INDEX]
    
    # 特徴量を計算
    features = get_pseudo_likelihood_features(sentence, model, tokenizer, device)
    
    # 結果をDataFrameの対応するセルに格納 (O列から)
    for i, feature in enumerate(features):
        df.loc[index, OUTPUT_COLUMN_START_INDEX + i] = feature

# --- 5. 結果をファイルに上書き保存 ---
try:
    print(f"\n処理結果を '{EXCEL_FILE}' に保存しています...") # tqdmの表示と重ならないように改行を追加
    # 元のファイルにヘッダーとインデックスなしで上書き
    df.to_excel(EXCEL_FILE, index=False, header=False)
    print("処理が正常に完了しました。")
except Exception as e:
    print(f"ファイルの保存中にエラーが発生しました: {e}")