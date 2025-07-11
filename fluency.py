import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

# === 1. モデルとトークナイザの準備 ===
model_name = "liwii/fluency-score-classification-ja"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# === 2. 入力ファイルの読み込み（CSVまたはExcel） ===
# ファイルパスと列名を指定してください
input_file = "input.xlsx"  # 例: input.csv や input.xlsx
text_column = "文"         # 文が入っている列名に変更

if input_file.endswith(".csv"):
    df = pd.read_csv(input_file)
elif input_file.endswith(".xlsx"):
    df = pd.read_excel(input_file)
else:
    raise ValueError("CSVまたはExcel形式のファイルを指定してください。")

# === 3. 流暢さスコアを計算 ===
fluency_scores = []

for text in tqdm(df[text_column], desc="Calculating fluency scores"):
    if not isinstance(text, str) or text.strip() == "":
        fluency_scores.append(None)
        continue

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        score = probs[:, 1].item()  # クラス1（流暢）の確率
        fluency_scores.append(score)

# === 4. 結果をDataFrameに追加して保存 ===
df["fluency_score"] = fluency_scores

# 保存
output_file = "output_with_fluency_score.xlsx"
df.to_excel(output_file, index=False)
print(f"✔️ スコア付きのデータを保存しました：{output_file}")
