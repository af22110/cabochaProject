import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# SBERTモデル（変更可）
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# --- Step 1: train2_data.xlsx から平均差分ベクトルを作成 ---
train_df = pd.read_excel('train2_data.xlsx')  # 誤文, 正文, ラベル

label_diff_vectors = {}
for label in train_df.iloc[:, 2].unique():
    group = train_df[train_df.iloc[:, 2] == label]
    diffs = []
    for _, row in group.iterrows():
        wrong_vec = model.encode(row.iloc[0])   # 誤文
        correct_vec = model.encode(row.iloc[1]) # 正文
        diff = correct_vec - wrong_vec
        diffs.append(diff)
    label_diff_vectors[label] = np.mean(diffs, axis=0)

# ラベル順を固定（特徴量の順序がブレないように）
sorted_labels = sorted(label_diff_vectors.keys())

# --- Step 2: test2_data.xlsx の誤文に特徴量を適用 ---
test_df = pd.read_excel('test2_data.xlsx')  # 1列目に誤文がある前提

feature_rows = []
for wrong_text in test_df.iloc[:, 0]:
    vec = model.encode(wrong_text)
    features = []
    for label in sorted_labels:
        diff_vec = label_diff_vectors[label]
        sim = cosine_similarity([vec], [diff_vec])[0][0]
        features.append(sim)
    feature_rows.append(features)

# 特徴量DataFrameの作成
feature_df = pd.DataFrame(feature_rows, columns=[f'feature_{label}' for label in sorted_labels])

# --- Step 3: test_df に特徴量を3列目以降に追加して保存 ---
# 2列目までは元のtest2_data.xlsxのままにしておく（1列目: 誤文、2列目: 空または任意）
df_out = pd.concat([test_df, feature_df], axis=1)

df_out.to_excel('test3_data.xlsx', index=False)

print(" test3_data.xlsx に特徴量を保存しました。")
