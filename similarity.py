import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- SBERTモデルの指定 ---
model_name = 'jinaai/jina-embeddings-v3'

print(f"Loading SBERT model: {model_name}...")

try:
    model = SentenceTransformer(model_name, trust_remote_code=True)
    print(f"Model '{model_name}' loaded successfully.")
except Exception as e:
    print(f"\n--- エラー ---\nモデルの読み込みに失敗しました。\n詳細: {e}")
    exit()

# --- Step 1: train2_data.xlsx から平均差分ベクトルを作成 ---
print("\nReading train2_data.xlsx and creating mean difference vectors...")

train_df = pd.read_excel('train2_data.xlsx')  # 誤文, 正文, ラベル

label_diff_vectors = {}
for label in sorted(train_df.iloc[:, 2].unique()):
    print(f"Processing label: {label}...")
    group = train_df[train_df.iloc[:, 2] == label]

    wrong_texts = group.iloc[:, 0].tolist()
    correct_texts = group.iloc[:, 1].tolist()

    wrong_vecs = model.encode(wrong_texts, show_progress_bar=True)
    correct_vecs = model.encode(correct_texts, show_progress_bar=True)

    diffs = correct_vecs - wrong_vecs
    label_diff_vectors[label] = np.mean(diffs, axis=0)

sorted_labels = sorted(label_diff_vectors.keys())
print("\nMean difference vectors created.")

# --- Step 2: test2_data.xlsx の1列目（新しい誤文）に特徴量を適用 ---
print("\nApplying features to test data (test2_data.xlsx)...")

test_df = pd.read_excel('test2_data.xlsx')  # 新しい文章が1列目にある前提
test_texts = test_df.iloc[:, 0].tolist()
test_vecs = model.encode(test_texts, show_progress_bar=True)

feature_rows = []
print("\nCalculating similarity features...")
for i, vec in enumerate(test_vecs):
    if i % 100 == 0:
        print(f"Processing {i+1}/{len(test_vecs)} samples...")

    similarity_features = []
    for label in sorted_labels:
        diff_vec = label_diff_vectors[label]
        sim = cosine_similarity([vec], [diff_vec])[0][0]
        similarity_features.append(sim)
    
    feature_rows.append(similarity_features)

# 特徴量DataFrameの作成
similarity_df = pd.DataFrame(feature_rows, columns=[f'similarity_{label}' for label in sorted_labels])

# --- Step 3: test2_data.xlsx の結果に特徴量を結合して保存 ---
print("\nCombining original test data with features...")
df_out = pd.concat([test_df, similarity_df], axis=1)

# 出力
df_out.to_excel('t.xlsx', index=False)

print("\nProcess completed.")
print("test2_with_features.xlsx に以下の特徴量が保存されました:")
for i, label in enumerate(sorted_labels):
    print(f"  {label}: similarity_{label} (列{chr(66+i)})")
