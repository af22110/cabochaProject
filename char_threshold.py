import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 1) セグメント一覧（漢字2文字以上）
segments = [
    # 正しいもの 10（ラベル0）
    "情報", "運動会", "観光", "環境", "文字列",
    "漢字", "図書館", "人間", "交通", "新聞",
    # 誤り 10（ラベル1）
    "自然再害", "福署長", "技術革真", "解革", "幣害",
    "鳥龍茶", "完壁", "指適", "成積", "重復"
]

# 2) 各セグメントに対する言語モデルから得られた log 確率スコア（例）
#    （実際は Hugging Face LM などで計算してください）
scores = np.array([
    -4.0,  -3.5,  -4.2,  -5.1,  -3.8,
    -4.5,  -3.9,  -4.1,  -4.3,  -3.7,
    -9.8, -10.2,  -9.5, -11.0, -10.8,
   -12.1, -10.5, -11.3,  -9.9, -10.0
])

# 3) 正解ラベル（0=正しい／1=誤り）
labels = np.array([0]*10 + [1]*10)

# 4) 閾値をスイープして accuracy を計算
thresholds = np.linspace(scores.min() - 1, scores.max() + 1, 200)
accuracies = []

for th in thresholds:
    # スコア<th を「誤り(1)」、>=th を「正しい(0)」と判定
    preds = (scores < th).astype(int)
    accuracies.append(accuracy_score(labels, preds))

accuracies = np.array(accuracies)

# 5) 最適閾値（最高 accuracy）を求める
best_idx = np.argmax(accuracies)
best_threshold = thresholds[best_idx]
best_accuracy = accuracies[best_idx]

print(f"最適閾値: {best_threshold:.2f}")
print(f"最高 Accuracy: {best_accuracy:.3f}")

# 6) Accuracy vs Threshold グラフ
plt.figure(figsize=(8,4))
plt.plot(thresholds, accuracies, label="Accuracy")
plt.axvline(best_threshold, color="red", linestyle="--",
            label=f"Best Threshold = {best_threshold:.2f}")
plt.xlabel("Threshold (log probability)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
