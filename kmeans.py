import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import time

def kmeans_accuracy_search(
    real_data_file, 
    artificial_data_file,
    k_min=4,
    k_max=15
):
    """
    K-Meansクラスタリングのパイプラインを実行し、異なるk値に対するクラスタリングの正解率を評価する。
    指定された範囲内で最適なkを探索する。
    """
    print("指定されたファイル構造でデータを読み込んでいます...")
    df_artificial = pd.read_excel(artificial_data_file, header=0)
    X_artificial = df_artificial.iloc[:, 2:17].values
    y_artificial = df_artificial.iloc[:, 1].values
    feature_names = df_artificial.columns[2:17].tolist()

    df_real = pd.read_excel(real_data_file, header=0)
    X_real = df_real.iloc[:, 2:17].values

    # 特徴量数のチェック
    if X_real.shape[1] != X_artificial.shape[1]:
        print(f"\nエラー: 特徴量数が一致しません。実データ: {X_real.shape[1]}個, 人工データ: {X_artificial.shape[1]}個")
        return

    # スケーリングと重み付け
    X_combined = np.vstack((X_real, X_artificial))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    weights_dict = {'辞書照合': 1.8, '自然度スコア(liwii)': 1.8}
    weights = np.array([weights_dict.get(name, 1.0) for name in feature_names])
    X_weighted = X_scaled * weights

    print("\n--- kごとのクラスタリング正解率を計算します ---")
    results = []
    for k in range(k_min, k_max + 1):
        print(f"\n--- k={k} ---")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(X_weighted)
        artificial_cluster_labels = cluster_labels[len(X_real):]
        df_map = pd.DataFrame({'cluster': artificial_cluster_labels, 'true_label': y_artificial})

        cluster_to_label_map = {}
        for i in range(k):
            labels_in_cluster = df_map[df_map['cluster'] == i]['true_label']
            if not labels_in_cluster.empty:
                cluster_to_label_map[i] = labels_in_cluster.mode()[0]
            else:
                cluster_to_label_map[i] = '分類不能'

        # 人工データで評価
        artificial_eval_indices = [i for i, label in enumerate(artificial_cluster_labels) if cluster_to_label_map.get(label, '分類不能') != '分類不能']
        y_true_eval = [str(y_artificial[i]) for i in artificial_eval_indices]
        y_pred_eval = [str(cluster_to_label_map[artificial_cluster_labels[i]]) for i in artificial_eval_indices]
        if len(y_true_eval) > 0:
            accuracy = accuracy_score(y_true_eval, y_pred_eval)
            print(f"k={k} の人工データでのクラスタリング正解率 (Accuracy): {accuracy:.4f}")
            results.append((k, accuracy))
        else:
            print(f"k={k} では評価可能なクラスタがありません。")
            results.append((k, None))

    print("\n=== kごとの正解率一覧 ===")
    for k, acc in results:
        if acc is not None:
            print(f"k={k}: Accuracy={acc:.4f}")
        else:
            print(f"k={k}: 評価不可")

# --- メイン実行ブロック ---
if __name__ == '__main__':
    kmeans_accuracy_search(
        real_data_file='all features 969.xlsx',
        artificial_data_file='all features 1020.xlsx',
        k_min=4,
        k_max=20
    )