import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import itertools
import time

def kmeans_weight_search(
    real_data_file, 
    artificial_data_file,
    n_clusters=19
):
    """
    K-Meansクラスタリングのパイプラインを実行し、異なる特徴量の重みに対するクラスタリングの正解率を評価する。
    指定されたクラスタ数に対して最適な重みの組み合わせを探索する。
    """
    print("指定されたファイル構造でデータを読み込んでいます...")
    df_artificial = pd.read_excel(artificial_data_file, header=0)
    X_artificial = df_artificial.iloc[:, 2:17].values
    y_artificial = df_artificial.iloc[:, 1].values
    feature_names = df_artificial.columns[2:17].tolist()

    df_real = pd.read_excel(real_data_file, header=0)
    y_real_true = df_real.iloc[:, 2].values
    X_real = df_real.iloc[:, 3:18].values

    # 特徴量数のチェック
    if X_real.shape[1] != X_artificial.shape[1]:
        print(f"\nエラー: 特徴量数が一致しません。実データ: {X_real.shape[1]}個, 人工データ: {X_artificial.shape[1]}個")
        return

    # スケーリング
    X_combined = np.vstack((X_real, X_artificial))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # 重みのグリッドサーチ設定（例：3種類の重みを3特徴量に対して変化させる）
    param_grid = {
        '熟語編集距離': [1.7],
        '単語3-gram異常度': [1.0],
        'Token依存距離分散': [1.05],
        '係り受けスコア分散':  [0.95,1.0,1.05],
        '辞書照合':  [1.8],
        'feature_名詞':  [1.0],
        'feature_意味が通ってしまう':  [ 1.1],
        'feature_打ち間違い': [1.0],
        'feature_文法（±0）':  [ 1.0], 
        'feature_文法（±1）':  [1.0],
        'feature_漢字':  [1.1],
        '自然度スコア(liwii)':  [1.0],
        '平均(対数尤度)':  [1.0],
        '分散(対数尤度)':  [1.0],
        '最小値(対数尤度)': [1.0],
    }
    # 残りの特徴量は1.0で埋める
    for name in feature_names:
        if name not in param_grid:
            param_grid[name] = [1.0]

    keys = feature_names
    values = [param_grid[name] for name in keys]
    weight_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"\n重み組み合わせ総数: {len(weight_combinations)}")

    best_accuracy = 0.0
    best_weights = None
    start_time = time.time()

    for i, combo in enumerate(weight_combinations):
        weights = np.array([combo[name] for name in feature_names])
        X_weighted = X_scaled * weights

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(X_weighted)
        artificial_cluster_labels = cluster_labels[len(X_real):]
        df_map = pd.DataFrame({'cluster': artificial_cluster_labels, 'true_label': y_artificial})

        cluster_to_label_map = {}
        for cluster_id in range(n_clusters):
            labels_in_cluster = df_map[df_map['cluster'] == cluster_id]['true_label']
            if not labels_in_cluster.empty:
                cluster_to_label_map[cluster_id] = labels_in_cluster.mode()[0]
            else:
                cluster_to_label_map[cluster_id] = '分類不能'

        real_cluster_labels = cluster_labels[:len(X_real)]
        pseudo_labels = [cluster_to_label_map.get(c, '分類不能') for c in real_cluster_labels]

        eval_indices = [j for j, label in enumerate(pseudo_labels) if label != '分類不能']
        y_true_eval = [str(y_real_true[j]) for j in eval_indices]
        y_pred_eval = [str(pseudo_labels[j]) for j in eval_indices]
        if len(y_true_eval) > 0:
            accuracy = accuracy_score(y_true_eval, y_pred_eval)
        else:
            accuracy = 0.0

        print(f"試行 {i+1}/{len(weight_combinations)}: Accuracy={accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = combo
            print(f"  ★★★ 最高精度を更新！ ★★★")

    end_time = time.time()
    print(f"\n探索完了。所要時間: {(end_time - start_time):.2f} 秒")
    print("\n" + "="*70)
    print("【K-Means重み探索 結果】")
    print("="*70)
    if best_weights:
        print(f"最良の正解率 (Accuracy): {best_accuracy:.4f}")
        print("\n最適な重みの組み合わせ:")
        for feature, weight in best_weights.items():
            print(f"  {feature}: {weight}")
    else:
        print("有効な結果が見つかりませんでした。")

# --- メイン実行ブロック ---
if __name__ == '__main__':
    kmeans_weight_search(
        real_data_file='all features 150.xlsx',
        artificial_data_file='all features 1020.xlsx',
        n_clusters=19
    )