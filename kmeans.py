import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import time

def kmeans_full_pipeline_with_k_search(
    real_data_file, 
    artificial_data_file, 
    output_heatmap_file='kmeans.png',
    output_excel_file='kmeans.xlsx',
    output_real_pca_file='kmeans2.png'
):
    # --- 1. データの準備 ---
    try:
        print("データを読み込んでいます...")
        df_real = pd.read_excel(real_data_file, header=0)
        X_real, y_real_true = df_real.iloc[:, 3:15].values, df_real.iloc[:, 2].values
        df_artificial = pd.read_excel(artificial_data_file, header=0)
        X_artificial, y_artificial = df_artificial.iloc[:, 2:14].values, df_artificial.iloc[:, 1].values
        feature_names = df_artificial.columns[2:14].tolist()
    except Exception as e:
        print(f"データ読み込みエラー: {e}"); return
        
    # --- 2. スケーリングと重み付け ---
    X_combined = np.vstack((X_real, X_artificial))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    weights_dict = {'辞書照合': 1.8, '自然度スコア(liwii)': 1.8}
    weights = np.array([weights_dict.get(name, 1.0) for name in feature_names])
    X_weighted = X_scaled * weights
    print(f"合計データ数: {len(X_weighted)} で処理を開始します。")

    # --- 3. 最適なクラスタ数の探索 ---
    print("\n--- 最適なクラスタ数を探索します ---")
    k_range = range(2, 16)
    wcss_list, silhouette_scores, db_scores = [], [], []
    start_time = time.time()
    for k in k_range:
        print(f"k={k}を試行中...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_weighted)
        wcss_list.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_weighted, kmeans.labels_))
        db_scores.append(davies_bouldin_score(X_weighted, kmeans.labels_))
    end_time = time.time()
    print(f"クラスタ数探索 完了。所要時間: {(end_time - start_time):.2f} 秒")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle('最適なクラスタ数 探索結果', fontsize=16)
    axes[0].plot(k_range, wcss_list, marker='o'); axes[0].set_title('エルボー法 (WCSS)'); axes[0].set_xlabel('クラスタ数'); axes[0].grid(True)
    axes[1].plot(k_range, silhouette_scores, marker='o', color='g'); axes[1].set_title('シルエット係数 (高いほど良い)'); axes[1].set_xlabel('クラスタ数'); axes[1].grid(True)
    axes[2].plot(k_range, db_scores, marker='o', color='r'); axes[2].set_title('Davies-Bouldin指数 (低いほど良い)'); axes[2].set_xlabel('クラスタ数'); axes[2].grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('c.png', dpi=300)
    print("探索結果のグラフを 'c.png' に保存しました。")
    plt.show()

    # --- 4. K-Meansによる疑似ラベリング ---
    n_clusters = int(input("グラフを見て、使用する最適なクラスタ数 (k) を入力してください: "))
    print(f"\n--- k={n_clusters}としてK-Meansクラスタリングを実行し、疑似ラベルを付与します ---")
    
    kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans_final.fit_predict(X_weighted)
    
    artificial_cluster_labels = cluster_labels[len(X_real):]
    df_map = pd.DataFrame({'cluster': artificial_cluster_labels, 'true_label': y_artificial})
    cluster_to_label_map = {i: (labels.mode()[0] if not labels.empty else '分類不能') 
                            for i, labels in df_map.groupby('cluster')['true_label']}
    
    real_cluster_labels = cluster_labels[:len(X_real)]
    pseudo_labels = [cluster_to_label_map.get(c, '分類不能') for c in real_cluster_labels]

    # --- 5. 詳細結果をExcelに出力 ---
    print(f"\n疑似ラベル付けの結果を '{output_excel_file}' に出力します...")
    df_real_with_labels = df_real.copy()
    df_real_with_labels['クラスタラベル'] = real_cluster_labels
    df_real_with_labels['疑似ラベル'] = pseudo_labels
    df_real_with_labels.to_excel(output_excel_file, index=False)
    print(f"'{output_excel_file}' の保存が完了しました。")

    # --- 6. 性能評価 ---
    print("\n" + "="*70)
    print("【K-Means疑似ラベリング性能評価 (最適重み適用後)】")
    eval_indices = [i for i, label in enumerate(pseudo_labels) if label != '分類不能']
    y_real_true_eval = y_real_true[eval_indices]
    pseudo_labels_eval = [pseudo_labels[i] for i in eval_indices]
    if len(y_real_true_eval) > 0:
        accuracy = accuracy_score(y_real_true_eval, pseudo_labels_eval)
        print(f"正解率 (Accuracy): {accuracy:.4f}")
        print("\n--- 詳細レポート (クラスごと) ---\n", classification_report(y_real_true_eval, pseudo_labels_eval, digits=4, zero_division=0))
        labels_for_matrix = sorted(list(set(y_real_true_eval) | set(pseudo_labels_eval)))
        cm = confusion_matrix(y_real_true_eval, pseudo_labels_eval, labels=labels_for_matrix)
        plt.figure(figsize=(12, 10)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_for_matrix, yticklabels=labels_for_matrix)
        plt.title(f'K-Means疑似ラベリングの混同行列 (k={n_clusters})', fontsize=16)
        plt.ylabel('正解ラベル'); plt.xlabel('予測された疑似ラベル')
        plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
        plt.savefig(output_heatmap_file, dpi=300); plt.close()
        print(f"\n混同行列のヒートマップを '{output_heatmap_file}' に保存しました。")
    else: print("評価対象データなし。")

    # --- 7. 実データのみのPCA可視化 ---
    print("\n【実データのみのクラスタ分布可視化】")
    pca = PCA(n_components=2, random_state=42)
    X_real_weighted = X_weighted[:len(X_real)]
    X_real_pca = pca.fit_transform(X_real_weighted)
    plt.figure(figsize=(14, 10));
    scatter = plt.scatter(X_real_pca[:, 0], X_real_pca[:, 1], c=real_cluster_labels, cmap='tab10', alpha=0.8, s=60)
    plt.title(f'実データのクラスタ分布 (PCA可視化, k={n_clusters})', fontsize=18)
    plt.xlabel('主成分1'); plt.ylabel('主成分2')
    legend_handles = scatter.legend_elements(num=n_clusters)[0]
    legend_labels_text = [f'クラスタ {i}' for i in range(n_clusters)]
    plt.legend(handles=legend_handles, labels=legend_labels_text, title="所属クラスタ"); plt.grid(True)
    plt.savefig(output_real_pca_file, dpi=300)
    print(f"実データのクラスタ分布PCA画像を '{output_real_pca_file}' に保存しました。")
    plt.close()

if __name__ == '__main__':
    kmeans_full_pipeline_with_k_search(
        real_data_file='train2_data.xlsx',
        artificial_data_file='test2_data.xlsx',
        output_heatmap_file='kmeans.png',
        output_excel_file='kmeans.xlsx',
        output_real_pca_file='kmeans2.png'
    )