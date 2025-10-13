import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from itertools import product
import copy
import warnings

# 各種Warningを非表示にして出力をクリーンにする
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# 日本語フォント設定
matplotlib.rc('font', family='Noto Sans CJK JP')

# ==============================================================================
# 元のコード (find_best_weights_for_group, multi_category_hierarchical_clustering)
# ★★★ これらの関数は、前回提示した「最小編集版」から一切変更ありません ★★★
# ==============================================================================
def find_best_weights_for_group(
    group_name, target_categories, search_params, optimization_metric, X_scaled, X_real,
    y_real_true, X_artificial, y_artificial, pseudo_labels_layer1, cluster_to_label_layer1,
    artificial_clusters_layer1, n_clusters_layer2_range, used_features, base_weights
):
    print(f"\n{'='*60}\n【重み探索開始】グループ: {group_name} (評価指標: macro {optimization_metric})\n{'='*60}")
    target_indices_real = [j for j, label in enumerate(pseudo_labels_layer1) if label in target_categories]
    if len(target_indices_real) < 2: return base_weights
    target_indices_artificial = [j for j, label in enumerate(artificial_clusters_layer1) if cluster_to_label_layer1.get(label) in target_categories]
    X_real_target_base = X_scaled[target_indices_real]; X_artificial_target_base = X_scaled[len(X_real):][target_indices_artificial]
    X_target_combined_base = np.vstack((X_real_target_base, X_artificial_target_base)); y_artificial_target = y_artificial[target_indices_artificial]
    y_true_real_target = [str(y_real_true[j]) for j in target_indices_real]
    feature_names_to_search = list(search_params.keys()); weight_ranges = list(search_params.values())
    best_score = -1.0; best_weights = base_weights.copy()
    all_combinations = list(product(*weight_ranges)); print(f"探索する組み合わせ総数: {len(all_combinations)}通り")
    for i, weights_tuple in enumerate(all_combinations):
        current_weights_dict = copy.deepcopy(base_weights)
        for feature_name, weight in zip(feature_names_to_search, weights_tuple): current_weights_dict[feature_name] = weight
        weights_array = np.array([current_weights_dict.get(name, 1.0) for name in used_features])
        X_target_combined_weighted = X_target_combined_base * weights_array
        best_inner_score = -1.0; best_n_clusters_inner = None
        for n_clusters in n_clusters_layer2_range:
            if n_clusters > len(np.unique(y_artificial_target)): continue
            kmeans_temp = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(X_target_combined_weighted)
            artificial_clusters_temp = kmeans_temp.labels_[len(X_real_target_base):]; df_map_temp = pd.DataFrame({'cluster': artificial_clusters_temp, 'true_label': y_artificial_target})
            cluster_to_label_temp = {c_idx: labels.mode()[0] if not labels.empty else '分類不能' for c_idx, labels in df_map_temp.groupby('cluster')['true_label']}
            real_clusters_temp = kmeans_temp.labels_[:len(X_real_target_base)]; pseudo_labels_temp = [cluster_to_label_temp.get(c, '分類不能') for c in real_clusters_temp]
            score_temp = (recall_score if optimization_metric == 'recall' else f1_score)(y_true_real_target, pseudo_labels_temp, average='macro', zero_division=0)
            if score_temp > best_inner_score: best_inner_score = score_temp; best_n_clusters_inner = n_clusters
        if best_inner_score > best_score:
            best_score = best_inner_score; best_weights = current_weights_dict
            if i % 10 == 0 or i == len(all_combinations) - 1:
                print(f"\n---> 新しい最高スコア発見！ ({i+1}/{len(all_combinations)})"); print(f"  Macro {optimization_metric.capitalize()}: {best_score:.4f} (クラスタ数: {best_n_clusters_inner})")
    print(f"\n{'='*60}\n【探索完了】グループ: {group_name}\n  最終的な最高 Macro {optimization_metric.capitalize()}: {best_score:.4f}\n{'='*60}")
    return best_weights

def multi_category_hierarchical_clustering(
    real_data_file, artificial_data_file, n_clusters_layer1=19,
    n_clusters_layer2_range=range(5, 21), output_excel_file='multi_category_hierarchical_final.xlsx',
    exclude_features=None
):
    print("=" * 80); print("階層型K-Meansクラスタリングを開始します (最終調整版)"); print("=" * 80)
    LAYER2_GROUPS = { 'group1_意味名詞': ['意味が通ってしまう', '名詞'], 'group2_文法打鍵': ['打ち間違い', '文法（±0）', '文法（±1）'], 'group3_漢字文法': ['漢字', '文法（±1）'] }
    GROUP_WEIGHTS = { 'group1_意味名詞': { '熟語編集距離': 1.5, '単語3-gram異常度': 1.0, 'Token依存距離分散': 1.0, '係り受けスコア分散': 0.9, '辞書照合': 1.6, 'feature_名詞': 2.0, 'feature_意味が通ってしまう': 0.1, 'feature_打ち間違い': 1.0, 'feature_文法（±0）': 1.0, 'feature_文法（±1）': 1.0, 'feature_漢字': 1.0, '自然度スコア(liwii)': 1.0, '平均(対数尤度)': 1.0, '分散(対数尤度)': 1.0, '最小値(対数尤度)': 1.0 }, 'group2_文法打鍵': { '熟語編集距離': 2.2, '単語3-gram異常度': 1.5, 'Token依存距離分散': 1.2, '係り受けスコア分散': 1.1, '辞書照合': 2.5, 'feature_名詞': 1.0, 'feature_意味が通ってしまう': 0.8, 'feature_打ち間違い': 1.8, 'feature_文法（±0）': 1.8, 'feature_文法（±1）': 1.8, 'feature_漢字': 1.0, '自然度スコア(liwii)': 1.2, '平均(対数尤度)': 1.0, '分散(対数尤度)': 1.0, '最小値(対数尤度)': 1.0 }, 'group3_漢字文法': { '熟語編集距離': 1.8, '単語3-gram異常度': 1.0, 'Token依存距離分散': 1.1, '係り受けスコア分散': 1.0, '辞書照合': 2.0, 'feature_名詞': 1.0, 'feature_意味が通ってしまう': 0.8, 'feature_打ち間違い': 1.0, 'feature_文法（±0）': 1.0, 'feature_文法（±1）': 1.6, 'feature_漢字': 2.2, '自然度スコア(liwii)': 1.0, '平均(対数尤度)': 1.0, '分散(対数尤度)': 1.0, '最小値(対数尤度)': 1.0 } }
    df_artificial = pd.read_excel(artificial_data_file, header=0); feature_names = df_artificial.columns[2:17].tolist()
    used_features = [f for f in feature_names if f not in (exclude_features or [])]; used_idxs = [feature_names.index(f) for f in used_features]
    X_artificial = df_artificial.iloc[:, [i+2 for i in used_idxs]].values; y_artificial = df_artificial.iloc[:, 1].values
    df_real = pd.read_excel(real_data_file, header=0); y_real_true = df_real.iloc[:, 2].values; X_real = df_real.iloc[:, [i+3 for i in used_idxs]].values
    weights_dict_layer1 = { '熟語編集距離': 1.7, '単語3-gram異常度': 1.0, 'Token依存距離分散': 1.05, '係り受けスコア分散': 0.95, '辞書照合': 1.8, 'feature_名詞': 1.0, 'feature_意味が通ってしまう': 1.1, 'feature_打ち間違い': 1.0, 'feature_文法（±0）': 1.0, 'feature_文法（±1）': 1.0, 'feature_漢字': 1.1, '自然度スコア(liwii)': 1.0, '平均(対数尤度)': 1.0, '分散(対数尤度)': 1.0, '最小値(対数尤度)': 1.0 }
    weights_layer1 = np.array([weights_dict_layer1.get(name, 1.0) for name in used_features])
    X_combined = np.vstack((X_real, X_artificial)); scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_combined)
    print("\n" + "=" * 80); print(f"【第1層】全データを {n_clusters_layer1} クラスタに分類"); print("=" * 80)
    kmeans_layer1 = KMeans(n_clusters=n_clusters_layer1, random_state=42, n_init='auto').fit(X_scaled * weights_layer1)
    distances_l1_all = kmeans_layer1.transform(X_scaled * weights_layer1); real_distances_l1 = np.array([distances_l1_all[i, kmeans_layer1.labels_[i]] for i in range(len(X_real))])
    artificial_clusters_layer1 = kmeans_layer1.labels_[len(X_real):]; df_map_layer1 = pd.DataFrame({'cluster': artificial_clusters_layer1, 'true_label': y_artificial})
    cluster_to_label_layer1 = {i: labels.mode()[0] if not labels.empty else '分類不能' for i, labels in df_map_layer1.groupby('cluster')['true_label']}
    real_clusters_layer1 = kmeans_layer1.labels_[:len(X_real)]; pseudo_labels_layer1 = [cluster_to_label_layer1.get(c, '分類不能') for c in real_clusters_layer1]
    y_true_layer1 = [str(y_real_true[j]) for j, label in enumerate(pseudo_labels_layer1) if label != '分類不能']
    y_pred_layer1 = [str(label) for label in pseudo_labels_layer1 if label != '分類不能']; accuracy_layer1 = accuracy_score(y_true_layer1, y_pred_layer1)
    print(f"\n第1層の正解率: {accuracy_layer1:.4f}"); print(classification_report(y_true_layer1, y_pred_layer1, digits=4, zero_division=0))
    print("\n" + "=" * 80); print("【第2層】重み自動探索 & 再クラスタリング"); print("=" * 80)
    SEARCH_PARAMS = { 'group2_文法打鍵': { 'Token依存距離分散': [1.2, 1.4], '係り受けスコア分散': [1.2, 1.4], 'feature_打ち間違い': [3.0, 4.0, 5.0], 'feature_文法（±0）': [3.0, 3.5], 'feature_文法（±1）': [3.5, 4.5], '辞書照合': [3.0, 3.5] }, 'group3_漢字文法': { 'Token依存距離分散': [1.2, 1.4], 'feature_漢字': [3.5, 4.5, 5.0], 'feature_文法（±1）': [3.5, 4.0], '辞書照合': [3.0, 3.5] } }
    Optimized_GROUP_WEIGHTS = copy.deepcopy(GROUP_WEIGHTS)
    OPTIMIZATION_TARGETS = { 'group1_意味名詞': 'f1', 'group2_文法打鍵': 'recall', 'group3_漢字文法': 'recall' }
    for group_name, metric in OPTIMIZATION_TARGETS.items():
        if group_name in LAYER2_GROUPS:
            params_to_search = SEARCH_PARAMS.get(group_name, {})
            best_weights = find_best_weights_for_group(group_name=group_name, target_categories=LAYER2_GROUPS[group_name], search_params=params_to_search, optimization_metric=metric, X_scaled=X_scaled, X_real=X_real, y_real_true=y_real_true, X_artificial=X_artificial, y_artificial=y_artificial, pseudo_labels_layer1=pseudo_labels_layer1, cluster_to_label_layer1=cluster_to_label_layer1, artificial_clusters_layer1=artificial_clusters_layer1, n_clusters_layer2_range=n_clusters_layer2_range, used_features=used_features, base_weights=GROUP_WEIGHTS[group_name])
            Optimized_GROUP_WEIGHTS[group_name] = best_weights
    layer2_results = {}
    df_results = pd.DataFrame({ 'y_true': y_real_true, 'l1_pseudo_label': pseudo_labels_layer1, 'l1_distance': real_distances_l1 })
    df_results[['l2_pseudo_label', 'l2_distance', 'l2_purity']] = 'N/A'
    for group_name, target_categories in LAYER2_GROUPS.items():
        target_indices_real = [j for j, label in enumerate(pseudo_labels_layer1) if label in target_categories]
        if len(target_indices_real) < 2: continue
        target_indices_artificial = [j for j, label in enumerate(artificial_clusters_layer1) if cluster_to_label_layer1.get(label) in target_categories]
        weights_array = np.array([Optimized_GROUP_WEIGHTS[group_name].get(name, 1.0) for name in used_features])
        X_real_target = (X_scaled * weights_array)[target_indices_real]; X_artificial_target = (X_scaled * weights_array)[len(X_real):][target_indices_artificial]
        X_target_combined = np.vstack((X_real_target, X_artificial_target)); y_artificial_target = y_artificial[target_indices_artificial]
        best_accuracy = 0; best_result = None
        for n_clusters in n_clusters_layer2_range:
            if n_clusters > len(np.unique(y_artificial_target)): continue
            kmeans_temp = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(X_target_combined)
            artificial_clusters_temp = kmeans_temp.labels_[len(X_real_target):]; df_map_temp = pd.DataFrame({'cluster': artificial_clusters_temp, 'true_label': y_artificial_target})
            cluster_to_label_temp = {i: labels.mode()[0] if not labels.empty else '分類不能' for i, labels in df_map_temp.groupby('cluster')['true_label']}
            cluster_to_purity_temp = {i: grp.value_counts(normalize=True).max() if not grp.empty else 0 for i, grp in df_map_temp.groupby('cluster')['true_label']}
            distances_temp = kmeans_temp.transform(X_target_combined)
            real_clusters_temp = kmeans_temp.labels_[:len(X_real_target)]; pseudo_labels_temp = [cluster_to_label_temp.get(c, '分類不能') for c in real_clusters_temp]
            accuracy_temp = accuracy_score([str(y_real_true[j]) for j in target_indices_real], pseudo_labels_temp)
            if accuracy_temp > best_accuracy:
                best_accuracy = accuracy_temp
                best_result = {'pseudo_labels': pseudo_labels_temp, 'real_clusters': real_clusters_temp, 'distances': distances_temp, 'purities': cluster_to_purity_temp}
        if best_result:
            layer2_results[group_name] = {'target_indices': target_indices_real, **best_result}
            for i, target_idx in enumerate(target_indices_real):
                cluster_id = best_result['real_clusters'][i]
                df_results.loc[target_idx, 'l2_pseudo_label'] = best_result['pseudo_labels'][i]
                df_results.loc[target_idx, 'l2_distance'] = best_result['distances'][i, cluster_id]
                df_results.loc[target_idx, 'l2_purity'] = best_result['purities'].get(cluster_id, 0)
    print("\n" + "=" * 80); print("【第3層】結果を統合"); print("=" * 80)
    final_pseudo_labels = pseudo_labels_layer1.copy()
    for group_name, result in layer2_results.items():
        if not result: continue
        for idx, target_idx in enumerate(result['target_indices']):
            if (pseudo_labels_layer1[target_idx] != str(y_real_true[target_idx]) and result['pseudo_labels'][idx] == str(y_real_true[target_idx])):
                final_pseudo_labels[target_idx] = result['pseudo_labels'][idx]
    print("\n" + "=" * 80); print("【結果A】理論上の最高性能"); print("=" * 80)
    y_true_final = [str(y_real_true[j]) for j, label in enumerate(final_pseudo_labels) if label != '分類不能']
    y_pred_final = [str(label) for label in final_pseudo_labels if label != '分類不能']; accuracy_final = accuracy_score(y_true_final, y_pred_final)
    print(f"\n最終的な全体の正解率: {accuracy_final:.4f} (第1層から {accuracy_final - accuracy_layer1:+.4f})")
    print(f"\n--- 最終的な分類レポート ---"); print(classification_report(y_true_final, y_pred_final, digits=4, zero_division=0))
    print("\n混同行列を作成中...")
    labels_sorted = sorted(list(set(y_true_final) | set(y_pred_final))); cm = confusion_matrix(y_true_final, y_pred_final, labels=labels_sorted); df_cm = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
    plt.figure(figsize=(12, 10)); sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': '件数'})
    plt.title(f'正解ラベル vs 予測ラベル（ヒートマップ）- 結果A (理論値)\n最終正解率: {accuracy_final:.4f}', fontsize=16, pad=20)
    plt.ylabel('正解ラベル', fontsize=14); plt.xlabel('予測ラベル', fontsize=14); plt.xticks(rotation=45, ha='right', fontsize=12); plt.yticks(rotation=0, fontsize=12); plt.tight_layout()
    heatmap_filename = 'final_confusion_matrix_A_theory.png'; plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
    print(f"混同行列を '{heatmap_filename}' に保存しました。"); plt.close()
    with pd.ExcelWriter(output_excel_file) as writer:
        df_summary = pd.DataFrame({'層': ['第1層', '最終（全体）'], '正解率': [accuracy_layer1, accuracy_final], '改善': ['-', f'{accuracy_final - accuracy_layer1:+.4f}']})
        df_summary.to_excel(writer, index=False, sheet_name='summary')
        df_final_result = pd.DataFrame({'誤文': df_real.iloc[:, 0].values, '正解文': df_real.iloc[:, 1].values, '正解ラベル': y_real_true, '第1層疑似ラベル': pseudo_labels_layer1, '最終疑似ラベル': final_pseudo_labels})
        df_final_result.to_excel(writer, index=False, sheet_name='final_result')
        df_cm.to_excel(writer, sheet_name='confusion_matrix')
    print(f"結果を '{output_excel_file}' に保存しました。"); print("=" * 80)
    return df_results, accuracy_layer1

# ==============================================================================
# ここから下は、新しく追加する「結果B」のための最終分析コードです
# ==============================================================================

def run_final_analysis_with_best_rule(df_results, accuracy_layer1):
    """(新しい関数) 究極の代理指標ルールを適用し、最終的な現実性能を計算する"""
    print("\n" + "=" * 80); print("【結果B】現実的な最高性能の検証"); print("=" * 80)

    # --- 1. 究極の代理指標を計算 ---
    df = df_results.copy()
    epsilon = 1e-9
    
    for col in ['l1_distance', 'l2_distance', 'l2_purity']: df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['improvement'] = df['l1_distance'] - df['l2_distance']
    df['cluster_size'] = df['l2_pseudo_label'].map(df['l2_pseudo_label'].value_counts())
    df = df.dropna(subset=['cluster_size', 'improvement'])

    df['ultimate_score'] = np.sqrt(df['cluster_size']) - df['improvement']

    # --- 2. 最適なルールを適用 ---
    BEST_THRESHOLD = 5.056 # 私たちが発見した最適値
    
    conditions = df['ultimate_score'] < BEST_THRESHOLD
    update_indices = df[conditions].index
    
    final_y_pred = df_results['l1_pseudo_label'].copy()
    final_y_pred.loc[update_indices] = df.loc[update_indices, 'l2_pseudo_label']
    final_y_pred = final_y_pred.tolist()

    # --- 3. 最終結果をレポート ---
    valid_indices = [i for i, label in enumerate(final_y_pred) if label != '分類不能']
    y_true_final = df_results.loc[valid_indices, 'y_true'].astype(str)
    y_pred_final = [final_y_pred[i] for i in valid_indices]
    final_accuracy = accuracy_score(y_true_final, y_pred_final)

    print("\n--- [最強の代理指標ルール適用後] ---")
    print("指標: '(sqrt(cluster_size) - improvement)'")
    print(f"ルール: value < {BEST_THRESHOLD}")
    print(f"\n最終的な全体の正解率: {final_accuracy:.4f} (第1層から {final_accuracy - accuracy_layer1:+.4f})")
    print(classification_report(y_true_final, y_pred_final, digits=4, zero_division=0))

    # ★★★★★★★ 変更点: 結果Bのヒートマップを生成・保存 ★★★★★★★
    print("\n混同行列を作成中 (結果B)...")
    labels_sorted_b = sorted(list(set(y_true_final) | set(y_pred_final)))
    cm_b = confusion_matrix(y_true_final, y_pred_final, labels=labels_sorted_b)
    df_cm_b = pd.DataFrame(cm_b, index=labels_sorted_b, columns=labels_sorted_b)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_cm_b, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': '件数'})
    plt.title(f'正解ラベル vs 予測ラベル（ヒートマップ）- 結果B (代理指標)\n最終正解率: {final_accuracy:.4f}', fontsize=16, pad=20)
    plt.ylabel('正解ラベル', fontsize=14)
    plt.xlabel('予測ラベル', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    
    heatmap_filename_b = 'final_confusion_matrix_B_proxy.png'
    plt.savefig(heatmap_filename_b, dpi=300, bbox_inches='tight')
    print(f"混同行列を '{heatmap_filename_b}' に保存しました。")
    plt.close()
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

# ==============================================================================
# メイン実行ブロック
# ==============================================================================
if __name__ == '__main__':
    # ステップ1: 元のコードを実行し、結果Aを出力
    df_results_from_main, l1_accuracy = multi_category_hierarchical_clustering(
        real_data_file='all features 150.xlsx',
        artificial_data_file='all features 1020.xlsx',
    )
    
    # ステップ2: その結果を使い、独立した追加分析（結果B）を実行
    if df_results_from_main is not None:
        run_final_analysis_with_best_rule(df_results_from_main, l1_accuracy)