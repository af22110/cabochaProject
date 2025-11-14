import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, f1_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib
from itertools import product
import copy
import warnings
import seaborn as sns

# 各種Warningを非表示にして出力をクリーンにする
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# 日本語フォント設定
matplotlib.rc('font', family='Noto Sans CJK JP')

# ==============================================================================
# 階層クラスタリング実行部 (find_best_weights_for_groupは変更なし)
# ==============================================================================
def find_best_weights_for_group(
    group_name, target_categories, search_params, optimization_metric, X_scaled, X_real,
    y_real_true, X_artificial, y_artificial, pseudo_labels_layer1, cluster_to_label_layer1,
    artificial_clusters_layer1, n_clusters_layer2_range, used_features, base_weights
):
    # (この関数は変更なし)
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
            model_temp = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(X_target_combined_weighted)
            labels_temp = model_temp.labels_
            artificial_clusters_temp = labels_temp[len(X_real_target_base):]; df_map_temp = pd.DataFrame({'cluster': artificial_clusters_temp, 'true_label': y_artificial_target})
            cluster_to_label_temp = {c_idx: labels.mode()[0] if not labels.empty else '分類不能' for c_idx, labels in df_map_temp.groupby('cluster')['true_label']}
            real_clusters_temp = labels_temp[:len(X_real_target_base)]; pseudo_labels_temp = [cluster_to_label_temp.get(c, '分類不能') for c in real_clusters_temp]
            score_temp = accuracy_score([str(y_real_true[j]) for j in target_indices_real], pseudo_labels_temp)
            if score_temp > best_inner_score: best_inner_score = score_temp; best_n_clusters_inner = n_clusters
        if best_inner_score > best_score:
            best_score = best_inner_score; best_weights = current_weights_dict
            if i % 10 == 0 or i == len(all_combinations) - 1:
                print(f"\n---> 新しい最高スコア発見！ ({i+1}/{len(all_combinations)})"); print(f"  Accuracy: {best_score:.4f} (クラスタ数: {best_n_clusters_inner})")
    print(f"\n{'='*60}\n【探索完了】グループ: {group_name}\n  最終的な最高 Accuracy: {best_score:.4f}\n{'='*60}")
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
    X_artificial_raw = df_artificial.iloc[:, [i+2 for i in used_idxs]].values; y_artificial = df_artificial.iloc[:, 1].values
    df_real = pd.read_excel(real_data_file, header=0); y_real_true = df_real.iloc[:, 2].values; X_real_raw = df_real.iloc[:, [i+3 for i in used_idxs]].values
    weights_dict_layer1 = { '熟語編集距離': 1.7, '単語3-gram異常度': 1.0, 'Token依存距離分散': 1.05, '係り受けスコア分散': 0.95, '辞書照合': 1.8, 'feature_名詞': 1.0, 'feature_意味が通ってしまう': 1.1, 'feature_打ち間違い': 1.0, 'feature_文法（±0）': 1.0, 'feature_文法（±1）': 1.0, 'feature_漢字': 1.1, '自然度スコア(liwii)': 1.0, '平均(対数尤度)': 1.0, '分散(対数尤度)': 1.0, '最小値(対数尤度)': 1.0 }
    weights_layer1 = np.array([weights_dict_layer1.get(name, 1.0) for name in used_features])
    X_combined = np.vstack((X_real_raw, X_artificial_raw)); scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_combined)
    X_real, X_artificial = X_scaled[:len(X_real_raw)], X_scaled[len(X_real_raw):]
    
    print("\n" + "=" * 80); print(f"【第1層】全データを {n_clusters_layer1} クラスタに分類"); print("=" * 80)
    kmeans_layer1 = KMeans(n_clusters=n_clusters_layer1, random_state=42, n_init='auto').fit(X_scaled * weights_layer1)
    labels_layer1 = kmeans_layer1.labels_
    artificial_clusters_layer1 = labels_layer1[len(X_real):]; df_map_layer1 = pd.DataFrame({'cluster': artificial_clusters_layer1, 'true_label': y_artificial})
    
    unique_artificial_clusters = set(df_map_layer1['cluster'].unique())
    cluster_to_label_layer1 = {}
    unclassifiable_count = 0
    for i in range(n_clusters_layer1):
        if i in unique_artificial_clusters:
            labels_in_cluster = df_map_layer1[df_map_layer1['cluster'] == i]['true_label']
            if not labels_in_cluster.empty: cluster_to_label_layer1[i] = labels_in_cluster.mode()[0]
            else: cluster_to_label_layer1[i] = '分類不能'; unclassifiable_count += 1
        else: cluster_to_label_layer1[i] = '分類不能'; unclassifiable_count += 1
    print(f"★ 人工データが含まれない（分類不能）クラスタ数: {unclassifiable_count} / {n_clusters_layer1}")

    real_clusters_layer1 = labels_layer1[:len(X_real)]; pseudo_labels_layer1 = [cluster_to_label_layer1.get(c, '分類不能') for c in real_clusters_layer1]
    y_true_layer1 = [str(y_real_true[j]) for j, label in enumerate(pseudo_labels_layer1) if label != '分類不能']
    y_pred_layer1 = [str(label) for label in pseudo_labels_layer1 if label != '分類不能']; accuracy_layer1 = accuracy_score(y_true_layer1, y_pred_layer1)
    print(f"\n第1層の正解率: {accuracy_layer1:.4f}"); print(classification_report(y_true_layer1, y_pred_layer1, digits=4, zero_division=0))
    print("\n" + "=" * 80); print("【第2層】重み自動探索 & 再クラスタリング"); print("=" * 80)
    
    SEARCH_PARAMS = { 'group1_意味名詞': { 'feature_名詞': [2.0, 3.0, 4.0], 'feature_意味が通ってしまう': [0.4], '辞書照合': [1.6, 2.5, 3.5] }, 'group2_文法打鍵': { 'Token依存距離分散': [1.2, 1.4], '係り受けスコア分散': [1.2], 'feature_打ち間違い': [ 4.4], 'feature_文法（±0）': [3.2], 'feature_文法（±1）': [ 4.5], '辞書照合': [3.1] }, 'group3_漢字文法': { 'Token依存距離分散': [0.9], 'feature_漢字': [3.0], 'feature_文法（±1）': [3.5], '辞書照合': [3.0] } }
    Optimized_GROUP_WEIGHTS = copy.deepcopy(GROUP_WEIGHTS)
    OPTIMIZATION_TARGETS = { 'group1_意味名詞': 'f1', 'group2_文法打鍵': 'recall', 'group3_漢字文法': 'recall' }
    for group_name, metric in OPTIMIZATION_TARGETS.items():
        if group_name in LAYER2_GROUPS:
            params_to_search = SEARCH_PARAMS.get(group_name, {})
            best_weights = find_best_weights_for_group(group_name=group_name, target_categories=LAYER2_GROUPS[group_name], search_params=params_to_search, optimization_metric=metric, X_scaled=X_scaled, X_real=X_real, y_real_true=y_real_true, X_artificial=X_artificial, y_artificial=y_artificial, pseudo_labels_layer1=pseudo_labels_layer1, cluster_to_label_layer1=cluster_to_label_layer1, artificial_clusters_layer1=artificial_clusters_layer1, n_clusters_layer2_range=range(5, 21), used_features=used_features, base_weights=GROUP_WEIGHTS[group_name])
            Optimized_GROUP_WEIGHTS[group_name] = best_weights
            
    df_results = pd.DataFrame({ 'y_true': y_real_true, 'l1_pseudo_label': pseudo_labels_layer1 })
    
    accuracy_history = {}
    layer2_results = {}
    for group_name, target_categories in LAYER2_GROUPS.items():
        print(f"\n--- グループ [{group_name}] の最適クラスタ数を探索中 ---")
        target_indices_real = [j for j, label in enumerate(pseudo_labels_layer1) if label in target_categories]
        if len(target_indices_real) < 2: print(" -> 対象となる実データが2件未満のため、このグループの探索をスキップします。"); continue
            
        target_indices_artificial = [j for j, label in enumerate(artificial_clusters_layer1) if cluster_to_label_layer1.get(label) in target_categories]
        y_artificial_target = y_artificial[target_indices_artificial]
        
        max_clusters_allowed = len(np.unique(y_artificial_target))
        print(f" -> このグループ内の人工データのカテゴリ数は {max_clusters_allowed} 種類です。")
        print(f" -> そのため、クラスタ数の探索は最大 {max_clusters_allowed} までとなります。")
        
        weights_array = np.array([Optimized_GROUP_WEIGHTS[group_name].get(name, 1.0) for name in used_features])
        X_real_target_weighted = X_real[target_indices_real] * weights_array
        X_artificial_target_weighted = X_artificial[target_indices_artificial] * weights_array
        X_target_combined = np.vstack((X_real_target_weighted, X_artificial_target_weighted))
        
        best_accuracy = 0; best_result = None; group_accuracy_list = []

        for n_clusters in n_clusters_layer2_range:
            if n_clusters > max_clusters_allowed: break
            
            kmeans_temp = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(X_target_combined)
            artificial_clusters_temp = kmeans_temp.labels_[len(X_real_target_weighted):]; df_map_temp = pd.DataFrame({'cluster': artificial_clusters_temp, 'true_label': y_artificial_target})
            cluster_to_label_temp = {i: labels.mode()[0] if not labels.empty else '分類不能' for i, labels in df_map_temp.groupby('cluster')['true_label']}
            real_clusters_temp = kmeans_temp.labels_[:len(X_real_target_weighted)]; pseudo_labels_temp = [cluster_to_label_temp.get(c, '分類不能') for c in real_clusters_temp]
            accuracy_temp = accuracy_score([str(y_real_true[j]) for j in target_indices_real], pseudo_labels_temp)
            group_accuracy_list.append((n_clusters, accuracy_temp))
            
            if accuracy_temp > best_accuracy:
                best_accuracy = accuracy_temp
                best_result = {'pseudo_labels': pseudo_labels_temp, 'kmeans_obj': kmeans_temp, 'n_clusters': n_clusters}
        
        print(f"探索完了。最高精度: {best_accuracy:.4f} (最適クラスタ数: {best_result.get('n_clusters') if best_result else 'N/A'})")
        accuracy_history[group_name] = group_accuracy_list
        if best_result: layer2_results[group_name] = {'target_indices': target_indices_real, **best_result}

    return df_results, accuracy_layer1, X_real, kmeans_layer1, layer2_results, Optimized_GROUP_WEIGHTS, used_features, weights_layer1, accuracy_history

# ==============================================================================
# グラフ描画関数
# ==============================================================================
def plot_accuracy_history(accuracy_history):
    # (変更なし)
    print("\n" + "=" * 80); print("【グラフ作成】第2層クラスタ数探索結果の可視化"); print("=" * 80)
    for group_name, history in accuracy_history.items():
        if not history: print(f"グループ '{group_name}' の履歴データがありません。スキップします。"); continue
        x_vals = [item[0] for item in history]; y_vals = [item[1] for item in history]
        best_point = max(history, key=lambda item: item[1]); best_n_cluster, best_acc = best_point[0], best_point[1]
        plt.figure(figsize=(12, 7))
        plt.plot(x_vals, y_vals, marker='o', linestyle='-', label='探索時の精度')
        plt.scatter(best_n_cluster, best_acc, color='red', s=100, zorder=5, label=f'最高精度 ({best_acc:.4f})\nクラスタ数: {best_n_cluster}')
        plt.title(f'第2層 クラスタ数と精度の関係 ({group_name})', fontsize=16)
        plt.xlabel('クラスタ数 (Number of Clusters)', fontsize=12); plt.ylabel('分類精度 (Accuracy)', fontsize=12)
        plt.xticks(np.arange(min(x_vals), max(x_vals)+1, 1)); plt.grid(True, linestyle='--', alpha=0.6); plt.legend(); plt.tight_layout()
        filename = f'cluster_accuracy_{group_name}.png'; plt.savefig(filename)
        print(f"-> グラフを '{filename}' として保存しました。"); plt.close()

# ★★★★★ ここからが新規追加部分 ★★★★★
def plot_threshold_search_history(threshold_accuracy_history):
    """分析フェーズの閾値探索における精度変化をグラフ化して保存する"""
    print("\n" + "=" * 80); print("【グラフ作成】改善率(R)の閾値探索結果の可視化"); print("=" * 80)
    
    for group_name, history in threshold_accuracy_history.items():
        if not history:
            print(f"グループ '{group_name}' の閾値履歴データがありません。スキップします。")
            continue
            
        x_vals = [item[0] for item in history] # 閾値
        y_vals = [item[1] for item in history] # 精度
        
        # 最高精度だった点を見つける
        best_point = max(history, key=lambda item: item[1])
        best_threshold = best_point[0]
        best_acc = best_point[1]
        
        plt.figure(figsize=(12, 7))
        plt.plot(x_vals, y_vals, linestyle='-', label='探索時の精度')
        plt.scatter(best_threshold, best_acc, color='red', s=100, zorder=5, 
                    label=f'最高精度 ({best_acc:.4f})\n閾値: {best_threshold:.3f}')
        
        plt.title(f'改善率(R)の閾値とグループ内精度の関係 ({group_name})', fontsize=16)
        plt.xlabel('改善率の閾値 (Improvement Ratio Threshold R)', fontsize=12)
        plt.ylabel('グループ内精度 (Intra-Group Accuracy)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        
        filename = f'threshold_search_{group_name}.png'
        plt.savefig(filename)
        print(f"-> グラフを '{filename}' として保存しました。")
        plt.close()
# ★★★★★ ここまでが新規追加部分 ★★★★★

# ==============================================================================
# 分析コード (V1.3)
# ==============================================================================
def run_analysis_v1_3(df_results, accuracy_layer1, X_real_scaled, 
                      kmeans_layer1, layer2_results, optimized_weights, 
                      used_features, weights_layer1):
    print("\n" + "=" * 80); print("【分析 V1.3】改善策: 最大改善率ルール適用"); print("=" * 80)
    df = df_results.copy(); epsilon = 1e-9
    print("全レイヤー・全グループのスコアと予測を事前計算中...")
    df['score_l1'] = [b / (a + epsilon) for a, b in [calculate_ab_scores_weighted(X_real_scaled[i], kmeans_layer1, weights_layer1) for i in range(len(X_real_scaled))]]
    for group_name, result in layer2_results.items():
        df[f'score_l2_{group_name}'] = np.nan; df[f'l2_pseudo_label_{group_name}'] = 'N/A'
        w2 = np.array([optimized_weights[group_name].get(name, 1.0) for name in used_features])
        for i, idx in enumerate(result['target_indices']):
            a2, b2 = calculate_ab_scores_weighted(X_real_scaled[idx], result['kmeans_obj'], w2)
            df.loc[idx, f'score_l2_{group_name}'] = b2 / (a2 + epsilon)
            df.loc[idx, f'l2_pseudo_label_{group_name}'] = result['pseudo_labels'][i]
            
    print("グループごとに最適な改善倍率閾値(R >= 1.0)を探索中...")
    best_thresholds = {}
    
    # ★★★ 変更点: 閾値探索の履歴を保存する辞書を追加 ★★★
    threshold_accuracy_history = {}

    for group_name, result in layer2_results.items():
        target_indices = result['target_indices']; target_df = df.loc[target_indices].dropna(subset=[f'score_l2_{group_name}'])
        if target_df.empty: best_thresholds[group_name] = 999.9; continue
        
        target_df['R'] = target_df[f'score_l2_{group_name}'] / target_df['score_l1']
        base_acc = accuracy_score(df.loc[target_indices, 'y_true'].astype(str), df.loc[target_indices, 'l1_pseudo_label'].astype(str))
        best_acc_group, best_th_group = base_acc, 999.9
        
        max_r = target_df['R'].max()
        if max_r < 1.0: best_thresholds[group_name] = 999.9; continue
            
        search_range = np.unique(np.concatenate(([1.0], np.linspace(1.0, min(max_r, 5.0), 200))))
        
        # ★★★ 追加: このグループの閾値探索履歴を保存するリスト ★★★
        group_threshold_history = []
        
        for threshold in search_range:
            temp_pred = df.loc[target_indices, 'l1_pseudo_label'].copy()
            update_indices = target_df[target_df['R'] > threshold].index
            temp_pred.loc[update_indices] = df.loc[update_indices, f'l2_pseudo_label_{group_name}']
            acc = accuracy_score(df.loc[target_indices, 'y_true'].astype(str), temp_pred.astype(str))
            
            # ★★★ 追加: 履歴を記録 ★★★
            group_threshold_history.append((threshold, acc))
            
            if acc >= best_acc_group: best_acc_group, best_th_group = acc, threshold
        
        best_thresholds[group_name] = best_th_group
        threshold_accuracy_history[group_name] = group_threshold_history

    print("最大改善率ルールに基づいて最終的な予測を生成中...")
    final_y_pred = df['l1_pseudo_label'].copy()
    for idx in range(len(df)):
        best_r = 1.0; best_group_for_this_data = None
        for group_name in layer2_results.keys():
            score_l2 = df.loc[idx, f'score_l2_{group_name}']
            if not pd.isna(score_l2):
                r_group = score_l2 / df.loc[idx, 'score_l1']
                if r_group > best_r: best_r = r_group; best_group_for_this_data = group_name
        if best_group_for_this_data is not None:
            threshold = best_thresholds.get(best_group_for_this_data, 999.9)
            if best_r > threshold: final_y_pred.loc[idx] = df.loc[idx, f'l2_pseudo_label_{best_group_for_this_data}']

    # ★★★ 変更点: 閾値履歴をレポート関数に渡す ★★★
    print_final_report("V1.3", "最大改善率ルール", best_thresholds, final_y_pred.tolist(), df['y_true'].tolist(), accuracy_layer1, threshold_accuracy_history)

def print_final_report(version, metric_name, best_th, final_y_pred, y_true, l1_acc, threshold_history):
    # (変更なしの部分は簡略化)
    print(f"\n--- 【最終レポート {version}】 ---"); print(f"指標: {metric_name}")
    if isinstance(best_th, dict):
        print("最適なルール (グループ別):")
        for g, t in best_th.items(): val = "(変更なし)" if t >= 999 else f"R > {t:.3f}"; print(f"  {g}: {val}")
    else: print(f"最適なルール: R > {best_th:.3f}" if best_th is not None else "最適なルールなし")
    valid_indices = [i for i, label in enumerate(final_y_pred) if label != '分類不能']
    y_true_final = [str(y_true[i]) for i in valid_indices]; y_pred_final = [str(final_y_pred[i]) for i in valid_indices]
    final_accuracy = accuracy_score(y_true_final, y_pred_final)
    precision_val = precision_score(y_true_final, y_pred_final, average='weighted', zero_division=0)
    f1_val = f1_score(y_true_final, y_pred_final, average='weighted', zero_division=0)
    print(f"\n最終的な全体の Accuracy      : {final_accuracy:.4f} (第1層から {final_accuracy - l1_acc:+.4f})")
    print(f"最終的な全体の Weighted Precision: {precision_val:.4f}"); print(f"最終的な全体の Weighted F1 Score : {f1_val:.4f}")
    print(classification_report(y_true_final, y_pred_final, digits=4, zero_division=0))
    
    # ★★★ 追加: 閾値探索グラフ描画関数を呼び出す ★★★
    plot_threshold_search_history(threshold_history)

    try:
        labels = sorted(list(set(y_true_final) | set(y_pred_final))); cm = confusion_matrix(y_true_final, y_pred_final, labels=labels)
        plt.figure(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('予測ラベル (Predicted Label)'); plt.ylabel('正解ラベル (True Label)'); plt.title(f'混同行列のヒートマップ ')
        plt.tight_layout(); filename = f'confusion_matrix_{version}.png'; plt.savefig(filename)
        print(f"\n混同行列のヒートマップを '{filename}' として保存しました。"); plt.close()
    except Exception as e: print(f"\nヒートマップの描画中にエラーが発生しました: {e}")
    print("-" * 40)

def calculate_ab_scores_weighted(X_point, kmeans_obj, weights_array=None):
    if kmeans_obj is None: return np.nan, np.nan
    point_weighted = X_point.reshape(1, -1) * weights_array if weights_array is not None else X_point.reshape(1, -1)
    centers_weighted = kmeans_obj.cluster_centers_
    distances = cdist(point_weighted, centers_weighted).flatten()
    cluster_idx = kmeans_obj.predict(point_weighted)[0]
    a_x = distances[cluster_idx]
    distances[cluster_idx] = np.inf
    b_x = np.min(distances)
    return a_x, b_x

# ==============================================================================
# メイン実行ブロック
# ==============================================================================
if __name__ == '__main__':
    analysis_objects = multi_category_hierarchical_clustering(
        real_data_file='all features 150.xlsx',
        artificial_data_file='all features 1020.xlsx',
    )
    
    if analysis_objects:
        df_res, l1_acc, X, km1, l2_res, opt_w, used_f, w1, acc_hist = analysis_objects
        
        plot_accuracy_history(acc_hist)
        
        run_analysis_v1_3(df_res, l1_acc, X, km1, l2_res, opt_w, used_f, w1)