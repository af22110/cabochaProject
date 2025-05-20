# word3gram.py
import os
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# --- 日本語フォント設定 (変更なし、前回のものをそのまま使用) ---
try:
    font_candidates = [
        'IPAexGothic', 'TakaoPGothic', 'Noto Sans CJK JP', 'Yu Gothic', 'MS Gothic',
        'Hiragino Sans', 'Meiryo', 'VL Gothic', 'IPAGothic', 'TakaoGothic'
    ]
    font_path = None
    font_name_used = "System Default"
    for font_name_candidate in font_candidates:
        try:
            font_path_candidate = fm.findfont(fm.FontProperties(family=font_name_candidate), fallback_to_default=False)
            if font_path_candidate and os.path.exists(font_path_candidate):
                plt.rcParams['font.family'] = font_name_candidate
                font_name_used = font_name_candidate
                font_path = font_path_candidate
                print(f"Using font: {font_name_used} (Path: {font_path})")
                break
        except Exception:
            continue
    if not font_path:
        print(f"Warning: None of the Japanese font candidates found: {font_candidates}. Using system default (likely DejaVu Sans).")
        print("         Graph labels (annotations) might not display Japanese characters correctly.")
        try:
            default_font_path = fm.findfont(fm.FontProperties(family="DejaVu Sans"))
            print(f"         Falling back to: DejaVu Sans (Path: {default_font_path}) for non-Japanese text and plot elements.")
        except:
            print(f"         Could not even find DejaVu Sans. Matplotlib might use a very basic font.")
except Exception as e:
    print(f"Warning: Critical error during Japanese font setup for matplotlib: {e}")

# --- SudachiPyのインポートと設定 (変更なし) ---
try:
    from sudachipy import tokenizer
    from sudachipy import dictionary
    sudachi_tokenizer = dictionary.Dictionary().create()
    sudachi_mode = tokenizer.Tokenizer.SplitMode.C
    print("SudachiPy tokenizer loaded.")
except ImportError:
    print("Error: SudachiPy or sudachidict_core not found. Please ensure they are installed in your environment.")
    exit()
except Exception as e:
    print(f"Error loading SudachiPy tokenizer: {e}")
    exit()


if __name__ == "__main__":
    start_overall_time = time.time()

    # --- 設定項目 (変更なし) ---
    pickle_data_path = os.path.expanduser('~/cabochaProject/nwc2010-ngrams/word/over999/')
    input_text = "猫が道路を歩いています。"
    SMOOTHING_K = 1.0
    UNNATURAL_THRESHOLD_PROB = 0.00001
    GRAPH_OUTPUT_FILENAME = "word3gram_smoothed.png"
    MAX_TEXT_LENGTH_FOR_NUMERIC_XTICKS_ALL = 40 # 全ての文字位置に数値目盛りを表示する最大テキスト長
    NUMERIC_XTICKS_INTERVAL_LONG_TEXT = 5      # 長いテキストの場合の数値目盛りの間隔

    # --- 設定表示 (変更なし) ---
    print(f"\n--- Settings ---")
    print(f"Input text: \"{input_text}\"")
    print(f"N-gram data path: {pickle_data_path}")
    print(f"Smoothing parameter (k): {SMOOTHING_K}")
    print(f"Unnatural probability threshold (P): {UNNATURAL_THRESHOLD_PROB:.2e}")
    if UNNATURAL_THRESHOLD_PROB > 0:
        LOG_PROB_THRESHOLD_DISPLAY = np.log(UNNATURAL_THRESHOLD_PROB)
        print(f"Log probability display threshold (log P): {LOG_PROB_THRESHOLD_DISPLAY:.4f}")
    else:
        LOG_PROB_THRESHOLD_DISPLAY = -float('inf')
        print(f"Log probability display threshold (log P): -inf (as P_threshold is 0 or less)")

    # --- N-gramデータのロード (変更なし、前回のものをそのまま使用) ---
    bigram_counts = None
    trigram_counts = None
    total_vocab_size = 0 # V
    load_success = True
    pickle_file_2gm_data = os.path.join(pickle_data_path, 'bigrams_data.pkl')
    pickle_file_3gm_data = os.path.join(pickle_data_path, 'trigrams_data.pkl')
    try:
        start_load_time = time.time()
        with open(pickle_file_2gm_data, 'rb') as f:
            loaded_bigram_data = pickle.load(f)
        bigram_counts = loaded_bigram_data.get('counts', {})
        v_from_bigram = loaded_bigram_data.get('total_vocab_size', 0)
        print(f"Loaded {len(bigram_counts)} unique 2-grams (V={v_from_bigram} in this file) from {pickle_file_2gm_data}. (Time: {time.time() - start_load_time:.2f}s)")
        if not bigram_counts: print(f"Warning: No 2-gram counts found in {pickle_file_2gm_data}.")
        if v_from_bigram == 0 and len(bigram_counts) > 0: print(f"Warning: Bigram vocabulary size (V) is 0 in {pickle_file_2gm_data}, but counts exist.")
    except FileNotFoundError:
        print(f"Error: {pickle_file_2gm_data} not found. Please run pre_word3gram.py first to generate it.")
        load_success = False
    except Exception as e: print(f"Error loading {pickle_file_2gm_data}: {e}"); load_success = False
    if load_success:
        try:
            start_load_time = time.time()
            with open(pickle_file_3gm_data, 'rb') as f:
                loaded_trigram_data = pickle.load(f)
            trigram_counts = loaded_trigram_data.get('counts', {})
            total_vocab_size = loaded_trigram_data.get('total_vocab_size', 0)
            print(f"Loaded {len(trigram_counts)} unique 3-grams (V={total_vocab_size} in this file) from {pickle_file_3gm_data}. (Time: {time.time() - start_load_time:.2f}s)")
            if not trigram_counts: print(f"Warning: No 3-gram counts found in {pickle_file_3gm_data}.")
            if total_vocab_size == 0 and len(trigram_counts) > 0: print(f"Warning: Trigram vocabulary size (V) is 0 in {pickle_file_3gm_data}, but counts exist.")
        except FileNotFoundError:
            print(f"Error: {pickle_file_3gm_data} not found. Please run pre_word3gram.py first to generate it.")
            load_success = False
        except Exception as e: print(f"Error loading {pickle_file_3gm_data}: {e}"); load_success = False
    if not load_success: print("\nFailed to load necessary N-gram data. Exiting analysis."); exit()
    V = total_vocab_size
    if V == 0 and SMOOTHING_K > 0: print("Critical Warning: Total vocabulary size (V) is 0. Add-k smoothing with k > 0 will lead to division by zero or incorrect probabilities. Exiting."); exit()
    elif V == 0 and SMOOTHING_K == 0: print("Warning: Total vocabulary size (V) is 0 and k=0 (no smoothing). Probabilities might be undefined if context counts are also zero.")

    # --- Text Analysis (変更なし、前回のものをそのまま使用) ---
    print("\n--- Text Analysis ---")
    morpheme_objects = []
    try:
        morpheme_objects = list(sudachi_tokenizer.tokenize(input_text, sudachi_mode))
        morphemes_surface = [m.surface() for m in morpheme_objects]
        print(f"Tokenized by SudachiPy (mode: {sudachi_mode}, {len(morphemes_surface)} morphemes): {morphemes_surface}")
    except Exception as e: print(f"Error during SudachiPy tokenization: {e}"); exit()

    if len(morpheme_objects) < 3:
        print("\nTokenized text is too short (less than 3 morphemes) for 3-gram analysis.")
    else:
        print(f"\n--- Conditional Probability Analysis (3-grams with Add-{SMOOTHING_K} Smoothing, V={V}) ---")
        found_unnatural_sequences = []
        trigram_strings_for_annotation = []
        log_probabilities_for_graph = []
        x_char_positions_for_graph = [] 
        min_log_prob_overall = float('inf') 
        min_log_prob_info = None

        for i in range(len(morpheme_objects) - 2):
            m1, m2, m3 = morpheme_objects[i], morpheme_objects[i+1], morpheme_objects[i+2]
            w1, w2, w3 = m1.surface(), m2.surface(), m3.surface()
            current_trigram_tuple = (w1, w2, w3)
            current_bigram_tuple = (w1, w2)
            raw_count_w1_w2_w3 = trigram_counts.get(current_trigram_tuple, 0) if trigram_counts else 0
            raw_count_w1_w2 = bigram_counts.get(current_bigram_tuple, 0) if bigram_counts else 0
            numerator = raw_count_w1_w2_w3 + SMOOTHING_K
            denominator = raw_count_w1_w2 + (SMOOTHING_K * V)
            smoothed_prob, log_prob = 0.0, -float('inf')
            if denominator > 0:
                smoothed_prob = numerator / denominator
                if smoothed_prob > 0: log_prob = np.log(smoothed_prob)
            elif SMOOTHING_K == 0 and raw_count_w1_w2 == 0: smoothed_prob, log_prob = 0.0, -float('inf')
            
            # (略: print文や最小確率更新、不自然判定のロジックは前回のまま)
            print(f"  Sequence: ('{w1}', '{w2}', '{w3}') at char_pos_start={m1.begin()}")
            print(f"    Raw Counts: Trigram={raw_count_w1_w2_w3}, Bigram_Ctx={raw_count_w1_w2}")
            print(f"    Smoothed P( '{w3}' | '{w1}', '{w2}' ) = {smoothed_prob:.6e} (Num={numerator:.1f}, Denom={denominator:.1f})")
            print(f"    Smoothed Log P(...) = {log_prob:.4f}")

            if log_prob < min_log_prob_overall:
                min_log_prob_overall = log_prob
                min_log_prob_info = {
                    "sequence": (w1, w2, w3), "smoothed_probability": smoothed_prob, "log_probability": log_prob,
                    "raw_trigram_count": raw_count_w1_w2_w3, "raw_bigram_count": raw_count_w1_w2, "start_char_position": m1.begin()
                }
            trigram_strings_for_annotation.append(f"{w1}\n{w2}\n{w3}")
            log_probabilities_for_graph.append(log_prob)
            x_char_positions_for_graph.append(m1.begin())
            is_unnatural = False
            reason = ""
            if smoothed_prob < UNNATURAL_THRESHOLD_PROB:
                is_unnatural = True
                reason = f"Smoothed P ({smoothed_prob:.2e}) < Threshold ({UNNATURAL_THRESHOLD_PROB:.1e}). Raw Cnt: 3g={raw_count_w1_w2_w3}, 2g_ctx={raw_count_w1_w2}."
            if is_unnatural:
                # (略: unnatural_info 作成部分は前回のまま)
                context_window = 5; start_idx_morph = max(0, i - context_window//2 +1); end_idx_morph = min(len(morphemes_surface), i + 2 + context_window//2 +1)
                context_snippet_list = []
                for k_morph_idx in range(start_idx_morph, end_idx_morph):
                    surface_form = morphemes_surface[k_morph_idx]; prefix = "  "; suffix = ""
                    if k_morph_idx == i: prefix = ">>"
                    if k_morph_idx == i+2 : suffix = "<<"
                    context_snippet_list.append(f"{prefix}{surface_form}{suffix}")
                context_snippet = " ".join(context_snippet_list)
                unnatural_info = {"sequence": (w1, w2, w3), "probability": smoothed_prob, "log_probability": log_prob, "reason": reason, 
                                  "original_text_snippet": context_snippet, "start_char_position": m1.begin(),
                                  "raw_trigram_count": raw_count_w1_w2_w3, "raw_bigram_count": raw_count_w1_w2}
                found_unnatural_sequences.append(unnatural_info)
                print(f"    >> Unnatural sequence detected! {reason}")
        # (略: 不自然箇所表示、最小確率表示は前回のまま)
        if found_unnatural_sequences:
            print("\n--- Detected Unnatural Sequences (based on smoothed probability) ---")
            for item in found_unnatural_sequences: print(f"  - Seq: {item['sequence']} (char_pos: {item['start_char_position']})\n    LogP: {item['log_probability']:.4f} (P: {item['probability']:.2e}), Reason: {item['reason']}")
        else: print("\nNo sequences detected below the unnaturalness threshold based on smoothed probabilities.")
        if min_log_prob_info:
            print("\n--- Most Statistically Unlikely 3-gram Sequence (Lowest Smoothed Log Probability) ---")
            item = min_log_prob_info; print(f"  Sequence: {item['sequence']} (starts at char {item['start_char_position']})\n  Smoothed Log Prob: {item['log_probability']:.4f} (Smoothed Prob: {item['smoothed_probability']:.6e})\n  Raw Counts: Trigram={item['raw_trigram_count']}, Bigram_Ctx={item['raw_bigram_count']}")
        else: print("\nCould not determine the most unlikely sequence (no 3-grams processed or all had -inf log_prob).")

        # --- グラフ描画 ---
        if log_probabilities_for_graph:
            print(f"\n--- Generating graph: {GRAPH_OUTPUT_FILENAME} ---")
            try:
                fig, ax = plt.subplots(figsize=(max(12, len(input_text) * 0.25 + 3), 7)) # X軸ラベル表示のため少し高さを確保
                
                plot_x_values = np.array(x_char_positions_for_graph, dtype=float)
                plot_y_values = np.array(log_probabilities_for_graph, dtype=float)

                finite_log_probs = plot_y_values[np.isfinite(plot_y_values)]
                min_finite_log_prob = finite_log_probs.min() if len(finite_log_probs) > 0 else LOG_PROB_THRESHOLD_DISPLAY - 5
                plot_min_y = min(LOG_PROB_THRESHOLD_DISPLAY - 2 , min_finite_log_prob - 2) if np.isfinite(LOG_PROB_THRESHOLD_DISPLAY) else min_finite_log_prob - 2
                plot_y_values[plot_y_values == -np.inf] = plot_min_y -1 

                ax.plot(plot_x_values, plot_y_values, 
                           marker='o', linestyle='-', label='3-gram Smoothed Log Prob.', color='dodgerblue', markersize=7, zorder=3)
                
                for i, txt_label in enumerate(trigram_strings_for_annotation):
                    ax.annotate(txt_label, 
                                (plot_x_values[i], plot_y_values[i]),
                                textcoords="offset points", xytext=(0,10), 
                                ha='center', fontsize=7, color='dimgray',
                                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.5, ec="none"))
                
                if np.isfinite(LOG_PROB_THRESHOLD_DISPLAY):
                    ax.axhline(y=LOG_PROB_THRESHOLD_DISPLAY, color='r', linestyle='--', 
                               label=f'Threshold ({LOG_PROB_THRESHOLD_DISPLAY:.2f}, P≈{UNNATURAL_THRESHOLD_PROB:.1e})', zorder=2)
                
                # X軸のラベルを「文字インデックス」に変更
                ax.set_xlabel('Character Index in Original Text (0-indexed)')
                ax.set_ylabel(f'Smoothed Log Probability (k={SMOOTHING_K}, V={V})')
                ax.set_title(f'3-gram Conditional Log Probability (Add-{SMOOTHING_K} Smoothed)')
                
                if input_text:
                    ax.set_xlim(-0.5, len(input_text) - 0.5) # X軸の範囲を少し広げる
                    
                    # X軸の目盛りを数値（文字インデックス）に変更
                    if len(input_text) <= MAX_TEXT_LENGTH_FOR_NUMERIC_XTICKS_ALL:
                        # テキストが短い場合は全ての文字位置に目盛りを表示
                        ticks = np.arange(len(input_text))
                        ax.set_xticks(ticks)
                        ax.set_xticklabels([str(t) for t in ticks], fontsize=8)
                    else:
                        # テキストが長い場合は一定間隔で目盛りを表示
                        ticks = np.arange(0, len(input_text), NUMERIC_XTICKS_INTERVAL_LONG_TEXT)
                        if len(input_text)-1 not in ticks : # 最後尾のインデックスも追加
                            ticks = np.append(ticks, len(input_text)-1)
                        ax.set_xticks(ticks)
                        ax.set_xticklabels([str(t) for t in ticks], fontsize=8, rotation=30, ha="right")
                    
                    # X軸のマイナーグリッド（必要に応じて）
                    # ax.set_xticks(np.arange(len(input_text)), minor=True)
                    # ax.grid(True, linestyle=':', alpha=0.3, which='minor', axis='x')

                if len(finite_log_probs) > 0 or np.isfinite(LOG_PROB_THRESHOLD_DISPLAY) :
                    current_min_y = plot_y_values.min()
                    current_max_y = plot_y_values[np.isfinite(plot_y_values)].max() if len(plot_y_values[np.isfinite(plot_y_values)]) > 0 else 0
                    y_bottom = current_min_y - 1
                    y_top = max(current_max_y + 1, (LOG_PROB_THRESHOLD_DISPLAY +1 if np.isfinite(LOG_PROB_THRESHOLD_DISPLAY) else 0) )
                    if y_top <= y_bottom : y_top = y_bottom +2 
                    ax.set_ylim(y_bottom, y_top)

                fig.suptitle(f'Input: "{input_text}"', fontsize=10, y=0.99) 
                ax.legend(loc='best')
                ax.grid(True, linestyle=':', alpha=0.6, zorder=1, which='major') 
                
                fig.tight_layout(rect=[0, 0.03, 1, 0.96]) 
                plt.savefig(GRAPH_OUTPUT_FILENAME)
                print(f"Graph saved to {GRAPH_OUTPUT_FILENAME}")
            except Exception as e:
                print(f"Error generating graph: {e}")
        else:
            print("\nNo valid probability data to generate graph (all log_probs might be -inf or no 3-grams).")

    print(f"\nTotal analysis time: {time.time() - start_overall_time:.2f}s")