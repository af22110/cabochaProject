import CaboCha
import math

def anomaly_metrics_token_level(text: str):
    parser = CaboCha.Parser()
    tree   = parser.parse(text)

    # 1) 文節(Chunk)リストの取得
    chunks = [tree.chunk(i) for i in range(tree.chunk_size())]

    # 2) トークン→文節マッピングを作る
    #    token_to_chunk[i] = トークンiが属する文節番号
    token_to_chunk = [None] * tree.size()
    for ci, ch in enumerate(chunks):
        start = ch.token_pos
        size  = ch.token_size
        for ti in range(start, start + size):
            token_to_chunk[ti] = ci

    # 3) トークン依存距離を計算
    token_distances = []
    for ti in range(tree.size()):
        ci = token_to_chunk[ti]      # 自分の文節
        head_ci = chunks[ci].link    # 係り先文節番号
        if head_ci >= 0:
            # 係り先文節の先頭トークン位置との距離
            head_token_pos = chunks[head_ci].token_pos
            token_distances.append(abs(ti - head_token_pos))

    # トークン距離の分散
    if token_distances:
        mean_td = sum(token_distances) / len(token_distances)
        var_td  = sum((d - mean_td) ** 2 for d in token_distances) / len(token_distances)
    else:
        var_td = 0.0

    # 係り受けスコアの分散（従来どおり文節単位）
    scores = [ch.score for ch in chunks]
    if scores:
        mean_sc = sum(scores) / len(scores)
        var_sc  = sum((s - mean_sc) ** 2 for s in scores) / len(scores)
    else:
        var_sc = 0.0

    return var_td, var_sc

if __name__ == "__main__":
    samples = [
        "太郎は花子が読んでいる本を次郎に渡した。",
        "走り空飛ぶ魚車大雨召喚"
    ]
    for text in samples:
        v_td, v_sc = anomaly_metrics_token_level(text)
        print(f"[{text}]")
        print(f"  Token依存距離の分散: {v_td:.4f}")
        print(f"  係り受けスコアの分散: {v_sc:.4f}")
