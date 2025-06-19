import matplotlib.pyplot as plt
import numpy as np
from typing import List
from matplotlib.patches import Rectangle

def plot_topk(
    tokens: List[str],
    values: List[float],
    ids: List[int],
    chosen: int,
    top_k: int,
    temperature: float,
    title: str = "Top-K Distribution"
):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#e63946" if i == chosen else "#457b9d" for i in ids]
    ax.barh(tokens[::-1], values[::-1], color=list(reversed(colors)))
    xlabel = "Logit Score" if temperature < 1e-5 else "Probability"
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
    mn, mx = min(values), max(values)
    for i, v in enumerate(values[::-1]):
        label = f"{v:.0f}" if temperature < 1e-5 else f"{v:.2f}"
        offset = (mx - mn) * 0.02 if temperature < 1e-5 else mx * 0.01
        ax.text(v + offset, i, f"{v:.2f}", va='center', ha='left', fontsize=8)
    if temperature < 1e-5:
        ax.set_xlim(mn - abs(mn) * 0.1, mx + abs(mx) * 0.1)
    return fig

def plot_logits(
    tokens: List[str],
    logits: List[float],
    ids: List[int],
    chosen: int,
    title: str = "Logits（Softmax前のスコア）"
):
    """
    Softmax前のLogitsスコアを棒グラフとして表示する。
    """
    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = ["#e63946" if i == chosen else "#457b9d" for i in ids]
    ax.barh(tokens[::-1], logits[::-1], color=list(reversed(colors)))
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.invert_yaxis()

    # 値ラベルを追加
    mn, mx = min(logits), max(logits)
    for i, v in enumerate(logits[::-1]):
        offset = (mx - mn) * 0.01
        ax.text(v + offset, i, f"{v:.2f}", va='center', ha='left', fontsize=8)

    ax.set_xlim(mn - abs(mn) * 0.1, mx + abs(mx) * 0.1)
    return fig

def plot_attention(attn: np.ndarray, tokens: List[str], title: str = "Average"):
    """
    Attention ヒートマップ（Full N×N）を表示する関数。

    - 行 (Query): 全トークン
    - 列 (Key): 全トークン
    - 未来への attention (上三角) を灰色でマスク
    - 最終行 (新トークン) をシアンでハイライト
    """
    n = attn.shape[0]
    tokens_q = tokens[1:n+1]
    tokens_k = tokens[:n]

    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    data = np.ma.array(attn, mask=mask)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    cmap = plt.cm.get_cmap('inferno').copy()
    cmap.set_bad(color='lightgray')
    im = ax.imshow(data, cmap=cmap,
                   vmin=attn.min(), vmax=attn.max(),
                   interpolation='nearest')

    ax.set_title(f"Attention Map ({title})", pad=12)
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(tokens_k, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(tokens_q, fontsize=8)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                continue
            val = attn[i, j]
            color = 'white' if val > (attn.max() * 0.5) else 'black'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=6, color=color)

    ax.add_patch(
        Rectangle((-0.5, n - 1.5), n, 1,
                      fill=False, edgecolor='cyan', lw=2)
    )

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', rotation=270, labelpad=15)

    fig.subplots_adjust(bottom=0.25, left=0.25)
    plt.tight_layout()
    return fig
