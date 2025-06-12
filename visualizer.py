import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_topk(
    tokens: List[str],
    values: List[float],
    ids: List[int],
    chosen: int,
    top_k: int,
    temperature: float,
):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#e63946" if i == chosen else "#457b9d" for i in ids]
    ax.barh(tokens[::-1], values[::-1], color=list(reversed(colors)))
    xlabel = "Logit Score" if temperature < 1e-5 else "Probability"
    ax.set_title(f"Top-{top_k} Candidates ({xlabel})", pad=12)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
    mn, mx = min(values), max(values)
    for i, v in enumerate(values[::-1]):
        label = f"{v:.0f}" if temperature < 1e-5 else f"{v:.2f}"
        offset = (mx - mn) * 0.02 if temperature < 1e-5 else mx * 0.01
        ax.text(v + offset, i, label, va='center', ha='left', fontsize=8)
    if temperature < 1e-5:
        ax.set_xlim(mn - abs(mn) * 0.1, mx + abs(mx) * 0.1)
    return fig

def plot_attention(attn: np.ndarray, tokens: List[str]):
    """
    Attention ヒートマップ:
    正方形表示し、右上三角部分（j >= i）を灰色で置き換え、
    左下は元の注意重みをそのまま表示、新トークン行（最後の行）をハイライト
    """
    n = attn.shape[0]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')

    # マスク領域: j >= i（上三角+対角線）
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)

    # データをマスク
    data = np.ma.array(attn, mask=mask)

    # カラーマップ設定: bad 値（マスク部分）を灰色に
    cmap = plt.cm.get_cmap('inferno').copy()
    cmap.set_bad(color='lightgray')

    # マスク処理したデータを描画
    im = ax.imshow(data, cmap=cmap, vmin=attn.min(), vmax=attn.max(), interpolation='nearest')

    # 軸ラベルとタイトル
    ax.set_title('Attention Map (Last Layer Avg)', pad=12)
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)

    # 注釈（マスク部分除外）
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                continue
            weight = attn[i, j]
            color = 'white' if weight > (attn.max() * 0.5) else 'black'
            ax.text(j, i, f'{weight:.2f}', ha='center', va='center', fontsize=6, color=color)

    # 新トークン行ハイライト
    new_idx = n - 1
    rect = plt.Rectangle((-0.5, new_idx - 0.5), n, 1, fill=False, edgecolor='cyan', lw=2)
    ax.add_patch(rect)

    # 軽いグリッド線
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)

    # カラーバー
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', rotation=270, labelpad=15)

    plt.tight_layout()
    return fig