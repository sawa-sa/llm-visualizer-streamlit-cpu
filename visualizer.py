import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

def plot_topk(tokens: List[str], values: List[float], ids: List[int], chosen: int, top_k: int, temperature: float):
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
    fig, ax = plt.subplots(figsize=(6, 6))
    n = attn.shape[0]
    im = ax.imshow(attn, cmap="inferno", aspect='auto', vmin=0.0, vmax=attn.max())
    ax.set_title("Attention Map (Last Layer Avg)", pad=12)
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    for r in range(n):
        ci = int(np.argmax(attn[r]))
        rect = plt.Rectangle((ci-0.5, r-0.5), 1, 1, fill=False, edgecolor='white', lw=1.5)
        ax.add_patch(rect)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Attention Weight", rotation=270, labelpad=15)
    plt.tight_layout()
    return fig