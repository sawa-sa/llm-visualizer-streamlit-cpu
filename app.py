import numpy as np
import streamlit as st
from config import DEFAULT_PROMPTS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K
from model_loader import load_model
from generator import generate_step
from visualizer import plot_topk, plot_attention
from devinfo import show_device_info


device = show_device_info()
model, tokenizer, device = load_model()

# ─── セッションステートの初期化 ───────────────────────────
state = st.session_state
for key, default in [
    ("input_ids", None),
    ("steps", []),
    ("step_index", 0),
    ("prompt", DEFAULT_PROMPTS[0]),
    ("prompt_selector", DEFAULT_PROMPTS[0]),
    ("prompt_input", DEFAULT_PROMPTS[0]),
    ("lock_params", False)
]:
    if key not in state:
        state[key] = default

# ─── 初期プロンプト自動適用関数 ────────────────────────────
def init_with_template():
    state.prompt = state.prompt_selector.strip().replace("\n", " ").replace("\r", "")
    # state.prompt_input = state.prompt_selector
    state.input_ids = tokenizer.encode(state.prompt, return_tensors="pt").to(device)
    state.steps = []
    state.step_index = 0
    state.lock_params = False

def on_template_change():
    state.prompt_input = state.prompt_selector  # テンプレート選択時のみカスタム入力に反映
    init_with_template()


def init_with_custom():
    state.prompt = state.prompt_input.strip().replace("\n", " ").replace("\r", "")
    state.input_ids = tokenizer.encode(state.prompt, return_tensors="pt").to(device)
    state.steps = []
    state.step_index = 0
    state.lock_params = False

# 初回ロード時にデフォルトプロンプトを設定
if state.input_ids is None:
    init_with_template()

# ─── UI設定: タイトルと探索モード ─────────────────────────
st.title("🔍 GPT-2 Medium 可視化デモ")
explore_mode = st.checkbox(
    "🔀 探索モード: 途中でパラメータ変更を許可",
    value=False,
    help="オフにすると生成後にパラメータがロックされます"
)
locked = state.lock_params and not explore_mode

# ─── プロンプト選択 & 編集 ─────────────────────────────────
st.selectbox(
    "🧪 プロンプトテンプレート",
    DEFAULT_PROMPTS,
    index=DEFAULT_PROMPTS.index(state.prompt) if state.prompt in DEFAULT_PROMPTS else 0,
    key="prompt_selector",
    disabled=locked,
    on_change=on_template_change
)
st.text_input(
    "または自分で入力",
    value=state.prompt,
    key="prompt_input",
    disabled=locked,
    on_change=init_with_custom
)
# 初期化ボタン
st.button(
    "🔄 プロンプト初期化",
    on_click=init_with_template,
    disabled=False
)

# ─── パラメータ設定 ─────────────────────────────────────
temperature = st.slider(
    "Temperature",
    0.0, 2.0,
    value=DEFAULT_TEMPERATURE,
    step=0.1,
    disabled=locked
)
ntop_p = st.slider(
    "Top-p (Nucleus)",
    0.0, 1.0,
    value=DEFAULT_TOP_P,
    step=0.01,
    disabled=locked or temperature <= 0.0
)
ntop_k = st.slider(
    "Top-K Sampling",
    1, 50,
    value=DEFAULT_TOP_K,
    step=1,
    disabled=locked or ntop_p < 1.0 or temperature <= 0.0
)

st.markdown("---")
if locked:
    st.info("🔒 パラメータロック中: プロンプト変更で解除")
elif temperature <= 0.0:
    st.warning("⚠️ Temperature=0 のため Greedy Decoding")
elif ntop_p < 1.0:
    st.warning("⚠️ Top-p Mode: Top-K 無効")
else:
    st.warning("⚠️ Top-K Mode: Top-p 無効")

chart_ph = st.empty()
heatmap_ph = st.empty()

# ─── トークン生成 & ロックコールバック ────────────────────────
def generate_and_lock():
    prev_sel = state.get(f"head_select_{state.step_index}", "Average")
    result = generate_step(
        state.input_ids, model, tokenizer,
        temperature, ntop_p, ntop_k, device
    )
    state.input_ids = result["input_ids"]
    state.steps.append(result["step_data"])
    new_idx = len(state.steps) - 1
    state.step_index = new_idx
    state[f"head_select_{new_idx}"] = prev_sel
    if not explore_mode:
        state.lock_params = True

st.button(
    "▶️ トークン生成",
    on_click=generate_and_lock
)

# ─── ステップナビゲーション & 可視化 ───────────────────────
if state.steps:
    idx = state.step_index
    step = state.steps[idx]

    c1, _, c3 = st.columns([1, 2, 1])
    c1.button(
        "← 前へ",
        on_click=lambda: setattr(state, 'step_index', max(idx-1, 0)),
        disabled=(idx == 0),
        key='prev'
    )
    c3.button(
        "次へ →",
        on_click=lambda: setattr(state, 'step_index', min(idx+1, len(state.steps)-1)),
        disabled=(idx == len(state.steps)-1),
        key='next'
    )
    st.markdown(f"**Step {idx+1}/{len(state.steps)}**")

    # Top-K プロット
    if temperature <= 0.0:
        title, limit = "Top-1 (Greedy)", 1
    elif ntop_p < 1.0:
        title, limit = f"Top-p (p={ntop_p:.2f})", 10
    else:
        title, limit = f"Top-K (k={ntop_k})", ntop_k
    fig = plot_topk(
        tokens=step["tokens"],
        values=step["values"],
        ids=step["ids"],
        chosen=step["chosen"],
        top_k=limit,
        temperature=temperature,
        title=title
    )
    chart_ph.pyplot(fig)

    # Attention ヒートマップ
    attn = step["attn"]
    if attn.ndim == 2:
        attn = attn[np.newaxis, ...]
    options = ["Average"] + [f"Head {i}" for i in range(attn.shape[0])]
    key = f"head_select_{idx}"
    sel = st.selectbox(
        "Attention Head",
        options,
        key=key
    )
    mat = attn.mean(axis=0) if sel == "Average" else attn[int(sel.split()[1])]
    heat_fig = plot_attention(mat, step["all_toks"], title=sel)
    heatmap_ph.pyplot(heat_fig, clear_figure=False)

# ─── 最終出力を表示 ───────────────────────────────────────
# st.markdown("### 🧠 最終アウトプット")
# st.write(tokenizer.decode(state.input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
# 注意: 'Q:' のような記号付き先頭トークンは ['Q', ':', ...] に分割されるため、
# decode 時に不可視文字（例: U+2028）として復元され、改行に見えることがある。
# st.text() よりも st.code() での表示が推奨される。

final_text = tokenizer.decode(state.input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
final_text = final_text.replace("\n", " ").replace("\r", "").replace("\u2028", " ").replace("\u2029", " ")
st.text(final_text)
