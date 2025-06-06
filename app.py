import streamlit as st
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.rcParams["font.family"] = "IPAexGothic"

# ——————————————
# セッション状態の初期化
# ——————————————
if "input_ids" not in st.session_state:
    st.session_state.input_ids = None
if "generated_tokens" not in st.session_state:
    st.session_state.generated_tokens = []
if "mode" not in st.session_state:
    st.session_state.mode = "ステップバイステップ"
if "steps" not in st.session_state:
    # 各ステップごとのデータを保持するリスト
    # 要素は dict: {"topk_tokens","topk_values","chosen_id","attn_avg","tokens_all"}
    st.session_state.steps = []
if "step_index" not in st.session_state:
    st.session_state.step_index = 0

# プロンプト選択用のセッションキー初期化
if "prompt" not in st.session_state:
    st.session_state.prompt = "The cat sat on the"
if "prompt_initialized" not in st.session_state:
    st.session_state.prompt_initialized = False

@st.cache_resource
def load_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium", output_attentions=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

st.title("🔍 GPT-2 可視化デモ：2モード切替＋ステップナビゲーション")

# ——————————————
# プロンプト選択＆入力欄
# ——————————————
example_prompt = st.selectbox(
    "🧪 試してみたいプロンプトを選んでください（編集も可能）",
    [
        "（←選んでください）",
        "Once upon a time, there was a",
        "In the future, artificial intelligence will",
        "The quick brown fox jumps over the",
        "I can't believe that she actually",
        "This is the reason why you should never",
        "The meaning of life is",
        "If I were the president, I would",
        "She looked at him and said",
    ],
    key="prompt_selector"
)
if example_prompt != "（←選んでください）" and not st.session_state.prompt_initialized:
    st.session_state.prompt = example_prompt
    st.session_state.prompt_initialized = True

prompt = st.text_input(
    "プロンプト",
    value=st.session_state.prompt
)
# ユーザーが直接編集したら上書き
st.session_state.prompt = prompt

temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)

# モードごとの設定
mode = st.radio(
    "生成モードを選択してください",
    ("ステップバイステップ", "まとめて生成")
)
st.session_state.mode = mode

if mode == "ステップバイステップ":
    top_p = st.slider(
        "Top-p (Nucleus sampling)", 0.0, 1.0, 1.0, 0.01,
        help="Top-p < 1.0 のときは Top-p サンプリング、Top-p = 1.0 のときは Top-K サンプリング"
    )
    top_k = st.slider(
        "Top-K sampling", 1, 50, 10, 1,
        help="Top-p = 1.0 のときのみ有効", disabled=(top_p < 1.0)
    )
    if top_p < 1.0:
        st.markdown("⚠️ Top-K は現在無効です（Top-p 有効）")
    else:
        st.markdown("⚠️ Top-p は現在無効です（Top-K 有効）")
else:
    gen_count = st.slider(
        "まとめて生成するトークン数", 1, 50, 20, 1,
        help="生成したいトークン数を指定"
    )
    top_p = st.slider(
        "Top-p (Nucleus sampling)", 0.0, 1.0, 0.9, 0.01,
        help="まとめて生成時の Top-p 設定"
    )
    top_k = st.slider(
        "Top-K sampling", 1, 50, 40, 1,
        help="まとめて生成時の Top-K 設定（Top-p = 1.0 のとき有効）", disabled=(top_p < 1.0)
    )
    if top_p < 1.0:
        st.markdown("⚠️ まとめて生成では Top-K は無効です（Top-p 有効）")
    else:
        st.markdown("⚠️ まとめて生成では Top-p は無効です（Top-K 有効）")

st.markdown("---")

# プロンプト初期化ボタン
if st.button("🔄 プロンプト初期化"):
    st.session_state.input_ids = tokenizer.encode(st.session_state.prompt, return_tensors="pt")
    st.session_state.generated_tokens = []
    st.session_state.steps = []
    st.session_state.step_index = 0

if st.session_state.input_ids is None:
    st.warning("まずはプロンプトを初期化してください。")
    st.stop()

# 可視化用のプレースホルダー
chart_placeholder = st.empty()
attention_placeholder = st.empty()

if mode == "ステップバイステップ":
    # ステップ単位生成ボタン
    if st.button("▶️ トークン生成"):
        input_ids = st.session_state.input_ids
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
            logits = outputs.logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            # Top-p または Top-K でフィルタリング
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cumulative_probs > top_p
                cutoff_idx = torch.argmax(cutoff.int()).item() + 1
                nucleus_indices = sorted_indices[0, :cutoff_idx]
                filtered_probs = torch.zeros_like(probs)
                filtered_probs[0, nucleus_indices] = probs[0, nucleus_indices]
            else:
                top_probs, top_indices = torch.topk(probs, top_k)
                filtered_probs = torch.zeros_like(probs)
                filtered_probs[0, top_indices[0]] = probs[0, top_indices[0]]

            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(filtered_probs, num_samples=1)
            st.session_state.input_ids = torch.cat([input_ids, next_token], dim=1)
            st.session_state.generated_tokens.append(next_token.item())

            # Top-K（可視化用として常に Top-K を取得）
            topk_probs, topk_indices = torch.topk(probs, top_k)
            topk_tokens = [tokenizer.decode([i]).strip() for i in topk_indices[0]]
            topk_values = topk_probs[0].tolist()
            chosen_id = next_token.item()

            # Attention 行列（最終層の平均）
            attn = outputs.attentions[-1][0]  # shape: [n_head, seq_len, seq_len]
            attn_avg = attn.mean(dim=0).cpu().numpy()
            tokens_all = [tokenizer.decode([i]).strip() for i in st.session_state.input_ids[0].tolist()]

            # ステップデータを保存
            step_data = {
                "topk_tokens": topk_tokens,
                "topk_values": topk_values,
                "topk_ids": topk_indices[0].tolist(),
                "chosen_id": chosen_id,
                "attn_avg": attn_avg,
                "tokens_all": tokens_all,
            }
            st.session_state.steps.append(step_data)
            st.session_state.step_index = len(st.session_state.steps) - 1

    # ステップナビゲーション
    if st.session_state.steps:
        idx = st.session_state.step_index
        step = st.session_state.steps[idx]
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            prev_disabled = idx == 0
            if st.button("← 前へ", disabled=prev_disabled) and idx > 0:
                st.session_state.step_index = idx - 1
        with col3:
            next_disabled = idx == len(st.session_state.steps) - 1
            if st.button("次へ →", disabled=next_disabled) and idx < len(st.session_state.steps) - 1:
                st.session_state.step_index = idx + 1
        st.markdown(f"**Step {idx+1}/{len(st.session_state.steps)}**")

        # グラフタイトルを動的に切り替え
        if top_p < 1.0:
            title = f"Step {idx+1}: Next Token Candidates (Top-p)"
        else:
            title = f"Step {idx+1}: Next Token Candidates (Top-K)"

        # 分布棒グラフの再描画
        fig, ax = plt.subplots()
        colors = [
            "red" if tok_id == step["chosen_id"] else "gray"
            for tok_id in step["topk_ids"]
        ]
        ax.barh(step["topk_tokens"][::-1], step["topk_values"][::-1], color=colors[::-1])
        ax.set_title(title)
        ax.set_xlabel("Probability")
        ax.invert_yaxis()
        chart_placeholder.pyplot(fig)

        # Attention ヒートマップ再描画
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        im = ax2.imshow(step["attn_avg"], cmap="viridis", vmin=0.0, vmax=0.2)
        ax2.set_xticks(range(len(step["tokens_all"])))
        ax2.set_xticklabels(step["tokens_all"], rotation=90, fontsize=6)
        ax2.set_yticks(range(len(step["tokens_all"])))
        ax2.set_yticklabels(step["tokens_all"], fontsize=6)
        ax2.set_title(f"Step {idx+1}: Attention Map")
        fig2.colorbar(im, ax=ax2)
        attention_placeholder.pyplot(fig2)

else:
    if st.button("▶️ まとめて生成"):
        input_ids = st.session_state.input_ids
        with torch.no_grad():
            if top_p < 1.0:
                # Top-p サンプリング
                generated_ids = model.generate(
                    input_ids,
                    max_new_tokens=gen_count,
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    pad_token_id=tokenizer.eos_token_id
                )
            else:
                # Top-K サンプリング
                generated_ids = model.generate(
                    input_ids,
                    max_new_tokens=gen_count,
                    do_sample=True,
                    top_k=top_k,
                    temperature=temperature,
                    pad_token_id=tokenizer.eos_token_id
                )

        # 生成部分をリストで抽出
        new_ids = generated_ids[0][input_ids.size(1):].tolist()
        seq = input_ids
        st.session_state.steps = []

        # 各生成ステップごとにデータを保存
        for idx, tok_id in enumerate(new_ids):
            with torch.no_grad():
                outputs = model(seq, output_attentions=True)
                logits = outputs.logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)

                topk_probs, topk_indices = torch.topk(probs, top_k)
                topk_tokens = [tokenizer.decode([i]).strip() for i in topk_indices[0]]
                topk_values = topk_probs[0].tolist()
                topk_ids = topk_indices[0].tolist()
                chosen_id = tok_id

                attn = outputs.attentions[-1][0]
                attn_avg = attn.mean(dim=0).cpu().numpy()
                tokens_all = [tokenizer.decode([i]).strip() for i in seq[0].tolist()]

                step_data = {
                    "topk_tokens": topk_tokens,
                    "topk_values": topk_values,
                    "topk_ids": topk_ids,
                    "chosen_id": chosen_id,
                    "attn_avg": attn_avg,
                    "tokens_all": tokens_all,
                }
                st.session_state.steps.append(step_data)
                seq = torch.cat([seq, torch.tensor([[tok_id]])], dim=1)

        st.session_state.input_ids = seq.clone()
        st.session_state.step_index = 0

    # まとめて生成後のステップナビゲーション
    if st.session_state.steps:
        idx = st.session_state.step_index
        step = st.session_state.steps[idx]
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            prev_disabled = idx == 0
            if st.button("← 前へ", disabled=prev_disabled) and idx > 0:
                st.session_state.step_index = idx - 1
        with col3:
            next_disabled = idx == len(st.session_state.steps) - 1
            if st.button("次へ →", disabled=next_disabled) and idx < len(st.session_state.steps) - 1:
                st.session_state.step_index = idx + 1
        st.markdown(f"**Step {idx+1}/{len(st.session_state.steps)}**")

        # グラフタイトルを動的に切り替え
        if top_p < 1.0:
            title = f"Step {idx+1}: Next Token Candidates (Top-p)"
        else:
            title = f"Step {idx+1}: Next Token Candidates (Top-K)"

        # 分布棒グラフの再描画
        fig, ax = plt.subplots()
        colors = [
            "red" if tok_id == step["chosen_id"] else "gray"
            for tok_id in step["topk_ids"]
        ]
        ax.barh(step["topk_tokens"][::-1], step["topk_values"][::-1], color=colors[::-1])
        ax.set_title(title)
        ax.set_xlabel("Probability")
        ax.invert_yaxis()
        chart_placeholder.pyplot(fig)

        # Attention ヒートマップ再描画
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        im = ax2.imshow(step["attn_avg"], cmap="viridis", vmin=0.0, vmax=0.2)
        ax2.set_xticks(range(len(step["tokens_all"])))
        ax2.set_xticklabels(step["tokens_all"], rotation=90, fontsize=6)
        ax2.set_yticks(range(len(step["tokens_all"])))
        ax2.set_yticklabels(step["tokens_all"], fontsize=6)
        ax2.set_title(f"Step {idx+1}: Attention Map")
        fig2.colorbar(im, ax=ax2)
        attention_placeholder.pyplot(fig2)

# 最終出力文
st.markdown("### 🧠 最終的な出力文")
st.write(tokenizer.decode(st.session_state.input_ids[0], skip_special_tokens=True))
