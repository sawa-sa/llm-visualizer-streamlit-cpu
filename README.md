# LLM Visualizer Streamlit

**LLM Visualizer Streamlit** は、大規模言語モデル（LLM）の生成挙動に影響を与える各種パラメータ（Temperature / Top-p / Top-k）を調整し、その出力結果や注意機構（Attention）を視覚的に観察できるツールです。  
CPU環境でのローカル実行および無料のクラウド実行を前提としており、モデルは `gpt2-medium` に限定されています。

📍 **公開デモ（Streamlit Cloud 無料プラン）**  
👉 [https://llm-visualizer-app-cpu-fzjyvfhna7sszlugnhuu7u.streamlit.app/](https://llm-visualizer-app-cpu-fzjyvfhna7sszlugnhuu7u.streamlit.app/)

---

## 特徴

### 🔹 `gpt2-medium` モデル限定
Hugging Face の `gpt2-medium` を使用。モデルの切り替え機能は搭載していません。

### 🔹 出力パラメータのインタラクティブ調整
以下の生成パラメータをスライダーで調整できます：

- `Temperature`
- `Top-p`
- `Top-k`

※ Top-p が 1 の場合、Top-k スライダーは自動で無効化されます（UI制御済み）

### 🔹 Attention のヒートマップ可視化
Transformerの自己注意（Self-Attention）をヒートマップで可視化し、各トークン間の依存関係を視覚的に確認できます。

---

## 対応環境

- Python 3.10 推奨
- **CPU 環境のみ対応（GPU 非対応）**
- Hugging Face Transformers / Streamlit / Matplotlib などを使用

▶️ GPU対応版はこちら：  
[https://github.com/sawa-sa/llm-visualizer-streamlit](https://github.com/sawa-sa/llm-visualizer-streamlit)

---

## デプロイ環境について（Streamlit Community Cloud）

このアプリは **Streamlit Cloud（無料アカウント）** を使ってホスティングされています。

### ⚠️ 無料プランでの注意点

- **初回アクセス時に起動ラグ（最大1分ほど）**が発生する場合があります（スリープからの復帰のため）。
- **CPUリソース・メモリが限定的**なため、処理が遅く感じることがあります。
- **1時間アクセスがないと自動的にスリープ**状態になります。

### 💡 快適に使うコツ

- アプリ起動まで数十秒待ってから操作を始めてください。
- ページが固まった場合は**リロード**してください。
- 長文プロンプトは避け、まずは短めの入力で試してください。

---

## 想定ユースケース

- LLM の生成挙動を**学習・研究目的で観察**したい
- Temperature / Top-p / Top-k の効果を**視覚的に比較**したい
- Attention 可視化を通じて**モデル内部の挙動を理解**したい
- **授業やワークショップでのデモンストレーション**として活用したい

---

## 今後の展望

- `repetition_penalty` の調整スライダー追加
- モデル選択機能（ELYZA、GPT-Neo等）への対応

---

> ※本READMEの文章は OpenAI の ChatGPT-4o によって生成され、内容は開発者によって確認・修正された上で掲載されています。
