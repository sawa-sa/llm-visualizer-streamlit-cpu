import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st

@st.cache_resource
def load_model(model_name: str = "gpt2-medium"):
    """
    GPT-2 モデルとトークナイザーをロードしてキャッシュする
    """
    model = GPT2LMHeadModel.from_pretrained(model_name, output_attentions=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model.eval()
    return model, tokenizer