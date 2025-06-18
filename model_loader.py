import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st

@st.cache_resource
def load_model(model_name: str = "gpt2-medium", device: str = "cpu"):
    device = torch.device(device)
    model = GPT2LMHeadModel.from_pretrained(model_name, output_attentions=True)
    model.to(device)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model, tokenizer, device
