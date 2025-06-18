import torch
import streamlit as st

def show_device_info():
    st.sidebar.write("CPUモードで実行中")
    return "cpu"