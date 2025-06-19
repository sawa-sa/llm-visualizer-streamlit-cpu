import torch
import torch.nn.functional as F
from typing import Dict
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_step(
    input_ids: torch.Tensor,
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    temperature: float,
    top_p: float,
    top_k: int,
    device: torch.device,
) -> Dict:
    """
    モデルの次トークンをステップごとに生成し、可視化データを構築する
    Returns a dict with keys: input_ids, step_data
    """
    with torch.no_grad():
        input_ids = input_ids.to(device)
        outputs = model(input_ids)
        raw_logits = outputs.logits[:, -1, :]

        # Greedy (temp=0)
        if temperature < 1e-5:
            next_token = torch.argmax(raw_logits, dim=-1, keepdim=True)
            probs = None
        else:
            scaled_logits = raw_logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            # Top-p or Top-k フィルタリング
            if top_p < 1.0:
                sp, si = torch.sort(probs, descending=True)
                cum = torch.cumsum(sp, dim=-1)
                cutoff = int((cum > top_p).int().argmax()) + 1
                keep = si[0, :cutoff]
                mask = torch.zeros_like(probs)
                mask[0, keep] = probs[0, keep]
                filt = mask / mask.sum()
                p_cutoff_index = cutoff
            else:
                tv, ti = torch.topk(probs, top_k)
                mask = torch.zeros_like(probs)
                mask[0, ti[0]] = probs[0, ti[0]]
                filt = mask / mask.sum()
                p_cutoff_index = top_k
            next_token = torch.multinomial(filt, num_samples=1)

        chosen_id = next_token.item()
        new_input_ids = torch.cat([input_ids, next_token], dim=1)

        # 可視化用データ作成
        vals, inds = torch.topk(probs if probs is not None else raw_logits, top_k)
        sorted_vals, sort_idx = vals[0].sort(descending=True)
        sorted_inds = inds[0][sort_idx]

        def safe_token_label(i):
            decoded = tokenizer.decode([i])
            return decoded if decoded.strip() != "" else f"[id {i}]"

        toks = [safe_token_label(i) for i in sorted_inds]
        vals_list = sorted_vals.tolist()
        inds = sorted_inds.unsqueeze(0)  # 重要: 他の処理で参照されるため





        attn = outputs.attentions[-1][0].cpu().numpy()  # shape: (num_heads, seq_len, seq_len)

        all_toks = [tokenizer.decode([i]).strip() for i in new_input_ids[0].tolist()]

        return {
            "input_ids": new_input_ids,
            "step_data": {
                "tokens": toks,
                "values": vals_list,
                "ids": inds[0].tolist(),
                "chosen": chosen_id,
                "attn": attn,
                "all_toks": all_toks,
                "p_cutoff_index": p_cutoff_index,
                "raw_logits": raw_logits.squeeze().tolist()  # 全トークンに対するスコア
            },
        }
