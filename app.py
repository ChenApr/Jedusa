import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import time
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from medusa_jittor.model.medusa_model import MedusaModel

# ================= 配置区 =================
baseline_model_path = "../models/vicuna-7b-v1.33"
medusa_model_path = "../models/vicuna-7b-v1.33"
medusa_weights_path = "medusa_jittor/checkpoints_vicuna_jtclean_compare/medusa_epoch_0.pkl"

BASE_DEVICE = "cuda:0"
MEDUSA_DEVICE = "cuda:1"

# ================= 工具函数 =================
def _to_torch_on_device(x, device):
    if hasattr(x, "numpy"): x = x.numpy()
    if isinstance(x, np.ndarray): x = torch.from_numpy(x)
    return x.to(device) if torch.is_tensor(x) else torch.tensor(x, device=device)

def load_medusa_head_weights_strict(model, weights_path: str):
    import jittor as jt
    weights = jt.load(weights_path)
    state_dict = {k: _to_torch_on_device(v, model.device) for k, v in weights.items()}
    
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("heads."): k = k[len("heads."):]
        if k.startswith("medusa_head."): k = k[len("medusa_head."):]
        cleaned[k] = v
    
    model.medusa_head.load_state_dict(cleaned, strict=True)
    print(f"✅ Medusa weights loaded on {model.device}")

# ================= 模型加载 =================
tokenizer = AutoTokenizer.from_pretrained(baseline_model_path, use_fast=False)

print(f"Loading Baseline on {BASE_DEVICE}...")
baseline_model = AutoModelForCausalLM.from_pretrained(
    baseline_model_path, torch_dtype=torch.float16, device_map={"": BASE_DEVICE}
)

print(f"Loading Medusa on {MEDUSA_DEVICE}...")
medusa_model = MedusaModel.from_pretrained(
    medusa_model_path, medusa_num_heads=3, torch_dtype=torch.float16, 
    device_map={"": MEDUSA_DEVICE}, low_cpu_mem_usage=True
)
load_medusa_head_weights_strict(medusa_model, medusa_weights_path)
medusa_model.eval()

# ================= 交互逻辑 =================

def run_comparison(prompt, max_tokens, temp, top_p):
    # 1. Baseline 生成 (同步)
    inputs_b = tokenizer(prompt, return_tensors="pt").to(BASE_DEVICE)
    b_start = time()
    with torch.no_grad():
        output_ids = baseline_model.generate(
            **inputs_b, do_sample=True, temperature=temp, top_p=top_p, max_new_tokens=max_tokens
        )
    b_duration = time() - b_start
    gen_ids = output_ids[0][inputs_b["input_ids"].shape[1]:]
    b_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    b_tps = len(gen_ids) / b_duration if b_duration > 0 else 0
    
    res_baseline = f"{b_text}\n\n" + "-"*30 + f"\n⏱️ Time: {b_duration:.2f}s\n🚀 Speed: {b_tps:.2f} tokens/s"

    # 2. Medusa 生成 (流式 yield)
    inputs_m = tokenizer(prompt, return_tensors="pt").to(MEDUSA_DEVICE)
    m_start = time()
    
    # 这里的 gen_results 是一个 generator
    gen_results = medusa_model.medusa_generate(
        inputs_m["input_ids"], temperature=temp, top_p=top_p, max_steps=max_tokens
    )

    last_text = ""
    for item in gen_results:
        # 修复点：直接覆盖而不是累加，解决“阶梯状重复”
        if isinstance(item, str):
            current_text = item
        elif isinstance(item, dict):
            current_text = item.get("text", "")
        else:
            current_text = str(item)
        
        last_text = current_text
        
        # 计算中间状态的 TPS
        m_curr_dur = time() - m_start
        curr_tokens = len(tokenizer.encode(current_text, add_special_tokens=False))
        curr_tps = curr_tokens / m_curr_dur if m_curr_dur > 0 else 0
        
        res_medusa = f"{current_text}\n\n" + "-"*30 + f"\n⏱️ 正在生成...\n🚀 实时速度: {curr_tps:.2f} tokens/s"
        
        # 实时推送到前端
        yield res_baseline, res_medusa

    # 3. 最终统计
    m_duration = time() - m_start
    m_tokens = len(tokenizer.encode(last_text, add_special_tokens=False))
    m_tps = m_tokens / m_duration if m_duration > 0 else 0
    speedup = m_tps / b_tps if b_tps > 0 else 0
    
    res_medusa_final = f"{last_text}\n\n" + "-"*30 + f"\n⏱️ 总耗时: {m_duration:.2f}s\n🚀 最终速度: {m_tps:.2f} tokens/s\n📈 加速比: {speedup:.2f}x"
    
    yield res_baseline, res_medusa_final

# ================= Gradio UI =================

with gr.Blocks(title="Medusa vs Baseline") as iface:
    gr.Markdown("# 🚀 Medusa 加速对比测试")
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="输入 Prompt", lines=3, value="Write a Python script to merge two sorted lists.")
            with gr.Row():
                max_tokens = gr.Slider(64, 512, value=256, step=64, label="最大 Token 数")
                temp = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="Temperature")
            btn = gr.Button("开始对比", variant="primary")

    with gr.Row():
        out_b = gr.Textbox(label="标准 Vicuna-7B (Baseline)", lines=12)
        out_m = gr.Textbox(label="Medusa 加速版", lines=12)

    btn.click(run_comparison, inputs=[prompt_input, max_tokens, temp, gr.State(0.95)], outputs=[out_b, out_m])

if __name__ == "__main__":
    # 启动命令: CUDA_VISIBLE_DEVICES=0,1 python app.py
    iface.launch(server_name="0.0.0.0", share=True)