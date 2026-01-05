import os
# 必须最先：离线 + GPU
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import jittor as jt
jt.flags.use_cuda = 1

import gradio as gr
import time
import torch
# ✅ Fix: Ensure torch.version is available for transformers/torch serialization
import importlib
if not hasattr(torch, "version"):
    try:
        torch.version = importlib.import_module("torch.version")
    except ImportError:
        pass

if not hasattr(torch, "_utils"):
    try:
        torch._utils = importlib.import_module("torch._utils")
    except ImportError:
        pass

import numpy as np
import json
from medusa_jittor.model.medusa_model import MedusaModel
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download

# Paths
base_model_path = "../models/vicuna-7b-v1.33"
medusa_head_path = "medusa_jittor/checkpoints_vicuna_jtclean_compare/medusa_epoch_0.pkl"

# Print paths correctly
print(f"Base Model Path: {base_model_path}")
print(f"Medusa Head Path: {medusa_head_path}")

def load_torch_base_model(model, path):
    print(f"Loading PyTorch base model from {path}...")
    try:
        shard_files = []
        # Check for index file
        index_file = os.path.join(path, "pytorch_model.bin.index.json")
        if os.path.exists(index_file):
            with open(index_file, "r") as f:
                index = json.load(f)
            shard_files = sorted(list(set(index["weight_map"].values())))
            shard_files = [os.path.join(path, f) for f in shard_files]
        else:
            # Check for single bin file
            bin_file = os.path.join(path, "pytorch_model.bin")
            if os.path.exists(bin_file):
                shard_files = [bin_file]
            else:
                # Check for safetensors
                safe_index = os.path.join(path, "model.safetensors.index.json")
                if os.path.exists(safe_index):
                     with open(safe_index, "r") as f:
                        index = json.load(f)
                     shard_files = sorted(list(set(index["weight_map"].values())))
                     shard_files = [os.path.join(path, f) for f in shard_files]
                else:
                    safe_file = os.path.join(path, "model.safetensors")
                    if os.path.exists(safe_file):
                        shard_files = [safe_file]
                    else:
                        raise ValueError("No checkpoint found")

        for shard_file in shard_files:
            print(f"Loading shard: {shard_file}")
            if shard_file.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(shard_file)
            else:
                state_dict = torch.load(shard_file, map_location="cpu")
            
            jt_state_dict = {}
            for k, v in state_dict.items():
                # Convert to float16 numpy to save memory
                if v.dtype == torch.float32 or v.dtype == torch.float16:
                    jt_state_dict[k] = jt.array(v.to(torch.float16).numpy())
                else:
                    jt_state_dict[k] = jt.array(v.numpy())
            
            model.load_state_dict(jt_state_dict, strict=False)
            del state_dict
            del jt_state_dict
            import gc
            gc.collect()
            jt.gc()

        print("Loaded base model weights.")

    except Exception as e:
        print(f"Failed to load PyTorch base model: {e}")
        import traceback
        traceback.print_exc()

print("Initializing MedusaModel...")
try:
    # Initialize model structure with the correct argument name
    config = AutoConfig.from_pretrained(base_model_path, local_files_only=True)  # Ensure it's loaded offline
    config.medusa_num_heads = 3  # Set number of heads explicitly
    config.medusa_num_layers = 1  # Set number of layers explicitly

    model = MedusaModel.from_pretrained(
        pretrained_model_name_or_path=base_model_path,  # Correct argument name
        config=config,  # Pass in config with offline setting
        medusa_num_heads=3,  # Assuming 3 heads based on previous comments
        medusa_num_layers=1,  # Number of layers for Medusa, adjust as needed
        torch_dtype=torch.float16,  # Ensuring dtype
        local_files_only=True  # Ensure it uses local files only
    )

    tokenizer = model.get_tokenizer()
    
    # Load base model weights (fallback to PyTorch if needed)
    load_torch_base_model(model, base_model_path)
    
    # Load Medusa Head
    model.load_medusa_head(medusa_head_path)
    
    # Force garbage collection
    jt.gc()

except Exception as e:
    print(f"Error initializing model: {e}")
    import traceback
    traceback.print_exc()
    model = None
    tokenizer = None

def generate_baseline(prompt, max_new_tokens=128, temperature=0.7):
    if model is None:
        return "Model not loaded.", 0.0
    
    input_ids = jt.array(tokenizer.encode(prompt)).unsqueeze(0)
    start_time = time.time()
    
    # Reset medusa mode just in case
    from medusa_jittor.model.utils import reset_medusa_mode
    reset_medusa_mode(model)
    
    # Simple autoregressive loop
    cur_len = 0
    past_key_values = None
    
    with jt.no_grad():
        while cur_len < max_new_tokens:
            if past_key_values is None:
                outputs = model(input_ids)
                logits = outputs.logits
                past_key_values = outputs.past_key_values
            else:
                outputs = model(input_ids[:, -1:], past_key_values=past_key_values)
                logits = outputs.logits
                past_key_values = outputs.past_key_values
            
            next_token_logits = logits[:, -1, :]
            if temperature > 0:
                probs = jt.nn.softmax(next_token_logits / temperature, dim=-1)
                next_token = jt.argmax(probs, dim=-1)[0].unsqueeze(1)
            else:
                next_token = jt.argmax(next_token_logits, dim=-1)[0].unsqueeze(1)
            
            input_ids = jt.concat([input_ids, next_token], dim=1)
            cur_len += 1
            
            if int(next_token.numpy()[0]) == tokenizer.eos_token_id:
                break

    end_time = time.time()
    duration = end_time - start_time
    speed = cur_len / duration

    output_text = tokenizer.decode(input_ids[0].numpy(), skip_special_tokens=True)
    return output_text, speed

def generate_medusa_ui(prompt, max_new_tokens=128, temperature=0.7):
    if model is None:
        return "Model not loaded.", 0.0
    
    input_ids = jt.array(tokenizer.encode(prompt)).unsqueeze(0)
    start_time = time.time()
    
    with jt.no_grad():
        output_ids = model.medusa_generate(
            input_ids,
            max_steps=max_new_tokens,
            temperature=temperature
        )
    
    end_time = time.time()
    duration = end_time - start_time
    new_tokens = output_ids.shape[1] - input_ids.shape[1]
    speed = new_tokens / duration
    
    output_text = tokenizer.decode(output_ids[0].numpy(), skip_special_tokens=True)
    return output_text, speed

with gr.Blocks() as demo:
    gr.Markdown("# Medusa Jittor Acceleration Demo")
    gr.Markdown(f"Base Model: {base_model_path}")
    gr.Markdown(f"Medusa Head: {medusa_head_path}")
    
    with gr.Row():
        prompt_input = gr.Textbox(label="Prompt", value="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: Hello, who are you?\nASSISTANT:")
        
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Baseline (Vicuna)")
            baseline_output = gr.Textbox(label="Output")
            baseline_speed = gr.Number(label="Speed (tokens/sec)")
            baseline_btn = gr.Button("Generate Baseline")
            
        with gr.Column():
            gr.Markdown("## Medusa")
            medusa_output = gr.Textbox(label="Output")
            medusa_speed = gr.Number(label="Speed (tokens/sec)")
            medusa_btn = gr.Button("Generate Medusa")
            
    baseline_btn.click(generate_baseline, inputs=[prompt_input], outputs=[baseline_output, baseline_speed])
    medusa_btn.click(generate_medusa_ui, inputs=[prompt_input], outputs=[medusa_output, medusa_speed])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
