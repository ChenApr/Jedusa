import argparse
import os
import sys
import json
import torch
import numpy as np

from fastchat.serve.cli import SimpleChatIO, RichChatIO, ProgrammaticChatIO
from fastchat.model.model_adapter import get_conversation_template
from fastchat.conversation import get_conv_template

# Add project root to sys.path to allow importing medusa_jittor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from medusa_jittor.model.medusa_model import MedusaModel


def _to_torch_on_device(x, device):
    """Convert jittor/numpy/torch to torch.Tensor on target device."""
    if hasattr(x, "numpy"):  # jittor Var
        x = x.numpy()
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if torch.is_tensor(x):
        return x.to(device)
    # Fallback: if somehow it's a python number/list
    return torch.tensor(x, device=device)


def load_medusa_head_weights(model, weights_path):
    """
    Load medusa head weights into model.medusa_head with strict key matching.

    Supports:
      - .npz (numpy)
      - .pkl (jittor checkpoint)
      - .pt / .bin (torch checkpoint)

    Key normalization:
      - 'heads.xxx' -> 'xxx'
      - 'medusa_head.xxx' -> 'xxx'
    """
    print(f"Loading Medusa heads from: {weights_path}")

    state_dict = None

    if weights_path.endswith(".npz"):
        weights = np.load(weights_path)
        state_dict = {k: torch.from_numpy(v) for k, v in weights.items()}

    elif weights_path.endswith((".pkl", ".pt", ".bin")):
        if weights_path.endswith(".pkl"):
            import jittor as jt
            print(f"Attempting to load {weights_path} with Jittor...")
            weights = jt.load(weights_path)
            state_dict = {k: _to_torch_on_device(v, model.device).detach().cpu() for k, v in weights.items()}
            print("Weights loaded with Jittor.")
        else:
            # torch load
            state_dict = torch.load(weights_path, map_location="cpu")

    else:
        raise ValueError(f"Unsupported weight format: {weights_path}")

    if state_dict is None:
        raise RuntimeError("Error: Could not load weights (state_dict is None).")

    # ---- Key cleanup for loading into the submodule model.medusa_head ----
    cleaned = {}
    for k, v in state_dict.items():
        # normalize key prefix
        if k.startswith("heads."):
            k = k[len("heads."):]
        if k.startswith("medusa_head."):
            k = k[len("medusa_head."):]
        cleaned[k] = v

    # Move tensors to model.device
    cleaned = {k: _to_torch_on_device(v, model.device) for k, v in cleaned.items()}

    # ---- Strict load: mismatch => error, so you KNOW it's the trained head ----
    model.medusa_head.load_state_dict(cleaned, strict=True)
    print("✅ Medusa head loaded successfully (strict=True).")


def main(args):
    if args.style == "simple":
        chatio = SimpleChatIO(args.multiline)
    elif args.style == "rich":
        chatio = RichChatIO(args.multiline, args.mouse)
    elif args.style == "programmatic":
        chatio = ProgrammaticChatIO()
    else:
        raise ValueError(f"Invalid style for console: {args.style}")

    try:
        print(f"Loading base model: {args.base_model}")

        # Handle device mapping
        if args.device.startswith("cuda:"):
            device_id = int(args.device.split(":")[-1])
            device_map = {"": device_id}
        else:
            device_map = "auto"

        model = MedusaModel.from_pretrained(
            args.base_model,
            medusa_num_heads=args.medusa_num_heads,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device_map,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
        )

        # Ensure model is on the correct device if not using auto
        if args.device != "auto":
            model = model.to(torch.device(args.device))

        model.eval()
        tokenizer = model.get_tokenizer()

        # Load Medusa weights (STRICT)
        load_medusa_head_weights(model, args.medusa_weights)

        conv = None

        def new_chat():
            if args.conv_template:
                c = get_conv_template(args.conv_template)
            else:
                c = get_conversation_template(args.base_model)
            if args.conv_system_msg:
                c.set_system_message(args.conv_system_msg)
            return c

        def reload_conv(c):
            """Reprints the conversation from the start."""
            for message in c.messages[c.offset:]:
                chatio.prompt_for_output(message[0])
                chatio.print_output(message[1])

        while True:
            if not conv:
                conv = new_chat()

            try:
                inp = chatio.prompt_for_input(conv.roles[0])
            except EOFError:
                inp = ""

            if inp == "!!exit" or not inp:
                print("exit...")
                break
            elif inp == "!!reset":
                print("resetting...")
                conv = new_chat()
                continue
            elif inp == "!!remove":
                print("removing last message...")
                if len(conv.messages) > conv.offset:
                    # Assistant
                    if conv.messages[-1][0] == conv.roles[1]:
                        conv.messages.pop()
                    # User
                    if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0]:
                        conv.messages.pop()
                    reload_conv(conv)
                else:
                    print("No messages to remove.")
                continue
            elif inp == "!!regen":
                print("regenerating last message...")
                if len(conv.messages) > conv.offset:
                    # Assistant
                    if conv.messages[-1][0] == conv.roles[1]:
                        conv.messages.pop()
                    # User
                    if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0]:
                        reload_conv(conv)
                        inp = conv.messages.pop()[1]
                    else:
                        print("No user message to regenerate from.")
                        continue
                else:
                    print("No messages to regenerate.")
                    continue
            elif inp.startswith("!!save"):
                args_split = inp.split(" ", 1)
                if len(args_split) != 2:
                    print("usage: !!save <filename>")
                    continue
                filename = args_split[1]
                if "." not in filename:
                    filename += ".json"
                print("saving...", filename)
                with open(filename, "w") as outfile:
                    json.dump(conv.dict(), outfile)
                continue
            elif inp.startswith("!!load"):
                args_split = inp.split(" ", 1)
                if len(args_split) != 2:
                    print("usage: !!load <filename>")
                    continue
                filename = args_split[1]

                if not os.path.exists(filename):
                    if (not filename.endswith(".json")) and os.path.exists(filename + ".json"):
                        filename += ".json"
                    else:
                        print("file not found:", filename)
                        continue

                print("loading...", filename)
                with open(filename, "r") as infile:
                    new_conv = json.load(infile)

                conv = get_conv_template(new_conv["template_name"])
                conv.set_system_message(new_conv["system_message"])
                conv.messages = new_conv["messages"]
                reload_conv(conv)
                continue

            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            try:
                chatio.prompt_for_output(conv.roles[1])
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
                    model.base_model.device
                )

                with torch.inference_mode():
                    outputs = chatio.stream_output(
                        model.medusa_generate(
                            input_ids,
                            temperature=args.temperature,
                            max_steps=args.max_steps,
                        )
                    )
                conv.update_last_message(outputs.strip())

            except KeyboardInterrupt:
                print("stopped generation.")
                if conv.messages[-1][1] is None:
                    conv.messages.pop()
                    if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0]:
                        conv.messages.pop()
                    reload_conv(conv)

    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model (Vicuna)")
    parser.add_argument("--medusa_weights", type=str, required=True, help="Path to trained weights (.pkl/.npz/.pt/.bin)")
    parser.add_argument("--medusa_num_heads", type=int, default=3, help="Number of Medusa heads")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to load the model on (e.g. cuda:0)")

    parser.add_argument("--load-in-8bit", dest="load_in_8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", help="Use 4-bit quantization")

    parser.add_argument("--conv-template", type=str, default=None, help="Conversation prompt template.")
    parser.add_argument("--conv-system-msg", type=str, default=None, help="Conversation system message.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--no-history", action="store_true")

    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        choices=["simple", "rich", "programmatic"],
        help="Display style.",
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="[Rich Style]: Enable mouse support for cursor positioning.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print useful debug information (e.g., prompts)",
    )

    args = parser.parse_args()
    main(args)
