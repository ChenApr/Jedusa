import argparse
import os
import sys
import torch
import numpy as np
import json
from fastchat.serve.cli import SimpleChatIO, RichChatIO, ProgrammaticChatIO
from fastchat.model.model_adapter import get_conversation_template
from fastchat.conversation import get_conv_template

# Add project root to sys.path to allow importing medusa_jittor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from medusa_jittor.model.medusa_model import MedusaModel

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

        # Load Medusa weights
        print(f"Loading Medusa heads from: {args.medusa_weights}")
        state_dict = None
        if args.medusa_weights.endswith(".npz"):
            weights = np.load(args.medusa_weights)
            state_dict = {k: torch.from_numpy(v).to(model.device) for k, v in weights.items()}
        elif args.medusa_weights.endswith(".pkl") or args.medusa_weights.endswith(".pt") or args.medusa_weights.endswith(".bin"):
            try:
                if args.medusa_weights.endswith(".pkl"):
                    import jittor as jt
                    print(f"Attempting to load {args.medusa_weights} with Jittor...")
                    weights = jt.load(args.medusa_weights)
                    state_dict = {}
                    for k, v in weights.items():
                        if k.startswith("heads."):
                            new_k = k.replace("heads.", "medusa_head.")
                        else:
                            new_k = k
                        
                        if hasattr(v, "numpy"):
                            state_dict[new_k] = torch.from_numpy(v.numpy()).to(model.device)
                        elif isinstance(v, np.ndarray):
                            state_dict[new_k] = torch.from_numpy(v).to(model.device)
                        else:
                            state_dict[new_k] = v
                    print(f"Weights loaded with Jittor.")
            except Exception as e:
                print(f"Jittor load failed: {e}. Falling back to torch.load...")
                state_dict = torch.load(args.medusa_weights, map_location=model.device)

        if state_dict is not None:
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("heads."):
                    new_k = k.replace("heads.", "medusa_head.")
                else:
                    new_k = k
                new_state_dict[new_k] = v
            
            missing, unexpected = model.medusa_head.load_state_dict(new_state_dict, strict=False)
            print(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        else:
            print("Error: Could not load weights.")
            return

        conv = None

        def new_chat():
            if args.conv_template:
                conv = get_conv_template(args.conv_template)
            else:
                conv = get_conversation_template(args.base_model)
            if args.conv_system_msg:
                conv.set_system_message(args.conv_system_msg)
            return conv

        def reload_conv(conv):
            """
            Reprints the conversation from the start.
            """
            for message in conv.messages[conv.offset :]:
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
                    if conv.messages[-1][0] == conv.roles[0]:
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
                    if conv.messages[-1][0] == conv.roles[0]:
                        reload_conv(conv)
                        # Set inp to previous message
                        inp = conv.messages.pop()[1]
                    else:
                        # Shouldn't happen in normal circumstances
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
                else:
                    filename = args_split[1]

                # Add .json if extension not present
                if not "." in filename:
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
                else:
                    filename = args_split[1]

                # Check if file exists and add .json if needed
                if not os.path.exists(filename):
                    if (not filename.endswith(".json")) and os.path.exists(
                        filename + ".json"
                    ):
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
                
                # Medusa Generation
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
                # If generation didn't finish
                if conv.messages[-1][1] is None:
                    conv.messages.pop()
                    # Remove last user message, so there isn't a double up
                    if conv.messages[-1][0] == conv.roles[0]:
                        conv.messages.pop()

                    reload_conv(conv)

    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model (Vicuna)")
    parser.add_argument("--medusa_weights", type=str, required=True, help="Path to trained .npz weights")
    parser.add_argument("--medusa_num_heads", type=int, default=3, help="Number of Medusa heads")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to load the model on (e.g. cuda:0)")
    
    parser.add_argument(
        "--load-in-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true", help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--conv-system-msg", type=str, default=None, help="Conversation system message."
    )
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
    main(args)
