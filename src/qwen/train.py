"""Train one feature-specific Qwen LoRA adapter with a plain PyTorch loop."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import random

from src.files import file_hash, write_json
from src.qwen.features import field_name, feature_instance, feature_prompt, load_feature_data
from src.qwen.run import build_messages
from src.video import sample_video


def target_modules(model, mode):
    """Select language attention, vision, merger, or combined projections."""
    import torch

    names = []
    attention = ["q_proj", "k_proj", "v_proj", "o_proj"]
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        visual = ".visual." in "." + name + "."
        merger = visual and "merger" in name
        language_attention = "language_model" in name and name.split(".")[-1] in attention
        selected = mode == "attention" and language_attention
        selected = selected or (mode == "vision" and visual and not merger)
        selected = selected or (mode == "projector" and merger)
        selected = selected or (mode == "vision_language" and (visual or language_attention))
        if selected:
            names.append(name)
    if not names:
        raise ValueError("No adaptable linear layers found for " + mode)
    return sorted(names)


def load_training_model(checkpoint, revision, device, dtype, attention, quantized,
                        targets="attention", rank=8, alpha=16, dropout=0.0):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if quantized and not device.startswith("cuda"):
        raise ValueError("This 4-bit training command uses CUDA. Use --no-quantization for another device.")
    kwargs = {"revision": revision, "device_map": {"": device},
              "dtype": getattr(torch, dtype), "attn_implementation": attention,
              "trust_remote_code": False}
    if quantized:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, dtype), bnb_4bit_use_double_quant=False,
        )
    model = Qwen3VLForConditionalGeneration.from_pretrained(checkpoint, **kwargs)
    processor = AutoProcessor.from_pretrained(checkpoint, revision=revision, trust_remote_code=False)
    names = target_modules(model, targets)
    if quantized:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    adapter = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                         bias="none", task_type="CAUSAL_LM", target_modules=names)
    model = get_peft_model(model, adapter)
    model.train()
    environment = {"checkpoint": checkpoint, "requested_revision": revision,
                   "resolved_revision": getattr(model.config, "_commit_hash", None),
                   "device": device, "dtype": dtype, "attention": attention,
                   "quantization": "nf4" if quantized else None,
                   "target_modules": names, "torch": torch.__version__}
    for package in ["transformers", "peft", "qwen-vl-utils", "bitsandbytes"]:
        try:
            environment[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment[package] = None
    return {"model": model, "processor": processor, "environment": environment}


def answer_labels(prefix_ids, full_ids, attention_mask=None):
    """Mask the complete user/video prompt while keeping answer and turn-end tokens."""
    import torch

    length = prefix_ids.shape[1]
    if full_ids.shape[1] <= length or not torch.equal(full_ids[:, :length], prefix_ids):
        raise ValueError("The chat template does not preserve the prompt prefix before the answer")
    labels = full_ids.clone()
    labels[:, :length] = -100
    if attention_mask is not None:
        labels[attention_mask == 0] = -100
    if not torch.any(labels != -100):
        raise ValueError("No supervised answer tokens remain")
    return labels


def training_inputs(bundle, instance, answer, video_root=None):
    """Decode once and prepare matching prompt-only and answered conversations."""
    import torch
    from qwen_vl_utils import process_vision_info

    sampled = []
    for spec in instance["videos"]:
        sampled.append(sample_video(spec, video_root))
    messages = build_messages(instance, sampled)
    processor = bundle["processor"]
    prefix_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, video_items, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True,
    )
    videos = []
    metadata = []
    for video, information in video_items:
        videos.append(video)
        metadata.append(information)
    prefix = processor(text=[prefix_text], images=images, videos=videos, video_metadata=metadata,
                       padding=True, return_tensors="pt", do_resize=False, **video_kwargs)
    prefix_ids = prefix["input_ids"].clone()
    del prefix
    full_messages = list(messages)
    full_messages.append({"role": "assistant", "content": json.dumps({"answer": answer}, ensure_ascii=False)})
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[full_text], images=images, videos=videos, video_metadata=metadata,
                       padding=True, return_tensors="pt", do_resize=False, **video_kwargs)
    inputs["labels"] = answer_labels(prefix_ids, inputs["input_ids"], inputs.get("attention_mask"))
    for key in inputs:
        if torch.is_tensor(inputs[key]):
            inputs[key] = inputs[key].to(bundle["model"].device)
    return inputs


def train(bundle, examples, prompt, output, epochs=1, learning_rate=0.0002,
          accumulation=8, weight_decay=0.01, frames=32, seed=42, max_steps=None):
    """One video per microbatch; the final short group has the correct weight."""
    import torch

    model = bundle["model"]
    trainable = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable.append(parameter)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    randomizer = random.Random(seed)
    examples = list(examples)
    updates = 0
    output = Path(output)
    with (output / "losses.jsonl").open("w", encoding="utf-8") as handle:
        for epoch in range(epochs):
            randomizer.shuffle(examples)
            for start in range(0, len(examples), accumulation):
                group = examples[start:start + accumulation]
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                for example in group:
                    instance = feature_instance(example, prompt, frames)
                    inputs = training_inputs(bundle, instance, example["value"])
                    device_type = model.device.type
                    mixed = bundle["environment"]["dtype"] == "bfloat16"
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=mixed):
                        loss = model(**inputs).loss
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite training loss for " + example["video_id"])
                    total_loss += float(loss.detach())
                    (loss / len(group)).backward()
                    del inputs, loss
                optimizer.step()
                updates += 1
                record = {"epoch": epoch + 1, "update": updates, "examples": len(group),
                          "mean_loss": total_loss / len(group)}
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                print(f"epoch={epoch + 1} update={updates} loss={record['mean_loss']:.5f}", flush=True)
                if max_steps is not None and updates >= max_steps:
                    break
            if max_steps is not None and updates >= max_steps:
                break
    model.save_pretrained(output)
    bundle["processor"].save_pretrained(output)
    return updates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", required=True)
    parser.add_argument("--metadata", default="resources/semlex_metadata.csv")
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True, help="New or empty adapter directory")
    parser.add_argument("--checkpoint", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--targets", choices=["attention", "vision", "projector", "vision_language"], default="attention")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--attention", choices=["sdpa", "flash_attention_2", "eager"], default="sdpa")
    parser.add_argument("--no-quantization", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Limit training examples for a small run")
    parser.add_argument("--max-steps", type=int, help="Limit optimizer updates")
    args = parser.parse_args()
    for name in ["epochs", "accumulation", "rank", "alpha", "frames"]:
        if getattr(args, name) < 1:
            parser.error("--" + name + " must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1:
        parser.error("Invalid learning rate, weight decay, or dropout")
    if args.limit is not None and args.limit < 1 or args.max_steps is not None and args.max_steps < 1:
        parser.error("--limit and --max-steps must be positive")
    field = field_name(args.field)
    splits, labels, counts = load_feature_data(args.metadata, args.video_root, field)
    examples = splits["train"]
    if args.limit is not None:
        examples = examples[:args.limit]
    if not examples:
        raise ValueError("No labelled training videos are available")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty output directory")
    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    bundle = load_training_model(args.checkpoint, args.revision, args.device, args.dtype,
                                 args.attention, not args.no_quantization, args.targets,
                                 args.rank, args.alpha, args.dropout)
    prompt = feature_prompt(field, labels)
    configuration = vars(args).copy()
    configuration.update({"field": field, "allowed_labels": labels, "selection": counts,
                          "training_examples": len(examples), "microbatch_size": 1,
                          "prompt": prompt,
                          "metadata_sha256": file_hash(args.metadata),
                          "environment": bundle["environment"]})
    write_json(output / "feature_config.json", configuration)
    updates = train(bundle, examples, prompt, output, args.epochs,
                    args.learning_rate, args.accumulation, args.weight_decay, args.frames,
                    args.seed, args.max_steps)
    print(f"Saved adapter to {output}; optimizer updates: {updates}")


if __name__ == "__main__":
    main()
