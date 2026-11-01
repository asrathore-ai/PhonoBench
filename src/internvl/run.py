"""InternVL3 inference with one synchronized, 28-frame composite video."""

import importlib.metadata
import time

from src.video import composite_frames, sample_video


def load_model(checkpoint="OpenGVLab/InternVL3-8B-hf", revision=None,
               dtype="bfloat16", device="auto", attention="sdpa", max_new_tokens=32):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if dtype not in ["bfloat16", "float16", "float32"]:
        raise ValueError("dtype must be bfloat16, float16, or float32.")
    if int(max_new_tokens) < 1:
        raise ValueError("max_new_tokens must be positive.")
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, revision=revision, dtype=getattr(torch, dtype),
        device_map=device, attn_implementation=attention,
        trust_remote_code=False,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        checkpoint, revision=revision, trust_remote_code=False,
    )
    environment = {
        "checkpoint": checkpoint, "requested_revision": revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "dtype": dtype, "device": device, "attention": attention,
        "max_new_tokens": int(max_new_tokens), "do_sample": False,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "interface": "synchronized_composite", "frames_per_source": 28,
        "video_processor_size": dict(processor.video_processor.size),
    }
    for package in ["torchvision", "numpy", "Pillow", "opencv-python-headless", "accelerate"]:
        try:
            environment[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment[package] = None
    return {"model": model, "processor": processor, "environment": environment,
            "max_new_tokens": int(max_new_tokens)}


def build_messages(instance):
    content = []
    specs = instance.get("videos", [])
    if specs:
        mapping = "The video shows synchronized panels arranged left to right.\n"
        for index in range(len(specs)):
            role = specs[index].get("role", "Source")
            mapping += f"Panel {index + 1} is Video {index + 1} ({role}).\n"
        content.append({"type": "text", "text": mapping})
        # Only the token placeholder is needed: frames are passed below.
        content.append({"type": "video"})
    content.append({"type": "text", "text": instance["prompt"]})
    return [{"role": "user", "content": content}]


def prepare_inputs(bundle, instance, video_root=None):
    import numpy as np
    import torch

    sampled = []
    for spec in instance.get("videos", []):
        sampled.append(sample_video(spec, video_root, num_frames=28))
    processor = bundle["processor"]
    text = processor.apply_chat_template(
        build_messages(instance), tokenize=False, add_generation_prompt=True,
    )
    kwargs = {"text": [text], "padding": True, "return_tensors": "pt"}
    if sampled:
        frames = composite_frames(sampled)
        arrays = []
        for frame in frames:
            arrays.append(np.asarray(frame))
        kwargs["videos"] = [np.stack(arrays)]
        # Keep the synchronized frames selected above.
        kwargs["videos_kwargs"] = {"do_sample_frames": False}
    inputs = processor(**kwargs)
    model = bundle["model"]
    for key in inputs:
        if torch.is_tensor(inputs[key]):
            if inputs[key].is_floating_point():
                inputs[key] = inputs[key].to(device=model.device, dtype=model.dtype)
            else:
                inputs[key] = inputs[key].to(model.device)
    audits = []
    for video in sampled:
        audits.append(video["audit"])
    return inputs, audits


def generate(bundle, instance, video_root=None):
    import torch

    started = time.perf_counter()
    inputs, audits = prepare_inputs(bundle, instance, video_root)
    input_tokens = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode():
        output = bundle["model"].generate(
            **inputs, max_new_tokens=bundle["max_new_tokens"],
            do_sample=False, use_cache=True,
        )
    suffix = output[:, input_tokens:]
    text = bundle["processor"].batch_decode(
        suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    return {"raw_output": text, "input_tokens": input_tokens,
            "output_tokens": int(suffix.shape[-1]), "sampled_videos": audits,
            "elapsed_seconds": time.perf_counter() - started}
