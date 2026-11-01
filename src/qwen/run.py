"""A small wrapper around the official Qwen3-VL Transformers interface."""

import importlib.metadata
import time

from src.video import sample_video


def load_model(checkpoint="Qwen/Qwen3-VL-8B-Instruct", revision=None,
               dtype="bfloat16", device="auto", attention="sdpa", max_new_tokens=32,
               adapter=None, load_4bit=False):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if dtype not in ["bfloat16", "float16", "float32"]:
        raise ValueError("dtype must be bfloat16, float16, or float32.")
    if int(max_new_tokens) < 1:
        raise ValueError("max_new_tokens must be positive.")
    options = {}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, dtype),
        )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint, revision=revision, dtype=getattr(torch, dtype),
        device_map=device, attn_implementation=attention,
        trust_remote_code=False, **options,
    )
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        checkpoint, revision=revision, trust_remote_code=False,
    )
    environment = {
        "checkpoint": checkpoint, "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "adapter": adapter, "load_in_4bit": load_4bit,
        "dtype": dtype, "device": device, "attention": attention,
        "max_new_tokens": int(max_new_tokens), "do_sample": False,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "qwen_vl_utils": importlib.metadata.version("qwen-vl-utils"),
        "interface": "native_multi_video",
    }
    for package in ["torchvision", "numpy", "Pillow", "opencv-python-headless", "accelerate", "peft", "bitsandbytes"]:
        try:
            environment[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment[package] = None
    return {"model": model, "processor": processor, "environment": environment,
            "max_new_tokens": int(max_new_tokens)}


def build_messages(instance, sampled):
    content = []
    specs = instance.get("videos", [])
    if len(specs) != len(sampled):
        raise ValueError("Every video role must have one sampled source.")
    for index in range(len(specs)):
        spec = specs[index]
        video = sampled[index]
        role = spec.get("role", "Source")
        content.append({"type": "text", "text": f"Video {index + 1} ({role}):"})
        item = {"type": "video", "video": video["frames"],
                "sample_fps": video["audit"]["sample_fps"]}
        min_pixels = spec.get("min_pixels")
        max_pixels = spec.get("max_pixels")
        if min_pixels is None:
            min_pixels = 16384
        if max_pixels is None:
            max_pixels = 307200
        item["min_pixels"] = int(min_pixels)
        item["max_pixels"] = int(max_pixels)
        item["total_pixels"] = int(max_pixels) * len(video["frames"])
        content.append(item)
    content.append({"type": "text", "text": instance["prompt"]})
    return [{"role": "user", "content": content}]


def prepare_inputs(bundle, instance, video_root=None):
    import torch
    from qwen_vl_utils import process_vision_info

    sampled = []
    for spec in instance.get("videos", []):
        sampled.append(sample_video(spec, video_root))
    messages = build_messages(instance, sampled)
    processor = bundle["processor"]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    images, videos_with_metadata, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True,
        return_video_metadata=True,
    )
    videos = None
    metadata = None
    if videos_with_metadata is not None:
        videos = []
        metadata = []
        for video, information in videos_with_metadata:
            videos.append(video)
            metadata.append(information)
    inputs = processor(
        text=[text], images=images, videos=videos, video_metadata=metadata,
        padding=True, return_tensors="pt", do_resize=False, **video_kwargs,
    )
    device = bundle["model"].device
    for key in inputs:
        if torch.is_tensor(inputs[key]):
            inputs[key] = inputs[key].to(device)
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


def extract_embedding(bundle, video_spec, video_root=None):
    """Mean-pool the primary projected video tokens."""
    import torch

    instance = {"videos": [video_spec], "prompt": "Represent this sign video."}
    inputs, audits = prepare_inputs(bundle, instance, video_root)
    model = bundle["model"]
    with torch.inference_mode():
        features = model.get_video_features(
            pixel_values_videos=inputs["pixel_values_videos"],
            video_grid_thw=inputs["video_grid_thw"],
        )
    if not isinstance(features, tuple) or len(features) != 2:
        raise RuntimeError("Unexpected Qwen video-feature API; use Transformers 4.57.x.")
    primary, deepstack = features
    if not isinstance(primary, (list, tuple)) or len(primary) != 1:
        raise RuntimeError("Expected projected tokens for exactly one video.")
    embedding = primary[0].float().mean(dim=0).cpu().numpy()
    return embedding, audits[0]
