"""Check decoding and interventions with a short synthetic video; no GPU needed."""

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.video import composite_frames, resolve_video, sample_video, uniform_indices


def check_video(directory):
    import cv2
    import numpy as np
    from PIL import Image

    path = directory / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 12, (64, 48))
    if not writer.isOpened():
        raise RuntimeError("This OpenCV installation cannot write the FFV1 test video")
    for index in range(12):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :] = [200, 20, index * 10]
        writer.write(frame)
    writer.release()
    spec = {"path": str(path), "num_frames": 8, "sampling": "uniform",
            "metadata": {"video_id": "sample", "split": "test"}}
    sampled = sample_video(spec)
    assert sampled["audit"]["selected_indices"] == [0, 2, 3, 5, 6, 8, 9, 11]
    assert sampled["frames"][0].getpixel((0, 0)) == (0, 20, 200)
    assert sampled["frames"][-1].getpixel((0, 0)) == (110, 20, 200)
    assert uniform_indices(2, 5) == [0, 0, 0, 1, 1]

    reversed_spec = dict(spec)
    reversed_spec["sampling"] = "reverse"
    reversed_video = sample_video(reversed_spec)
    indices = list(sampled["audit"]["selected_indices"])
    indices.reverse()
    assert reversed_video["audit"]["selected_indices"] == indices
    shuffled_spec = dict(spec)
    shuffled_spec.update({"sampling": "shuffle", "seed": 19})
    shuffled = sample_video(shuffled_spec)
    assert shuffled["audit"]["selected_indices"] == sample_video(shuffled_spec)["audit"]["selected_indices"]
    assert sorted(shuffled["audit"]["selected_indices"]) == sorted(indices)

    black_spec = dict(spec)
    black_spec["black"] = True
    for frame in sample_video(black_spec)["frames"]:
        assert not np.asarray(frame).any()
    single_spec = dict(spec)
    single_spec["repeat_middle"] = True
    repeated = sample_video(single_spec)
    for frame in repeated["frames"]:
        assert np.array_equal(np.asarray(frame), np.asarray(repeated["frames"][0]))
    assert len(repeated["frames"]) == 8

    crop_spec = dict(spec)
    crop_spec["crop_box"] = [4, 6, 28, 30]
    cropped = sample_video(crop_spec)
    assert cropped["audit"]["crop_box"] == crop_spec["crop_box"]
    for frame in cropped["frames"]:
        assert frame.size == (24, 24)
    crop_spec["crop_box"] = [0, 0, 100, 100]
    try:
        sample_video(crop_spec)
    except ValueError:
        pass
    else:
        raise AssertionError("Out-of-bounds crop was accepted")
    automatic_spec = dict(spec)
    automatic_spec["crop"] = "signing_space"
    automatic = sample_video(automatic_spec)
    assert automatic["audit"]["crop_box"] == sample_video(automatic_spec)["audit"]["crop_box"]
    for frame in automatic["frames"]:
        assert frame.size == automatic["frames"][0].size

    panels = [{"frames": [Image.new("RGB", (10, 20), "red")]},
              {"frames": [Image.new("RGB", (15, 10), "blue")]}]
    composite = composite_frames(panels)[0]
    assert composite.size == (25, 20)
    assert composite.getpixel((5, 10)) == (255, 0, 0)
    assert composite.getpixel((15, 10)) == (0, 0, 255)
    assert composite.getpixel((15, 0)) == (0, 0, 0)

    missing_root = directory / "empty"
    missing_root.mkdir()
    try:
        resolve_video(spec, missing_root)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("A different video root silently reused an absolute source path")
    flat = directory / "train" / "token.mp4"
    flat.parent.mkdir()
    flat.touch()
    assert resolve_video({"metadata": {"video_id": "token", "split": "train"}}, directory) == flat


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        check_video(Path(directory))
    print("Video RGB decoding, sampling, interventions, crops, composites, and paths passed.")
