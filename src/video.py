"""Video decoding, frame sampling and visual interventions for PhonoBench.

The public data structure is a dictionary. Heavy video/image dependencies are
imported only by the functions that need them.
"""

import math
from pathlib import Path


_CROP_BOXES = {}


def resolve_video(spec, video_root=None):
    """Resolve an explicit path or a Sem-Lex video ID without scanning a disk."""
    path = Path(spec.get("path", ""))
    if video_root is None:
        if path.is_file():
            return path
        raise FileNotFoundError(f"Video not found: {path}. Supply --video-root.")

    root = Path(video_root).expanduser()
    candidates = []
    if not path.is_absolute():
        candidates.append(root / path)
    metadata = spec.get("metadata", {})
    video_id = str(metadata.get("video_id", spec.get("video_id", "")))
    if not video_id:
        raise ValueError("A video_root requires metadata.video_id or video_id.")
    if Path(video_id).name != video_id:
        raise ValueError("video_id must be a single directory or file name.")
    roots = [root]
    split = str(metadata.get("split", ""))
    if split in ["train", "val", "test"]:
        roots.append(root / split)
    names = ["rgb_clean_480p.mp4", "rgb_480.mp4"]
    if path.name and path.suffix:
        names.insert(0, path.name)
    for current_root in roots:
        for name in names:
            candidates.append(current_root / video_id / "videos" / name)
            candidates.append(current_root / video_id / name)
        candidates.append(current_root / (video_id + ".mp4"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find video {video_id!r} under {root}.")


def uniform_indices(total, count):
    """Sample round(k*(T-1)/(K-1)) for k from 0 to K-1."""
    if total < 1 or count < 1:
        raise ValueError("The video and requested sample must contain frames.")
    if count == 1:
        return [(total - 1) // 2]
    indices = []
    for index in range(count):
        indices.append(round(index * (total - 1) / (count - 1)))
    return indices


def decode_video(path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No decodable frames in {path}")
    # Use a finite fallback when the container has no source frame rate.
    if not math.isfinite(fps) or fps <= 0:
        fps = 24.0
    return frames, fps


def event_indices(frames, count):
    """Half uniform, half high motion, with a two-frame separation when possible."""
    import cv2
    import numpy as np

    total = len(frames)
    if total <= count or count < 2:
        return uniform_indices(total, count)
    uniform_count = max(2, round(count * 0.5))
    selected = set(uniform_indices(total, uniform_count))
    scores = np.zeros(total, dtype=np.float64)
    previous = None
    for index in range(total):
        gray = cv2.cvtColor(frames[index], cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
        gray = gray.astype(np.float32)
        if previous is not None:
            scores[index] = float(np.mean(np.abs(gray - previous)))
        previous = gray
    for index in np.argsort(-scores):
        if len(selected) >= count:
            break
        separated = True
        for existing in selected:
            if abs(int(index) - existing) < 2:
                separated = False
                break
        if separated:
            selected.add(int(index))
    for index in uniform_indices(total, count * 2):
        if len(selected) == count:
            break
        selected.add(index)
    # Defensive termination for a pathological short sequence.
    for index in range(total):
        if len(selected) == count:
            break
        selected.add(index)
    return sorted(selected)


def select_indices(frames, count, sampling="uniform", seed=0):
    indices = uniform_indices(len(frames), count)
    if sampling == "event":
        return event_indices(frames, count)
    if sampling == "reverse":
        indices.reverse()
    elif sampling == "shuffle":
        import numpy as np

        np.random.default_rng(seed).shuffle(indices)
    elif sampling == "single":
        indices = []
        for index in range(count):
            indices.append(len(frames) // 2)
    elif sampling != "uniform":
        raise ValueError(f"Unsupported sampling mode: {sampling}")
    return indices


def signing_space_box(frames):
    """Estimate one fixed face-and-motion region for the entire video."""
    import cv2
    import numpy as np

    height, width = frames[0].shape[:2]
    scale = min(1.0, 320 / width)
    small_width, small_height = max(1, round(width * scale)), max(1, round(height * scale))
    positions = uniform_indices(len(frames), min(48, len(frames)))
    motion = np.zeros((small_height, small_width), dtype=np.float32)
    previous = None
    grays = []
    for position in positions:
        gray = cv2.cvtColor(frames[position], cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (small_width, small_height), interpolation=cv2.INTER_AREA)
        grays.append(gray)
        if previous is not None:
            motion = np.maximum(motion, np.abs(gray.astype(np.float32) - previous))
        previous = gray.astype(np.float32)
    threshold = max(12.0, float(np.quantile(motion, 0.88)))
    mask = (motion > threshold).astype(np.uint8)
    components, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for index in range(1, components):
        x, y, w, h, area = stats[index]
        if area >= max(4, small_width * small_height * 0.0005):
            boxes.append([int(x), int(y), int(x + w), int(y + h)])
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if not detector.empty():
        for position in uniform_indices(len(grays), min(3, len(grays))):
            faces = detector.detectMultiScale(grays[position], scaleFactor=1.1, minNeighbors=4)
            for x, y, w, h in faces:
                boxes.append([int(x), int(y), int(x + w), int(y + h)])
    if not boxes:
        return [0, 0, width, height], "full_frame_no_detection"
    left, top, right, bottom = small_width, small_height, 0, 0
    for box in boxes:
        left, top = min(left, box[0]), min(top, box[1])
        right, bottom = max(right, box[2]), max(bottom, box[3])
    center_x, center_y = (left + right) / (2 * scale), (top + bottom) / (2 * scale)
    crop_width = max(width * 0.45, (right - left) * 1.5 / scale)
    crop_height = max(height * 0.55, (bottom - top) * 1.5 / scale)
    crop_width = max(crop_width, crop_height * width / height)
    crop_width = min(width, crop_width)
    crop_height = crop_width * height / width
    x0 = max(0, min(width - round(crop_width), round(center_x - crop_width / 2)))
    y0 = max(0, min(height - round(crop_height), round(center_y - crop_height / 2)))
    return [x0, y0, x0 + round(crop_width), y0 + round(crop_height)], "face_and_motion"


def sample_video(spec, video_root=None, num_frames=None):
    """Decode a video, apply one fixed crop, and sample the requested frames."""
    from PIL import Image

    path = resolve_video(spec, video_root)
    frames, source_fps = decode_video(path)
    source_count = len(frames)
    source_height, source_width = frames[0].shape[:2]
    count = int(spec.get("num_frames", 32) if num_frames is None else num_frames)
    if count < 1:
        raise ValueError("num_frames must be positive.")
    box = spec.get("crop_box")
    crop_method = "supplied_box" if box is not None else "none"
    crop_mode = spec.get("crop", "none")
    if crop_mode not in ["none", "signing_space"]:
        raise ValueError("crop must be none or signing_space")
    if crop_mode == "signing_space" and box is None:
        file_info = path.stat()
        key = (str(path.resolve()), file_info.st_size, file_info.st_mtime_ns)
        if key not in _CROP_BOXES:
            _CROP_BOXES[key] = signing_space_box(frames)
        box, crop_method = _CROP_BOXES[key]
    if box is not None:
        if len(box) != 4:
            raise ValueError("crop_box must be [x0, y0, x1, y1].")
        for coordinate in box:
            if not isinstance(coordinate, int):
                raise ValueError("crop_box coordinates must be integers")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= source_width and 0 <= y0 < y1 <= source_height):
            raise ValueError(f"Invalid crop {box} for {source_width}x{source_height}.")
        cropped = []
        for frame in frames:
            cropped.append(frame[y0:y1, x0:x1])
        frames = cropped

    sampling = spec.get("sampling", "uniform")
    indices = select_indices(frames, count, sampling, int(spec.get("seed", 0)))

    selected = []
    for index in indices:
        selected.append(Image.fromarray(frames[index]))
    if spec.get("repeat_middle", False):
        middle = selected[len(selected) // 2]
        selected = []
        for index in indices:
            selected.append(middle.copy())
    if spec.get("black", False):
        black_frames = []
        for frame in selected:
            black_frames.append(Image.new("RGB", frame.size))
        selected = black_frames

    duration = max(source_count - 1, 1) / source_fps
    sample_fps = (count - 1) / duration if count > 1 else source_fps
    audit = {
        "path": str(path), "role": spec.get("role", ""),
        "num_frames": len(selected), "source_frame_count": source_count,
        "source_fps": source_fps, "sample_fps": sample_fps,
        "source_width": source_width, "source_height": source_height,
        "selected_indices": indices, "crop_box": box, "crop_method": crop_method,
        "sampling": sampling, "crop": crop_mode,
        "black": bool(spec.get("black", False)),
        "repeat_middle": bool(spec.get("repeat_middle", False)),
    }
    return {"frames": selected, "audit": audit}


def composite_frames(videos):
    """Pad source frames vertically and place panels left to right in role order."""
    from PIL import Image

    if not videos:
        raise ValueError("A composite needs at least one source video.")
    count = len(videos[0]["frames"])
    for video in videos:
        if len(video["frames"]) != count:
            raise ValueError("Composite source videos must have equal sample counts.")
    output = []
    for index in range(count):
        width = 0
        height = 0
        for video in videos:
            frame = video["frames"][index]
            width += frame.width
            height = max(height, frame.height)
        canvas = Image.new("RGB", (width, height))
        left = 0
        for video in videos:
            frame = video["frames"][index]
            canvas.paste(frame, (left, (height - frame.height) // 2))
            left += frame.width
        output.append(canvas)
    return output
