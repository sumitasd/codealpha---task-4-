"""
Real-time object detection and simple centroid-based tracking.

Usage:
  python object_detection_tracking.py --source 0
  python object_detection_tracking.py --source path/to/video.mp4
  python object_detection_tracking.py --source 0 --backend yolov5 --model yolov5m

Dependencies:
  - OpenCV, torch, torchvision, ultralytics, numpy (see requirements.txt)

This script uses YOLO-World by default so common custom labels like clothes and
mobile phone can be requested. YOLOv5 is also available for COCO classes.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

try:
    import numpy as np
except ImportError:
    print("ERROR: NumPy is not installed. Run: pip install numpy")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV is not installed. Run: pip install opencv-python")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("ERROR: PyTorch is not installed. Run: pip install torch torchvision")
    sys.exit(1)

DEFAULT_WORLD_CLASSES = [
    "person",
    "mobile phone",
    "cell phone",
    "smartphone",
    "clothing",
    "clothes",
    "shirt",
    "t-shirt",
    "pants",
    "trousers",
    "jeans",
    "jacket",
    "cap",
    "hat",
    "shoe",
    "bag",
    "backpack",
    "laptop",
    "bottle",
]

LABEL_ALIASES = {
    "cell phone": "mobile",
    "mobile phone": "mobile",
    "smartphone": "mobile",
    "clothing": "cloth",
    "clothes": "cloth",
    "shirt": "cloth",
    "t-shirt": "cloth",
    "pants": "cloth",
    "trousers": "cloth",
    "jeans": "cloth",
    "jacket": "cloth",
    "hat": "cap",
}


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    label: str
    confidence: float


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def confidence(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from exc
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def parse_classes(value: str | None) -> List[str]:
    if not value:
        return DEFAULT_WORLD_CLASSES
    classes = [item.strip() for item in value.split(",") if item.strip()]
    if not classes:
        raise argparse.ArgumentTypeError("must contain at least one class name")
    return classes


def normalize_label(label: str) -> str:
    return LABEL_ALIASES.get(label.strip().lower(), label.strip().lower())


def is_valid_person_box(
    bbox: Tuple[int, int, int, int],
    frame_shape,
    min_area_ratio: float,
    min_aspect_ratio: float,
) -> bool:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    frame_height, frame_width = frame_shape[:2]
    area_ratio = (width * height) / max(1, frame_width * frame_height)
    aspect_ratio = height / width
    return area_ratio >= min_area_ratio and aspect_ratio >= min_aspect_ratio


def load_model(
    conf: float,
    backend: str = "yolo-world",
    model_name: str | None = None,
    weights: str | None = None,
    yolov5_repo: str | None = None,
    world_classes: List[str] | None = None,
):
    try:
        if backend == "yolo-world":
            try:
                from ultralytics import YOLOWorld
            except ImportError as exc:
                raise ImportError("ultralytics is not installed. Run: pip install ultralytics") from exc

            world_weights = weights or model_name or "yolov8s-world.pt"
            model = YOLOWorld(world_weights)
            model.set_classes(world_classes or DEFAULT_WORLD_CLASSES)
            return {"backend": backend, "model": model, "conf": conf}

        if backend == "yolov5":
            yolov5_model = model_name or "yolov5m"
            if weights:
                weights_path = Path(weights)
                if not weights_path.exists():
                    raise FileNotFoundError(f"weights file not found: {weights_path}")

                if yolov5_repo:
                    repo_path = Path(yolov5_repo)
                    if not repo_path.exists():
                        raise FileNotFoundError(f"YOLOv5 repo folder not found: {repo_path}")
                    model = torch.hub.load(str(repo_path), "custom", path=str(weights_path), source="local")
                else:
                    model = torch.hub.load("ultralytics/yolov5", "custom", path=str(weights_path))
            else:
                model = torch.hub.load("ultralytics/yolov5", yolov5_model, pretrained=True)

            model.conf = conf
            return {"backend": backend, "model": model, "conf": conf}

        raise ValueError(f"unsupported backend: {backend}")
    except Exception as exc:
        print("ERROR: Could not load the detection model.")
        print(f"Reason: {exc}")
        print(f"Python used: {sys.executable}")
        print()
        print("Fix options:")
        print("  1. Run this script with the project virtual environment:")
        print(r"     .\.venv\Scripts\python.exe object_detection_tracking.py --source 0")
        print("  2. Or activate the virtual environment first, then run python:")
        print(r"     .\.venv\Scripts\Activate.ps1")
        print("     python object_detection_tracking.py --source 0")
        print("  3. If you are intentionally using this Python, install missing packages into it:")
        print(f"     \"{sys.executable}\" -m pip install -r requirements.txt")
        print("  4. Check your internet connection so the selected model can download.")
        print("  5. Install Git if torch.hub reports that git is missing.")
        print("  6. Or use local files:")
        print("     python object_detection_tracking.py --backend yolov5 --weights yolov5s.pt --yolov5-repo path/to/yolov5")
        sys.exit(1)


def detect_objects(
    detector,
    image_rgb,
    person_conf: float,
    person_min_area: float,
    person_min_aspect: float,
) -> List[Detection]:
    backend = detector["backend"]
    model = detector["model"]
    conf = detector["conf"]
    detections: List[Detection] = []

    if backend == "yolo-world":
        results = model.predict(image_rgb, conf=conf, verbose=False)
        if not results:
            return detections

        result = results[0]
        names = getattr(result, "names", getattr(model, "names", {}))
        for box in result.boxes:
            score = float(box.conf[0])
            class_id = int(box.cls[0])
            if isinstance(names, dict):
                raw_label = names.get(class_id, str(class_id))
            else:
                raw_label = names[class_id] if class_id < len(names) else str(class_id)
            label = normalize_label(raw_label)
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            bbox = (x1, y1, x2, y2)
            if label == "person" and (
                score < person_conf
                or not is_valid_person_box(bbox, image_rgb.shape, person_min_area, person_min_aspect)
            ):
                continue
            detections.append(Detection(bbox, label, score))
        return detections

    results = model(image_rgb)
    rows = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, conf, cls
    names = getattr(results, "names", getattr(model, "names", {}))
    for *xyxy, score, cls in rows:
        score = float(score)
        if score < conf:
            continue
        class_id = int(cls)
        if isinstance(names, dict):
            raw_label = names.get(class_id, str(class_id))
        else:
            raw_label = names[class_id] if class_id < len(names) else str(class_id)
        label = normalize_label(raw_label)
        x1, y1, x2, y2 = map(int, xyxy)
        bbox = (x1, y1, x2, y2)
        if label == "person" and (
            score < person_conf
            or not is_valid_person_box(bbox, image_rgb.shape, person_min_area, person_min_aspect)
        ):
            continue
        detections.append(Detection(bbox, label, score))
    return detections


class CentroidTracker:
    def __init__(self, max_disappeared: int = 30):
        self.next_object_id = 0
        self.objects: Dict[int, Tuple[int, int, int, int]] = {}
        self.centroids: Dict[int, Tuple[int, int]] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared

    def register(self, bbox: Tuple[int, int, int, int]):
        self.objects[self.next_object_id] = bbox
        x1, y1, x2, y2 = bbox
        cX = int((x1 + x2) / 2.0)
        cY = int((y1 + y2) / 2.0)
        self.centroids[self.next_object_id] = (cX, cY)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id: int):
        del self.objects[object_id]
        del self.centroids[object_id]
        del self.disappeared[object_id]

    def update(self, rects: List[Tuple[int, int, int, int]]):
        # rects: list of (x1, y1, x2, y2)
        if len(rects) == 0:
            # increase disappeared counter for existing objects
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = []
        for (x1, y1, x2, y2) in rects:
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids.append((cX, cY))

        if len(self.centroids) == 0:
            for i in range(len(rects)):
                self.register(rects[i])
        else:
            # build distance matrix between existing centroids and new input centroids
            object_ids = list(self.centroids.keys())
            object_centroids = list(self.centroids.values())
            D = np.zeros((len(object_centroids), len(input_centroids)), dtype="float")
            for i, (ox, oy) in enumerate(object_centroids):
                for j, (ix, iy) in enumerate(input_centroids):
                    D[i, j] = np.linalg.norm(np.array((ox, oy)) - np.array((ix, iy)))

            # greedy match: for each existing object find nearest input centroid
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            assigned_rows = set()
            assigned_cols = set()

            for row, col in zip(rows, cols):
                if row in assigned_rows or col in assigned_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = rects[col]
                self.centroids[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                assigned_rows.add(row)
                assigned_cols.add(col)

            # process unassigned existing objects
            unused_rows = set(range(0, D.shape[0])) - assigned_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # register new objects for unassigned input centroids
            unused_cols = set(range(0, D.shape[1])) - assigned_cols
            for col in unused_cols:
                self.register(rects[col])

        return self.objects


def draw_boxes(
    frame,
    tracked_objects: Dict[int, Tuple[int, int, int, int]],
    labels: Dict[int, str],
    scores: Dict[int, float],
    colors: Dict[str, Tuple[int, int, int]],
):
    for object_id, bbox in tracked_objects.items():
        x1, y1, x2, y2 = bbox
        # label available per-object if we store it in labels map
        label = labels.get(object_id, "object")
        color = colors.get(label, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        score = scores.get(object_id)
        text = f"{label} {score:.2f} ID:{object_id}" if score is not None else f"{label} ID:{object_id}"
        cv2.putText(frame, text, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Video source. 0 for webcam or path to video file")
    parser.add_argument("--conf", type=confidence, default=0.35, help="Confidence threshold for detections")
    parser.add_argument("--person-conf", type=confidence, default=0.65, help="Higher threshold for person detections")
    parser.add_argument("--person-min-area", type=confidence, default=0.04, help="Minimum frame area ratio for person boxes")
    parser.add_argument("--person-min-aspect", type=float, default=1.15, help="Minimum height/width ratio for person boxes")
    parser.add_argument("--skip", type=positive_int, default=1, help="Process every Nth frame to speed up")
    parser.add_argument("--backend", choices=["yolo-world", "yolov5"], default="yolo-world", help="Detection backend")
    parser.add_argument("--model", help="Model name. Examples: yolov8s-world.pt, yolov8m-world.pt, yolov5m, yolov5x")
    parser.add_argument("--world-classes", type=parse_classes, help="Comma-separated classes for YOLO-World")
    parser.add_argument("--weights", help="Optional local model weights file")
    parser.add_argument("--yolov5-repo", help="Optional local ultralytics/yolov5 repo folder for offline loading")
    args = parser.parse_args()

    print("Loading detection model (this may take a moment)...")
    detector = load_model(
        args.conf,
        backend=args.backend,
        model_name=args.model,
        weights=args.weights,
        yolov5_repo=args.yolov5_repo,
        world_classes=args.world_classes,
    )

    # video source
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("ERROR: Cannot open video source", src)
        return

    tracker = CentroidTracker(max_disappeared=30)
    # mapping from tracked object id -> label (string)
    id_labels: Dict[int, str] = {}
    id_scores: Dict[int, float] = {}
    rng = np.random.default_rng(42)
    colors: Dict[str, Tuple[int, int, int]] = {}

    frame_idx = 0
    fps_start = time.time()
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % args.skip != 0:
            # still draw previously tracked boxes
            draw_boxes(frame, tracker.objects, id_labels, id_scores, colors)
            cv2.imshow("Detection+Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # convert BGR->RGB for model
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections = detect_objects(
            detector,
            img,
            args.person_conf,
            args.person_min_area,
            args.person_min_aspect,
        )
        rects = [detection.bbox for detection in detections]
        det_labels = [detection.label for detection in detections]
        det_scores = [detection.confidence for detection in detections]

        # update tracker
        objects = tracker.update(rects)

        # assign labels to tracked ids by matching bbox centers to input rects
        for obj_id in list(objects.keys()):
            # find closest rect to this object's centroid
            bx = objects[obj_id]
            bx_cx = int((bx[0] + bx[2]) / 2.0)
            bx_cy = int((bx[1] + bx[3]) / 2.0)
            best_idx = None
            best_dist = float("inf")
            for i, r in enumerate(rects):
                rx_cx = int((r[0] + r[2]) / 2.0)
                rx_cy = int((r[1] + r[3]) / 2.0)
                d = (rx_cx - bx_cx) ** 2 + (rx_cy - bx_cy) ** 2
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx is not None and best_idx < len(det_labels):
                id_labels[obj_id] = det_labels[best_idx]
                id_scores[obj_id] = det_scores[best_idx]
                if id_labels[obj_id] not in colors:
                    colors[id_labels[obj_id]] = tuple(map(int, rng.integers(0, 255, size=3)))

        draw_boxes(frame, objects, id_labels, id_scores, colors)

        processed_frames += 1
        if processed_frames % 30 == 0:
            fps_now = processed_frames / max(1e-6, (time.time() - fps_start))
            cv2.putText(frame, f"FPS: {fps_now:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Detection+Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
