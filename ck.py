import torch
import time
from rfdetr import RFDETRSmall
from PIL import Image
from glob import glob
import numpy as np
import cv2

MODEL_PATH = "checkpoints/checkpoint_best_ema.pth"
CONF = 0.5
CLASSES = ["emr", "emr", "nem", "emr"]

model = RFDETRSmall(pretrain_weights=MODEL_PATH, num_classes=len(CLASSES))
model.model.model.eval()

# -----------------------------
# 1) INFERENCE SPEED TEST
# -----------------------------
imgs = [Image.open(p).convert("RGB") for p in glob("data/valid/-_jpeg.rf.bb943c84f9f0485d3e5eaf388f9c04f1.jpg")]

t0 = time.time()
N = len(imgs)

with torch.no_grad():
    for img in imgs:
        _ = model.predict(img, threshold=CONF)

t1 = time.time()

avg_time = (t1 - t0) / N
fps = 1.0 / avg_time

print("Inference Speed:")
print(f" Avg time/image: {avg_time*1000:.2f} ms")
print(f" FPS: {fps:.2f}")

# -----------------------------
# 2) mAP / Precision / Recall
# -----------------------------
from torchmetrics.detection.mean_ap import MeanAveragePrecision

metric = MeanAveragePrecision()

def load_annotations(path):
    boxes = []
    labels = []
    with open(path.replace(".jpg", ".txt")) as f:
        for line in f:
            c, x, y, w, h = map(float, line.split())
            labels.append(int(c))
            boxes.append([x, y, w, h])
    return boxes, labels

img_paths = glob("val/*.jpg")

with torch.no_grad():
    for p in img_paths:
        img = Image.open(p).convert("RGB")
        result = model.predict(img, threshold=CONF)

        pred_boxes = []
        pred_labels = []
        pred_scores = []

        for s, c, b in zip(result.confidence, result.class_id, result.bbox):
            pred_scores.append(float(s))
            pred_labels.append(int(c))
            pred_boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])

        gt_boxes, gt_labels = load_annotations(p)

        metric.update(
            [
                {
                    "boxes": torch.tensor(pred_boxes),
                    "scores": torch.tensor(pred_scores),
                    "labels": torch.tensor(pred_labels)
                }
            ],
            [
                {
                    "boxes": torch.tensor(gt_boxes),
                    "labels": torch.tensor(gt_labels)
                }
            ]
        )

final_scores = metric.compute()
print("\nDetection Metrics:")
print(final_scores)
