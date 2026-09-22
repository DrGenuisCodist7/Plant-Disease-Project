"""Adaptive multi-threshold enhancement for plant-disease classification.

Expected dataset layout:
dataset/
  train/class_name/*.jpg
  val/class_name/*.jpg
  test/class_name/*.jpg

Workflow:
1. Train a classifier using random enhancement augmentation.
2. Calibrate a shallow decision-tree selector on the validation set.
3. Compare raw, fixed preprocessing, and adaptive preprocessing on the test set.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


METHODS = ["none", "clahe", "gamma_bright", "gamma_dark", "hist_eq"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_statistics(rgb: np.ndarray) -> np.ndarray:
    """Features used by the multi-threshold selector (no class-label features)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    luminance = lab[:, :, 0].astype(np.float32) / 255.0
    hist = cv2.calcHist([(luminance * 255).astype(np.uint8)], [0], None, [256], [0, 256])
    probability = hist.ravel() / max(float(hist.sum()), 1.0)
    probability = probability[probability > 0]
    entropy = float(-(probability * np.log2(probability)).sum() / 8.0)
    p05, p95 = np.percentile(luminance, [5, 95])
    return np.array(
        [
            luminance.mean(),                         # brightness
            luminance.std(),                          # RMS contrast
            entropy,                                  # normalized entropy
            np.mean(luminance <= 0.03),               # shadow clipping
            np.mean(luminance >= 0.97),               # highlight clipping
            p95 - p05,                                # robust dynamic range
        ],
        dtype=np.float32,
    )


def enhance(rgb: np.ndarray, method: str) -> np.ndarray:
    """Enhance only luminance so disease-related colour is less distorted."""
    if method == "none":
        return rgb.copy()

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    luminance = lab[:, :, 0]

    if method == "clahe":
        luminance = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(luminance)
    elif method in {"gamma_bright", "gamma_dark"}:
        gamma = 0.65 if method == "gamma_bright" else 1.45
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        luminance = cv2.LUT(luminance, table)
    elif method == "hist_eq":
        luminance = cv2.equalizeHist(luminance)
    else:
        raise ValueError(f"Unknown method: {method}")

    lab[:, :, 0] = luminance
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


class RandomEnhancement:
    """Makes the classifier compatible with every candidate preprocessing path."""

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"))
        return Image.fromarray(enhance(rgb, random.choice(METHODS)))


class SelectedDataset(torch.utils.data.Dataset):
    def __init__(self, folder: str, transform, selector=None, fixed_method: str | None = None):
        self.base = datasets.ImageFolder(folder)
        self.transform = transform
        self.selector = selector
        self.fixed_method = fixed_method

    @property
    def classes(self):
        return self.base.classes

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        path, label = self.base.samples[index]
        rgb = np.asarray(Image.open(path).convert("RGB"))
        if self.selector is not None:
            method = METHODS[int(self.selector.predict(image_statistics(rgb)[None, :])[0])]
        else:
            method = self.fixed_method or "none"
        return self.transform(Image.fromarray(enhance(rgb, method))), label


def transforms_for(size: int = 224):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose(
        [
            RandomEnhancement(),
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tf = transforms.Compose(
        [transforms.Resize((size, size)), transforms.ToTensor(), normalize]
    )
    return train_tf, eval_tf


def make_model(number_of_classes: int) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, number_of_classes)
    return model


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf, eval_tf = transforms_for(args.image_size)
    train_data = datasets.ImageFolder(Path(args.data) / "train", transform=train_tf)
    val_data = datasets.ImageFolder(Path(args.data) / "val", transform=eval_tf)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False, num_workers=args.workers)

    model = make_model(len(train_data.classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            val_loss, val_accuracy = run_epoch(model, val_loader, device)
        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                {"state_dict": model.state_dict(), "classes": train_data.classes},
                args.model,
            )


@torch.no_grad()
def method_scores(model, rgb, true_label, transform, device):
    """Return true-class probability for each candidate method."""
    batch = torch.stack([transform(Image.fromarray(enhance(rgb, m))) for m in METHODS]).to(device)
    return torch.softmax(model(batch), dim=1)[:, true_label].cpu().numpy()


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = make_model(len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint["classes"]


def calibrate(args):
    """Learn interpretable thresholds from validation images only."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, eval_tf = transforms_for(args.image_size)
    model, classes = load_model(args.model, device)
    data = datasets.ImageFolder(Path(args.data) / "val")
    if data.classes != classes:
        raise ValueError("Class order differs between model and validation data")

    features, best_methods = [], []
    for path, label in data.samples:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        features.append(image_statistics(rgb))
        best_methods.append(int(np.argmax(method_scores(model, rgb, label, eval_tf, device))))

    selector = DecisionTreeClassifier(
        max_depth=args.selector_depth,
        min_samples_leaf=args.min_leaf,
        class_weight="balanced",
        random_state=args.seed,
    )
    selector.fit(np.stack(features), np.asarray(best_methods))
    joblib.dump(selector, args.selector)
    print(export_text(selector, feature_names=[
        "brightness", "contrast", "entropy", "shadow_clip", "highlight_clip", "dynamic_range"
    ]))
    counts = np.bincount(best_methods, minlength=len(METHODS))
    print("Calibration method labels:", dict(zip(METHODS, counts.tolist())))


@torch.no_grad()
def predict_loader(model, loader, device):
    truth, prediction = [], []
    for images, labels in loader:
        prediction.extend(model(images.to(device)).argmax(1).cpu().tolist())
        truth.extend(labels.tolist())
    return np.asarray(truth), np.asarray(prediction)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, eval_tf = transforms_for(args.image_size)
    model, classes = load_model(args.model, device)
    selector = joblib.load(args.selector)
    results = {}
    strategies = METHODS + ["adaptive"]
    for strategy in strategies:
        dataset = SelectedDataset(
            str(Path(args.data) / "test"),
            eval_tf,
            selector=selector if strategy == "adaptive" else None,
            fixed_method=None if strategy == "adaptive" else strategy,
        )
        loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.workers)
        truth, prediction = predict_loader(model, loader, device)
        results[strategy] = {
            "accuracy": float(accuracy_score(truth, prediction)),
            "confusion_matrix": confusion_matrix(truth, prediction).tolist(),
            "report": classification_report(
                truth, prediction, target_names=classes, output_dict=True, zero_division=0
            ),
        }
        print(f"{strategy:>12}: accuracy={results[strategy]['accuracy']:.4f}")
    Path(args.results).write_text(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "calibrate", "evaluate"])
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--model", default="plant_model.pt")
    parser.add_argument("--selector", default="enhancement_selector.joblib")
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--selector-depth", type=int, default=3)
    parser.add_argument("--min-leaf", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    {"train": train, "calibrate": calibrate, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
