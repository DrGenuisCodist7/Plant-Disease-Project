"""
FINAL ADAPTIVE PREPROCESSING FRAMEWORK
======================================

Research idea:
For each plant-leaf image, determine whether enhancement is required.
If enhancement is useful, select the most suitable enhancement method
using an interpretable Decision Tree based on image statistics.

Run:
    python adaptive_preprocessing_framework.py

Required dataset structure:

dataset/
    train/
        class_1/
        class_2/
        ...
    val/
        class_1/
        class_2/
        ...
    test/
        class_1/
        class_2/
        ...

Outputs:
    plant_model_final.pt
    training_history.csv
    selector_training_data.json
    selector_training_data.csv
    adaptive_selector_final.joblib
    decision_tree_rules.txt
    final_results.json
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn

from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from sklearn.tree import (
    DecisionTreeClassifier,
    export_text,
)

from torch.utils.data import (
    DataLoader,
    Dataset,
)

from torchvision import (
    datasets,
    models,
    transforms,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "dataset"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"


MODEL_FILE = (
    PROJECT_DIR /
    "plant_model_final.pt"
)

TRAINING_HISTORY_FILE = (
    PROJECT_DIR /
    "training_history.csv"
)

SELECTOR_DATA_JSON = (
    PROJECT_DIR /
    "selector_training_data.json"
)

SELECTOR_DATA_CSV = (
    PROJECT_DIR /
    "selector_training_data.csv"
)

SELECTOR_FILE = (
    PROJECT_DIR /
    "adaptive_selector_final.joblib"
)

TREE_RULES_FILE = (
    PROJECT_DIR /
    "decision_tree_rules.txt"
)

RESULTS_FILE = (
    PROJECT_DIR /
    "final_results.json"
)


# ============================================================
# IMPORTANT EXPERIMENT SETTINGS
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 6

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

SEED = 42

WORKERS = 0


# ============================================================
# DECISION TREE SETTINGS
# ============================================================

SELECTOR_DEPTH = 4

MIN_SAMPLES_LEAF = 20


# ============================================================
# MINIMUM REQUIRED BENEFIT OF ENHANCEMENT
#
# Example:
# raw confidence = 0.80
# enhanced confidence = 0.81
#
# improvement = 0.01
#
# Since 0.01 < 0.02,
# choose NO ENHANCEMENT.
# ============================================================

BENEFIT_THRESHOLD = 0.02


# ============================================================
# PREPROCESSING OPTIONS
# ============================================================

METHODS = [
    "none",
    "clahe",
    "gamma_bright",
    "gamma_dark",
    "hist_eq",
]


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. IMAGE STATISTICS
# ============================================================

def image_statistics(rgb):

    """
    Extract statistical characteristics
    used by the adaptive selector.
    """

    lab = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2LAB
    )

    luminance = (
        lab[:, :, 0].astype(np.float32)
        / 255.0
    )


    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    brightness = float(
        np.mean(luminance)
    )


    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    contrast = float(
        np.std(luminance)
    )


    # --------------------------------------------------------
    # Entropy
    # --------------------------------------------------------

    histogram = cv2.calcHist(
        [
            (luminance * 255)
            .astype(np.uint8)
        ],
        [0],
        None,
        [256],
        [0, 256],
    )

    probability = histogram.ravel()

    probability = (
        probability /
        max(
            float(probability.sum()),
            1.0
        )
    )

    probability = probability[
        probability > 0
    ]

    entropy = float(
        -(
            probability *
            np.log2(probability)
        ).sum() / 8.0
    )


    # --------------------------------------------------------
    # Shadow clipping
    # --------------------------------------------------------

    shadow_clip = float(
        np.mean(
            luminance <= 0.03
        )
    )


    # --------------------------------------------------------
    # Highlight clipping
    # --------------------------------------------------------

    highlight_clip = float(
        np.mean(
            luminance >= 0.97
        )
    )


    # --------------------------------------------------------
    # Robust dynamic range
    # --------------------------------------------------------

    p05, p95 = np.percentile(
        luminance,
        [5, 95]
    )

    dynamic_range = float(
        p95 - p05
    )


    return np.array(
        [
            brightness,
            contrast,
            entropy,
            shadow_clip,
            highlight_clip,
            dynamic_range,
        ],
        dtype=np.float32,
    )


# ============================================================
# 4. IMAGE ENHANCEMENT METHODS
# ============================================================

def enhance(rgb, method):

    """
    Apply the selected enhancement method.
    Enhancement is performed mainly on luminance
    to reduce unnecessary colour distortion.
    """

    if method == "none":

        return rgb.copy()


    lab = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2LAB
    )

    luminance = lab[:, :, 0]


    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    if method == "clahe":

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        luminance = clahe.apply(
            luminance
        )


    # --------------------------------------------------------
    # Gamma brightening
    # --------------------------------------------------------

    elif method == "gamma_bright":

        gamma = 0.65

        table = np.array(
            [
                (
                    (i / 255.0) ** gamma
                ) * 255
                for i in range(256)
            ],
            dtype=np.float32
        ).astype(np.uint8)

        luminance = cv2.LUT(
            luminance,
            table
        )


    # --------------------------------------------------------
    # Gamma darkening
    # --------------------------------------------------------

    elif method == "gamma_dark":

        gamma = 1.45

        table = np.array(
            [
                (
                    (i / 255.0) ** gamma
                ) * 255
                for i in range(256)
            ],
            dtype=np.float32
        ).astype(np.uint8)

        luminance = cv2.LUT(
            luminance,
            table
        )


    # --------------------------------------------------------
    # Histogram equalization
    # --------------------------------------------------------

    elif method == "hist_eq":

        luminance = cv2.equalizeHist(
            luminance
        )


    else:

        raise ValueError(
            f"Unknown method: {method}"
        )


    lab[:, :, 0] = luminance


    return cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )


# ============================================================
# 5. RANDOM CANDIDATE PREPROCESSING
#
# Used only while training the disease classifier.
# This enables one classifier to work with images produced
# through different candidate preprocessing paths.
# ============================================================

class RandomCandidatePreprocessing:

    def __call__(self, image):

        rgb = np.asarray(
            image.convert("RGB")
        )

        method = random.choice(
            METHODS
        )

        processed = enhance(
            rgb,
            method
        )

        return Image.fromarray(
            processed
        )


# ============================================================
# 6. IMAGE TRANSFORMS
# ============================================================

def get_transforms():

    normalize = transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ],
    )


    train_transform = transforms.Compose(
        [
            RandomCandidatePreprocessing(),

            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            ),

            transforms.RandomHorizontalFlip(),

            transforms.RandomRotation(
                12
            ),

            transforms.ToTensor(),

            normalize,
        ]
    )


    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            ),

            transforms.ToTensor(),

            normalize,
        ]
    )


    return (
        train_transform,
        evaluation_transform
    )


# ============================================================
# 7. CREATE DISEASE CLASSIFICATION MODEL
# ============================================================

def create_model(number_of_classes):

    weights = (
        models
        .EfficientNet_B0_Weights
        .DEFAULT
    )


    model = models.efficientnet_b0(
        weights=weights
    )


    model.classifier[1] = nn.Linear(

        model.classifier[1].in_features,

        number_of_classes
    )


    return model


# ============================================================
# 8. TRAIN / VALIDATION EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    device,
    optimizer=None
):

    training = (
        optimizer is not None
    )


    model.train(
        training
    )


    loss_function = (
        nn.CrossEntropyLoss()
    )


    total_loss = 0.0

    correct = 0

    total = 0


    for images, labels in loader:

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )


        if training:

            optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = loss_function(
            outputs,
            labels
        )


        if training:

            loss.backward()

            optimizer.step()


        total_loss += (
            loss.item()
            *
            labels.size(0)
        )


        predictions = outputs.argmax(
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


    return (
        total_loss / total,
        correct / total
    )


# ============================================================
# 9. PHASE 1
# TRAIN DISEASE CLASSIFIER
# ============================================================

def train_disease_classifier():

    print()
    print("=" * 70)
    print("PHASE 1: TRAINING DISEASE CLASSIFIER")
    print("=" * 70)


    if MODEL_FILE.exists():
        print()
        print(f"Found existing trained model: {MODEL_FILE.name}")
        print("Skipping Phase 1 classifier training and proceeding to next phases.")
        return

    set_seed(
        SEED
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "Device:",
        device
    )


    train_transform, eval_transform = (
        get_transforms()
    )


    train_dataset = datasets.ImageFolder(

        TRAIN_DIR,

        transform=train_transform
    )


    val_dataset = datasets.ImageFolder(

        VAL_DIR,

        transform=eval_transform
    )


    print(
        "Training images:",
        len(train_dataset)
    )


    print(
        "Validation images:",
        len(val_dataset)
    )


    print(
        "Classes:",
        len(train_dataset.classes)
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=WORKERS
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=WORKERS
    )


    model = create_model(

        len(
            train_dataset.classes
        )

    ).to(device)


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )


    best_validation_accuracy = -1.0


    history = []


    for epoch in range(
        1,
        EPOCHS + 1
    ):


        train_loss, train_accuracy = (
            run_epoch(

                model,

                train_loader,

                device,

                optimizer
            )
        )


        with torch.no_grad():

            val_loss, val_accuracy = (
                run_epoch(

                    model,

                    val_loader,

                    device
                )
            )


        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"Train Loss={train_loss:.4f}  "
            f"Train Acc={train_accuracy*100:.2f}%  "
            f"Val Loss={val_loss:.4f}  "
            f"Val Acc={val_accuracy*100:.2f}%"
        )


        history.append(
            {
                "epoch": epoch,
                "train_loss":
                    train_loss,
                "train_accuracy":
                    train_accuracy,
                "validation_loss":
                    val_loss,
                "validation_accuracy":
                    val_accuracy,
            }
        )


        if (
            val_accuracy >
            best_validation_accuracy
        ):

            best_validation_accuracy = (
                val_accuracy
            )


            torch.save(
                {
                    "state_dict":
                        model.state_dict(),

                    "classes":
                        train_dataset.classes,

                    "best_validation_accuracy":
                        best_validation_accuracy,
                },

                MODEL_FILE
            )


            print(
                "   Best model saved."
            )


    # --------------------------------------------------------
    # Save complete training history
    # --------------------------------------------------------

    with TRAINING_HISTORY_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:


        writer = csv.writer(
            handle
        )


        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_accuracy",
                "validation_loss",
                "validation_accuracy",
            ]
        )


        for item in history:

            writer.writerow(
                [
                    item["epoch"],
                    item["train_loss"],
                    item["train_accuracy"],
                    item["validation_loss"],
                    item[
                        "validation_accuracy"
                    ],
                ]
            )


    print()
    print(
        "Training completed."
    )


    print(
        "Best validation accuracy:",
        f"{best_validation_accuracy*100:.3f}%"
    )


    print(
        "Model:",
        MODEL_FILE.name
    )


    print(
        "Training history:",
        TRAINING_HISTORY_FILE.name
    )


# ============================================================
# 10. LOAD TRAINED MODEL
# ============================================================

def load_trained_model(
    device
):

    checkpoint = torch.load(

        MODEL_FILE,

        map_location=device
    )


    classes = checkpoint[
        "classes"
    ]


    model = create_model(
        len(classes)
    )


    model.load_state_dict(
        checkpoint[
            "state_dict"
        ]
    )


    model = model.to(
        device
    )


    model.eval()


    return (
        model,
        classes
    )


# ============================================================
# 11. CORRECT-CLASS CONFIDENCE
# ============================================================

@torch.no_grad()
def correct_class_confidence(

    model,

    rgb,

    method,

    true_label,

    transform,

    device
):


    processed = enhance(
        rgb,
        method
    )


    tensor = transform(
        Image.fromarray(
            processed
        )
    )


    tensor = tensor.unsqueeze(
        0
    ).to(device)


    logits = model(
        tensor
    )


    probability = torch.softmax(

        logits,

        dim=1
    )


    return float(
        probability[
            0,
            true_label
        ].cpu()
    )


# ============================================================
# 12. PHASE 2
# BUILD ADAPTIVE-SELECTOR TRAINING DATA
# ============================================================

def build_selector_training_data():

    print()
    print("=" * 70)
    print("PHASE 2: BUILDING ADAPTIVE SELECTOR TRAINING DATA")
    print("=" * 70)


    set_seed(
        SEED
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    _, transform = (
        get_transforms()
    )


    model, classes = (
        load_trained_model(
            device
        )
    )


    val_dataset = (
        datasets.ImageFolder(
            VAL_DIR
        )
    )


    if (
        val_dataset.classes
        != classes
    ):

        raise ValueError(
            "Class order mismatch between model and validation dataset."
        )


    records = []

    method_counter = Counter()


    total_images = len(
        val_dataset.samples
    )


    for index, (
        image_path,
        true_label
    ) in enumerate(
        val_dataset.samples,
        start=1
    ):


        rgb = np.asarray(

            Image.open(
                image_path
            ).convert("RGB")
        )


        stats = image_statistics(
            rgb
        )


        # ----------------------------------------------------
        # No-enhancement confidence
        # ----------------------------------------------------

        raw_confidence = (
            correct_class_confidence(

                model,

                rgb,

                "none",

                true_label,

                transform,

                device
            )
        )


        best_method = "none"

        best_confidence = (
            raw_confidence
        )


        # ----------------------------------------------------
        # Test every enhancement
        # ----------------------------------------------------

        for method in METHODS:

            if method == "none":

                continue


            confidence = (
                correct_class_confidence(

                    model,

                    rgb,

                    method,

                    true_label,

                    transform,

                    device
                )
            )


            if (
                confidence >
                best_confidence
            ):

                best_confidence = (
                    confidence
                )

                best_method = (
                    method
                )


        # ----------------------------------------------------
        # Improvement relative to original image
        # ----------------------------------------------------

        improvement = (
            best_confidence
            -
            raw_confidence
        )


        # ----------------------------------------------------
        # Enhancement accepted only when benefit is meaningful
        # ----------------------------------------------------

        if (
            improvement
            <
            BENEFIT_THRESHOLD
        ):

            final_method = "none"

        else:

            final_method = (
                best_method
            )


        method_counter[
            final_method
        ] += 1


        record = {

            "image":
                str(image_path),

            "brightness":
                float(stats[0]),

            "contrast":
                float(stats[1]),

            "entropy":
                float(stats[2]),

            "shadow_clip":
                float(stats[3]),

            "highlight_clip":
                float(stats[4]),

            "dynamic_range":
                float(stats[5]),

            "raw_confidence":
                raw_confidence,

            "best_confidence":
                best_confidence,

            "improvement":
                improvement,

            "best_enhancement_before_threshold":
                best_method,

            "selected_method":
                final_method,
        }


        records.append(
            record
        )


        if (
            index % 100 == 0
            or
            index == total_images
        ):

            print(
                f"Processed "
                f"{index}/{total_images}"
            )


    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    SELECTOR_DATA_JSON.write_text(

        json.dumps(
            records,
            indent=2
        ),

        encoding="utf-8"
    )


    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    fieldnames = [
        "image",
        "brightness",
        "contrast",
        "entropy",
        "shadow_clip",
        "highlight_clip",
        "dynamic_range",
        "raw_confidence",
        "best_confidence",
        "improvement",
        "best_enhancement_before_threshold",
        "selected_method",
    ]


    with SELECTOR_DATA_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:


        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames
        )


        writer.writeheader()


        writer.writerows(
            records
        )


    print()
    print(
        "Selector data completed."
    )


    print()
    print(
        "Selected-method distribution:"
    )


    for method in METHODS:

        print(
            f"{method:15s}: "
            f"{method_counter[method]}"
        )


    print()
    print(
        "Saved:",
        SELECTOR_DATA_JSON.name
    )


    print(
        "Saved:",
        SELECTOR_DATA_CSV.name
    )


# ============================================================
# 13. PHASE 3
# TRAIN DECISION TREE ADAPTIVE SELECTOR
# ============================================================

def train_adaptive_selector():

    print()
    print("=" * 70)
    print("PHASE 3: TRAINING ADAPTIVE DECISION TREE SELECTOR")
    print("=" * 70)


    records = json.loads(

        SELECTOR_DATA_JSON.read_text(
            encoding="utf-8"
        )
    )


    feature_names = [

        "brightness",

        "contrast",

        "entropy",

        "shadow_clip",

        "highlight_clip",

        "dynamic_range",
    ]


    X = np.asarray(
        [
            [
                item[name]
                for name
                in feature_names
            ]

            for item in records
        ],

        dtype=np.float32
    )


    y = np.asarray(
        [
            item[
                "selected_method"
            ]

            for item
            in records
        ]
    )


    print(
        "Selector samples:",
        len(y)
    )


    print(
        "Selector classes:",
        sorted(
            set(y)
        )
    )


    selector = DecisionTreeClassifier(

        max_depth=
            SELECTOR_DEPTH,

        min_samples_leaf=
            MIN_SAMPLES_LEAF,

        class_weight=
            "balanced",

        random_state=
            SEED
    )


    selector.fit(
        X,
        y
    )


    joblib.dump(

        selector,

        SELECTOR_FILE
    )


    rules = export_text(

        selector,

        feature_names=
            feature_names
    )


    TREE_RULES_FILE.write_text(

        rules,

        encoding="utf-8"
    )


    training_accuracy = (
        selector.score(
            X,
            y
        )
    )


    print()
    print(
        "Selector training accuracy:",
        f"{training_accuracy*100:.3f}%"
    )


    print()
    print(
        "Learned Decision Tree Rules:"
    )


    print(
        rules
    )


    print(
        "Selector saved:",
        SELECTOR_FILE.name
    )


    print(
        "Decision rules saved:",
        TREE_RULES_FILE.name
    )


# ============================================================
# 14. FINAL EVALUATION DATASET
# ============================================================

class AdaptiveEvaluationDataset(
    Dataset
):

    def __init__(
        self,
        folder,
        transform,
        strategy,
        selector=None
    ):

        self.base = (
            datasets.ImageFolder(
                folder
            )
        )


        self.transform = (
            transform
        )


        self.strategy = (
            strategy
        )


        self.selector = (
            selector
        )


    @property
    def classes(self):

        return self.base.classes


    def __len__(self):

        return len(
            self.base
        )


    def __getitem__(
        self,
        index
    ):


        path, label = (
            self.base.samples[
                index
            ]
        )


        rgb = np.asarray(

            Image.open(
                path
            ).convert("RGB")
        )


        # ----------------------------------------------------
        # Adaptive decision
        # ----------------------------------------------------

        if (
            self.strategy
            ==
            "adaptive"
        ):


            stats = (
                image_statistics(
                    rgb
                )
            )


            method = str(

                self.selector.predict(

                    stats.reshape(
                        1,
                        -1
                    )

                )[0]
            )


        else:

            method = (
                self.strategy
            )


        processed = enhance(

            rgb,

            method
        )


        tensor = self.transform(

            Image.fromarray(
                processed
            )
        )


        return (
            tensor,
            label,
            method
        )


# ============================================================
# 15. PHASE 4
# FINAL TEST-SET EVALUATION
# ============================================================

@torch.no_grad()
def final_evaluation():

    print()
    print("=" * 70)
    print("PHASE 4: FINAL TEST-SET EVALUATION")
    print("=" * 70)


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    _, transform = (
        get_transforms()
    )


    model, classes = (
        load_trained_model(
            device
        )
    )


    selector = joblib.load(
        SELECTOR_FILE
    )


    strategies = [

        "none",

        "clahe",

        "gamma_bright",

        "gamma_dark",

        "hist_eq",

        "adaptive",
    ]


    all_results = {}


    for strategy in strategies:


        dataset = (
            AdaptiveEvaluationDataset(

                TEST_DIR,

                transform,

                strategy,

                selector
                if strategy
                == "adaptive"
                else None
            )
        )


        if (
            dataset.classes
            != classes
        ):

            raise ValueError(
                "Class order mismatch between model and test dataset."
            )


        loader = DataLoader(

            dataset,

            batch_size=
                BATCH_SIZE,

            shuffle=False,

            num_workers=
                WORKERS
        )


        truth = []

        predictions = []

        selected_methods = []


        for (
            images,
            labels,
            methods
        ) in loader:


            images = images.to(
                device
            )


            logits = model(
                images
            )


            predicted = logits.argmax(
                dim=1
            ).cpu()


            truth.extend(
                labels.tolist()
            )


            predictions.extend(
                predicted.tolist()
            )


            selected_methods.extend(
                list(methods)
            )


        accuracy = accuracy_score(

            truth,

            predictions
        )


        report = (
            classification_report(

                truth,

                predictions,

                target_names=
                    classes,

                output_dict=True,

                zero_division=0
            )
        )


        matrix = confusion_matrix(

            truth,

            predictions
        )


        result = {

            "accuracy":
                float(accuracy),

            "macro_precision":
                float(
                    report[
                        "macro avg"
                    ][
                        "precision"
                    ]
                ),

            "macro_recall":
                float(
                    report[
                        "macro avg"
                    ][
                        "recall"
                    ]
                ),

            "macro_f1":
                float(
                    report[
                        "macro avg"
                    ][
                        "f1-score"
                    ]
                ),

            "weighted_f1":
                float(
                    report[
                        "weighted avg"
                    ][
                        "f1-score"
                    ]
                ),

            "confusion_matrix":
                matrix.tolist(),

            "classification_report":
                report,
        }


        if (
            strategy
            ==
            "adaptive"
        ):

            result[
                "selected_methods"
            ] = dict(
                Counter(
                    selected_methods
                )
            )


        all_results[
            strategy
        ] = result


        print(
            f"{strategy:15s} | "
            f"Accuracy = "
            f"{accuracy*100:.3f}% | "
            f"Macro F1 = "
            f"{result['macro_f1']*100:.3f}%"
        )


        if (
            strategy
            ==
            "adaptive"
        ):

            print(
                "Adaptive selections:"
            )


            for (
                method,
                count
            ) in sorted(

                Counter(
                    selected_methods
                ).items()
            ):

                print(
                    f"   "
                    f"{method:15s}"
                    f"{count}"
                )


    RESULTS_FILE.write_text(

        json.dumps(
            all_results,
            indent=2
        ),

        encoding="utf-8"
    )


    print()
    print(
        "Final results saved:",
        RESULTS_FILE.name
    )


# ============================================================
# 16. CHECK PROJECT BEFORE STARTING
# ============================================================

def check_project():

    print()
    print("=" * 70)
    print("CHECKING PROJECT")
    print("=" * 70)


    required_folders = [

        TRAIN_DIR,

        VAL_DIR,

        TEST_DIR,
    ]


    for folder in required_folders:


        if not folder.is_dir():

            raise SystemExit(
                f"\nERROR:\n"
                f"Required folder not found:\n"
                f"{folder}\n"
            )


        print(
            "Found:",
            folder
        )


    print()
    print(
        "Dataset folders are ready."
    )


# ============================================================
# 17. COMPLETE AUTOMATIC PIPELINE
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ADAPTIVE PREPROCESSING FOR PLANT DISEASE CLASSIFICATION")
    print("=" * 70)

    print()
    print(
        "The complete experiment will run automatically."
    )

    print()
    print(
        "Stages:"
    )

    print(
        "1. Disease classifier training"
    )

    print(
        "2. Adaptive-selector dataset generation"
    )

    print(
        "3. Decision Tree selector training"
    )

    print(
        "4. Final test-set evaluation"
    )


    check_project()


    train_disease_classifier()


    build_selector_training_data()


    train_adaptive_selector()


    final_evaluation()


    print()
    print("=" * 70)
    print("COMPLETE EXPERIMENT FINISHED SUCCESSFULLY")
    print("=" * 70)


    print()
    print(
        "Generated files:"
    )

    print(
        "1.",
        MODEL_FILE.name
    )

    print(
        "2.",
        TRAINING_HISTORY_FILE.name
    )

    print(
        "3.",
        SELECTOR_DATA_JSON.name
    )

    print(
        "4.",
        SELECTOR_DATA_CSV.name
    )

    print(
        "5.",
        SELECTOR_FILE.name
    )

    print(
        "6.",
        TREE_RULES_FILE.name
    )

    print(
        "7.",
        RESULTS_FILE.name
    )


# ============================================================
# 18. START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()