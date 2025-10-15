# ========================================================================
# EFFICIENTNET-B0 FOR PEPPER DISEASE DETECTION
# Dataset from local ZIP file
# ========================================================================

# ========================================================================
# CELL 1: SETUP AND DEPENDENCIES
# ========================================================================

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_memory:.1f} GB")
else:
    print("WARNING: No GPU detected")

# Install dependencies
!pip install -q fastai timm
!pip install -q matplotlib seaborn scikit-learn

# Import libraries
from fastai.vision.all import *
from fastai.metrics import *
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import json
import zipfile
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("Setup completed")

# ========================================================================
# CELL 2: CONFIGURE PATHS
# ========================================================================

ZIP_PATH = Path('/content/dataset.zip')
RESULTS_PATH = Path('/content/pepper_results')
MODELS_PATH = Path('/content/pepper_models')

RESULTS_PATH.mkdir(exist_ok=True)
MODELS_PATH.mkdir(exist_ok=True)

print(f"ZIP Dataset: {ZIP_PATH}")
print(f"Results: {RESULTS_PATH}")
print(f"Models: {MODELS_PATH}")

if ZIP_PATH.exists():
    zip_size = ZIP_PATH.stat().st_size / (1024*1024)
    print(f"ZIP found: {zip_size:.1f} MB")
else:
    print("ZIP not found")
    print("Available files in /content/:")
    for item in Path('/content').iterdir():
        if item.name.endswith('.zip'):
            print(f"  {item.name}")

# ========================================================================
# CELL 3: EXTRACT AND SETUP DATASET
# ========================================================================

def extract_and_setup_dataset(zip_path, extract_to='/content/pepper_dataset'):
    """Extract ZIP and configure dataset structure"""
    extract_path = Path(extract_to)

    print(f"Extracting ZIP...")
    print(f"From: {zip_path}")
    print(f"To: {extract_path}")

    if extract_path.exists():
        shutil.rmtree(extract_path)
    extract_path.mkdir(parents=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("ZIP extracted successfully")
    except Exception as e:
        print(f"Error extracting ZIP: {e}")
        return None

    print(f"Exploring structure...")

    def find_dataset_structure(base_path):
        """Find train and valid folders recursively"""
        for root, dirs, files in os.walk(base_path):
            root_path = Path(root)
            if 'train' in dirs and 'valid' in dirs:
                return root_path
        return None

    dataset_root = find_dataset_structure(extract_path)

    if dataset_root is None:
        print("train and valid folders not found")
        return None

    print(f"Dataset found at: {dataset_root}")

    # Move to root if necessary
    if dataset_root != extract_path:
        print(f"Moving dataset to root...")
        temp_path = extract_path.parent / 'temp_dataset'
        shutil.move(str(dataset_root), str(temp_path))
        shutil.rmtree(extract_path)
        shutil.move(str(temp_path), str(extract_path))

    # Count images
    print(f"VERIFYING STRUCTURE:")
    total_images = 0

    for split in ['train', 'valid']:
        split_path = extract_path / split
        if split_path.exists():
            print(f"\n{split.upper()}:")
            split_total = 0
            for class_dir in split_path.iterdir():
                if class_dir.is_dir():
                    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
                    images = [f for f in class_dir.iterdir()
                             if f.is_file() and f.suffix.lower() in image_extensions]
                    count = len(images)
                    split_total += count
                    total_images += count
                    print(f"  {class_dir.name}: {count} images")
            print(f"  Total {split}: {split_total} images")
        else:
            print(f"{split} not found")
            return None

    print(f"\nTOTAL IMAGES: {total_images}")

    train_classes = [d.name for d in (extract_path / 'train').iterdir() if d.is_dir()]
    valid_classes = [d.name for d in (extract_path / 'valid').iterdir() if d.is_dir()]

    print(f"Classes found:")
    print(f"  Train: {train_classes}")
    print(f"  Valid: {valid_classes}")

    return extract_path

print("Configuring dataset from local ZIP...")
dataset_path = extract_and_setup_dataset(ZIP_PATH)

if dataset_path is None:
    raise Exception("Dataset setup failed")

print(f"Dataset ready at: {dataset_path}")

# ========================================================================
# CELL 4: CREATE DATALOADERS
# ========================================================================

def create_pepper_dataloaders(path, batch_size='auto', img_size=224):
    """Create DataLoaders for pepper classification"""

    # Auto-detect optimal batch size
    if batch_size == 'auto':
        try:
            gpu_name = torch.cuda.get_device_name(0).lower()
            if 'v100' in gpu_name or 'a100' in gpu_name:
                batch_size = 64
            elif 't4' in gpu_name or 'p100' in gpu_name:
                batch_size = 32
            else:
                batch_size = 16
        except:
            batch_size = 16

    print(f"Batch size: {batch_size}")

    item_tfms = [Resize(img_size)]
    batch_tfms = [
        *aug_transforms(
            size=img_size,
            do_flip=True,
            flip_vert=True,
            max_rotate=25,
            max_zoom=1.3,
            max_lighting=0.4,
            max_warp=0.15,
            p_affine=0.8,
            p_lighting=0.8
        ),
        Normalize.from_stats(*imagenet_stats)
    ]

    try:
        dls = ImageDataLoaders.from_folder(
            path,
            train='train',
            valid='valid',
            item_tfms=item_tfms,
            batch_tfms=batch_tfms,
            bs=batch_size,
            num_workers=2,
            pin_memory=True
        )
        return dls
    except Exception as e:
        print(f"Error creating DataLoaders: {e}")
        return None

print("Creating DataLoaders...")
dls = create_pepper_dataloaders(dataset_path)

if dls is None:
    raise Exception("Failed to create DataLoaders")

print(f"\nDATASET INFO:")
print(f"Classes: {dls.vocab}")
print(f"Number of classes: {len(dls.vocab)}")
print(f"Training images: {len(dls.train_ds)}")
print(f"Validation images: {len(dls.valid_ds)}")
print(f"Batch size: {dls.bs}")
print(f"Image size: 224x224")

# Class distribution
train_counts = {}
valid_counts = {}

for class_name in dls.vocab:
    train_path = dataset_path / 'train' / class_name
    valid_path = dataset_path / 'valid' / class_name

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    train_images = [f for f in train_path.glob('*')
                   if f.is_file() and f.suffix.lower() in image_extensions]
    valid_images = [f for f in valid_path.glob('*')
                   if f.is_file() and f.suffix.lower() in image_extensions]

    train_counts[class_name] = len(train_images)
    valid_counts[class_name] = len(valid_images)

print(f"\nCLASS DISTRIBUTION:")
print("TRAINING:")
total_train = sum(train_counts.values())
for class_name, count in train_counts.items():
    percentage = (count / total_train) * 100 if total_train > 0 else 0
    print(f"  {class_name}: {count} ({percentage:.1f}%)")

print("VALIDATION:")
total_valid = sum(valid_counts.values())
for class_name, count in valid_counts.items():
    percentage = (count / total_valid) * 100 if total_valid > 0 else 0
    print(f"  {class_name}: {count} ({percentage:.1f}%)")

# ========================================================================
# CELL 5: VISUALIZE DATA
# ========================================================================

print("\nDATASET EXAMPLES:")
try:
    dls.show_batch(figsize=(12, 8), max_n=12)
except Exception as e:
    print(f"Error showing batch: {e}")

# ========================================================================
# CELL 6: CREATE EFFICIENTNET-B0 MODEL
# ========================================================================

def create_efficientnet_model(dls):
    """Create EfficientNet-B0 model for binary classification"""

    metrics = [
        accuracy,
        Precision(pos_label=1, average='binary'),
        Recall(pos_label=1, average='binary'),
        F1Score(pos_label=1, average='binary'),
    ]

    learn = vision_learner(
        dls,
        'efficientnet_b0',
        metrics=metrics,
        pretrained=True,
        normalize=True,
    )

    # Enable mixed precision if GPU available
    if torch.cuda.is_available():
        try:
            learn = learn.to_fp16()
            print("Mixed precision (FP16) enabled")
        except:
            print("Using FP32")

    return learn

print("\nCREATING EFFICIENTNET-B0 MODEL...")
learn = create_efficientnet_model(dls)
print("Architecture: EfficientNet-B0")

# ========================================================================
# CELL 7: FIND OPTIMAL LEARNING RATE
# ========================================================================

print("\nFINDING OPTIMAL LEARNING RATE...")

try:
    lr_find_result = learn.lr_find()
    if hasattr(lr_find_result, 'valley'):
        suggested_lr = lr_find_result.valley
    else:
        suggested_lr = lr_find_result[0] if isinstance(lr_find_result, tuple) else 1e-3
    print(f"Selected LR: {suggested_lr:.2e}")
except Exception as e:
    suggested_lr = 1e-3
    print(f"Using default LR: {suggested_lr:.2e}")

# ========================================================================
# CELL 8: TRAINING PHASE 1 - FROZEN
# ========================================================================

learn.freeze()

frozen_epochs = 5
frozen_callbacks = [
    SaveModelCallback(monitor='valid_loss', fname='best_frozen'),
    EarlyStoppingCallback(monitor='valid_loss', patience=3),
]

print(f"Training {frozen_epochs} epochs with frozen backbone...")
print(f"Learning Rate: {suggested_lr:.2e}")

learn.fit_one_cycle(
    frozen_epochs,
    lr_max=suggested_lr,
    wd=0.01,
    cbs=frozen_callbacks
)

print("\nPHASE 1 RESULTS:")
try:
    frozen_results = learn.recorder.values[-1]
    print(f"Validation loss: {frozen_results['valid_loss']:.4f}")
    print(f"Accuracy: {frozen_results['accuracy']:.4f}")
except:
    print("Results logged in training")

print("\nSAMPLE PREDICTIONS AFTER PHASE 1:")
try:
    learn.show_results(max_n=6, figsize=(12, 8))
except Exception as e:
    print(f"Error showing results: {e}")

# ========================================================================
# CELL 9: TRAINING PHASE 2 - UNFROZEN
# ========================================================================

learn.unfreeze()

print("Finding new LR for fine-tuning...")
try:
    new_lr_find = learn.lr_find()
    if hasattr(new_lr_find, 'valley'):
        new_lr = new_lr_find.valley
    else:
        new_lr = new_lr_find[0] if isinstance(new_lr_find, tuple) else suggested_lr / 5
    print(f"New LR: {new_lr:.2e}")
except:
    new_lr = suggested_lr / 5
    print(f"Using default LR: {new_lr:.2e}")

unfrozen_epochs = 15
unfrozen_callbacks = [
    SaveModelCallback(monitor='valid_loss', fname='best_final'),
    EarlyStoppingCallback(monitor='valid_loss', patience=6),
    ReduceLROnPlateau(monitor='valid_loss', patience=3, factor=0.5),
]

print(f"Fine-tuning {unfrozen_epochs} epochs with discriminative LR...")
print(f"LR range: {new_lr/10:.2e} -> {new_lr:.2e}")

learn.fit_one_cycle(
    unfrozen_epochs,
    lr_max=slice(new_lr/10, new_lr),
    wd=0.01,
    cbs=unfrozen_callbacks
)

print("\nTRAINING COMPLETED")

# ========================================================================
# CELL 10: FULL EVALUATION
# ========================================================================

print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

interp = ClassificationInterpretation.from_learner(learn)

# Confusion matrix
print("Confusion Matrix:")
try:
    plt.figure(figsize=(8, 6))
    interp.plot_confusion_matrix(figsize=(8, 6), normalize=True, cmap='Blues')
    plt.title("Confusion Matrix - Pepper (Healthy vs Bacterial Spot)", fontsize=14)
    plt.savefig(RESULTS_PATH / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"Error generating confusion matrix: {e}")

# Top losses
print("\nMost difficult cases:")
try:
    interp.plot_top_losses(9, figsize=(15, 10))
    plt.savefig(RESULTS_PATH / 'top_losses.png', dpi=300, bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"Error showing top losses: {e}")

# Final results
print("\nFinal Results:")
try:
    learn.show_results(max_n=12, figsize=(15, 10))
    plt.savefig(RESULTS_PATH / 'final_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"Error showing final results: {e}")

# ========================================================================
# CELL 11: DETAILED METRICS
# ========================================================================

print("\nDETAILED METRICS:")

preds, targs = learn.get_preds()
y_pred = preds.argmax(dim=1).cpu().numpy()
y_true = targs.cpu().numpy()
y_prob = torch.softmax(preds, dim=1)[:, 1].cpu().numpy()

accuracy_final = accuracy_score(y_true, y_pred)
precision_final = precision_score(y_true, y_pred)
recall_final = recall_score(y_true, y_pred)
f1_final = f1_score(y_true, y_pred)
mcc_final = matthews_corrcoef(y_true, y_pred)

try:
    auc_final = roc_auc_score(y_true, y_prob)
except:
    auc_final = 0.0

print(f"\nFINAL RESULTS:")
print(f"{'='*55}")
print(f"Accuracy:              {accuracy_final:.4f} ({accuracy_final*100:.2f}%)")
print(f"Precision:             {precision_final:.4f}")
print(f"Recall (Sensitivity):  {recall_final:.4f}")
print(f"F1-Score:              {f1_final:.4f}")
print(f"Matthews Corr Coeff:   {mcc_final:.4f}")
if auc_final > 0:
    print(f"ROC-AUC:               {auc_final:.4f}")

print(f"\nCLASSIFICATION REPORT:")
class_names = ['Bacterial Spot', 'Healthy']
print(classification_report(y_true, y_pred, target_names=class_names))

# ========================================================================
# CELL 12: SAVE MODEL AND RESULTS
# ========================================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
model_name = f"pepper_efficientb0_acc{accuracy_final:.3f}_{timestamp}"

print(f"Saving to: {MODELS_PATH}")
print(f"Model name: {model_name}")

learn.save(MODELS_PATH / model_name)
learn.export(MODELS_PATH / f"{model_name}_export.pkl")

results_dict = {
    "model_info": {
        "architecture": "EfficientNet-B0",
        "task": "Bell Pepper Disease Detection",
        "classes": list(learn.dls.vocab),
        "timestamp": timestamp,
        "dataset_source": "Local ZIP"
    },
    "final_metrics": {
        "accuracy": float(accuracy_final),
        "precision": float(precision_final),
        "recall": float(recall_final),
        "f1_score": float(f1_final),
        "matthews_corr_coeff": float(mcc_final),
        "roc_auc": float(auc_final) if auc_final > 0 else None
    }
}

with open(MODELS_PATH / f"{model_name}_results.json", 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"\nSAVED SUCCESSFULLY")
print(f"{model_name}.pth (training checkpoint)")
print(f"{model_name}_export.pkl (production model)")
print(f"{model_name}_results.json (detailed metrics)")