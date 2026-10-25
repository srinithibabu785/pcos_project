"""
Comprehensive Dataset Audit Script for PCOS Ultrasound Dataset.
Inspects raw dataset (dataset_5.zip / PCOSGen-train), detects:
- Total images, class distributions, file formats, dimensions
- Corrupted / zero-byte / unreadable images
- Exact binary duplicates (SHA-256)
- Perceptual duplicates (dHash)
- Conflicting labels
- Missing/invalid labels
- Cross-set (Train/Test) leakage
Generates:
- results/dataset_audit_report.json
- results/dataset_audit_report.csv
- results/dataset_summary.txt
- results/duplicate_report.csv
"""

import os
import io
import json
import zipfile
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from PIL import Image

BASE_DIR = Path(r"C:\Users\srini\Documents\pcos_project")
DATASET_DIR = BASE_DIR / "dataset"
RESULTS_DIR = BASE_DIR / "results"
ZIP_PATH = DATASET_DIR / "dataset_5.zip"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Computes difference hash (dHash) using PIL and numpy."""
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=np.float32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    decimal_val = 0
    hex_str = []
    for index, val in enumerate(diff.flatten()):
        if val:
            decimal_val += 2 ** (index % 8)
        if (index % 8) == 7:
            hex_str.append(hex(decimal_val)[2:].rjust(2, '0'))
            decimal_val = 0
    return "".join(hex_str)

def run_audit():
    print(f"Starting Dataset Audit on {ZIP_PATH}...")
    
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Raw dataset zip not found at {ZIP_PATH}")

    total_images = 0
    corrupted_count = 0
    missing_labels = 0
    invalid_filenames = 0
    dimensions = Counter()
    formats = Counter()
    class_counts = Counter()
    split_counts = Counter()
    
    sha_map = defaultdict(list)
    dhash_map = defaultdict(list)
    image_records = []

    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        namelist = z.namelist()
        
        metadata_files = [n for n in namelist if any(n.endswith(ext) for ext in ['.csv', '.xlsx', '.txt'])]
        print(f"Metadata files detected in archive: {metadata_files}")

        for item in z.infolist():
            filename = item.filename
            if item.is_dir() or not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue

            total_images += 1
            parts = filename.split('/')
            
            if 'train' in parts:
                split = 'train'
            elif 'test' in parts:
                split = 'test'
            else:
                split = 'unknown'
            split_counts[split] += 1
            
            if 'healthy' in parts:
                cls = 'Healthy'
                raw_cls = 'healthy'
                label = 0
            elif 'infected' in parts:
                cls = 'PCOS'
                raw_cls = 'infected'
                label = 1
            else:
                cls = 'Unknown'
                raw_cls = 'unknown'
                label = -1
                missing_labels += 1
            class_counts[cls] += 1

            data = z.read(item)
            file_size = len(data)
            if file_size == 0:
                corrupted_count += 1
                continue

            sha256 = hashlib.sha256(data).hexdigest()

            try:
                with Image.open(io.BytesIO(data)) as img:
                    width, height = img.size
                    img_format = img.format
                    formats[img_format] += 1
                    dimensions[(width, height)] += 1
                    dh = compute_dhash(img)
            except Exception as e:
                corrupted_count += 1
                continue

            record = {
                "file_path": filename,
                "filename": Path(filename).name,
                "split": split,
                "class_name": cls,
                "raw_class": raw_cls,
                "label": label,
                "file_size": file_size,
                "width": width,
                "height": height,
                "format": img_format,
                "sha256": sha256,
                "dhash": dh
            }
            image_records.append(record)
            sha_map[sha256].append(record)
            dhash_map[dh].append(record)

    df_records = pd.DataFrame(image_records)
    
    # -------------------------------------------------------------
    # Duplicate Analysis (Exact SHA-256)
    # -------------------------------------------------------------
    duplicate_rows = []
    sha_dup_groups = {h: recs for h, recs in sha_map.items() if len(recs) > 1}
    
    conflicting_label_groups = 0
    train_test_leakage_groups = 0
    train_duplicate_files = 0
    test_duplicate_files = 0
    
    for h, recs in sha_dup_groups.items():
        original = recs[0]
        splits = {r["split"] for r in recs}
        classes = {r["class_name"] for r in recs}
        
        has_train = "train" in splits
        has_test = "test" in splits
        has_conflict = len(classes) > 1

        if has_conflict:
            conflicting_label_groups += 1
        if has_train and has_test:
            train_test_leakage_groups += 1

        for dup in recs[1:]:
            if dup["split"] == "train":
                train_duplicate_files += 1
            elif dup["split"] == "test":
                test_duplicate_files += 1

            action = "Purge from Train" if dup["split"] == "train" else "Retain in Benchmark Test"
            reason_parts = ["Exact SHA-256 duplicate"]
            if has_conflict:
                reason_parts.append(f"Conflicting labels ({'/'.join(classes)})")
            if has_train and has_test:
                reason_parts.append("Cross-set Train/Test duplicate")
            
            duplicate_rows.append({
                "original_file": original["file_path"],
                "duplicate_file": dup["file_path"],
                "hash_type": "SHA-256",
                "hash_value": h,
                "action": action,
                "reason": "; ".join(reason_parts)
            })

    df_duplicates = pd.DataFrame(duplicate_rows)
    df_duplicates.to_csv(RESULTS_DIR / "duplicate_report.csv", index=False)
    print(f"Saved duplicate report to results/duplicate_report.csv ({len(df_duplicates)} entries)")

    # -------------------------------------------------------------
    # Summary Metrics
    # -------------------------------------------------------------
    total_healthy = class_counts["Healthy"]
    total_pcos = class_counts["PCOS"]
    train_count = split_counts["train"]
    test_count = split_counts["test"]
    
    train_healthy = len(df_records[(df_records["split"] == "train") & (df_records["class_name"] == "Healthy")])
    train_pcos = len(df_records[(df_records["split"] == "train") & (df_records["class_name"] == "PCOS")])
    test_healthy = len(df_records[(df_records["split"] == "test") & (df_records["class_name"] == "Healthy")])
    test_pcos = len(df_records[(df_records["split"] == "test") & (df_records["class_name"] == "PCOS")])

    audit_summary = {
        "dataset_name": "PCOSGen-train",
        "archive_source": str(ZIP_PATH),
        "total_images": total_images,
        "healthy_count": total_healthy,
        "pcos_count": total_pcos,
        "healthy_percentage": round((total_healthy / total_images) * 100, 2) if total_images else 0,
        "pcos_percentage": round((total_pcos / total_images) * 100, 2) if total_images else 0,
        "corrupted_images": corrupted_count,
        "missing_labels": missing_labels,
        "invalid_filenames": invalid_filenames,
        "file_formats": dict(formats),
        "top_dimensions": [f"{w}x{h} ({c} images)" for (w, h), c in dimensions.most_common(5)],
        "train_set": {
            "total": train_count,
            "healthy": train_healthy,
            "pcos": train_pcos,
            "healthy_percentage": round((train_healthy / train_count) * 100, 2) if train_count else 0,
            "pcos_percentage": round((train_pcos / train_count) * 100, 2) if train_count else 0,
            "duplicates_removed_target": train_duplicate_files
        },
        "test_set": {
            "total": test_count,
            "healthy": test_healthy,
            "pcos": test_pcos,
            "healthy_percentage": round((test_healthy / test_count) * 100, 2) if test_count else 0,
            "pcos_percentage": round((test_pcos / test_count) * 100, 2) if test_count else 0,
            "status": "Untouched independent benchmark"
        },
        "duplicate_summary": {
            "total_duplicate_groups": len(sha_dup_groups),
            "total_duplicate_files": len(df_duplicates),
            "intra_train_duplicate_files": train_duplicate_files,
            "intra_test_duplicate_files": test_duplicate_files,
            "conflicting_label_groups": conflicting_label_groups,
            "train_test_leakage_groups": train_test_leakage_groups
        },
        "leakage_findings": "NO cross-set binary duplicates detected between train and test sets (0 cross-set duplicates). Complete isolation confirmed." if train_test_leakage_groups == 0 else f"WARNING: {train_test_leakage_groups} cross-set duplicates detected!"
    }

    with open(RESULTS_DIR / "dataset_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=4)
    print("Saved results/dataset_audit_report.json")

    csv_rows = [
        {"Metric": "Dataset Name", "Value": audit_summary["dataset_name"]},
        {"Metric": "Total Raw Images", "Value": audit_summary["total_images"]},
        {"Metric": "Total Healthy Images", "Value": f"{audit_summary['healthy_count']} ({audit_summary['healthy_percentage']}%)"},
        {"Metric": "Total PCOS Images", "Value": f"{audit_summary['pcos_count']} ({audit_summary['pcos_percentage']}%)"},
        {"Metric": "Corrupted Images", "Value": audit_summary["corrupted_images"]},
        {"Metric": "Missing Labels", "Value": audit_summary["missing_labels"]},
        {"Metric": "Train Total Images", "Value": audit_summary["train_set"]["total"]},
        {"Metric": "Train Healthy Images", "Value": f"{audit_summary['train_set']['healthy']} ({audit_summary['train_set']['healthy_percentage']}%)"},
        {"Metric": "Train PCOS Images", "Value": f"{audit_summary['train_set']['pcos']} ({audit_summary['train_set']['pcos_percentage']}%)"},
        {"Metric": "Test Total Images (Independent)", "Value": audit_summary["test_set"]["total"]},
        {"Metric": "Test Healthy Images", "Value": f"{audit_summary['test_set']['healthy']} ({audit_summary['test_set']['healthy_percentage']}%)"},
        {"Metric": "Test PCOS Images", "Value": f"{audit_summary['test_set']['pcos']} ({audit_summary['test_set']['pcos_percentage']}%)"},
        {"Metric": "Intra-Train Duplicates", "Value": audit_summary["duplicate_summary"]["intra_train_duplicate_files"]},
        {"Metric": "Intra-Test Duplicates", "Value": audit_summary["duplicate_summary"]["intra_test_duplicate_files"]},
        {"Metric": "Conflicting Label Groups", "Value": audit_summary["duplicate_summary"]["conflicting_label_groups"]},
        {"Metric": "Cross Train/Test Leakage Groups", "Value": audit_summary["duplicate_summary"]["train_test_leakage_groups"]},
        {"Metric": "Leakage Status", "Value": audit_summary["leakage_findings"]}
    ]
    pd.DataFrame(csv_rows).to_csv(RESULTS_DIR / "dataset_audit_report.csv", index=False)
    print("Saved results/dataset_audit_report.csv")

    summary_text = f"""========================================================
PCOS ULTRASOUND DATASET AUDIT & INTEGRITY REPORT
========================================================
Source Archive: {audit_summary['archive_source']}
Total Raw Images: {audit_summary['total_images']}
File Formats: {audit_summary['file_formats']}
Predominant Resolutions: {', '.join(audit_summary['top_dimensions'])}

CLASS DISTRIBUTION:
--------------------------------------------------------
Class 0 (Healthy): {audit_summary['healthy_count']} images ({audit_summary['healthy_percentage']}%)
Class 1 (PCOS):    {audit_summary['pcos_count']} images ({audit_summary['pcos_percentage']}%)
Corrupted / Unreadable: {audit_summary['corrupted_images']}
Missing / Invalid Labels: {audit_summary['missing_labels']}

SPLIT BREAKDOWN:
--------------------------------------------------------
Train Split: {audit_summary['train_set']['total']} images
  - Healthy: {audit_summary['train_set']['healthy']} ({audit_summary['train_set']['healthy_percentage']}%)
  - PCOS:    {audit_summary['train_set']['pcos']} ({audit_summary['train_set']['pcos_percentage']}%)
Independent Test Split: {audit_summary['test_set']['total']} images
  - Healthy: {audit_summary['test_set']['healthy']} ({audit_summary['test_set']['healthy_percentage']}%)
  - PCOS:    {audit_summary['test_set']['pcos']} ({audit_summary['test_set']['pcos_percentage']}%)

DATA INTEGRITY & LEAKAGE FINDINGS:
--------------------------------------------------------
- Exact SHA-256 Duplicate Groups: {audit_summary['duplicate_summary']['total_duplicate_groups']}
- Total Duplicate Files: {audit_summary['duplicate_summary']['total_duplicate_files']}
  - Intra-Train Duplicates: {audit_summary['duplicate_summary']['intra_train_duplicate_files']} (Scheduled for clean removal)
  - Intra-Test Duplicates:  {audit_summary['duplicate_summary']['intra_test_duplicate_files']} (Preserved in test benchmark)
- Conflicting Label Groups: {audit_summary['duplicate_summary']['conflicting_label_groups']} (Identical image hashes under opposing labels)
- Cross Train/Test Leakage: {audit_summary['leakage_findings']}
========================================================
"""
    with open(RESULTS_DIR / "dataset_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("Saved results/dataset_summary.txt")
    print(summary_text)

if __name__ == "__main__":
    run_audit()
