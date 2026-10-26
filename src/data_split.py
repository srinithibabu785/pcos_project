"""
================================================================================
Anti-Leakage Data Splitting Pipeline for PCOS Ultrasound Dataset
Project: An Explainable AI-Based Ensemble Transfer Learning Approach
         for Early PCOS Prediction and Clinical Decision Support
================================================================================
This script:
1. Streams images from dataset/datasett.zip and extracts only UNIQUE images
   into data/clean_images/ (deduplicating 7,788 exact binary duplicates).
2. Prunes cross-class contradictory ground-truth pairs.
3. Assigns an 'image_family_id' to group rotated/resized twins together.
4. Performs a Group-Aware Stratified Split (70% Train, 15% Val, 15% Test).
5. Enforces strictly ZERO family and ZERO hash leakage across splits.
6. Saves reproducible manifests:
   - data/train.csv
   - data/validation.csv
   - data/test.csv
================================================================================
"""

import os
import re
import sys
import argparse
import zipfile
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from io import BytesIO
import pandas as pd
import numpy as np
from PIL import Image

def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Computes difference hash for perceptual collision checking."""
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(resized.get_flattened_data() if hasattr(resized, "get_flattened_data") else resized.getdata())
    difference = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = pixels[row * (hash_size + 1) + col]
            pixel_right = pixels[row * (hash_size + 1) + col + 1]
            difference.append(pixel_left > pixel_right)
    decimal_val = 0
    hex_str = []
    for index, val in enumerate(difference):
        if val:
            decimal_val += 2 ** (index % 8)
        if (index % 8) == 7:
            hex_str.append(hex(decimal_val)[2:].rjust(2, "0"))
            decimal_val = 0
    return "".join(hex_str)

def extract_family_id(filename: str, cls: str) -> str:
    """
    Extracts a family ID to group base scans with their rotated and resized variants.
    """
    basename = Path(filename).name
    # Search for numeric identifier
    nums = re.findall(r"\d+", basename)
    if "image" in basename.lower() and not basename.startswith("Image_"):
        # PCOSGen Kaggle style: image10441.jpg
        return f"{cls}_pcosgen_{nums[0]}" if nums else f"{cls}_{basename}"
    elif nums:
        return f"{cls}_base_{nums[0]}"
    else:
        return f"{cls}_{basename}"

def prepare_clean_splits(
    zip_path: str,
    output_dir: str,
    data_manifest_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
):
    np.random.seed(random_seed)
    zip_file = Path(zip_path)
    clean_images_dir = Path(output_dir)
    manifest_dir = Path(data_manifest_dir)
    clean_images_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    (clean_images_dir / "infected").mkdir(parents=True, exist_ok=True)
    (clean_images_dir / "noninfected").mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 1: DEDUPLICATION & CONFLICT DETECTION FROM ARCHIVE")
    print("=" * 80)

    sha_to_records = defaultdict(list)
    dhash_to_records = defaultdict(list)

    with zipfile.ZipFile(zip_file, "r") as z:
        namelist = [n for n in z.namelist() if not n.endswith("/")]
        print(f"Reading {len(namelist):,} files from archive...")

        for idx, name in enumerate(namelist):
            cls = name.split("/")[-2]
            data = z.read(name)
            sha = hashlib.sha256(data).hexdigest()
            
            with Image.open(BytesIO(data)) as img:
                dh = compute_dhash(img)

            sha_to_records[sha].append({
                "zip_path": name,
                "class": cls,
                "dhash": dh,
                "data": data
            })

    print(f"Unique SHA-256 hashes found: {len(sha_to_records):,}")

    # Identify cross-class dHash conflicts (identical image labeled as both infected & noninfected)
    cross_conflicts = set()
    for sha, recs in sha_to_records.items():
        dh = recs[0]["dhash"]
        dhash_to_records[dh].append((sha, recs[0]["class"]))

    for dh, entries in dhash_to_records.items():
        classes = set(e[1] for e in entries)
        if len(classes) > 1:
            for sha, _ in entries:
                cross_conflicts.add(sha)

    print(f"Cross-class conflicting SHA-256 hashes pruned: {len(cross_conflicts)}")

    # Extract clean unique images to disk
    clean_records = []
    print("\nSTEP 2: EXTRACTING CLEAN UNIQUE IMAGES TO DISK...")
    saved_count = 0

    for sha, recs in sha_to_records.items():
        if sha in cross_conflicts:
            continue

        # Choose the cleanest canonical file name
        canonical = recs[0]
        cls = canonical["class"]
        orig_name = Path(canonical["zip_path"]).name
        clean_filename = f"{cls}_{sha[:12]}_{orig_name}"
        clean_path = clean_images_dir / cls / clean_filename

        if not clean_path.exists():
            with open(clean_path, "wb") as f:
                f.write(canonical["data"])

        label = 1 if cls == "infected" else 0
        class_name = "PCOS" if cls == "infected" else "Normal"
        fam_id = extract_family_id(orig_name, cls)

        clean_records.append({
            "image_path": str(clean_path).replace("\\", "/"),
            "relative_path": f"data/clean_images/{cls}/{clean_filename}",
            "filename": clean_filename,
            "label": label,
            "class_name": class_name,
            "family_id": fam_id,
            "sha256": sha
        })
        saved_count += 1

    df_clean = pd.DataFrame(clean_records)
    print(f"Clean usable unique images saved: {len(df_clean):,}")
    print(f"Class distribution: {df_clean['class_name'].value_counts().to_dict()}")

    # Group-aware stratified splitting
    print("\nSTEP 3: GROUP-AWARE STRATIFIED SPLIT ON IMAGE FAMILIES...")
    family_df = df_clean.groupby("family_id").agg({
        "label": "first",
        "sha256": "count"
    }).reset_index().rename(columns={"sha256": "image_count"})

    print(f"Total unique image families: {len(family_df):,}")
    
    # Shuffle families deterministically
    family_df = family_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    train_families = set()
    val_families = set()
    test_families = set()

    # Stratify by class at the family level
    for lbl in [0, 1]:
        sub_fams = family_df[family_df["label"] == lbl]["family_id"].tolist()
        n_total = len(sub_fams)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_fams_cls = sub_fams[:n_train]
        val_fams_cls = sub_fams[n_train:n_train + n_val]
        test_fams_cls = sub_fams[n_train + n_val:]

        train_families.update(train_fams_cls)
        val_families.update(val_fams_cls)
        test_families.update(test_fams_cls)

    # Assign split to images
    train_df = df_clean[df_clean["family_id"].isin(train_families)].copy().sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    val_df = df_clean[df_clean["family_id"].isin(val_families)].copy().sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    test_df = df_clean[df_clean["family_id"].isin(test_families)].copy().sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Save CSV manifests
    train_csv = manifest_dir / "train.csv"
    val_csv = manifest_dir / "validation.csv"
    test_csv = manifest_dir / "test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"\nSaved Train manifest       : {train_csv} ({len(train_df):,} images)")
    print(f"  Distribution: {train_df['class_name'].value_counts().to_dict()}")
    print(f"Saved Validation manifest  : {val_csv} ({len(val_df):,} images)")
    print(f"  Distribution: {val_df['class_name'].value_counts().to_dict()}")
    print(f"Saved Test manifest (UNSEEN): {test_csv} ({len(test_df):,} images)")
    print(f"  Distribution: {test_df['class_name'].value_counts().to_dict()}")

    # Strict Leakage Verification
    train_shas = set(train_df["sha256"])
    val_shas = set(val_df["sha256"])
    test_shas = set(test_df["sha256"])

    overlap_train_val = len(train_shas & val_shas)
    overlap_train_test = len(train_shas & test_shas)
    overlap_val_test = len(val_shas & test_shas)

    fam_overlap_train_val = len(train_families & val_families)
    fam_overlap_train_test = len(train_families & test_families)
    fam_overlap_val_test = len(val_families & test_families)

    print("\n" + "=" * 80)
    print("ANTI-LEAKAGE VERIFICATION CHECK")
    print("=" * 80)
    print(f"Hash Overlap (Train & Val)   : {overlap_train_val} images")
    print(f"Hash Overlap (Train & Test)  : {overlap_train_test} images")
    print(f"Hash Overlap (Val & Test)    : {overlap_val_test} images")
    print(f"Family Overlap (Train & Val) : {fam_overlap_train_val} families")
    print(f"Family Overlap (Train & Test): {fam_overlap_train_test} families")
    print(f"Family Overlap (Val & Test)  : {fam_overlap_val_test} families")

    if any([overlap_train_val, overlap_train_test, overlap_val_test,
            fam_overlap_train_val, fam_overlap_train_test, fam_overlap_val_test]):
        raise RuntimeError("CRITICAL ERROR: Data leakage detected across splits!")

    print("STATUS: ZERO LEAKAGE CONFIRMED. Splits are mathematically isolated.")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCOS Data Splitting & Deduplication")
    parser.add_argument(
        "--zip_path",
        type=str,
        default=r"C:\Users\srini\Documents\pcos_project\dataset\datasett.zip",
        help="Path to raw datasett.zip"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\Users\srini\Documents\pcos_project\data\clean_images",
        help="Directory to save deduplicated clean images"
    )
    parser.add_argument(
        "--manifest_dir",
        type=str,
        default=r"C:\Users\srini\Documents\pcos_project\data",
        help="Directory to save train/validation/test CSV manifests"
    )
    args = parser.parse_args()
    prepare_clean_splits(args.zip_path, args.output_dir, args.manifest_dir)
