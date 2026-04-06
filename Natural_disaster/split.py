# ======================================================
# split.py  —  RESEARCH-GRADE DATASET SPLITTER
#
# Features added vs original:
# ✅ Reproducibility seed (random + numpy)
# ✅ 3-way stratified split  →  train / val / test
# ✅ Skips non-image / corrupt files safely
# ✅ Prints per-class + overall summary table
# ✅ Verifies no image leakage across splits
# ======================================================

import os
import random
import shutil
import numpy as np

# ============================
# CONFIG  —  edit these paths
# ============================

SOURCE_DIR  = r"C:\Users\medeh\Desktop\Natural_disaster\Natural_disaster\disaster_dataset"
OUTPUT_DIR  = r"./final_dataset"

TRAIN_RATIO = 0.70   # 70 %
VAL_RATIO   = 0.15   # 15 %  ← NEW: validation split
TEST_RATIO  = 0.15   # 15 %  (train + val + test must sum to 1.0)

SEED = 42            # for full reproducibility

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')

# ============================
# SEED  —  must be set before
#           any random calls
# ============================

random.seed(SEED)
np.random.seed(SEED)

# ============================
# HELPERS
# ============================

def is_valid_image(path: str) -> bool:
    """Return True only for readable image files."""
    if not path.lower().endswith(IMG_EXTENSIONS):
        return False
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def copy_files(file_list: list, src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    for fname in file_list:
        shutil.copy2(os.path.join(src_dir, fname),
                     os.path.join(dst_dir, fname))


def verify_no_leakage(splits: dict):
    """Assert that no filename appears in more than one split."""
    seen = {}
    for split_name, files in splits.items():
        for f in files:
            if f in seen:
                raise RuntimeError(
                    f"DATA LEAKAGE: '{f}' appears in both "
                    f"'{seen[f]}' and '{split_name}'"
                )
            seen[f] = split_name
    print("  ✅ No data leakage detected across splits.")

# ============================
# MAIN SPLIT FUNCTION
# ============================

def split_dataset():
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, \
        "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0"

    print("=" * 55)
    print("  RESEARCH-GRADE DATASET SPLITTER")
    print(f"  Seed : {SEED}")
    print(f"  Split: {int(TRAIN_RATIO*100)}% train / "
          f"{int(VAL_RATIO*100)}% val / "
          f"{int(TEST_RATIO*100)}% test")
    print("=" * 55)

    classes = sorted([
        d for d in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, d))
    ])

    if not classes:
        raise FileNotFoundError(f"No class folders found in: {SOURCE_DIR}")

    summary_rows = []
    total_train = total_val = total_test = 0

    for cls in classes:
        class_src = os.path.join(SOURCE_DIR, cls)

        # ---- collect valid images only ----
        all_files = os.listdir(class_src)
        images = [f for f in all_files
                  if is_valid_image(os.path.join(class_src, f))]

        skipped = len(all_files) - len(images)
        n = len(images)

        if n == 0:
            print(f"\n⚠️  Skipping '{cls}': no valid images found.")
            continue

        # ---- stratified shuffle ----
        random.shuffle(images)

        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        # give any rounding remainder to test
        n_test  = n - n_train - n_val

        train_imgs = images[:n_train]
        val_imgs   = images[n_train : n_train + n_val]
        test_imgs  = images[n_train + n_val :]

        # ---- leakage check per class ----
        verify_no_leakage({
            "train": train_imgs,
            "val"  : val_imgs,
            "test" : test_imgs,
        })

        # ---- copy files ----
        copy_files(train_imgs, class_src,
                   os.path.join(OUTPUT_DIR, "train", cls))
        copy_files(val_imgs,   class_src,
                   os.path.join(OUTPUT_DIR, "val",   cls))
        copy_files(test_imgs,  class_src,
                   os.path.join(OUTPUT_DIR, "test",  cls))

        summary_rows.append((cls, n, n_train, n_val, n_test, skipped))
        total_train += n_train
        total_val   += n_val
        total_test  += n_test

    # ============================
    # SUMMARY TABLE
    # ============================

    col_w = max(len(c) for c, *_ in summary_rows) + 2
    header = (f"{'Class':<{col_w}}{'Total':>8}"
              f"{'Train':>8}{'Val':>8}{'Test':>8}{'Skipped':>9}")
    sep    = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for cls, n, nt, nv, ns, sk in summary_rows:
        print(f"{cls:<{col_w}}{n:>8}{nt:>8}{nv:>8}{ns:>8}{sk:>9}")

    print(sep)
    grand = total_train + total_val + total_test
    print(f"{'TOTAL':<{col_w}}{grand:>8}"
          f"{total_train:>8}{total_val:>8}{total_test:>8}")
    print(sep)

    print(f"\n🎉 Split complete!  Output → {os.path.abspath(OUTPUT_DIR)}")
    print(f"   final_dataset/train/  ({total_train} images)")
    print(f"   final_dataset/val/    ({total_val} images)")
    print(f"   final_dataset/test/   ({total_test} images)")


# ============================
# ENTRY POINT
# ============================

if __name__ == "__main__":
    split_dataset()