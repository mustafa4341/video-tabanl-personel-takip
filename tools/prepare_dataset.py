import os
import shutil

# Resolve base paths dynamically so script works from any working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Look for construction_safety dataset in project root datasets folder first, fallback to relative
source_candidates = [
    os.path.join(PROJECT_ROOT, "datasets", "construction_safety"),
    os.path.join(SCRIPT_DIR, "..", "datasets", "construction_safety"),
    "../datasets/construction_safety"
]

SOURCE = None
for candidate in source_candidates:
    if os.path.exists(candidate):
        SOURCE = candidate
        break

if SOURCE is None:
    SOURCE = os.path.join(PROJECT_ROOT, "datasets", "construction_safety")

TARGET = os.path.join(PROJECT_ROOT, "datasets", "ppe_dataset")

CLASS_MAP = {
    0: 0,  # Hardhat -> Helmet
    2: 1,  # NO-Hardhat -> No Helmet
    7: 2,  # Safety Vest -> Safety Vest
    4: 3   # NO-Safety Vest -> No Safety Vest
}


splits = ["train", "valid", "test"]


for split in splits:

    src_img = os.path.join(SOURCE, split, "images")
    src_lbl = os.path.join(SOURCE, split, "labels")

    dst_img = os.path.join(TARGET, split, "images")
    dst_lbl = os.path.join(TARGET, split, "labels")

    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)

    if not os.path.exists(src_lbl):
        print(f"Warning: Source labels directory not found: {src_lbl}")
        continue

    copied_count = 0
    for label_file in os.listdir(src_lbl):

        if not label_file.endswith(".txt"):
            continue

        base_name = os.path.splitext(label_file)[0]
        new_lines = []

        lbl_path = os.path.join(src_lbl, label_file)
        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            try:
                cls = int(parts[0])
            except ValueError:
                continue

            if cls in CLASS_MAP:
                parts[0] = str(CLASS_MAP[cls])
                new_lines.append(" ".join(parts))

        # Eğer dosyada PPE etiketi varsa resmi ve etiketi kopyala
        if len(new_lines) > 0:
            img_name = None
            for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
                candidate_img = base_name + ext
                if os.path.exists(os.path.join(src_img, candidate_img)):
                    img_name = candidate_img
                    break

            if img_name and os.path.exists(os.path.join(src_img, img_name)):
                shutil.copy(
                    os.path.join(src_img, img_name),
                    os.path.join(dst_img, img_name)
                )

                with open(os.path.join(dst_lbl, label_file), "w", encoding="utf-8") as f:
                    f.write("\n".join(new_lines))

                copied_count += 1
            else:
                print(f"Warning: Image file for label {label_file} not found in {src_img}")

    print(f"[{split}] Processed and copied {copied_count} files to {TARGET}")

# YOLO konfigürasyon dosyası (data.yaml) oluştur
yaml_path = os.path.join(TARGET, "data.yaml")
yaml_content = f"""path: {TARGET.replace('\\', '/')}
train: train/images
val: valid/images
test: test/images

nc: 4
names:
  0: Helmet
  1: No Helmet
  2: Safety Vest
  3: No Safety Vest
"""

with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"\n[OK] YOLO konfigürasyon dosyası oluşturuldu: {yaml_path}")
