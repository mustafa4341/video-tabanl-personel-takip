import os
import cv2
import shutil
import random
import argparse
from pathlib import Path
from ultralytics import YOLO

def smart_pose_filtered_autolabel_and_finetune(hard_examples_dir="outputs/hard_examples", epochs=15):
    """
    Pose (vücut noktaları) destekli akıllı filtreleme ile zor kareleri temiz bir şekilde etiketler ve fine-tune eder.
    Hatalı / gürültülü etiket oluşturulmasını engeller.
    """
    print("=" * 60)
    print(" 🤖 POSE DESTEKLİ AKILLI ETİKETLEME VE İNCE AYAR (FINE-TUNING)")
    print("=" * 60)

    if not os.path.exists(hard_examples_dir) or len(os.listdir(hard_examples_dir)) == 0:
        print(f"Hata: {hard_examples_dir} klasöründe işlenecek zor kare bulunamadı!")
        return

    dataset_dir = os.path.abspath("datasets/warehouse_ppe")
    img_train_dir = os.path.join(dataset_dir, "images/train")
    img_val_dir   = os.path.join(dataset_dir, "images/val")
    lbl_train_dir = os.path.join(dataset_dir, "labels/train")
    lbl_val_dir   = os.path.join(dataset_dir, "labels/val")

    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"📸 Zor Kare Klasörü : {hard_examples_dir}")
    print(f"📁 Hedef Dataset    : {dataset_dir}")
    print(f"🔥 Epoch Sayısı     : {epochs}")
    print("-" * 60)

    ppe_model = YOLO("models/best.pt")
    pose_model = YOLO("models/yolo11n-pose.pt")

    image_files = [f for f in os.listdir(hard_examples_dir) if f.endswith(('.jpg', '.png'))]
    print(f"🖼️ Bulunan Zor Kare Sayısı: {len(image_files)}")

    labeled_count = 0
    total_boxes = 0
    random.seed(42)

    for img_name in image_files:
        img_path = os.path.join(hard_examples_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        h, w, _ = frame.shape

        # 1. Pose Tespiti (Vücut anahtar noktaları)
        pose_res = pose_model.predict(source=frame, conf=0.30, verbose=False)[0]
        valid_head_boxes = []
        valid_torso_boxes = []

        if pose_res.keypoints is not None and len(pose_res.keypoints) > 0:
            kpts_all = pose_res.keypoints.xy.cpu().numpy()
            confs_all = pose_res.keypoints.conf.cpu().numpy()

            for kpts, kconfs in zip(kpts_all, confs_all):
                # Kafa bölgesi doğrulama (Burun: 0, Kulaklar: 3, 4)
                head_pts = [kpts[i] for i in [0, 3, 4] if i < len(kconfs) and kconfs[i] > 0.40]
                if len(head_pts) > 0:
                    xs = [p[0] for p in head_pts]
                    ys = [p[1] for p in head_pts]
                    valid_head_boxes.append((min(xs), min(ys), max(xs), max(ys)))

                # Gövde bölgesi doğrulama (Omuzlar: 5, 6, Kalça: 11, 12)
                torso_pts = [kpts[i] for i in [5, 6, 11, 12] if i < len(kconfs) and kconfs[i] > 0.40]
                if len(torso_pts) >= 2:
                    xs = [p[0] for p in torso_pts]
                    ys = [p[1] for p in torso_pts]
                    valid_torso_boxes.append((min(xs), min(ys), max(xs), max(ys)))

        # 2. PPE Modeli tahmini (%25+ güven skoru olanlar)
        ppe_results = ppe_model(frame, conf=0.25, verbose=False)[0]
        yolo_lines = []

        if ppe_results.boxes is not None and len(ppe_results.boxes) > 0:
            boxes = ppe_results.boxes
            xywhn = boxes.xywhn.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for i in range(len(cls_ids)):
                cid = cls_ids[i]
                cx, cy, nw, nh = xywhn[i]
                box_xyxy = xyxy[i]

                # Sadece Pose anatomi bölgesiyle çakışan yüksek kaliteli tahminleri ekle
                is_valid = True
                if cid in [0, 1]:  # Helmet / No Helmet -> Head bölgesi doğrulaması
                    if len(valid_head_boxes) > 0:
                        # Kafa yakındaysa kabul et
                        is_valid = any(
                            abs(cx * w - (hb[0] + hb[2]) / 2) < w * 0.15 for hb in valid_head_boxes
                        )
                elif cid in [2, 3]:  # Vest / No Vest -> Torso bölgesi doğrulaması
                    if len(valid_torso_boxes) > 0:
                        # Gövde yakındaysa kabul et
                        is_valid = any(
                            abs(cx * w - (tb[0] + tb[2]) / 2) < w * 0.20 for tb in valid_torso_boxes
                        )

                if is_valid:
                    yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        if len(yolo_lines) > 0:
            base_name = Path(img_name).stem
            is_val = (random.random() < 0.20)

            target_img_dir = img_val_dir if is_val else img_train_dir
            target_lbl_dir = lbl_val_dir if is_val else lbl_train_dir

            shutil.copy(img_path, os.path.join(target_img_dir, f"{base_name}.jpg"))
            with open(os.path.join(target_lbl_dir, f"{base_name}.txt"), "w", encoding="utf-8") as f:
                f.writelines(yolo_lines)

            labeled_count += 1
            total_boxes += len(yolo_lines)

    print(f"✅ Pose Doğrulamalı Etiketlenen Kare Sayısı : {labeled_count}")
    print(f"📦 Kaliteli Oluşturulan Kutu Sayısı       : {total_boxes}")

    # Data YAML
    clean_dataset_dir = dataset_dir.replace("\\", "/")
    yaml_content = f"""path: {clean_dataset_dir}
train: images/train
val: images/val

names:
  0: Helmet
  1: No Helmet
  2: Safety Vests
  3: No Safety Vest
"""
    yaml_path = os.path.join(dataset_dir, "ppe_data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"📄 Yapılandırma Dosyası                     : {yaml_path}")

    # Fine-Tuning Başlat
    print("\n" + "=" * 60)
    print(f" 🚀 GPU Üzerinde Güvenli Fine-Tuning Eğitimi Başlatılıyor ({epochs} Epoch)...")
    print("=" * 60 + "\n")

    train_model = YOLO("models/best.pt")
    train_model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=8,
        lr0=0.001,
        degrees=15.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        name="ppe_finetuned_smart_pose",
        exist_ok=True
    )

    trained_weight = os.path.abspath("runs/detect/ppe_finetuned_smart_pose/weights/best.pt")
    if os.path.exists(trained_weight):
        backup_path = "models/best_backup.pt"
        shutil.copy("models/best.pt", backup_path)
        print(f"💾 Eski temiz model yedeği tazelendi : {backup_path}")

        shutil.copy(trained_weight, "models/best.pt")
        print(f"🏆 Temiz Güncellenmiş Model Yüklendi: models/best.pt")

    print("\n" + "=" * 60)
    print(" 🎉 AKILLI POSE EĞİTİMİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose Destekli Akıllı Etiketleme ve Fine-Tuning")
    parser.add_argument("--dir", default="outputs/hard_examples", help="Zor karelerin bulunduğu klasör")
    parser.add_argument("--epochs", type=int, default=15, help="Eğitim epoch sayısı (Varsayılan: 15)")
    args = parser.parse_args()

    smart_pose_filtered_autolabel_and_finetune(hard_examples_dir=args.dir, epochs=args.epochs)
