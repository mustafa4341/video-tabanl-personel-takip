import os
import glob
import cv2
import numpy as np

def crop_and_upscale_dataset(source_dir, output_dir, crop_scale=1.8, min_box_size=40):
    """
    Uzakta kalan / küçük nesneli işçi görsellerini otomatik kırpıp (crop) 
    büyüterek (upscale) sentetik yakın eğitim verisi üreten script.
    """
    images_dir = os.path.join(source_dir, "images")
    labels_dir = os.path.join(source_dir, "labels")

    out_images = os.path.join(output_dir, "images")
    out_labels = os.path.join(output_dir, "labels")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    image_paths = glob.glob(os.path.join(images_dir, "*.jpg")) + glob.glob(os.path.join(images_dir, "*.png"))
    print(f"📦 Toplam {len(image_paths)} görsel işleniyor...")

    created_count = 0

    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{base_name}.txt")

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip().split() for line in f if line.strip()]

        if not lines:
            continue

        # Küçük nesneleri filtrele ve kırpma alanları oluştur
        for idx, line in enumerate(lines):
            cls_id = line[0]
            cx, cy, bw, bh = map(float, line[1:5])

            box_w_px = bw * w
            box_h_px = bh * h

            # Küçük veya orta boy nesneler için crop penceresi oluştur
            crop_w = int(box_w_px * crop_scale)
            crop_h = int(box_h_px * crop_scale)

            px_cx = int(cx * w)
            px_cy = int(cy * h)

            x1 = max(0, px_cx - crop_w // 2)
            y1 = max(0, px_cy - crop_h // 2)
            x2 = min(w, px_cx + crop_w // 2)
            y2 = min(h, px_cy + crop_h // 2)

            crop_img = img[y1:y2, x1:x2]
            if crop_img.shape[0] < 30 or crop_img.shape[1] < 30:
                continue

            # Crop görselini standart 640x640 boyutuna büyüt (Upscale)
            upscaled = cv2.resize(crop_img, (640, 640), interpolation=cv2.INTER_CUBIC)

            # Etiket koordinatlarını kırpılmış alana göre yeniden hesapla
            crop_box_w = x2 - x1
            crop_box_h = y2 - y1

            new_lines = []
            for item in lines:
                c_cls = item[0]
                icx, icy, ibw, ibh = map(float, item[1:5])

                icx_px = icx * w
                icy_px = icy * h
                ibw_px = ibw * w
                ibh_px = ibh * h

                # Kırpılan kutunun içine düşüp düşmediğini kontrol et
                if x1 <= icx_px <= x2 and y1 <= icy_px <= y2:
                    new_cx = (icx_px - x1) / crop_box_w
                    new_cy = (icy_px - y1) / crop_box_h
                    new_bw = ibw_px / crop_box_w
                    new_bh = ibh_px / crop_box_h

                    new_lines.append(f"{c_cls} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}")

            if new_lines:
                out_img_name = f"{base_name}_crop_{idx}.jpg"
                out_lbl_name = f"{base_name}_crop_{idx}.txt"

                cv2.imwrite(os.path.join(out_images, out_img_name), upscaled)
                with open(os.path.join(out_labels, out_lbl_name), "w", encoding="utf-8") as lf:
                    lf.write("\n".join(new_lines))

                created_count += 1

    print(f"✅ İşlem tamamlandı! Toplam {created_count} adet sentetik kırpılmış & büyütülmüş görsel üretildi -> {output_dir}")

if __name__ == "__main__":
    dataset_dir = "datasets/master_ppe_combined/train"
    out_dir = "datasets/master_ppe_combined/crop_augmented"
    if os.path.exists(dataset_dir):
        crop_and_upscale_dataset(dataset_dir, out_dir)
