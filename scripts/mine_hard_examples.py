import cv2
import os
import argparse
from pathlib import Path
from ultralytics import YOLO

def mine_hard_examples(video_path, ppe_model_path="models/best.pt", person_model_path="runs/detect/yolo11m_overhead_person/weights/best.pt", out_dir="outputs/hard_examples", low_conf=0.10, high_conf=0.45, every_n=5):
    """
    Zor Örnek Madenciliği (Hard Example Mining) Script'i.
    Modelin kararsız kaldığı (%10-%45 arası güven skoru ürettiği) veya zorlandığı kareleri otomatik tespit edip kaydeder.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(ppe_model_path):
        ppe_model_path = "models/best.pt"
    
    print(f"🚀 Zor Örnek Madenciliği Başlatılıyor...")
    print(f"📹 Video           : {video_path}")
    print(f"🧠 PPE Modeli      : {ppe_model_path}")
    print(f"🧠 İnsan Modeli    : {person_model_path if os.path.exists(person_model_path) else 'Varsayılan'}")
    print(f"🎯 Kararsız Eşik   : %{int(low_conf*100)} - %{int(high_conf*100)}")
    print(f"📁 Çıktı Klasörü   : {out_dir}")
    print("-" * 60)

    ppe_model = YOLO(ppe_model_path)
    person_model = YOLO(person_model_path) if os.path.exists(person_model_path) else None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Hata: Video açılamadı -> {video_path}")
        return 0

    frame_idx = 0
    saved_count = 0
    video_name = Path(video_path).stem

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if frame_idx % every_n == 0:
            is_hard_example = False
            reasons = []

            # 1. PPE Modeli Kararsızlık Taraması
            ppe_res = ppe_model.predict(source=frame, conf=0.05, verbose=False)[0]
            if ppe_res.boxes is not None and len(ppe_res.boxes) > 0:
                confs = ppe_res.boxes.conf.cpu().numpy()
                for c in confs:
                    if low_conf <= c <= high_conf:
                        is_hard_example = True
                        reasons.append(f"PPE Kararsız Skor (%{int(c*100)})")
                        break

            # 2. İnsan Tespit Modeli Kararsızlık Taraması (Eğer varsa)
            if person_model:
                person_res = person_model.predict(source=frame, conf=0.05, verbose=False)[0]
                if person_res.boxes is not None and len(person_res.boxes) > 0:
                    p_confs = person_res.boxes.conf.cpu().numpy()
                    for pc in p_confs:
                        if low_conf <= pc <= high_conf:
                            is_hard_example = True
                            reasons.append(f"İnsan Kararsız Skor (%{int(pc*100)})")
                            break

            # Eğer zor bir kare tespit edildiyse orijinal kareyi kaydet
            if is_hard_example:
                out_path = os.path.join(out_dir, f"{video_name}_frame{frame_idx:05d}.jpg")
                cv2.imwrite(out_path, frame)
                saved_count += 1
                reason_str = ", ".join(set(reasons))
                print(f"📸 Kare {frame_idx:05d} Kaydedildi -> {out_path} [{reason_str}]")

    cap.release()
    print("=" * 60)
    print(f"✅ İŞLEM TAMAMLANDI!")
    print(f"🖼️ Toplam Kaydedilen Zor Kare Sayısı: {saved_count}")
    print(f"📁 Kayıt Konumu: {out_dir}")
    print(f"💡 İpucu: Bu kareleri AnyLabeling veya Roboflow ile etiketleyip datasetinize ekleyebilirsiniz!")
    print("=" * 60)
    return saved_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zor Örnek Madenciliği (Hard Example Miner)")
    parser.add_argument("--video", required=True, help="Taranacak video dosyasının yolu")
    parser.add_argument("--out", default="outputs/hard_examples", help="Kaydedilecek klasör yolu")
    parser.add_argument("--every-n", type=int, default=5, help="Kaç karede bir taransın (Varsayılan: 5)")
    parser.add_argument("--low", type=float, default=0.10, help="Alt kararsızlık eşiği (Varsayılan: 0.10)")
    parser.add_argument("--high", type=float, default=0.45, help="Üst kararsızlık eşiği (Varsayılan: 0.45)")
    args = parser.parse_args()

    mine_hard_examples(
        video_path=args.video,
        out_dir=args.out,
        every_n=args.every_n,
        low_conf=args.low,
        high_conf=args.high
    )
