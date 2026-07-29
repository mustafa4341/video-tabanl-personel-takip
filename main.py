import cv2
import time
import argparse
from detectors.person_detector import PersonDetector
from detectors.ppe_detector import PPEDetector
from tracking.person_tracker import PersonTracker
from services.violation_service import ViolationService
from services.video_service import VideoWriter
from utils.drawing import drawing_person, draw_osd_panel
from utils.geometry import match_ppe_to_person, PPEHysteresisState


# ── Argüman: video dosyası yolu ──────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Video Tabanlı Personel Takibi ve PPE Kontrol Sistemi"
)
parser.add_argument("--video", required=True, help="İşlenecek video dosyasının yolu")
parser.add_argument("--output", default="outputs/result.mp4", help="Çıktı videosunun yolu")
args = parser.parse_args()

# ── Video Aç ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(f"Hata: Video açılamadı -> {args.video}")
    exit(1)

fps_video = cap.get(cv2.CAP_PROP_FPS)
if fps_video <= 0 or fps_video != fps_video:  # NaN veya geçersiz ise
    fps_video = 30.0

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {args.video}")
print(f"FPS: {fps_video:.1f} | Toplam Kare: {total_frames}")

# ── Nesneler ─────────────────────────────────────────────────────────────────
# BoT-SORT ReID + 3 kare filtresi hız kaybı olmadan ID stabilitesi sağlar
detector          = PersonDetector(model_path="yolo11m.pt", conf_threshold=0.25, imgsz=640)
ppe_detector      = PPEDetector(model_path="models/best.pt", conf_threshold=0.25, imgsz=640)
tracker           = PersonTracker(min_box_area=225, min_confirm_frames=3)
ppe_state         = PPEHysteresisState(missing_to_present_frames=2, present_to_missing_frames=30)
violation_service = ViolationService(threshold=15)
video_writer      = VideoWriter(output_path=args.output, fps=fps_video)

frame_index = 0

# ── Ana Döngü ─────────────────────────────────────────────────────────────────
while True:
    t_start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1
    frame_time = round(frame_index / fps_video, 2)

    # 1. Kişileri BoT-SORT (ReID destekli) ile tespit et ve takip et
    track_results = detector.track(frame, persist=True)

    # 2. PPE'leri tespit et
    ppe_detections = ppe_detector.detect(frame)

    # 3. Takip sonuçlarını 3 kare onay filtresi ve kutu temizliğinden geçir
    tracked_persons = tracker.update(persons=None, results_with_tracking=track_results)

    # 4. Her kişi için PPE eşleştir ve ihlal kontrolü yap
    for person in tracked_persons:
        pid   = person["tracker_id"]
        pbbox = person["bbox"]

        # Helmet, No Helmet, Safety Vests, No Safety Vest sınıflarını birlikte değerlendir
        raw_helmet, raw_vest, max_conf = match_ppe_to_person(ppe_detections, pbbox)

        # Titremeyi %100 engelleyen Hysteresis (Yapışkan Durum) filtresi uygula
        has_helmet, has_vest = ppe_state.update(pid, raw_helmet, raw_vest)

        person["has_helmet"] = has_helmet
        person["has_vest"]   = has_vest

        # İhlal servisini güncelle ve kareyi aktar (ihlal anında resim kaydı için)
        violation_service.update(
            tracker_id=pid,
            has_helmet=has_helmet,
            has_vest=has_vest,
            frame_time=frame_time,
            frame=frame,
            confidence=max_conf
        )

    # Anlık FPS hesabı
    proc_time = time.time() - t_start
    fps_live = 1.0 / proc_time if proc_time > 0 else fps_video

    # 5. Çiz ve göster
    frame = drawing_person(frame, tracked_persons)
    frame = draw_osd_panel(
        frame=frame,
        detected_count=len(tracked_persons),
        helmet_violations=violation_service.helmet_violation_count,
        vest_violations=violation_service.vest_violation_count,
        fps=fps_live
    )

    video_writer.write(frame)

    display_frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
    cv2.imshow("Personel Takip ve PPE Kontrol Sistemi", display_frame)

    target_delay = max(1, int(1000.0 / fps_video) - int(proc_time * 1000))
    if cv2.waitKey(target_delay) & 0xFF == ord("q"):
        break

# ── Kapat ve Kaydet ───────────────────────────────────────────────────────────
violation_service.save()
video_writer.release()
cap.release()
cv2.destroyAllWindows()

# ── Metrik Raporu ─────────────────────────────────────────────────────────────
unique_ids = tracker.get_unique_id_count()
print(f"\n{'='*50}")
print(f"  RAPOR")
print(f"{'='*50}")
print(f"  İşlenen Kare Sayısı    : {frame_index}")
print(f"  Toplam Benzersiz ID    : {unique_ids}")
print(f"  Toplam Kask İhlali     : {violation_service.helmet_violation_count}")
print(f"  Toplam Yelek İhlali    : {violation_service.vest_violation_count}")
print(f"  Çıktı Videosu          : {args.output}")
print(f"  İhlal Kayıtları        : outputs/violations.json")
print(f"{'='*50}")