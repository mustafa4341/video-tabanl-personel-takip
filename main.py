import cv2
import os
import time
import argparse
from detectors.person_detector import PersonDetector
from detectors.ppe_detector import PPEDetector
from detectors.pose_detector import PoseDetector
from tracking.person_tracker import PersonTracker
from services.violation_service import ViolationService
from services.video_service import VideoWriter
from utils.drawing import drawing_person, draw_osd_panel
from utils.geometry import match_ppe_to_person, StableStateTracker, compute_iou, is_center_in_box


# ── Argüman: video dosyası yolu ──────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Video Tabanlı Personel Takibi ve PPE Kontrol Sistemi"
)
parser.add_argument("--video", required=True, help="İşlenecek video dosyasının yolu")
parser.add_argument("--output", default="outputs/result.mp4", help="Çıktı videosunun yolu")
OVERHEAD_WEIGHTS = "runs/detect/yolo11m_overhead_person/weights/best.pt"
default_model = OVERHEAD_WEIGHTS if os.path.exists(OVERHEAD_WEIGHTS) else "yolo11m.pt"

parser.add_argument("--model", default=default_model, help="Kullanılacak model yolu (Varsayılan: yeni eğitilen tepeden insan modeli)")
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
print(f"Model: {args.model} | FPS: {fps_video:.1f} | Toplam Kare: {total_frames}")

# ── Nesneler ─────────────────────────────────────────────────────────────────
# BoT-SORT ReID + 3 kare filtresi hız kaybı olmadan ID stabilitesi sağlar
detector          = PersonDetector(model_path=args.model, conf_threshold=0.20, imgsz=1024)
ppe_detector      = PPEDetector(model_path="models/best.pt", conf_helmet=0.30, conf_vest=0.35, imgsz=640)
pose_detector     = PoseDetector(model_path="models/yolo11n-pose.pt", conf_threshold=0.30)
tracker           = PersonTracker(min_box_area=100, min_confirm_frames=3)
state_tracker     = StableStateTracker(fps=fps_video, window_sec=2.0, min_obs_sec=1.5, present_ratio=0.75, missing_ratio=0.35)
violation_service = ViolationService(threshold=15)
video_writer      = VideoWriter(output_path=args.output, fps=fps_video)

frame_index = 0

# ── Ekran Penceresi Hazırla (Yeniden boyutlandırılabilir cv2.WINDOW_NORMAL) ────
WINDOW_NAME = "Personel Takip ve PPE Kontrol Sistemi"
has_gui = True
try:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
except Exception:
    has_gui = False

window_initialized = False

# ── Ana Döngü ─────────────────────────────────────────────────────────────────
while True:
    t_start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1
    frame_time = round(frame_index / fps_video, 2)

    # Yüksek çözünürlüklü videoları 1080p sınırına çek (Dikey ve yatay videoları destekler)
    max_dim = max(frame.shape[0], frame.shape[1])
    if max_dim > 1920:
        scale_ratio = 1920.0 / max_dim
        frame = cv2.resize(frame, (int(frame.shape[1] * scale_ratio), int(frame.shape[0] * scale_ratio)))

    # 1. Kişileri BoT-SORT (ReID destekli) ile tespit et ve takip et
    track_results = detector.track(frame, persist=True)

    # 2. PPE ve Pose (Anatomi) tespiti (Her 2 karede bir çalıştırılır)
    if frame_index == 1 or frame_index % 2 == 1 or 'ppe_detections' not in locals():
        ppe_detections = ppe_detector.detect(frame)
        pose_data = pose_detector.predict(frame)

    # 3. Takip sonuçlarını 3 kare onay filtresi, kutu temizliği ve TrackStitcher'dan geçir
    tracked_persons = tracker.update(persons=None, results_with_tracking=track_results, frame_index=frame_index)

    # 4. Her kişi için Pose (Anatomi) ve PPE eşleştir
    for person in tracked_persons:
        pid   = person["tracker_id"]
        pbbox = person["bbox"]

        # Kişiye ait Pose (Keypoint) verisini bul (en yüksek IoU / merkez çakışması)
        best_pose = None
        best_iou = 0.0
        for pdata in pose_data:
            iou = compute_iou(pbbox, pdata["person_box"])
            if iou > best_iou:
                best_iou = iou
                best_pose = pdata

        # Anatomik Keypoint (Pose) ve duruşa duyarlı PPE eşleştirme
        raw_h_state, raw_v_state, max_conf = match_ppe_to_person(ppe_detections, pbbox, pose_info=best_pose)

        # El hareketi / gölge / tespit yokluğunda son bilinen onaylı durumu koruyan Debounce Filtresi
        has_helmet, has_vest = state_tracker.update(pid, raw_h_state, raw_v_state)

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

    # Anlık FPS hesabı (Titremeyi önlemek için Exponential Moving Average yumuşatması)
    proc_time = time.time() - t_start
    fps_raw = 1.0 / proc_time if proc_time > 0 else fps_video
    fps_live = 0.85 * fps_live + 0.15 * fps_raw if 'fps_live' in locals() else fps_raw

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

    # Pencere boyutunu ilk karede ekran yüksekliğine sığacak şekilde ayarla (Siyah barksız)
    if not window_initialized:
        h_cur, w_cur = frame.shape[:2]
        target_h = min(720, h_cur)
        scale_disp = target_h / float(h_cur)
        disp_w = int(w_cur * scale_disp)
        disp_h = target_h
        if has_gui:
            try:
                cv2.resizeWindow(WINDOW_NAME, disp_w, disp_h)
            except Exception:
                has_gui = False
        window_initialized = True

    if has_gui:
        try:
            cv2.imshow(WINDOW_NAME, frame)
            target_delay = max(1, int(1000.0 / fps_video) - int(proc_time * 1000))
            if cv2.waitKey(target_delay) & 0xFF == ord("q"):
                break
        except Exception:
            has_gui = False

# ── Kapat ve Kaydet ───────────────────────────────────────────────────────────
violation_service.save()
video_writer.release()
cap.release()
if has_gui:
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

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