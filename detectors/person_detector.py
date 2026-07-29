from ultralytics import YOLO
import os

# BoT-SORT yaml dosyasının yolu
TRACKER_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "trackers", "custom_botsort.yaml"
)


class PersonDetector:

    def __init__(self, model_path="yolo11m.pt", conf_threshold=0.25, imgsz=1024):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

    def detect(self, frame):
        """Sadece tespit (track olmadan) — yedek amaçlı."""
        results = self.model(
            frame, conf=self.conf_threshold, classes=[0],
            imgsz=self.imgsz, verbose=False
        )

        persons = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                persons.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence
                })
        return persons

    def track(self, frame, persist=True):
        """
        BoT-SORT (ReID destekli) takipli tespit.
        Ultralytics'in yerleşik model.track() API'sini kullanır.
        """
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            classes=[0],
            imgsz=self.imgsz,
            tracker=TRACKER_CONFIG,
            persist=persist,
            verbose=False
        )
        return results