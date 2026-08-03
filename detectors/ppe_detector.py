from ultralytics import YOLO

class PPEDetector:
    """
    Sınıf Bazlı Hassas KKD Tespitçisi.
    - Kask (küçük/parlak): conf_helmet = 0.30
    - Yelek (geniş/mat): conf_vest = 0.35
    """
    def __init__(self, model_path="models/best.pt", conf_helmet=0.30, conf_vest=0.35, imgsz=640):
        self.model = YOLO(model_path)
        self.conf_helmet = conf_helmet
        self.conf_vest = conf_vest
        self.min_conf = min(conf_helmet, conf_vest)
        self.imgsz = imgsz

    def detect(self, frame):
        results = self.model(frame, conf=self.min_conf, imgsz=self.imgsz, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])

                # Sınıf bazlı filtreleme
                if "Helmet" in class_name and confidence < self.conf_helmet:
                    continue
                if "Vest" in class_name and confidence < self.conf_vest:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence
                })

        return detections
