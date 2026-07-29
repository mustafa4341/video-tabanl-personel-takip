from ultralytics import YOLO

class PPEDetector:
    def __init__(self, model_path="models/best.pt", conf_threshold=0.25, imgsz=1024):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

    def detect(self, frame):
        results = self.model(frame, conf=self.conf_threshold, imgsz=self.imgsz, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:

                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence
                })

        return detections
