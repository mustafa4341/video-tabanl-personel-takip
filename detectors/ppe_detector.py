
from ultralytics import YOLO


class PPEDetector:

    def __init__(self, model_path="models/best.pt"):
        self.model = YOLO(model_path)


    def detect(self, frame):
        results = self.model(frame)

        return results[0]