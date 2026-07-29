import json
import os
import cv2

class ViolationService:
    def __init__(self, threshold=15):
        self.threshold = threshold
        self.counters = {}
        self.violations = []
        self.reported_violations = set()  # (tracker_id, violation_type)
        self.helmet_violation_count = 0
        self.vest_violation_count = 0
        
        os.makedirs("outputs/violations", exist_ok=True)

    def _format_time(self, seconds):
        """Saniye cinsinden süreyi 'HH:MM:SS' dizesine dönüştürür."""
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def update(self, tracker_id, has_helmet, has_vest, frame_time, frame=None, confidence=0.88):
        if tracker_id not in self.counters:
            self.counters[tracker_id] = {
                "helmet_missing": 0,
                "vest_missing": 0,
                "helmet_violated": False,
                "vest_violated": False
            }

        person_data = self.counters[tracker_id]

        # Kask takibi
        if not has_helmet:
            person_data["helmet_missing"] += 1
        else:
            person_data["helmet_missing"] = 0
            person_data["helmet_violated"] = False

        # Yelek takibi
        if not has_vest:
            person_data["vest_missing"] += 1
        else:
            person_data["vest_missing"] = 0
            person_data["vest_violated"] = False

        time_str = self._format_time(frame_time)

        # Kask ihlali kontrolü
        if person_data["helmet_missing"] >= self.threshold:
            person_data["helmet_violated"] = True
            key = (tracker_id, "helmet_missing")
            if key not in self.reported_violations:
                self.reported_violations.add(key)
                self.helmet_violation_count += 1

                img_path = f"outputs/violations/id_{tracker_id}_helmet.jpg"
                if frame is not None:
                    cv2.imwrite(img_path, frame)

                self.violations.append({
                    "person_id": tracker_id,
                    "violation": "helmet_missing",
                    "video_time": time_str,
                    "confidence": round(float(confidence), 2),
                    "image_path": img_path
                })

        # Yelek ihlali kontrolü
        if person_data["vest_missing"] >= self.threshold:
            person_data["vest_violated"] = True
            key = (tracker_id, "vest_missing")
            if key not in self.reported_violations:
                self.reported_violations.add(key)
                self.vest_violation_count += 1

                img_path = f"outputs/violations/id_{tracker_id}_vest.jpg"
                if frame is not None:
                    cv2.imwrite(img_path, frame)

                self.violations.append({
                    "person_id": tracker_id,
                    "violation": "vest_missing",
                    "video_time": time_str,
                    "confidence": round(float(confidence), 2),
                    "image_path": img_path
                })

    def is_person_in_violation(self, tracker_id):
        """Kişinin anlık ihlal durumunda (baret veya yelek eksik) olup olmadığını söyler."""
        if tracker_id not in self.counters:
            return False
        c = self.counters[tracker_id]
        return c.get("helmet_violated", False) or c.get("vest_violated", False)

    def save(self):
        output_file = "outputs/violations.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.violations, f, indent=4, ensure_ascii=False)
        print(f"Toplam {len(self.violations)} ihlal kaydedildi -> {output_file}")

        
        