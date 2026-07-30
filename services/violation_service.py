import json
import os
import cv2


class ViolationManager:
    """
    Ekran görüntüsü ve ihlal kaydı senkronizasyonu sağlayan hafıza yöneticisi.
    - 3 saniye içinde aynı kişiden gelen mükerrer ihlallerde diskte ekstra resim oluşturmaz.
    - İhlalin ilk görüldüğü ve son görüldüğü zamanları ve tekrar sayısını (occurrence_count) takip eder.
    """
    def __init__(self, time_gap=3.0):
        self.time_gap = time_gap
        self.recent_violations = []  # [{stable_id, violation, first_time, last_time, count, confidence, img_path}]

    def should_log_and_screenshot(self, stable_id, violation_type, frame_time, frame=None, confidence=0.88):
        """
        İhlal tetiklendiğinde çağrılır.
        - True dönerse ekran görüntüsü alınır ve kayda eklenir.
        - False dönerse 3 saniye içinde tekrarlandığı için yok sayılır (disk tasarrufu).
        """
        for v in self.recent_violations:
            if v["stable_id"] == stable_id and v["violation"] == violation_type:
                if frame_time - v["last_time_sec"] <= self.time_gap:
                    v["last_time_sec"] = frame_time
                    v["last_time_str"] = self._format_time(frame_time)
                    v["count"] += 1
                    return False  # Ekran görüntüsü ALMA, JSON'a YENİ KAYIT EKLEME

        # Yeni ihlal veya 3 saniyeden uzun süre geçti
        time_str = self._format_time(frame_time)
        time_clean = time_str.replace(":", "-")
        img_path = f"outputs/violations/id_{stable_id}_{violation_type}_{time_clean}.jpg"

        if frame is not None:
            cv2.imwrite(img_path, frame)

        self.recent_violations.append({
            "stable_id": stable_id,
            "violation": violation_type,
            "first_time_sec": frame_time,
            "first_time_str": time_str,
            "last_time_sec": frame_time,
            "last_time_str": time_str,
            "count": 1,
            "confidence": round(float(confidence), 2),
            "image_path": img_path
        })
        return True

    def _format_time(self, seconds):
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def get_final_violations(self):
        """Video bitiminde JSON'a yazılacak temiz listeyi döndürür."""
        return [{
            "person_id": v["stable_id"],
            "violation_type": v["violation"],
            "first_detected": v["first_time_str"],
            "last_detected": v["last_time_str"],
            "occurrence_count": v["count"],
            "confidence": v["confidence"],
            "image_path": v["image_path"]
        } for v in self.recent_violations]


class ViolationService:
    def __init__(self, threshold=15, time_gap=3.0):
        self.threshold = threshold
        self.counters = {}
        self.manager = ViolationManager(time_gap=time_gap)
        self.helmet_violation_count = 0
        self.vest_violation_count = 0

        os.makedirs("outputs/violations", exist_ok=True)

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

        # Kask ihlali kontrolü (15 kare eşiği)
        if person_data["helmet_missing"] >= self.threshold:
            person_data["helmet_violated"] = True
            is_new = self.manager.should_log_and_screenshot(
                stable_id=tracker_id,
                violation_type="helmet_missing",
                frame_time=frame_time,
                frame=frame,
                confidence=confidence
            )
            if is_new:
                self.helmet_violation_count += 1

        # Yelek ihlali kontrolü (15 kare eşiği)
        if person_data["vest_missing"] >= self.threshold:
            person_data["vest_violated"] = True
            is_new = self.manager.should_log_and_screenshot(
                stable_id=tracker_id,
                violation_type="vest_missing",
                frame_time=frame_time,
                frame=frame,
                confidence=confidence
            )
            if is_new:
                self.vest_violation_count += 1

    def is_person_in_violation(self, tracker_id):
        """Kişinin anlık ihlal durumunda olup olmadığını söyler."""
        if tracker_id not in self.counters:
            return False
        c = self.counters[tracker_id]
        return c.get("helmet_violated", False) or c.get("vest_violated", False)

    def save(self):
        output_file = "outputs/violations.json"
        final_list = self.manager.get_final_violations()
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_list, f, indent=4, ensure_ascii=False)
        print(f"Toplam {len(final_list)} benzersiz ihlal grubu kaydedildi -> {output_file}")


        
        