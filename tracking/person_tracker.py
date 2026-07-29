import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# BoT-SORT yaml dosyasının yolu
TRACKER_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "trackers", "custom_botsort.yaml"
)


class PersonTracker:
    """
    Ultralytics BoT-SORT (ReID destekli) tabanlı kişi takipçisi.
    - 3 kare onay filtresi: Tek karelik gürültü tespitleri yeni ID oluşturamaz.
    - Kutu boyut/oran filtresi: Anlamsız küçük veya garip kutular elenir.
    - ReID: Engel arkasından çıkan kişi eski ID'siyle tanınır.
    """

    def __init__(self, min_box_area=225, min_confirm_frames=3):
        self.min_box_area = min_box_area        # 15x15 px minimum alan
        self.min_confirm_frames = min_confirm_frames
        self.track_seen_count = {}               # tracker_id -> art arda görüldüğü kare sayısı
        self.all_seen_ids = set()                # Tüm benzersiz ID'lerin kaydı (metrik için)

    def filter_detections(self, persons):
        """Fiziksel olarak anlamsız kutuları eler."""
        filtered = []
        for p in persons:
            x1, y1, x2, y2 = p["bbox"]
            w = x2 - x1
            h = y2 - y1
            area = w * h

            # Çok küçük kutuları ele
            if area < self.min_box_area:
                continue

            # Anormal en-boy oranını ele (kişi dikdörtgeni genelde 0.2 - 2.5 arası)
            if h > 0:
                aspect = w / h
                if aspect > 3.0 or aspect < 0.15:
                    continue

            filtered.append(p)
        return filtered

    def update(self, persons, results_with_tracking):
        """
        Ultralytics model.track() çıktısını alır ve 3 kare onay filtresiyle süzer.
        
        Args:
            persons: Ham tespit listesi (filter için)
            results_with_tracking: model.track() sonucu (tracked boxes)
        
        Returns:
            Onaylanmış (3+ kare) takip sonuçları listesi
        """
        if results_with_tracking is None or len(results_with_tracking) == 0:
            # Görünmeyen track'lerin sayaçlarını sıfırla
            self._decay_counts()
            return []

        result_obj = results_with_tracking[0]

        # Takip sonucu yoksa
        if result_obj.boxes is None or result_obj.boxes.id is None:
            self._decay_counts()
            return []

        boxes = result_obj.boxes
        tracked_ids = boxes.id.int().cpu().numpy()
        tracked_xyxy = boxes.xyxy.cpu().numpy()
        tracked_conf = boxes.conf.cpu().numpy()

        # Şu an görünen ID'leri güncelle
        seen_this_frame = set()
        raw_results = []

        for i in range(len(tracked_ids)):
            tid = int(tracked_ids[i])
            x1, y1, x2, y2 = map(int, tracked_xyxy[i])
            conf = float(tracked_conf[i])
            w = x2 - x1
            h = y2 - y1

            # Kutu boyut/oran filtresi
            if w * h < self.min_box_area:
                continue
            if h > 0 and (w / h > 3.0 or w / h < 0.15):
                continue

            seen_this_frame.add(tid)

            # Sayacı artır
            if tid in self.track_seen_count:
                self.track_seen_count[tid] += 1
            else:
                self.track_seen_count[tid] = 1

            raw_results.append({
                "tracker_id": tid,
                "bbox": (x1, y1, x2, y2),
                "confidence": conf
            })

        # Bu karede görünmeyen ID'lerin sayaçlarını kademeli düşür (hemen 0 yapma, 1 karelik kaçırmalarda insan kaybolmasın)
        lost_ids = [k for k in self.track_seen_count if k not in seen_this_frame]
        for lid in lost_ids:
            self.track_seen_count[lid] = max(0, self.track_seen_count[lid] - 1)

        # 3 kare onay filtresi: Sadece min_confirm_frames kadar art arda görülenleri döndür
        confirmed = []
        for r in raw_results:
            tid = r["tracker_id"]
            if self.track_seen_count.get(tid, 0) >= self.min_confirm_frames:
                confirmed.append(r)
                self.all_seen_ids.add(tid)  # Sadece onaylı ID'leri say

        return confirmed

    def _decay_counts(self):
        """Hiçbir tespit yokken tüm sayaçları sıfırla."""
        for k in self.track_seen_count:
            self.track_seen_count[k] = 0

    def get_unique_id_count(self):
        """Video boyunca görülen toplam benzersiz ID sayısını döndürür."""
        return len(self.all_seen_ids)

            
        
        
