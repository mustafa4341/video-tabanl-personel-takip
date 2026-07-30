import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class TrackStitcher:
    """
    Ham BoT-SORT ID'lerini (raw_id) kararlı personel ID'lerine (stable_id) bağlayan akıllı dikiş katmanı.

    - Çözünürlük ve zamandan bağımsız hareket toleransı: Geçen kare sayısı arttıkça mesafe toleransı
      (dynamic_max_dist = max(120, w * 2.5 + frame_diff * 4.5)) oransal olarak genişler.
    - Esnek boyut değişimi: Kameraya yaklaşan/uzaklaşan işçilerin boyut değişimlerine (max_size_ratio = 4.0) tolerans gösterir.
    - Maliyet Tabanlı Eşleştirme (Cost-based matching): En düşük maliyetli geçmiş kayıtla dikiş yapar.
    """
    def __init__(self, max_gap_frames=120, max_size_ratio=4.0):
        self.max_gap_frames = max_gap_frames
        self.max_size_ratio = max_size_ratio

        self.raw_to_stable = {}         # raw_id -> stable_id
        self.stable_history = {}        # stable_id -> {"center", "size", "last_frame", "raw_id"}
        self.next_stable_id = 1
        self.confirmed_stable_ids = set()

    def _center_size(self, box):
        x1, y1, x2, y2 = box
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        return center, (w, h)

    def _cleanup_expired(self, current_frame):
        """Süresi geçen (120 kare) eski kayıtları bellekten temizler (RAM leak önleme)."""
        expired_raws = [
            rid for rid, sid in list(self.raw_to_stable.items())
            if sid in self.stable_history and (current_frame - self.stable_history[sid]["last_frame"] > self.max_gap_frames)
        ]
        for rid in expired_raws:
            del self.raw_to_stable[rid]

    def resolve(self, raw_id, box, current_frame):
        self._cleanup_expired(current_frame)
        center, (w, h) = self._center_size(box)
        area = w * h

        # 1. BoT-SORT bu raw_id'yi zaten takip ediyorsa -> Mevcut stable_id'yi koru ve güncelle
        if raw_id in self.raw_to_stable:
            sid = self.raw_to_stable[raw_id]
            self.stable_history[sid] = {
                "center": center,
                "size": (w, h),
                "last_frame": current_frame,
                "raw_id": raw_id
            }
            return sid

        # 2. Yeni bir raw_id oluşmuş: Maliyet tabanlı dikiş (stitching) yap
        best_sid = None
        best_cost = float("inf")

        for sid, info in list(self.stable_history.items()):
            frame_diff = current_frame - info["last_frame"]
            if frame_diff > self.max_gap_frames or frame_diff <= 0:
                continue

            dist = np.linalg.norm(center - info["center"])
            prev_area = info["size"][0] * info["size"][1]
            size_ratio = max(area, prev_area) / max(min(area, prev_area), 1.0)

            # Geçen zamana göre dinamik hareket toleransı (işçi yürüse dahi yakalar)
            dynamic_max_dist = max(120.0, w * 2.5 + frame_diff * 4.5)

            if dist < dynamic_max_dist and size_ratio < self.max_size_ratio:
                # Eşleşme Maliyeti: Mesafe Oranı + 0.2 * Boyut Oranı
                cost = (dist / dynamic_max_dist) + 0.2 * (size_ratio / self.max_size_ratio)
                if cost < best_cost and cost < 1.0:
                    best_cost = cost
                    best_sid = sid

        if best_sid is not None:
            # En uygun eski stable_id'ye dikiş yap!
            stable_id = best_sid
        else:
            # Gerçekten yeni bir kişi girmiş -> yeni stable_id ver
            stable_id = self.next_stable_id
            self.next_stable_id += 1

        # Haritayı güncelle
        self.raw_to_stable[raw_id] = stable_id
        self.stable_history[stable_id] = {
            "center": center,
            "size": (w, h),
            "last_frame": current_frame,
            "raw_id": raw_id
        }
        return stable_id


class PersonTracker:
    """
    Ultralytics BoT-SORT (ReID destekli) + TrackStitcher tabanlı kişi takipçisi.
    - 3 kare onay filtresi: Tek karelik gürültü tespitleri yeni ID oluşturamaz.
    - Kutu boyut/oran filtresi: Anlamsız küçük veya garip kutular elenir.
    - TrackStitcher: Ham ID değişse bile aynı kişiyi kararlı 'stable_id'ye bağlar.
    """

    def __init__(self, min_box_area=225, min_confirm_frames=3, max_gap_frames=120):
        self.min_box_area = min_box_area
        self.min_confirm_frames = min_confirm_frames
        self.track_seen_count = {}        # raw_id -> art arda görüldüğü kare sayısı
        self.stitcher = TrackStitcher(max_gap_frames=max_gap_frames)

    def filter_detections(self, persons):
        """Fiziksel olarak anlamsız kutuları eler."""
        filtered = []
        for p in persons:
            x1, y1, x2, y2 = p["bbox"]
            w = x2 - x1
            h = y2 - y1
            area = w * h

            if area < self.min_box_area:
                continue

            if h > 0:
                aspect = w / h
                if aspect > 3.0 or aspect < 0.15:
                    continue

            filtered.append(p)
        return filtered

    def update(self, persons, results_with_tracking, frame_index=0):
        """
        Ultralytics model.track() çıktısını alır, 3 kare onay ve TrackStitcher ile bağlar.
        """
        if results_with_tracking is None or len(results_with_tracking) == 0:
            return []

        result_obj = results_with_tracking[0]

        if result_obj.boxes is None or result_obj.boxes.id is None:
            return []

        boxes = result_obj.boxes
        tracked_ids = boxes.id.int().cpu().numpy()
        tracked_xyxy = boxes.xyxy.cpu().numpy()
        tracked_conf = boxes.conf.cpu().numpy()

        seen_this_frame = set()
        raw_results = []

        for i in range(len(tracked_ids)):
            raw_id = int(tracked_ids[i])
            x1, y1, x2, y2 = map(int, tracked_xyxy[i])
            conf = float(tracked_conf[i])
            w = x2 - x1
            h = y2 - y1

            # Kutu boyut/oran filtresi
            if w * h < self.min_box_area:
                continue
            if h > 0 and (w / h > 3.0 or w / h < 0.15):
                continue

            seen_this_frame.add(raw_id)

            # Sayacı artır
            if raw_id in self.track_seen_count:
                self.track_seen_count[raw_id] += 1
            else:
                self.track_seen_count[raw_id] = 1

            raw_results.append({
                "raw_id": raw_id,
                "bbox": (x1, y1, x2, y2),
                "confidence": conf
            })

        # 3 kare onay filtresi ve TrackStitcher resolve
        confirmed = []
        for r in raw_results:
            rid = r["raw_id"]
            if self.track_seen_count.get(rid, 0) >= self.min_confirm_frames:
                box = r["bbox"]
                stable_id = self.stitcher.resolve(rid, box, frame_index)
                self.stitcher.confirmed_stable_ids.add(stable_id)
                confirmed.append({
                    "tracker_id": stable_id,
                    "bbox": box,
                    "confidence": r["confidence"]
                })

        return confirmed

    def get_unique_id_count(self):
        """Video boyunca onaylanan kararlı benzersiz personel ID sayısını döndürür."""
        return len(self.stitcher.confirmed_stable_ids)




            
        
        
