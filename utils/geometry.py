from collections import deque


def compute_iou(boxA, boxB):
    """İki kutunun kesişim / birleşim oranını hesaplar."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    intersection = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - intersection

    if union == 0:
        return 0.0

    return intersection / union


def is_center_in_box(small_box, person_box):
    """Küçük nesnenin merkez noktasının kişi kutusu içinde olup olmadığını kontrol eder.
    Marj kişi boyutuna göre dinamik hesaplanır."""
    sx1, sy1, sx2, sy2 = small_box
    px1, py1, px2, py2 = person_box

    cx = (sx1 + sx2) / 2.0
    cy = (sy1 + sy2) / 2.0

    # Marj kişi boyutunun %12'si
    p_w = px2 - px1
    p_h = py2 - py1
    margin_x = max(12, p_w * 0.12)
    margin_y = max(12, p_h * 0.12)

    return (px1 - margin_x <= cx <= px2 + margin_x) and (py1 - margin_y <= cy <= py2 + margin_y)


def is_ppe_belonging_to_person(ppe_box, person_box, ppe_type="helmet"):
    """Baret veya yeleğin kişiye ait olup olmadığını bölgesel olarak doğrular.
    Bölge sınırları kişi boyutuna göre oransal hesaplanır."""
    px1, py1, px2, py2 = person_box
    sx1, sy1, sx2, sy2 = ppe_box
    cx = (sx1 + sx2) / 2.0
    cy = (sy1 + sy2) / 2.0

    p_w = px2 - px1
    p_h = py2 - py1

    # Yatay sınır: kişi kutusu + %18 marj
    margin_x = max(12, p_w * 0.18)
    if not (px1 - margin_x <= cx <= px2 + margin_x):
        return False

    # Dikey sınır: kişi kutusunun üstünden %12 genişletilmiş
    margin_y = max(12, p_h * 0.12)
    if not (py1 - margin_y <= cy <= py2 + margin_y):
        return False

    if ppe_type == "helmet":
        # Baret kişinin üst %48'inde olmalı (baş ve omuz bölgesi, el hareketlerine tolere)
        return cy <= py1 + p_h * 0.48
    elif ppe_type == "vest":
        # Yelek kişinin üst %15 ile alt %15 arası gövde bölgesinde olmalı
        return py1 + p_h * 0.15 <= cy <= py2 - p_h * 0.15
    return True


def match_ppe_to_person(ppe_detections, person_bbox):
    """
    Bir kişi için tüm PPE tespitlerini değerlendirip kask ve yelek durumunu belirler.
    'Helmet' tespiti = kask var, 'No Helmet' tespiti = kask kesinlikle yok.
    Hem pozitif hem negatif sınıfları kullanarak daha doğru sonuç verir.
    """
    has_helmet = False
    has_vest = False
    no_helmet_detected = False
    no_vest_detected = False
    max_conf = 0.0

    for ppe in ppe_detections:
        cname = ppe["class_name"]
        ppe_box = ppe["bbox"]
        conf = ppe["confidence"]

        # Merkez noktası kişi kutusunda mı?
        if not is_center_in_box(ppe_box, person_bbox):
            # IoU ile de kontrol et (küçük kişiler için yedek)
            if compute_iou(person_bbox, ppe_box) < 0.05:
                continue

        # Baret kontrolleri
        if cname == "Helmet" and is_ppe_belonging_to_person(ppe_box, person_bbox, "helmet"):
            has_helmet = True
            max_conf = max(max_conf, conf)
        elif cname == "No Helmet" and is_ppe_belonging_to_person(ppe_box, person_bbox, "helmet"):
            no_helmet_detected = True
            max_conf = max(max_conf, conf)

        # Yelek kontrolleri
        elif cname == "Safety Vests" and is_ppe_belonging_to_person(ppe_box, person_bbox, "vest"):
            has_vest = True
            max_conf = max(max_conf, conf)
        elif cname == "No Safety Vest" and is_ppe_belonging_to_person(ppe_box, person_bbox, "vest"):
            no_vest_detected = True
            max_conf = max(max_conf, conf)

    # Çakışma çözümü: Hem "Helmet" hem "No Helmet" tespiti varsa,
    # pozitif tespite öncelik ver (model onu zaten yüksek conf'la bulmuştur)
    if no_helmet_detected and not has_helmet:
        has_helmet = False
    if no_vest_detected and not has_vest:
        has_vest = False

    return has_helmet, has_vest, max_conf


class PPEHysteresisState:
    """
    Hysteresis (Yapışkan Durum) filtresi.
    - Bir kişinin başında kask bir kez (2 kare) doğrulandığında, el hareketi, gölge veya baş çevirme
      sebebiyle geçici olarak kask tespit edilemese dahi etiket KESİNLİKLE 'Missing' olmaz.
    - Etiketin 'Helmet: Present' durumundan 'Helmet: Missing' durumuna geçmesi için
      kişinin üst üste en az 30 kare (yaklaşık 1 saniye) boyunca kasksız kalması gerekir.
    - Etiketin 'Helmet: Missing' durumundan 'Helmet: Present' durumuna geçmesi için
      üst üste en az 2 kare kask tespit edilmesi gerekir.
    """
    def __init__(self, missing_to_present_frames=2, present_to_missing_frames=30):
        self.missing_to_present_frames = missing_to_present_frames
        self.present_to_missing_frames = present_to_missing_frames
        self.states = {}

    def update(self, tracker_id, raw_helmet, raw_vest):
        if tracker_id not in self.states:
            self.states[tracker_id] = {
                "helmet_state": raw_helmet,
                "helmet_consec_yes": 1 if raw_helmet else 0,
                "helmet_consec_no": 0 if raw_helmet else 1,
                "vest_state": raw_vest,
                "vest_consec_yes": 1 if raw_vest else 0,
                "vest_consec_no": 0 if raw_vest else 1,
            }
            return raw_helmet, raw_vest

        st = self.states[tracker_id]

        # ── KASK HYSTERESIS ──
        if raw_helmet:
            st["helmet_consec_yes"] += 1
            st["helmet_consec_no"] = 0
            if not st["helmet_state"] and st["helmet_consec_yes"] >= self.missing_to_present_frames:
                st["helmet_state"] = True
        else:
            st["helmet_consec_no"] += 1
            st["helmet_consec_yes"] = 0
            if st["helmet_state"] and st["helmet_consec_no"] >= self.present_to_missing_frames:
                st["helmet_state"] = False

        # ── YELEK HYSTERESIS ──
        if raw_vest:
            st["vest_consec_yes"] += 1
            st["vest_consec_no"] = 0
            if not st["vest_state"] and st["vest_consec_yes"] >= self.missing_to_present_frames:
                st["vest_state"] = True
        else:
            st["vest_consec_no"] += 1
            st["vest_consec_yes"] = 0
            if st["vest_state"] and st["vest_consec_no"] >= self.present_to_missing_frames:
                st["vest_state"] = False

        return st["helmet_state"], st["vest_state"]


