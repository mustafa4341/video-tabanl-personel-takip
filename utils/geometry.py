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

    p_w = px2 - px1
    p_h = py2 - py1
    margin_x = max(15, p_w * 0.15)
    margin_y = max(15, p_h * 0.15)

    return (px1 - margin_x <= cx <= px2 + margin_x) and (py1 - margin_y <= cy <= py2 + margin_y)


def is_ppe_belonging_to_person(ppe_box, person_box, ppe_type="helmet"):
    """
    Baret veya yeleğin kişiye ait olup olmadığını duruşa (dik/eğilmiş) göre bölgesel doğrular.
    Kişi eğildiğinde (p_w > p_h * 1.1) baş bölgesi kutunun yatay uçlarında kalabilir.
    """
    px1, py1, px2, py2 = person_box
    sx1, sy1, sx2, sy2 = ppe_box
    cx = (sx1 + sx2) / 2.0
    cy = (sy1 + sy2) / 2.0

    p_w = px2 - px1
    p_h = py2 - py1

    margin_x = max(15, p_w * 0.20)
    if not (px1 - margin_x <= cx <= px2 + margin_x):
        return False

    margin_y = max(15, p_h * 0.15)
    if not (py1 - margin_y <= cy <= py2 + margin_y):
        return False

    # Eğilmiş veya öne eğilmiş kişi kontrolü (Genişlik > Yükseklik * 0.70)
    is_bending = (p_w > p_h * 0.70)

    if ppe_type == "helmet":
        if is_bending:
            # Eğilmiş/eğik duran kişide baret kutunun herhangi bir yerinde olabilir
            return True
        else:
            # Ayaktaki kişide baret üst %35'lik kafa bölgesinde olmalı
            return cy <= py1 + p_h * 0.35
    elif ppe_type == "vest":
        # Yelek kutusu zaten insan kutusunun içinde ise (is_center_in_box) geçerli sayılır (eğilme/yan durma uyumlu)
        return True
    return True


def match_ppe_to_person(ppe_detections, person_bbox):
    """
    ÜÇ DURUMLU (Present / Missing / Unknown) PPE Eşleştirme.
    - 'Present': Bu karede Helmet / Safety Vest tespiti eşleşti.
    - 'Missing': Bu karede açıkça No Helmet / No Safety Vest tespiti eşleşti.
    - 'Unknown': Bu karede bu kişi için hiçbir PPE kutusu bulunamadı (el kalktı, gölge vs).
    """
    has_helmet = False
    has_vest = False
    no_helmet_detected = False
    no_vest_detected = False
    max_conf = 0.0

    helmet_box_found = False
    vest_box_found = False

    for ppe in ppe_detections:
        cname = ppe["class_name"]
        ppe_box = ppe["bbox"]
        conf = ppe["confidence"]

        if not is_center_in_box(ppe_box, person_bbox):
            if compute_iou(person_bbox, ppe_box) < 0.05:
                continue

        # Baret kontrolleri
        if cname == "Helmet" and is_ppe_belonging_to_person(ppe_box, person_bbox, "helmet"):
            has_helmet = True
            helmet_box_found = True
            max_conf = max(max_conf, conf)
        elif cname == "No Helmet" and is_ppe_belonging_to_person(ppe_box, person_bbox, "helmet"):
            no_helmet_detected = True
            helmet_box_found = True
            max_conf = max(max_conf, conf)

        # Yelek kontrolleri
        elif cname == "Safety Vests" and is_ppe_belonging_to_person(ppe_box, person_bbox, "vest"):
            has_vest = True
            vest_box_found = True
            max_conf = max(max_conf, conf)
        elif cname == "No Safety Vest" and is_ppe_belonging_to_person(ppe_box, person_bbox, "vest"):
            no_vest_detected = True
            vest_box_found = True
            max_conf = max(max_conf, conf)

    # Kask durumu (Present / Missing / Unknown)
    if has_helmet:
        helmet_raw_state = "Present"
    elif no_helmet_detected:
        helmet_raw_state = "Missing"
    elif helmet_box_found:
        helmet_raw_state = "Missing"
    else:
        helmet_raw_state = "Unknown"  # Karede hiçbir kask nesnesi bulunamadı

    # Yelek durumu (Present / Missing / Unknown)
    if has_vest:
        vest_raw_state = "Present"
    elif no_vest_detected:
        vest_raw_state = "Missing"
    elif vest_box_found:
        vest_raw_state = "Missing"
    else:
        vest_raw_state = "Unknown"  # Karede hiçbir yelek nesnesi bulunamadı

    return helmet_raw_state, vest_raw_state, max_conf


class StableStateTracker:
    """
    Her track_id için Helmet/Vest durumunu titremeye karşı koruyan Asimetrik Debounce Sınıfı.
    - 'Present' (Var): 5 kare boyunca tutarlı tespit alındığında durum 'Present' (Yeşil) olur.
    - 'Unknown' (Açı Değişimi/Gölge): Tespit alınamasa bile mevcut durum korunur (titreme yapmaz).
    - 'Missing' (İhlal Onayı): Eğer model kesintisiz 12 kare boyunca açıkça 'Missing' (Kask/Yelek Yok)
      tespit ederse, hatalı kilit kırılarak durum 'Missing' (Kırmızı - İhlal) yapılır.
    """
    def __init__(self, present_confirm_frames=5, missing_confirm_frames=12):
        self.present_confirm_frames = present_confirm_frames
        self.missing_confirm_frames = missing_confirm_frames
        self.confirmed_helmet = {}   # track_id -> True ("Present") / False ("Missing")
        self.pending_helmet = {}     # track_id -> (cand_state, count)
        self.confirmed_vest = {}
        self.pending_vest = {}

    def update(self, track_id, raw_helmet_state, raw_vest_state):
        h_bool = self._update_single(
            track_id, raw_helmet_state, self.confirmed_helmet, self.pending_helmet
        )
        v_bool = self._update_single(
            track_id, raw_vest_state, self.confirmed_vest, self.pending_vest
        )
        return h_bool, v_bool

    def _update_single(self, track_id, raw_state, confirmed_map, pending_map):
        # 1. İlk defa görülen ID
        if track_id not in confirmed_map:
            # Varsayılan olarak raw_state Present ise True, aksi halde False başlat
            initial = (raw_state == "Present")
            confirmed_map[track_id] = initial
            pending_map[track_id] = (raw_state, 1)
            return initial

        current = confirmed_map[track_id]

        # 2. Unknown (gölge, açı değişimi, tespit yok):
        # Mevcut onaylı durumu korur (titremeyi ve yanlış ihlalleri engeller)
        if raw_state == "Unknown":
            pending_map[track_id] = (raw_state, 0)
            return current

        bool_raw = (raw_state == "Present")

        # 3. Ham tespit onaylı durumla aynıysa sayaç sıfırlanır
        if bool_raw == current:
            pending_map[track_id] = (raw_state, 0)
            return current

        # 4. Zıt tespit alındıysa (Current Present iken Raw Missing veya tam tersi)
        cand_state, cand_count = pending_map.get(track_id, (raw_state, 0))

        if raw_state == cand_state:
            cand_count += 1
        else:
            cand_state, cand_count = raw_state, 1

        # Gerekli onay kare sayısı: Missing'e geçiş için 12 kare, Present'a geçiş için 5 kare
        required_frames = self.missing_confirm_frames if not bool_raw else self.present_confirm_frames

        if cand_count >= required_frames:
            current = bool_raw
            cand_count = 0
            confirmed_map[track_id] = current

        pending_map[track_id] = (cand_state, cand_count)

        return confirmed_map[track_id]



