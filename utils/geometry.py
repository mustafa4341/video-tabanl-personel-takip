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


def match_ppe_to_person(ppe_detections, person_bbox, pose_info=None):
    """
    Anatomik Keypoint (Pose) ve Geometrik PPE Eşleştirme.
    - pose_info varsa: Kask head_box (Burun/Kulaklar/Omuzlar), Yelek torso_box (Omuz-Kalça) üzerinden doğrulanır.
    """
    has_helmet = False
    has_vest = False
    no_helmet_detected = False
    no_vest_detected = False
    max_conf = 0.0

    helmet_box_found = False
    vest_box_found = False

    head_box = pose_info["head_box"] if pose_info else None
    torso_box = pose_info["torso_box"] if pose_info else None

    for ppe in ppe_detections:
        cname = ppe["class_name"]
        ppe_box = ppe["bbox"]
        conf = ppe["confidence"]

        # Pose kafa/gövde kutusu varsa öncelikli olarak pose kutusuyla doğrulama yap
        if head_box and ("Helmet" in cname):
            if not is_center_in_box(ppe_box, head_box) and compute_iou(head_box, ppe_box) < 0.05:
                # Kafa bölgesinin dışındaysa pas geç
                continue

        if torso_box and ("Vest" in cname):
            if not is_center_in_box(ppe_box, torso_box) and compute_iou(torso_box, ppe_box) < 0.05:
                # Gövde bölgesinin dışındaysa pas geç
                continue

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
        helmet_raw_state = "Missing"  # Model açıkça No Helmet tespit etti -> Negatif Kanıt
    elif pose_info and not pose_info.get("has_valid_head_keypoints", True):
        helmet_raw_state = "Unknown"  # Kafa keypoint güveni < 0.5 -> Göremedim (Nötr)
    else:
        helmet_raw_state = "Unknown"  # Model göremedi/açı kaybı var -> Kanıt Yok (Nötr)

    # Yelek durumu (Present / Missing / Unknown)
    if has_vest:
        vest_raw_state = "Present"
    elif no_vest_detected:
        vest_raw_state = "Missing"   # Model açıkça No Safety Vest tespit etti -> Negatif Kanıt
    elif pose_info and not pose_info.get("has_valid_torso_keypoints", True):
        vest_raw_state = "Unknown"   # Gövde keypoint güveni < 0.5 -> Göremedim (Nötr)
    else:
        vest_raw_state = "Unknown"   # Model göremedi/açı kaybı var -> Kanıt Yok (Nötr)

    return helmet_raw_state, vest_raw_state, max_conf


from collections import deque

class StableStateTracker:
    """
    Kayan Pencere (Rolling Window) tabanlı Zaman Odaklı KKD Takipçisi.
    - Zamana Dayalı (FPS-Independent): Sabit kare sayısı yerine videonun gerçek FPS değerine göre 
      pencere boyutunu dinamik hesaplar (her zaman gerçek 2.0s pencere, 1.5s min gözlem).
    - Asimetrik Eşikler:
      * present_ratio (0.75): "Var" (True) demek için pencerenin en az %75'i gerekli (Güçlü kanıt şartı).
      * missing_ratio (0.35): "Yok" (False) demek için %35 veya altı yeterli (Güvenlik öncelikli ihlal tarafına düşme).
      * %35 - %75 arası: Belirsiz ara bölge, son onaylı durumu korur (titremeyi engeller).
    """
    def __init__(self, fps=30.0, window_sec=2.0, min_obs_sec=1.5, present_ratio=0.75, missing_ratio=0.35):
        self.fps = fps if fps > 0 else 30.0
        self.window_size = max(10, int(self.fps * window_sec))
        self.min_observations = max(5, int(self.fps * min_obs_sec))
        self.present_ratio = present_ratio
        self.missing_ratio = missing_ratio

        self.helmet_windows = {}  # track_id -> deque(maxlen=window_size)
        self.vest_windows = {}    # track_id -> deque(maxlen=window_size)

        self.helmet_state = {}    # track_id -> bool
        self.vest_state = {}      # track_id -> bool

    def update(self, track_id, raw_helmet_state, raw_vest_state):
        h_bool = self._update_single(
            track_id, raw_helmet_state, self.helmet_windows, self.helmet_state
        )
        v_bool = self._update_single(
            track_id, raw_vest_state, self.vest_windows, self.vest_state
        )
        return h_bool, v_bool

    def _update_single(self, track_id, raw_state, windows_map, state_map):
        if track_id not in windows_map:
            windows_map[track_id] = deque(maxlen=self.window_size)
            # Yanlış ihlal yazılmaması için başlangıç nötr True kabul edilir
            state_map[track_id] = True

        w = windows_map[track_id]

        if raw_state == "Present":
            w.append(True)
            # Ayağa kalkma sonrası hızlı toparlanma: Son 3 kare üst üste Present geldiyse geçmişteki bükülme gürültülerini temizle
            if len(w) >= 3 and w[-1] is True and w[-2] is True and w[-3] is True:
                # Sadece True olanları tutarak kayar pencereyi hızlı doğrula
                new_w = deque([True] * len(w), maxlen=self.window_size)
                windows_map[track_id] = new_w
                w = new_w
        elif raw_state == "Missing":
            w.append(False)
        else:
            # Unknown (gölge, açı değişimi, eğilme): Pencerede veri varsa son onaylı durumu koru
            if len(w) > 0:
                w.append(w[-1])

        # ~1.5 saniye (min_observations) veri birikmeden ani durum değişikliği yapma
        if len(w) >= self.min_observations:
            true_ratio = sum(1 for item in w if item is True) / len(w)
            if true_ratio >= self.present_ratio:
                state_map[track_id] = True
            elif true_ratio <= self.missing_ratio:
                state_map[track_id] = False
            # Ara bölgede (%35 - %75): son onaylı durumu koru, ani sıçrama yapma

        return state_map[track_id]



