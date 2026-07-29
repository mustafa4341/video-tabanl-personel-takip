# Staj Defteri — 2. Gün
**Tarih:** 28 Temmuz 2026
**Proje:** Video Tabanlı Personel Takibi ve Koruyucu Ekipman Kontrol Sistemi

---

## Günün Özeti

Bu gün projenin temel modüllerini yazmaya devam ettim. Sadece kod yazmakla kalmayıp her satırın **neden** yazıldığını ve bilgisayarın o satırı çalıştırırken **neler yaptığını** anlamaya çalıştım. Proje mimarisini netleştirdim, klasör yapısını düzenledim ve birden fazla modülü tamamladım.

---

## 1. Öğrenilen Temel Kavramlar

### 1.1 Video ve Frame Kavramı

Bir video aslında art arda gelen fotoğraflardan oluşur. Saniyede 30 FPS olan bir video, her saniyede 30 ayrı görüntü (frame) içerir. Bilgisayar bu görüntüleri tek tek işler. Bu yüzden sistemimiz her kareyi ayrı ayrı analiz eder.

### 1.2 NumPy Array ve BGR

OpenCV ile okunan her frame aslında bir NumPy dizisidir. Boyutu `(yükseklik, genişlik, 3)` şeklindedir. Üçüncü boyut renk kanallarını temsil eder. OpenCV'de renkler RGB değil, **BGR** sırasındadır (Blue, Green, Red).

```
frame.shape → (1920, 1080, 3)
```

### 1.3 YOLO ve results Nesnesi

YOLO modeline bir frame gönderildiğinde geriye `results` adında bir liste döner. Bu liste içinde `Results` nesneleri bulunur. Her `Results` nesnesi şu bilgileri taşır:

- `boxes` → tespit edilen nesnelerin kutuları
- Her `box` içinde:
  - `cls` → nesne sınıfı (0 = Person gibi)
  - `conf` → güven skoru (0.0 ile 1.0 arası)
  - `xyxy` → kutunun koordinatları (sol üst ve sağ alt köşe)

### 1.4 Bounding Box (Sınırlayıcı Kutu)

Tespit edilen her nesne dört koordinatla ifade edilir:

```
(x1, y1) → sol üst köşe
(x2, y2) → sağ alt köşe
```

### 1.5 Confidence Score (Güven Skoru)

YOLO hiçbir zaman "kesin" demez. Her tespit için bir olasılık değeri üretir. Örneğin `0.94`, modelin %94 oranında emin olduğunu gösterir. Eşiğin altındaki tespitler filtrelenir.

### 1.6 ByteTrack ve Kişi Takibi

YOLO her karede sıfırdan tespit yapar ve kişilere ID vermez. Kişileri video boyunca takip etmek için **ByteTrack** algoritması kullanılır. ByteTrack her karede tespit edilen nesneleri bir önceki karedeki nesnelerle eşleştirir (IoU hesabıyla). Eşleşen nesnelere aynı ID verilir; böylece bir kişi hareket etse bile ID'si değişmez.

### 1.7 IoU (Intersection over Union)

İki bounding box'ın ne kadar örtüştüğünü ölçen bir metriktir.

```
IoU = Kesişim Alanı / Birleşim Alanı
```

- IoU yüksekse → iki kutu aynı nesneye aittir
- IoU düşükse → iki kutu farklı nesnelere aittir

### 1.8 Argparse

`argparse`, Python programlarına komut satırından parametre almayı sağlar:

```
python main.py --video test_video.mp4
```

Bu sayede video dosya adı koda sabit yazılmaz; dışarıdan verilir.

---

## 2. Proje Mimarisinin Netleşmesi

İki ayrı model kullanıldığı kesinleşti:

| Model | Görev | Sınıflar |
|---|---|---|
| `yolo11n.pt` | Kişi tespiti | Person (COCO class 0) |
| `models/best.pt` | PPE tespiti | Helmet, No Helmet, Safety Vests, No Safety Vest |

Kendi eğittiğim modelin sınıfları terminalde doğrulandı:

```python
from ultralytics import YOLO
model = YOLO("models/best.pt")
print(model.names)
# {0: 'Helmet', 1: 'No Helmet', 2: 'Safety Vests', 3: 'No Safety Vest'}
```

---

## 3. Klasör Yapısının Düzenlenmesi

Proje belgesine uygun olarak klasör yapısı yeniden düzenlendi:

```
project/
├── main.py
├── detectors/
│   ├── person_detector.py
│   └── ppe_detector.py
├── tracking/
│   └── person_tracker.py
├── services/
│   ├── violation_service.py
│   └── video_service.py
├── utils/
│   ├── drawing.py
│   └── geometry.py
├── outputs/
└── models/
    └── best.pt
```

---

## 4. Yazılan Modüller

### 4.1 `detectors/person_detector.py`
- YOLO ile kişi tespiti yapan sınıf yazıldı
- `conf` ve `classes=[0]` parametreleri model seviyesinde filtreleme için kullanıldı
- Dönüş formatı: `[{"bbox": (...), "confidence": ...}]`

### 4.2 `tracking/person_tracker.py`
- `supervision.ByteTrack` kullanıldı
- Dict listesi `supervision.Detections` formatına çevrildi
- Dönüş formatı: `[{"tracker_id": ..., "bbox": (...)}]`

### 4.3 `main.py`
- `argparse` ile `--video` parametresi alındı
- Detect → Track → Draw → Display akışı kuruldu
- Sanal ortam (venv) aktivasyon sorunu çözüldü

### 4.4 `utils/drawing.py`
- `cv2.rectangle` ile yeşil kutu çizildi
- `cv2.putText` ile ID yazısı eklendi

### 4.5 `detectors/ppe_detector.py`
- `models/best.pt` ile kask/yelek tespiti yapıldı
- Dönüş formatı: `[{"class_id": ..., "class_name": ..., "bbox": (...), "confidence": ...}]`

### 4.6 `utils/geometry.py`
- `compute_iou(boxA, boxB)` fonksiyonu yazıldı
- Kesişim ve birleşim alanı hesabı yapıldı

---

## 5. Karşılaşılan Sorunlar ve Çözümler

| Sorun | Sebep | Çözüm |
|---|---|---|
| `ModuleNotFoundError: supervision` | Sanal ortam aktif değil | `.venv\Scripts\activate` ile venv aktive edildi |
| Ekranda kutu görünmemesi | Frame resize'dan sonra çizim yapılıyordu | Önce çiz, sonra küçült |
| Video tam görünmüyor | Çok büyük çözünürlük | `cv2.resize(frame, None, fx=0.3, fy=0.3)` kullanıldı |
| Fonksiyon class içinde tanımsız | Girinti hatası | `Shift+Tab` ile düzeltildi |

---

## 6. Öğrenilen VS Code Kısayolları

| Kısayol | İşlev |
|---|---|
| `Tab` / `Shift+Tab` | Girinti ekle / kaldır |
| `Alt + ↑ / ↓` | Satırı taşı |
| `Ctrl + /` | Yorum satırı yap |
| `Ctrl + H` | Bul ve değiştir |
| `Shift + Alt + F` | Otomatik formatla |

---

## 7. Yarın Yapılacaklar

- `services/violation_service.py` tamamlanacak (15 kare kuralı)
- `services/video_service.py` yazılacak (output video kaydı)
- PPE eşleştirme mantığı `main.py`e eklenecek
- `outputs/violations.json` otomatik oluşturulacak
- `outputs/result.mp4` çıktı videosu oluşturulacak
- Sistem uçtan uca test edilecek

---

*Staj 2. Gün — Video Tabanlı Personel Takibi ve Koruyucu Ekipman Kontrol Sistemi*
