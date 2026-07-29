# Video Tabanlı Personel Takibi ve Koruyucu Ekipman (PPE) Kontrol Sistemi

Bu proje, kayıtlı video görüntüleri üzerinden çalışan personellerin tespit edilmesi, görüntü boyunca takip edilmesi (ByteTrack) ve gerekli kişisel koruyucu ekipmanları (baret ve reflektörlü iş yeleği) kullanıp kullanmadıklarının otomatik olarak kontrol edilmesi amacıyla geliştirilmiştir.

---

## 📌 Proje Özellikleri

- **Nesne Tespiti (Person Detection):** Videodaki kişileri yüksek hassasiyetle tespit eder.
- **Kişi Takibi (Person Tracking):** ByteTrack algoritması ile kişilere benzersiz `ID` değerleri atar ve nesne engellerin arkasına geçse dahi takibi korur.
- **Koruyucu Ekipman Kontrolü (PPE Check):** Özelleştirilmiş YOLO modeli (`models/best.pt`) ile baret (`Helmet`) ve reflektörlü yelek (`Safety Vests`) varlığını denetler.
- **Dinamik Renklendirme ve Bilgi Etiketleri:**
  - Ekipmanları eksiksiz kişilere **Yeşil**, ihlali olan kişilere **Kırmızı** çerçeve çizilir.
  - Kişilerin üzerinde `ID`, `Helmet: Present/Missing`, `Vest: Present/Missing` etiketleri canlı gösterilir.
- **Canlı Özet Bilgi Paneli (OSD Overlay):** Ekranın sol üst köşesinde `Detected Persons`, `Helmet Violations`, `Vest Violations` ve anlık `FPS` bilgileri yer alır.
- **Yanlış İhlal Önleme (15 Kare Kuralı):** Yanlış tespitleri engellemek için bir kişinin en az 15 kare boyunca ekipmansız görünmesi durumunda ihlal kaydı tetiklenir.
- **Otomatik İhlal Görüntüsü ve Raporlama:**
  - Tetiklenen ihlaller anında `outputs/violations/id_<ID>_<ihlal_türü>.jpg` formatında kaydedilir.
  - İhlaller `outputs/violations.json` dosyasına zaman (`HH:MM:SS`), güven skoru ve resim yolu ile aktarılır.
- **Çıktı Videosu:** İşlenmiş görüntüler `outputs/result.mp4` olarak kaydedilir.

---

## 🏗️ Proje Mimari Yapısı

```text
video tabanlı bilgisayar takip proje/
├── main.py                    # Ana uygulama ve akış kontrolü
├── detectors/
│   ├── person_detector.py     # YOLO ile insan tespiti (yolo11n.pt)
│   └── ppe_detector.py        # Özel eğitilmiş PPE modeli (models/best.pt)
├── tracking/
│   └── person_tracker.py      # supervision.ByteTrack ile nesne takibi
├── services/
│   ├── violation_service.py   # İhlal takibi, JSON ve ekran görüntüsü kaydı
│   └── video_service.py       # Çıktı videosu yazma (outputs/result.mp4)
├── utils/
│   ├── drawing.py             # Kutu renklendirme, etiketler ve OSD paneli
│   └── geometry.py            # Baret/yelek için bölgesel eşleştirme mantığı
├── models/
│   └── best.pt                # Özel eğitilmiş PPE YOLO modeli
└── outputs/
    ├── result.mp4             # İşlenmiş çıktı videosu
    ├── violations.json        # İhlal kayıtları JSON
    └── violations/            # İhlal anı ekran görüntüleri (.jpg)
```

---

## ⚙️ Model Seçimleri ve Gerekçeleri

1. **İnsan Tespiti (`yolo11n.pt`):**
   - COCO veri kümesinde eğitilmiş hafif ve hızlı nesne tespit modelidir. Gerçek zamanlı performans sağladığı ve yüksek `Person` recall oranına sahip olduğu için tercih edilmiştir.
2. **PPE Tespiti (`models/best.pt`):**
   - Özel koruyucu ekipman veri seti üzerinde 50 epoch eğitilmiş YOLO modelidir.
   - Sınıflar: `Helmet`, `No Helmet`, `Safety Vests`, `No Safety Vest`.
   - Kask tespiti mAP50 %92.2, Yelek tespiti mAP50 %88.3 başarı oranına sahiptir.

---

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimlerin Yüklenmesi
```bash
pip install ultralytics supervision opencv-python numpy
```

### 2. Uygulamanın Çalıştırılması
Video dosyanızın yolunu `--video` parametresi olarak vererek çalıştırabilirsiniz:

```bash
python main.py --video video/test_video.mp4
```

İsteğe bağlı olarak çıktı video yolunu değiştirebilirsiniz:
```bash
python main.py --video video/test_video.mp4 --output outputs/result.mp4
```

---

## 📊 Örnek İhlal JSON Çıktısı (`outputs/violations.json`)

```json
[
    {
        "person_id": 7,
        "violation": "helmet_missing",
        "video_time": "00:01:34",
        "confidence": 0.88,
        "image_path": "outputs/violations/id_7_helmet.jpg"
    }
]
```
