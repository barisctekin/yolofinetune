# Araç Görsel Analizi — Plaka Maskeleme

Bu repo, araç fotoğraflarını analiz etmek ve görsellerdeki plakaları gizlemek için geliştirdiğim projenin plaka maskeleme bölümü. Projenin diğer bölümü olan `image_analyze`, fotoğraflardan araç ve hasar bilgilerini yorumlayıp bir özet çıkarıyor. Burada ise fotoğraflardaki plakalar kapatılıyor ve görsellere filigran ekleniyor.

İki bölüm ayrı servisler olarak çalışıyor. Plaka bilgisiyle analiz yapılacaksa önce orijinal görsellerin analiz edilmesi, ardından maskeleme yapılması gerekiyor. Servisler arasında şu an otomatik bir aktarım yok.

## Nasıl çalışıyor?

Plakaların yerini bulmak için YOLO modelini, bulunan alanları kapatmak için OpenCV'yi kullandım. Python betiği `images/` klasöründeki fotoğrafları okuyor, tespit edilen plakaları siyah dikdörtgenle kapatıyor ve sonucu `output/` klasörüne kaydediyor. Görselin ortasına ayrıca `sompo sigorta` yazısı ekleniyor. Plaka bulunmayan fotoğraflara da bu filigran uygulanıyor.

Yapay zekâ bu bölümde plakanın konumunu tespit etmek için kullanılıyor. Maskeleme ve filigran işlemleri OpenCV ile yapılıyor. Uygulama `yolo/best.pt` dosyasındaki model ağırlıklarını yüklüyor; bu repoda model eğitim kodu bulunmuyor.

Kullanılan teknolojiler: **Python, Ultralytics YOLO, OpenCV, Node.js ve Express.**

## Kurulum

Python 3, Node.js ve npm kurulu olmalı. Proje klasöründe:

```sh
npm ci
python -m venv .venv
```

Sanal ortamı Windows PowerShell'de şu komutla açabilirsiniz:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux için:

```sh
source .venv/bin/activate
```

Ardından Python bağımlılıklarını kurun:

```sh
python -m pip install ultralytics opencv-python
```

Proje kökünde `images` klasörü oluşturup fotoğrafları içine koyun. `yolo/best.pt` dosyasının da yerinde olması gerekiyor. Çıktı klasörü otomatik oluşturuluyor.

## Çalıştırma

Klasördeki görselleri doğrudan işlemek için sanal ortam açıkken:

```sh
python yolo_license_plate.py
```

Sonuçlar `output/` klasörüne, kaynak dosyalarla aynı adlarda kaydedilir. Orijinal fotoğraflar değişmez; aynı adlı eski çıktılar varsa üzerlerine yazılır.

HTTP üzerinden çalıştırmak için yine proje kökünde ve sanal ortam açıkken:

```sh
node main.js
```

Sonra ayrı bir terminalden istek gönderin:

```sh
curl -X POST http://localhost:3000/mask-plates
```

PowerShell'de `curl` yerine `curl.exe` kullanabilirsiniz. Postman ile denemek için de aynı adrese boş gövdeli bir **POST** isteği yeterli. Servis fotoğraf yüklemesi almıyor; yerel `images/` klasörünü işliyor.

## Dosyalar ve ayarlar

- `yolo_license_plate.py`: Modeli yükleyen ve görselleri işleyen Python betiği.
- `main.js`: Python betiğini çağıran Express servisi.
- `yolo/best.pt`: Kullanılan model ağırlıkları. `last.pt` mevcut akışta kullanılmıyor.
- `images/` ve `output/`: Giriş görselleri ve işlenmiş sonuçlar.

Filigran metni ve opaklığı Python dosyasındaki `add_watermark()` fonksiyonundan değiştirilebilir. Servisin portu `main.js` içinde `3000` olarak tanımlı; bu bölüm için `.env` ayarı gerekmiyor.

## Mevcut durum

HTTP tarafında düzeltilmesi gereken bir tekrar var: Node.js her dosya için Python'u çağırıyor, ancak Python betiği gelen dosya argümanını kullanmayıp tüm klasörü işliyor. Bu nedenle toplu kullanımda şimdilik Python betiğini doğrudan bir kez çalıştırmak daha uygun.

`images/` içinde yalnızca görsel dosyaları bulunmalı; alt klasörler taranmıyor. Tespit edilmeyen plakalar açık kalabileceği için çıktıları kontrol etmek gerekiyor. Model veya dosya okuma hataları her zaman HTTP yanıtına yansımadığından, bir çıktı eksikse terminal kayıtlarına da bakılmalı.
