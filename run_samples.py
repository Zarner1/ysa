from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import time

ROOT = Path(__file__).parent
# Ağırlık dosyalarını öncelikle proje kökünde arar
weights_candidates = [ROOT / "best.pt", ROOT / "best8.pt"]
weights = None
for w in weights_candidates:
    if w.exists():
        weights = str(w)
        break

if weights is None:
    print("Ağırlık dosyası bulunamadı (best.pt veya best8.pt). Lütfen dosyayı proje köküne koyun.")
    raise SystemExit(1)

# Görüntü örnekleri: P*.jpg veya *.png
images = sorted(list(ROOT.glob("P*.jpg")) + list(ROOT.glob("P*.jpeg")) + list(ROOT.glob("P*.png")))
if not images:
    print("Örnek görüntü bulunamadı. Proje kökünde P*.jpg şeklinde görüntüler arandı.")
    raise SystemExit(1)

out_dir = ROOT / "outputs"
out_dir.mkdir(exist_ok=True)

print(f"Modeli yüklüyor: {weights}")
model = YOLO(weights)
print("Model yüklendi.")

for img_path in images:
    print(f"İşleniyor: {img_path.name}")
    t0 = time.perf_counter()
    try:
        results = model.predict(source=str(img_path), conf=0.3, iou=0.5, imgsz=640, verbose=False)
        arr = results[0].plot()  # BGR
        arr = arr[:, :, ::-1]
        out_img = Image.fromarray(arr)
        out_file = out_dir / img_path.name
        out_img.save(out_file)
        t1 = time.perf_counter()
        print(f"Kaydedildi: {out_file} (took {(t1-t0)*1000:.1f} ms)")
    except Exception as e:
        print(f"Hata işlenirken {img_path.name}: {e}")

print("Tamamlandı. Çıktılar 'outputs' klasöründe.")
