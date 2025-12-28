# train_rcnn_local.py
import os
import xml.etree.ElementTree as ET
import time
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend for saving without display

# ⚠️ BURAYI DEĞİŞTİRİN: VOC2007 KLASÖRÜNÜN YEREL YOLUNU GİRİN
# Örnek: 'C:/Users/KullaniciAdi/Desktop/XRayProject/VOC2007'
DATA_ROOT = "VOC2007"

# =========================================================================
# 2. YARDIMCI FONKSİYONLAR ve VOCDATASET SINIFI (Colab koduyla aynı)
# =========================================================================

def infer_classes_from_xmls(root_dir):
    names = set()
    def find_xml_files(directory):
        xmls = []
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                if f.lower().endswith('.xml'): xmls.append(os.path.join(dirpath, f))
        return xmls
    for xml_path in find_xml_files(root_dir):
        try:
            tree = ET.parse(xml_path); root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name: names.add(name)
        except ET.ParseError: continue
    return sorted(list(names))

def parse_voc_annotation(xml_path):
    tree = ET.parse(xml_path); root = tree.getroot()
    boxes = []; labels = []; iscrowd = []
    for obj in root.findall('object'):
        name = obj.find('name').text; bnd = obj.find('bndbox')
        xmin = float(bnd.find('xmin').text); ymin = float(bnd.find('ymin').text)
        xmax = float(bnd.find('xmax').text); ymax = float(bnd.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax]); labels.append(name); iscrowd.append(0)
    size = root.find('size'); width = int(size.find('width').text); height = int(size.find('height').text)
    filename = root.find('filename').text
    return {'filename': filename, 'width': width, 'height': height, 'boxes': boxes, 'labels': labels, 'iscrowd': iscrowd}

def box_iou(boxA, boxB):
    xa1, ya1, xa2, ya2 = boxA; xb1, yb1, xb2, yb2 = boxB
    inter_x1 = max(xa1, xb1); inter_y1 = max(ya1, yb1); inter_x2 = min(xa2, xb2); inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1); inter_h = max(0.0, inter_y2 - inter_y1); inter = inter_w * inter_h
    areaA = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1); areaB = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

class VOCDataset(Dataset):
    def __init__(self, voc_root, image_set='train', images_dir='JPEGImages', annotations_dir='Annotations', transforms=None, classes=None):
        self.voc_root = voc_root; self.images_dir = os.path.join(voc_root, images_dir); self.annotations_dir = os.path.join(voc_root, annotations_dir)
        self.transforms = transforms; self.image_set = image_set
        if classes is None: classes = infer_classes_from_xmls(os.path.join(voc_root, annotations_dir))
        self.classes = ['__background__'] + classes; self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        set_file = os.path.join(voc_root, 'ImageSets', 'Main', f'{image_set}.txt')
        with open(set_file) as f: self.file_names = [x.strip() for x in f.readlines()]
        self.xmls = [os.path.join(self.annotations_dir, f'{name}.xml') for name in self.file_names]

    def __len__(self): return len(self.file_names)

    def __getitem__(self, idx):
        img_id = self.file_names[idx]; xml_path = os.path.join(self.annotations_dir, f'{img_id}.xml'); ann = parse_voc_annotation(xml_path)
        img_path = os.path.join(self.images_dir, ann['filename'])
        if not os.path.exists(img_path): img_path = os.path.join(self.images_dir, f'{img_id}.jpg')
        img = Image.open(img_path).convert("RGB")
        
        boxes = []; labels = []
        for box, label_name in zip(ann['boxes'], ann['labels']):
            xmin, ymin, xmax, ymax = box; xmin -= 1; ymin -= 1; xmax -= 1; ymax -= 1;
            if xmax > xmin and ymax > ymin:
                 boxes.append([xmin, ymin, xmax, ymax]); labels.append(self.class_to_idx[label_name])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32); labels = torch.zeros((0,), dtype=torch.int64);
            area = torch.zeros((0,), dtype=torch.float32); iscrowd = torch.zeros((0,), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32); labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]); iscrowd = torch.as_tensor(ann['iscrowd'][:len(boxes)], dtype=torch.uint8)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx]), "area": area, "iscrowd": iscrowd}
        if self.transforms is not None: img = self.transforms(img)
        return img, target

# =========================================================================
# 3. ANA EĞİTİM MANTIĞI VE DÖNGÜSÜ (Yerel ortam için optimize edildi)
# =========================================================================

def collate_fn(batch): return tuple(zip(*batch))
def get_transform(train): 
    # Not: Görüntü boyutlandırma eklenmedi, eğer Colab'da kilitlenme yaşamıyorsanız, yerelde de yaşamazsınız.
    return T.Compose([T.ToTensor()]) 

def build_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1 
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Metrik Hesaplama Fonksiyonu
def evaluate_model(model, dataset, device, iou_thresh=0.5, score_thresh=0.25, out_dir='runs/fasterrcnn_eval', epoch=None, save_plots=False):
    """
    Modeli değerlendirip mAP, precision, recall, F1, karışıklık matrisi ve grafikleri kaydeder.
    save_plots=True ise grafikler PNG olarak kaydedilir, False ise sadece metrik değerleri döndürülür.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    n_classes = len(dataset.classes)  # includes '__background__' at index 0
    
    # Metrik takibi
    total_gt = np.zeros(n_classes, dtype=np.int32)
    total_tp = np.zeros(n_classes, dtype=np.int32)
    total_fp = np.zeros(n_classes, dtype=np.int32)
    total_fn = np.zeros(n_classes, dtype=np.int32)
    confmat = np.zeros((n_classes, n_classes), dtype=np.int32)
    
    # AP hesabı için
    per_class_scores = {c: [] for c in range(1, n_classes)}
    per_class_tp = {c: [] for c in range(1, n_classes)}
    per_class_fp = {c: [] for c in range(1, n_classes)}
    matched_ious = []

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            out = outputs[0]
            pred_boxes = out['boxes'].cpu().numpy()
            pred_scores = out['scores'].cpu().numpy()
            pred_labels = out['labels'].cpu().numpy()

            mask = pred_scores >= score_thresh
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_labels = pred_labels[mask]

            gt_boxes = targets[0]['boxes'].cpu().numpy()
            gt_labels = targets[0]['labels'].cpu().numpy()

            # GT sayısını kaydet
            for gl in gt_labels:
                total_gt[int(gl)] += 1

            # Greedy matching (score'a göre azalan sırada)
            if len(pred_boxes) > 0:
                order = np.argsort(-pred_scores)
            else:
                order = []
            used_gt = set()
            
            for k in order:
                c = int(pred_labels[k])
                if c == 0:  # background'u atla
                    continue
                    
                best_iou = 0.0
                best_j = -1
                for j in range(len(gt_boxes)):
                    if j in used_gt:
                        continue
                    iou = box_iou(pred_boxes[k], gt_boxes[j])
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                        
                if best_iou >= iou_thresh:
                    used_gt.add(best_j)
                    matched_ious.append(best_iou)
                    gl = int(gt_labels[best_j])
                    confmat[c, gl] += 1
                    
                    # AP için - sadece sınıf eşleşirse TP
                    if c == gl:
                        per_class_tp[c].append(1)
                        per_class_fp[c].append(0)
                        total_tp[c] += 1
                    else:
                        per_class_tp[c].append(0)
                        per_class_fp[c].append(1)
                        total_fp[c] += 1
                    per_class_scores[c].append(pred_scores[k])
                else:
                    # Eşleşmeyen tahmin -> FP
                    per_class_tp[c].append(0)
                    per_class_fp[c].append(1)
                    per_class_scores[c].append(pred_scores[k])
                    total_fp[c] += 1
            
            # Eşleşmeyen GT'ler -> FN
            for j in range(len(gt_boxes)):
                if j not in used_gt:
                    gl = int(gt_labels[j])
                    total_fn[gl] += 1

    # AP hesaplama (11-point interpolation)
    ap_scores = {}
    for c in range(1, n_classes):
        if total_gt[c] == 0:
            ap_scores[c] = 0.0
            continue
        scores = np.array(per_class_scores[c])
        tps = np.array(per_class_tp[c])
        fps = np.array(per_class_fp[c])
        
        if len(scores) == 0:
            ap_scores[c] = 0.0
            continue
            
        order = np.argsort(-scores)
        tps = tps[order]
        fps = fps[order]
        cum_tp = np.cumsum(tps)
        cum_fp = np.cumsum(fps)
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
        recall = cum_tp / float(total_gt[c])
        
        # 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = np.max(precision[recall >= t]) if np.any(recall >= t) else 0.0
            ap += p
        ap /= 11.0
        ap_scores[c] = ap

    # Genel metrikler
    mAP50 = float(np.mean([ap_scores[c] for c in range(1, n_classes) if total_gt[c] > 0])) if any(total_gt[1:] > 0) else 0.0
    mean_iou = float(np.mean(matched_ious)) if len(matched_ious) > 0 else 0.0
    
    # Precision, Recall, F1 (sınıf bazında)
    precision_per_class = {}
    recall_per_class = {}
    f1_per_class = {}
    
    for c in range(1, n_classes):
        if total_tp[c] + total_fp[c] > 0:
            prec = total_tp[c] / (total_tp[c] + total_fp[c])
        else:
            prec = 0.0
        if total_tp[c] + total_fn[c] > 0:
            rec = total_tp[c] / (total_tp[c] + total_fn[c])
        else:
            rec = 0.0
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0
        precision_per_class[c] = prec
        recall_per_class[c] = rec
        f1_per_class[c] = f1
    
    # Ortalama metrikler
    valid_classes = [c for c in range(1, n_classes) if total_gt[c] > 0]
    avg_precision = np.mean([precision_per_class[c] for c in valid_classes]) if valid_classes else 0.0
    avg_recall = np.mean([recall_per_class[c] for c in valid_classes]) if valid_classes else 0.0
    avg_f1 = np.mean([f1_per_class[c] for c in valid_classes]) if valid_classes else 0.0

    # ===== GRAFİKLERİ KAYDET (sadece save_plots=True ise) =====
    if save_plots:
        epoch_str = f'epoch_{epoch}' if epoch is not None else 'final'
        
        # 1. Karışıklık Matrisi
        if n_classes > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            cm_display = confmat[1:, 1:]  # background hariç
            im = ax.imshow(cm_display, cmap='Blues', aspect='auto')
            cls_names = dataset.classes[1:]
            ax.set_xticks(range(len(cls_names)))
            ax.set_yticks(range(len(cls_names)))
            ax.set_xticklabels(cls_names, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(cls_names, fontsize=9)
            ax.set_xlabel('Gerçek Sınıf', fontsize=11)
            ax.set_ylabel('Tahmin Edilen Sınıf', fontsize=11)
            ax.set_title(f'Karışıklık Matrisi ({epoch_str})', fontsize=13)
            
            # Hücre değerlerini yaz
            for i in range(len(cls_names)):
                for j in range(len(cls_names)):
                    text = ax.text(j, i, int(cm_display[i, j]), ha="center", va="center", 
                                  color="white" if cm_display[i, j] > cm_display.max()/2 else "black", fontsize=9)
            
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            cm_path = os.path.join(out_dir, f'confusion_matrix_{epoch_str}.png')
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'Karışıklık matrisi kaydedildi: {cm_path}')
        
        # 2. PR Eğrileri
        fig, ax = plt.subplots(figsize=(10, 7))
        for c in range(1, n_classes):
            if total_gt[c] == 0:
                continue
            scores = np.array(per_class_scores[c])
            if len(scores) == 0:
                continue
            tps = np.array(per_class_tp[c])
            fps = np.array(per_class_fp[c])
            order = np.argsort(-scores)
            cum_tp = np.cumsum(tps[order])
            cum_fp = np.cumsum(fps[order])
            precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
        
            recall = cum_tp / float(total_gt[c])
            ax.plot(recall, precision, linewidth=2, label=f'{dataset.classes[c]} (AP={ap_scores[c]:.3f})')
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Eğrileri (IoU≥{iou_thresh}) - {epoch_str}', fontsize=13)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        plt.tight_layout()
        pr_path = os.path.join(out_dir, f'pr_curves_{epoch_str}.png')
        plt.savefig(pr_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'PR eğrileri kaydedildi: {pr_path}')
        
        # 3. Metrik Tablosu (Bar Chart)
        fig, ax = plt.subplots(figsize=(12, 6))
        cls_names = [dataset.classes[c] for c in range(1, n_classes) if total_gt[c] > 0]
        if cls_names:
            x = np.arange(len(cls_names))
            width = 0.2
            precs = [precision_per_class[c] for c in range(1, n_classes) if total_gt[c] > 0]
            recs = [recall_per_class[c] for c in range(1, n_classes) if total_gt[c] > 0]
            f1s = [f1_per_class[c] for c in range(1, n_classes) if total_gt[c] > 0]
            aps = [ap_scores[c] for c in range(1, n_classes) if total_gt[c] > 0]
            
            ax.bar(x - 1.5*width, precs, width, label='Precision', alpha=0.8)
            ax.bar(x - 0.5*width, recs, width, label='Recall', alpha=0.8)
            ax.bar(x + 0.5*width, f1s, width, label='F1-Score', alpha=0.8)
            ax.bar(x + 1.5*width, aps, width, label='AP@0.5', alpha=0.8)
            
            ax.set_xlabel('Sınıflar', fontsize=12)
            ax.set_ylabel('Değer', fontsize=12)
            ax.set_title(f'Sınıf Bazında Metrikler - {epoch_str}', fontsize=13)
            ax.set_xticks(x)
            ax.set_xticklabels(cls_names, rotation=45, ha='right')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 1.05])
        plt.tight_layout()
        metrics_path = os.path.join(out_dir, f'metrics_bars_{epoch_str}.png')
        plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Metrik bar grafiği kaydedildi: {metrics_path}')
        
        # 4. Metrik CSV kaydet
        csv_path = os.path.join(out_dir, f'metrics_{epoch_str}.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('Sınıf,Precision,Recall,F1-Score,AP@0.5,TP,FP,FN,Total_GT\n')
            for c in range(1, n_classes):
                if total_gt[c] > 0:
                    f.write(f'{dataset.classes[c]},{precision_per_class[c]:.6f},{recall_per_class[c]:.6f},' 
                           f'{f1_per_class[c]:.6f},{ap_scores[c]:.6f},{total_tp[c]},{total_fp[c]},{total_fn[c]},{total_gt[c]}\n')
            f.write(f'\nOrtalamaMetrikler\n')
            f.write(f'mAP@0.5,{mAP50:.6f}\n')
            f.write(f'Avg_Precision,{avg_precision:.6f}\n')
            f.write(f'Avg_Recall,{avg_recall:.6f}\n')
            f.write(f'Avg_F1,{avg_f1:.6f}\n')
            f.write(f'Mean_IoU,{mean_iou:.6f}\n')
        print(f'Metrikler CSV olarak kaydedildi: {csv_path}')
    
    return mAP50, mean_iou, ap_scores, avg_precision, avg_recall, avg_f1

# Ana Eğitim Fonksiyonu
def train_model():
    epochs = 100; batch_size = 2; lr = 0.005; workers = 0; output_dir = './checkpoints'
    eval_dir = 'runs/fasterrcnn_eval'  # Değerlendirme sonuçları klasörü

    # ⚠️ AMD/Windows için CUDA DESTEĞİ YOK. Otomatik olarak CPU kullanılacaktır.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    print(f"\n--- YEREL EĞİTİM BAŞLADI ---")
    print(f"Kullanılan Cihaz: {device} (AMD GPU'nuz olduğu için CPU kullanılacaktır.)")

    if not os.path.isdir(DATA_ROOT):
        print(f"HATA: Veri yolu bulunamadı: {DATA_ROOT}")
        return

    annotations_path = os.path.join(DATA_ROOT, 'Annotations')
    classes = infer_classes_from_xmls(annotations_path)
    print(f"Tespit Edilen Sınıflar: {['__background__'] + classes}")
    
    train_dataset = VOCDataset(DATA_ROOT, image_set='train', transforms=get_transform(True), classes=classes)
    val_dataset = VOCDataset(DATA_ROOT, image_set='val', transforms=get_transform(False), classes=classes)
    num_classes = len(train_dataset.classes)
    
    # Yerel PC'de kilitlenmeyi önlemek için workers=0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = build_model(num_classes); model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Eğitim döngüsü
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0; iters = 0
        print(f"\n--- Epoch {epoch+1}/{epochs} Başladı ---")

        for i, (images, targets) in enumerate(train_loader):
            # Verinin CPU'dan CPU'ya veya (varsa) GPU'ya aktarılması
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad(); losses.backward(); optimizer.step();
            epoch_loss += losses.item(); iters += 1

            if i % 10 == 0:
                 print(f"   [Batch {i}/{len(train_loader)}] Kayıp: {losses.item():.4f}")


        lr_scheduler.step()
        avg_loss = epoch_loss / max(1, iters)
        print(f'Epoch [{epoch+1}/{epochs}]  Avg Loss: {avg_loss:.4f}  Total Time: {time.time()-start_time:.1f}s')

        # Checkpoint Kaydetme
        os.makedirs(output_dir, exist_ok=True)
        ckpt = os.path.join(output_dir, f'model_epoch_{epoch+1}.pth')
        torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, ckpt)

    print("\n--- EĞİTİM TAMAMLANDI ---")
    
    # Final değerlendirme - Tüm grafiklerle birlikte
    print("\n--- Final Değerlendirme Başladı (Tüm Grafikler Kaydediliyor) ---")
    final_mAP, final_iou, final_ap, final_prec, final_rec, final_f1 = evaluate_model(
        model, val_dataset, device, iou_thresh=0.5, score_thresh=0.25, 
        out_dir=eval_dir, epoch=None, save_plots=True  # Tüm grafikleri kaydet
    )
    print(f'Final Sonuçlar:')
    print(f'  mAP@0.5: {final_mAP:.4f} | Precision: {final_prec:.4f} | Recall: {final_rec:.4f} | F1: {final_f1:.4f} | IoU: {final_iou:.4f}')
    
    print(f'\nTüm sonuçlar şu klasöre kaydedildi: {eval_dir}')
    print("Kaydedilen dosyalar:")
    print("  - Karışıklık matrisi (confusion_matrix_final.png)")
    print("  - PR eğrileri (pr_curves_final.png)")
    print("  - Metrik bar grafikleri (metrics_bars_final.png)")
    print("  - Metrik CSV dosyası (metrics_final.csv)")

if __name__ == '__main__':
    train_model()