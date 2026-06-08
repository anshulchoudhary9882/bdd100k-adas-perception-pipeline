from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import csv
import json
import os
import shutil
from collections import defaultdict
import torch

# ==========================================================
# CONFIG
# ==========================================================

MODEL_PATH = "yolo11m.pt"

IMAGE_DIR = Path("bdd_yolo/images/val")
LABEL_DIR = Path("bdd_yolo/labels/val")
METADATA_CSV = Path("bdd_yolo/metadata/val_metadata.csv")

OUTPUT_DIR = Path("evaluation")
QUAL_DIR = OUTPUT_DIR / "qualitative"
ATTR_DIR = OUTPUT_DIR / "attribute_analysis"
BEST_DIR = OUTPUT_DIR / "best_cases"
FAIL_DIR = OUTPUT_DIR / "failure_cases"

IOU_THRESHOLD = 0.50
CONF_THRESHOLD = 0.45
QUAL_LIMIT = 12
TOP_K_CASES = 20

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

BDD_CLASSES = {
    0: "person",
    1: "rider",
    2: "car",
    3: "truck",
    4: "bus",
    5: "train",
    6: "bike",
    7: "motor",
    8: "traffic light",
    9: "traffic sign",
}

# COCO class id -> BDD class id
COCO_TO_BDD = {
    0: 0,    # person -> person
    1: 6,    # bicycl -> bike
    2: 2,    # car -> car
    3: 7,    # motorcycle -> motor
    5: 4,    # bus -> bus
    6: 5,    # train -> train
    7: 3,    # truck -> truck
    9: 8,    # traffic light -> traffic light
    11: 9,   # stop sign -> traffic sign
}

# ==========================================================
# HELPERS
# ==========================================================

def safe_div(a, b):
    return a / b if b else 0.0


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    QUAL_DIR.mkdir(parents=True, exist_ok=True)
    ATTR_DIR.mkdir(parents=True, exist_ok=True)
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    FAIL_DIR.mkdir(parents=True, exist_ok=True)


def load_metadata(metadata_csv):
    lookup = {}
    if not metadata_csv.exists():
        return lookup

    with open(metadata_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = (row.get("image_name") or "").strip()
            if not image_name:
                continue
            lookup[image_name] = row
            lookup[Path(image_name).stem] = row
    return lookup


def yolo_label_line_to_box(line, img_w, img_h):
    parts = line.strip().split()
    if len(parts) != 5:
        return None

    cls_id = int(float(parts[0]))
    xc = float(parts[1])
    yc = float(parts[2])
    bw = float(parts[3])
    bh = float(parts[4])

    x1 = (xc - bw / 2.0) * img_w
    y1 = (yc - bh / 2.0) * img_h
    x2 = (xc + bw / 2.0) * img_w
    y2 = (yc + bh / 2.0) * img_h

    return {
        "class": cls_id,
        "box": [x1, y1, x2, y2],
    }


def load_ground_truth(label_path, img_w, img_h):
    gts = []
    if not label_path.exists():
        return gts

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = yolo_label_line_to_box(line, img_w, img_h)
            if parsed is None:
                continue
            if parsed["class"] not in BDD_CLASSES:
                continue
            gts.append(parsed)
    return gts


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    denom = area1 + area2 - inter_area
    return inter_area / denom if denom > 0 else 0.0


def get_predictions(model, image_path, device):
    result = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        device=device,
        verbose=False
    )[0]

    preds = []
    if result.boxes is None or len(result.boxes) == 0:
        return preds

    for box in result.boxes:
        coco_cls = int(box.cls.item())
        if coco_cls not in COCO_TO_BDD:
            continue

        bdd_cls = COCO_TO_BDD[coco_cls]
        xyxy = box.xyxy.cpu().numpy()[0].tolist()
        conf = float(box.conf.item())

        preds.append({
            "class": bdd_cls,
            "box": xyxy,
            "confidence": conf,
        })

    return preds


def draw_boxes(image_path, boxes, out_path, mode="gt"):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for item in boxes:
        x1, y1, x2, y2 = item["box"]
        cls_id = item["class"]
        cls_name = BDD_CLASSES.get(cls_id, str(cls_id))

        if mode == "gt":
            text = cls_name
            outline = "green"
        else:
            conf = item.get("confidence", 0.0)
            text = f"{cls_name} {conf:.2f}"
            outline = "red"

        draw.rectangle([x1, y1, x2, y2], outline=outline, width=2)
        draw.text((x1, max(0, y1 - 10)), text, fill=outline, font=font)

    img.save(out_path)


def draw_side_by_side(image_path, gts, preds, out_path):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    left = img.copy()
    right = img.copy()

    draw_l = ImageDraw.Draw(left)
    draw_r = ImageDraw.Draw(right)
    font = ImageFont.load_default()

    for item in gts:
        x1, y1, x2, y2 = item["box"]
        cls_name = BDD_CLASSES.get(item["class"], str(item["class"]))
        draw_l.rectangle([x1, y1, x2, y2], outline="green", width=2)
        draw_l.text((x1, max(0, y1 - 10)), cls_name, fill="green", font=font)

    for item in preds:
        x1, y1, x2, y2 = item["box"]
        cls_name = BDD_CLASSES.get(item["class"], str(item["class"]))
        conf = item.get("confidence", 0.0)
        draw_r.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw_r.text((x1, max(0, y1 - 10)), f"{cls_name} {conf:.2f}", fill="red", font=font)

    canvas = Image.new("RGB", (w * 2, h), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (w, 0))

    draw_c = ImageDraw.Draw(canvas)
    draw_c.text((10, 10), "Ground Truth", fill="green", font=font)
    draw_c.text((w + 10, 10), "Prediction", fill="red", font=font)

    canvas.save(out_path)


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ==========================================================
# FIX: Class-aware matching
# ==========================================================
# The original code matched predictions to GTs using IoU alone,
# ignoring class. This caused cross-class overlaps (e.g. a car
# prediction matched to a pedestrian GT) to consume the GT slot,
# preventing any same-class prediction from claiming it as a TP.
# The result: classes that spatially overlap with cars (pedestrian,
# rider, motorcycle, bicycle) showed near-zero TPs in the diagonal.
#
# Fix: only allow a match between a prediction and a GT if they
# share the same class. Misclassified but overlapping boxes are
# counted as separate FP (pred) + FN (gt) — which is the standard
# detection evaluation convention (class-conditional matching).
# ==========================================================

def match_predictions_to_gts(preds, gts, iou_threshold):
    """
    Greedy matching: a prediction can only match a GT of the SAME class.
    Sorted by descending IoU so the best-overlapping pair is taken first.

    Returns:
        matches          : list of (pred_idx, gt_idx, iou_value)
        unmatched_preds  : set of pred indices with no matching GT
        unmatched_gts    : set of gt  indices with no matching pred
    """
    candidates = []
    for pi, pred in enumerate(preds):
        for gi, gt in enumerate(gts):
            # KEY FIX: only consider pairs of the same class
            if pred["class"] != gt["class"]:
                continue
            ov = iou(pred["box"], gt["box"])
            if ov >= iou_threshold:
                candidates.append((ov, pi, gi))

    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_preds = set()
    matched_gts = set()
    matches = []

    for ov, pi, gi in candidates:
        if pi in matched_preds or gi in matched_gts:
            continue
        matched_preds.add(pi)
        matched_gts.add(gi)
        matches.append((pi, gi, ov))

    unmatched_preds = set(range(len(preds))) - matched_preds
    unmatched_gts   = set(range(len(gts)))  - matched_gts

    return matches, unmatched_preds, unmatched_gts


# ==========================================================
# MAIN
# ==========================================================

def main():
    ensure_dirs()

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")
    if not LABEL_DIR.exists():
        raise FileNotFoundError(f"Label directory not found: {LABEL_DIR}")

    metadata_lookup = load_metadata(METADATA_CSV)

    print("Loading YOLO11m...")
    model = YOLO(MODEL_PATH)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    per_class = {
        cid: {"tp": 0, "fp": 0, "fn": 0}
        for cid in BDD_CLASSES
    }

    attribute_stats = {
        "weather":   defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "images": 0}),
        "scene":     defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "images": 0}),
        "timeofday": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "images": 0}),
    }

    # Confusion matrix: confusion[gt_cls][pred_cls] = count
    # A cell is incremented when a GT box has NO same-class match but a
    # different-class prediction overlaps it (IoU >= threshold).
    confusion = {
        gt: {pred: 0 for pred in BDD_CLASSES}
        for gt in BDD_CLASSES
    }

    image_rows = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    qual_saved = 0

    image_paths = sorted([
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    ])

    print(f"Found {len(image_paths)} validation images.")

    for idx, img_path in enumerate(image_paths, start=1):
        try:
            meta = (
                metadata_lookup.get(img_path.name)
                or metadata_lookup.get(img_path.stem)
                or {"weather": "unknown", "scene": "unknown", "timeofday": "unknown"}
            )

            label_path = LABEL_DIR / f"{img_path.stem}.txt"

            img = Image.open(img_path)
            img_w, img_h = img.size

            gts   = load_ground_truth(label_path, img_w, img_h)
            preds = get_predictions(model, img_path, device)

            # ---- class-aware TP/FP/FN matching ----
            matches, unmatched_pred_idxs, unmatched_gt_idxs = match_predictions_to_gts(
                preds, gts, IOU_THRESHOLD
            )

            image_tp = 0
            image_fp = 0
            image_fn = 0

            # Every match is a same-class TP (class equality is guaranteed by the matcher)
            for pi, gi, ov in matches:
                cls_id = gts[gi]["class"]           # == preds[pi]["class"]
                per_class[cls_id]["tp"] += 1
                confusion[cls_id][cls_id] += 1
                image_tp += 1

            # Unmatched predictions => FP
            for pi in unmatched_pred_idxs:
                pred_cls = preds[pi]["class"]
                per_class[pred_cls]["fp"] += 1
                image_fp += 1

            # Unmatched GTs => FN
            # Additionally check whether any pred (wrong class) overlaps
            # this GT to populate the off-diagonal confusion cells.
            for gi in unmatched_gt_idxs:
                gt_cls  = gts[gi]["class"]
                per_class[gt_cls]["fn"] += 1
                image_fn += 1

                # Find the highest-IoU prediction that overlaps this GT
                # (regardless of class) to attribute the confusion.
                best_iou   = 0.0
                best_pred_cls = None
                for pi, pred in enumerate(preds):
                    if pred["class"] == gt_cls:
                        continue  # same-class misses are pure FN, not confusions
                    ov = iou(pred["box"], gts[gi]["box"])
                    if ov >= IOU_THRESHOLD and ov > best_iou:
                        best_iou = ov
                        best_pred_cls = pred["class"]

                if best_pred_cls is not None:
                    confusion[gt_cls][best_pred_cls] += 1
                # If no overlapping pred at all, the box was simply missed (pure FN).
                # Those cases do NOT appear in the confusion matrix off-diagonal
                # because there is no competing prediction to attribute them to.

            total_tp += image_tp
            total_fp += image_fp
            total_fn += image_fn

            # Attribute aggregation
            for attr_name in ("weather", "scene", "timeofday"):
                attr_value = (meta.get(attr_name) or "unknown").strip() or "unknown"
                attribute_stats[attr_name][attr_value]["tp"]     += image_tp
                attribute_stats[attr_name][attr_value]["fp"]     += image_fp
                attribute_stats[attr_name][attr_value]["fn"]     += image_fn
                attribute_stats[attr_name][attr_value]["images"] += 1

            precision = safe_div(image_tp, image_tp + image_fp)
            recall    = safe_div(image_tp, image_tp + image_fn)
            f1        = safe_div(2 * precision * recall, precision + recall)

            image_rows.append({
                "image_name": img_path.name,
                "weather":    meta.get("weather",   "unknown"),
                "scene":      meta.get("scene",     "unknown"),
                "timeofday":  meta.get("timeofday", "unknown"),
                "gt_count":   len(gts),
                "pred_count": len(preds),
                "tp": image_tp,
                "fp": image_fp,
                "fn": image_fn,
                "precision": precision,
                "recall":    recall,
                "f1":        f1,
            })

            # Qualitative examples
            if qual_saved < QUAL_LIMIT:
                draw_boxes(img_path, gts,   QUAL_DIR / f"{img_path.stem}_gt.jpg",   mode="gt")
                draw_boxes(img_path, preds, QUAL_DIR / f"{img_path.stem}_pred.jpg", mode="pred")
                qual_saved += 1

            if idx % 250 == 0:
                print(f"Processed {idx}/{len(image_paths)} images...")

        except Exception as e:
            print(f"[WARN] Skipping {img_path.name}: {e}")

    # ======================================================
    # PER-CLASS METRICS
    # ======================================================

    class_rows = []
    supported_precisions = []
    supported_recalls    = []
    supported_f1s        = []

    for cid, name in BDD_CLASSES.items():
        tp = per_class[cid]["tp"]
        fp = per_class[cid]["fp"]
        fn = per_class[cid]["fn"]

        precision = safe_div(tp, tp + fp)
        recall    = safe_div(tp, tp + fn)
        f1        = safe_div(2 * precision * recall, precision + recall)
        support   = tp + fn

        class_rows.append({
            "class_id":   cid,
            "class_name": name,
            "tp": tp, "fp": fp, "fn": fn,
            "support":    support,
            "precision":  precision,
            "recall":     recall,
            "f1":         f1,
        })

        if support > 0:
            supported_precisions.append(precision)
            supported_recalls.append(recall)
            supported_f1s.append(f1)

    write_csv(
        OUTPUT_DIR / "per_class_metrics.csv",
        ["class_id", "class_name", "tp", "fp", "fn", "support", "precision", "recall", "f1"],
        class_rows
    )

    # ======================================================
    # OVERALL METRICS
    # ======================================================

    micro_precision = safe_div(total_tp, total_tp + total_fp)
    micro_recall    = safe_div(total_tp, total_tp + total_fn)
    micro_f1        = safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    macro_precision = safe_div(sum(supported_precisions), len(supported_precisions))
    macro_recall    = safe_div(sum(supported_recalls),    len(supported_recalls))
    macro_f1        = safe_div(sum(supported_f1s),        len(supported_f1s))

    summary = {
        "model": MODEL_PATH,
        "dataset_images_processed": len(image_paths),
        "iou_threshold":    IOU_THRESHOLD,
        "confidence_threshold": CONF_THRESHOLD,
        "micro_precision":  micro_precision,
        "micro_recall":     micro_recall,
        "micro_f1":         micro_f1,
        "macro_precision_supported_classes": macro_precision,
        "macro_recall_supported_classes":    macro_recall,
        "macro_f1_supported_classes":        macro_f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # ======================================================
    # ATTRIBUTE METRICS
    # ======================================================

    attribute_output = {}
    for attr_name, attr_values in attribute_stats.items():
        rows = []
        for attr_value, stats in sorted(attr_values.items(), key=lambda x: str(x[0])):
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]

            precision = safe_div(tp, tp + fp)
            recall    = safe_div(tp, tp + fn)
            f1        = safe_div(2 * precision * recall, precision + recall)

            rows.append({
                attr_name:  attr_value,
                "images":   stats["images"],
                "tp": tp, "fp": fp, "fn": fn,
                "precision": precision,
                "recall":    recall,
                "f1":        f1,
            })

        write_csv(
            ATTR_DIR / f"{attr_name}_metrics.csv",
            [attr_name, "images", "tp", "fp", "fn", "precision", "recall", "f1"],
            rows
        )
        attribute_output[attr_name] = rows

    # ======================================================
    # CONFUSION MATRIX CSV
    # ======================================================

    confusion_path = OUTPUT_DIR / "confusion_matrix.csv"
    with open(confusion_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["gt/pred"] + [BDD_CLASSES[cid] for cid in sorted(BDD_CLASSES)]
        writer.writerow(header)
        for gt_id in sorted(BDD_CLASSES):
            row = [BDD_CLASSES[gt_id]]
            for pred_id in sorted(BDD_CLASSES):
                row.append(confusion[gt_id][pred_id])
            writer.writerow(row)

    # ======================================================
    # IMAGE-LEVEL CSV
    # ======================================================

    write_csv(
        OUTPUT_DIR / "image_level_metrics.csv",
        ["image_name", "weather", "scene", "timeofday", "gt_count", "pred_count",
         "tp", "fp", "fn", "precision", "recall", "f1"],
        image_rows
    )

    # ======================================================
    # BEST / WORST CASES
    # ======================================================

    sorted_rows = sorted(image_rows, key=lambda r: r["f1"])
    worst_rows  = sorted_rows[:TOP_K_CASES]
    best_rows   = list(reversed(sorted_rows[-TOP_K_CASES:]))

    write_csv(
        FAIL_DIR / "worst_cases.csv",
        ["image_name", "weather", "scene", "timeofday", "gt_count", "pred_count",
         "tp", "fp", "fn", "precision", "recall", "f1"],
        worst_rows
    )
    write_csv(
        BEST_DIR / "best_cases.csv",
        ["image_name", "weather", "scene", "timeofday", "gt_count", "pred_count",
         "tp", "fp", "fn", "precision", "recall", "f1"],
        best_rows
    )

    for case_name, rows, out_dir in [("best", best_rows, BEST_DIR), ("worst", worst_rows, FAIL_DIR)]:
        for row in rows:
            img_path = IMAGE_DIR / row["image_name"]
            if not img_path.exists():
                continue
            try:
                img_w, img_h = Image.open(img_path).size
                gts   = load_ground_truth(LABEL_DIR / f"{img_path.stem}.txt", img_w, img_h)
                preds = get_predictions(model, img_path, device)
                draw_side_by_side(img_path, gts, preds, out_dir / f"{img_path.stem}_{case_name}.jpg")
            except Exception as e:
                print(f"[WARN] Could not save {case_name} case {img_path.name}: {e}")

    # ======================================================
    # REPORT
    # ======================================================

    worst_lines = []
    for attr_name, rows in attribute_output.items():
        if not rows:
            continue
        worst_row = min(rows, key=lambda r: r["f1"])
        best_row  = max(rows, key=lambda r: r["f1"])
        worst_lines.append(
            f"{attr_name}: worst={worst_row[attr_name]} (F1={worst_row['f1']:.4f}), "
            f"best={best_row[attr_name]} (F1={best_row['f1']:.4f})"
        )

    report_lines = [
        "FAILURE ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Model: {MODEL_PATH}",
        f"Images processed: {len(image_paths)}",
        f"IoU threshold: {IOU_THRESHOLD}",
        f"Confidence threshold: {CONF_THRESHOLD}",
        "",
        "OVERALL METRICS",
        f"Micro Precision : {micro_precision:.4f}",
        f"Micro Recall    : {micro_recall:.4f}",
        f"Micro F1        : {micro_f1:.4f}",
        f"Macro Precision : {macro_precision:.4f}",
        f"Macro Recall    : {macro_recall:.4f}",
        f"Macro F1        : {macro_f1:.4f}",
        "",
        "WORST / BEST ATTRIBUTE CONDITIONS",
        *worst_lines,
        "",
        "LIKELY FAILURE PATTERNS",
        "- Small objects such as distant traffic signs are harder to detect.",
        "- Night and low-light scenes usually reduce recall.",
        "- Crowded city scenes increase false positives and localization errors.",
        "- COCO stop-sign predictions are only a partial proxy for BDD traffic-sign labels.",
        "- COCO has no 'rider' class; rider GTs are always FN unless the model fires 'person' at them.",
        "- COCO 'person' maps to BDD 'pedestrian' only; riders on bikes are structurally missed.",
        "",
    ]

    with open(OUTPUT_DIR / "failure_analysis.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # ======================================================
    # OPTIONAL PLOTS
    # ======================================================

    try:
        import matplotlib.pyplot as plt

        # Per-class F1
        class_names = [row["class_name"] for row in class_rows]
        class_f1    = [row["f1"]         for row in class_rows]

        plt.figure(figsize=(12, 6))
        plt.bar(class_names, class_f1)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("F1")
        plt.title("Per-Class F1 Score")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "per_class_f1.png", dpi=200)
        plt.close()

        # Attribute F1
        for attr_name, rows in attribute_output.items():
            if not rows:
                continue
            labels = [str(r[attr_name]) for r in rows]
            f1s    = [r["f1"]           for r in rows]

            plt.figure(figsize=(10, 5))
            plt.bar(labels, f1s)
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("F1")
            plt.title(f"F1 by {attr_name}")
            plt.tight_layout()
            plt.savefig(ATTR_DIR / f"{attr_name}_f1.png", dpi=200)
            plt.close()

        # Confusion matrix heatmap
        labels = [BDD_CLASSES[cid] for cid in sorted(BDD_CLASSES)]
        matrix = [
            [confusion[gt_id][pred_id] for pred_id in sorted(BDD_CLASSES)]
            for gt_id in sorted(BDD_CLASSES)
        ]

        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, interpolation="nearest")
        plt.colorbar()
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("Confusion Matrix (off-diagonal = cross-class IoU overlap)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
        plt.close()

    except Exception as e:
        print(f"[WARN] Plot generation skipped: {e}")

    # ======================================================
    # PRINT SUMMARY
    # ======================================================

    print("\n===== OVERALL SUMMARY =====")
    print(json.dumps(summary, indent=4))
    print("\nSaved files:")
    print(f"- {OUTPUT_DIR / 'summary.json'}")
    print(f"- {OUTPUT_DIR / 'per_class_metrics.csv'}")
    print(f"- {OUTPUT_DIR / 'confusion_matrix.csv'}")
    print(f"- {OUTPUT_DIR / 'image_level_metrics.csv'}")
    print(f"- {OUTPUT_DIR / 'failure_analysis.txt'}")
    print(f"- {OUTPUT_DIR / 'confusion_matrix.png'}")
    print(f"- {ATTR_DIR}")
    print(f"- {QUAL_DIR}")
    print(f"- {BEST_DIR}")
    print(f"- {FAIL_DIR}")


if __name__ == "__main__":
    main()
