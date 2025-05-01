
import json
import numpy as np

def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

def evaluate(ground_truths, predictions, iou_threshold=0.95):
    gt_by_image = {gt["image_id"]: [] for gt in ground_truths}
    for gt in ground_truths:
        gt_by_image[gt["image_id"]].append(gt)

    aps = []
    for image_id, preds in predictions.items():
        preds = sorted(preds, key=lambda x: -x["score"])
        gt = gt_by_image.get(image_id, [])

        matched = set()
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        for i, pred in enumerate(preds):
            found_match = False
            for j, g in enumerate(gt):
                if j not in matched and pred["category_id"] == g["category_id"]:
                    if iou(pred["bbox"], g["bbox"]) >= iou_threshold:
                        found_match = True
                        matched.add(j)
                        break
            tp[i] = 1 if found_match else 0
            fp[i] = 0 if found_match else 1

        if len(tp) == 0:
            aps.append(0)
            continue

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / (len(gt) + 1e-6)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-6)

        ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[:-1])
        aps.append(ap)

    return {"mAP@0.95": np.mean(aps)}

def main():
    with open("ground_truths.json") as f:
        ground_truths = json.load(f)
    with open("predictions.json") as f:
        predictions_raw = json.load(f)

    predictions = {}
    for pred in predictions_raw:
        predictions.setdefault(pred["image_id"], []).append(pred)

    results = evaluate(ground_truths, predictions)
    with open("result.json", "w") as f:
        json.dump(results, f)
        
if __name__ == "__main__":
    main()
