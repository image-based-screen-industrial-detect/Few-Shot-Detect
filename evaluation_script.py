import os
import zipfile
import tempfile
import shutil
import xml.etree.ElementTree as ET
import numpy as np

def parse_voc_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        cls = obj.find('name').text
        bndbox = obj.find('bndbox')
        bbox = [
            int(bndbox.find('xmin').text),
            int(bndbox.find('ymin').text),
            int(bndbox.find('xmax').text),
            int(bndbox.find('ymax').text)
        ]
        boxes.append({'class': cls, 'bbox': bbox})
    return boxes

def compute_iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def evaluate(test_annotation_file, user_annotation_file, phase_codename, **kwargs):
    IOU_THRESHOLD = 0.95
    temp_dir = tempfile.mkdtemp()
    gt_dir = os.path.join(temp_dir, "gt")
    pred_dir = os.path.join(temp_dir, "pred")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(test_annotation_file, "r") as zip_ref:
            zip_ref.extractall(gt_dir)
        with zipfile.ZipFile(user_annotation_file, "r") as zip_ref:
            zip_ref.extractall(pred_dir)

        classwise_precisions = {}

        for file_name in os.listdir(gt_dir):
            if not file_name.endswith(".xml"):
                continue
            gt_path = os.path.join(gt_dir, file_name)
            pred_path = os.path.join(pred_dir, file_name)
            gt_objs = parse_voc_xml(gt_path)
            pred_objs = parse_voc_xml(pred_path) if os.path.exists(pred_path) else []

            gt_by_class = {}
            for obj in gt_objs:
                gt_by_class.setdefault(obj['class'], []).append(obj['bbox'])

            pred_by_class = {}
            for obj in pred_objs:
                pred_by_class.setdefault(obj['class'], []).append(obj['bbox'])

            for cls in set(gt_by_class.keys()).union(set(pred_by_class.keys())):
                gt_boxes = gt_by_class.get(cls, [])
                pred_boxes = pred_by_class.get(cls, [])

                matched_gt = set()
                tp = 0
                fp = 0

                for pb in pred_boxes:
                    match_found = False
                    for i, gb in enumerate(gt_boxes):
                        if i in matched_gt:
                            continue
                        if compute_iou(pb, gb) >= IOU_THRESHOLD:
                            matched_gt.add(i)
                            match_found = True
                            break
                    if match_found:
                        tp += 1
                    else:
                        fp += 1
                fn = len(gt_boxes) - len(matched_gt)
                precision = tp / (tp + fp + 1e-6)

                if cls not in classwise_precisions:
                    classwise_precisions[cls] = []
                classwise_precisions[cls].append(precision)

        ap_per_class = {cls: np.mean(prec) for cls, prec in classwise_precisions.items()}
        map_095 = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
        rounded_map = round(map_095 * 100, 2)

        return {
            "result": [{
                phase_codename: {
                    "mAP@0.95": rounded_map
                }
            }],
            "metadata": {
                "mean_ap_0.95": rounded_map,
                "ap_per_class": {k: round(v * 100, 2) for k, v in ap_per_class.items()}
            }
        }

    except Exception as e:
        return {
            "result": [{
                phase_codename: {
                    "mAP@0.95": 0.0
                }
            }],
            "error": str(e)
        }
    finally:
        shutil.rmtree(temp_dir)
