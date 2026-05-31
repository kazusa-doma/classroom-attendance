import cv2
import yaml
import time
import csv
import random
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


# ================= 配置区 =================
DATASET_ROOT = Path(r"D:\desktop\classroom\dataset_yolo_seg")
DATA_YAML = DATASET_ROOT / "data.yaml"

TEST_IMG = Path(r"D:\desktop\classroom\classroom_empty1.jpg")

OUT_DIR = Path("compare_kmeans_adaboost_results")
OUT_DIR.mkdir(exist_ok=True)

# 类别设置
DESK_ID = 1
CHAIR_ID = 2

# K-means参数
N_CLUSTERS = 7

# 候选框过滤参数
MIN_AREA = 800
MAX_AREA_RATIO = 0.25

# 训练样本数量控制
MAX_TRAIN_IMAGES = 400
MAX_NEGATIVE_PER_IMAGE = 15

# IoU阈值
IOU_POS_THRESH = 0.25
IOU_EVAL_THRESH = 0.5

# AdaBoost概率阈值
PROB_THRESH = 0.45

# NMS阈值
NMS_THRESH = 0.35

RANDOM_SEED = 42
# ==========================================


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_yaml():
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_yolo_seg_label(label_path, img_w, img_h):
    """
    读取YOLO分割标签，转为bbox格式。
    label格式:
    class_id x1 y1 x2 y2 ... xn yn
    """
    objects = []

    if not label_path.exists():
        return objects

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue

        cls_id = int(float(parts[0]))
        coords = list(map(float, parts[1:]))

        xs = coords[0::2]
        ys = coords[1::2]

        xs = [int(x * img_w) for x in xs]
        ys = [int(y * img_h) for y in ys]

        x1, y1 = max(0, min(xs)), max(0, min(ys))
        x2, y2 = min(img_w - 1, max(xs)), min(img_h - 1, max(ys))

        if x2 <= x1 or y2 <= y1:
            continue

        objects.append({
            "cls": cls_id,
            "bbox": [x1, y1, x2, y2]
        })

    return objects


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter
    if union <= 0:
        return 0.0

    return inter / union


def nms_boxes(boxes, scores, iou_thresh=0.35):
    """
    对候选框进行NMS去重
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def generate_kmeans_candidates(img, debug_name=None):
    """
    使用K-means颜色聚类生成候选区域
    """
    h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 降采样加速K-means
    small = cv2.resize(hsv, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)

    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_SEED,
        batch_size=4096,
        n_init=3
    )

    labels = kmeans.fit_predict(pixels)
    label_small = labels.reshape(small.shape[:2])
    label_img = cv2.resize(label_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    candidates = []

    debug_vis = img.copy()

    for k in range(N_CLUSTERS):
        mask = np.uint8(label_img == k) * 255

        # 去除图像上方区域，减少墙面、黑板、门窗干扰
        roi_mask = np.zeros_like(mask)
        roi_mask[int(h * 0.18):, :] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area_contour = cv2.contourArea(cnt)

            if area_contour < MIN_AREA:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            box_area = bw * bh

            if box_area < MIN_AREA:
                continue

            if box_area > h * w * MAX_AREA_RATIO:
                continue

            if y < int(h * 0.18):
                continue

            if bw <= 8 or bh <= 8:
                continue

            aspect = bw / max(bh, 1)

            # 宽高比过滤，避免极端长条区域
            if aspect < 0.4 or aspect > 6.5:
                continue

            extent = area_contour / max(box_area, 1)

            # 太稀疏的轮廓通常是噪声
            if extent < 0.15:
                continue

            candidates.append({
                "bbox": [x, y, x + bw, y + bh],
                "contour": cnt,
                "cluster_id": k,
                "extent": extent
            })

            cv2.rectangle(debug_vis, (x, y), (x + bw, y + bh), (255, 255, 0), 1)

    if debug_name is not None:
        cv2.imwrite(str(OUT_DIR / debug_name), debug_vis)

    return candidates


def extract_features(img, candidate):
    """
    提取候选区域的颜色、形状、位置特征
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = candidate["bbox"]

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    mean_hsv = hsv.reshape(-1, 3).mean(axis=0)
    std_hsv = hsv.reshape(-1, 3).std(axis=0)

    bw = x2 - x1
    bh = y2 - y1
    area = bw * bh

    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h

    aspect = bw / max(bh, 1)
    area_ratio = area / (w * h)

    contour_area = cv2.contourArea(candidate["contour"])
    extent = contour_area / max(area, 1)

    # 灰度纹理特征
    gray_mean = gray.mean()
    gray_std = gray.std()

    features = [
        mean_hsv[0], mean_hsv[1], mean_hsv[2],
        std_hsv[0], std_hsv[1], std_hsv[2],
        gray_mean, gray_std,
        aspect,
        area_ratio,
        cx,
        cy,
        extent
    ]

    return features


def assign_label(candidate_box, gt_objects):
    """
    根据候选框与GT框IoU分配标签：
    1 桌子
    2 椅子
    0 背景
    """
    best_iou = 0.0
    best_cls = 0

    for obj in gt_objects:
        if obj["cls"] not in [DESK_ID, CHAIR_ID]:
            continue

        iou = box_iou(candidate_box, obj["bbox"])
        if iou > best_iou:
            best_iou = iou
            best_cls = obj["cls"]

    if best_iou >= IOU_POS_THRESH:
        return best_cls

    return 0


def get_image_paths(img_dir):
    img_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        img_paths.extend(list(img_dir.glob(ext)))
    return img_paths


def build_train_data():
    """
    从训练集构建AdaBoost训练数据
    """
    cfg = load_yaml()
    train_img_dir = DATASET_ROOT / cfg["train"]
    train_label_dir = train_img_dir.parent / "labels"

    img_paths = get_image_paths(train_img_dir)
    random.shuffle(img_paths)
    img_paths = img_paths[:MAX_TRAIN_IMAGES]

    X, y = [], []

    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        label_path = train_label_dir / (img_path.stem + ".txt")
        gt_objects = parse_yolo_seg_label(label_path, w, h)

        candidates = generate_kmeans_candidates(img)

        neg_count = 0
        pos_count = 0

        for cand in candidates:
            feat = extract_features(img, cand)
            if feat is None:
                continue

            label = assign_label(cand["bbox"], gt_objects)

            # 控制背景样本数量，避免类别极度不平衡
            if label == 0:
                if neg_count >= MAX_NEGATIVE_PER_IMAGE:
                    continue
                neg_count += 1
            else:
                pos_count += 1

            X.append(feat)
            y.append(label)

        print(f"[{idx + 1}/{len(img_paths)}] {img_path.name} 当前总样本: {len(y)} 正样本: {pos_count} 负样本: {neg_count}")

    return np.array(X), np.array(y)


def train_adaboost(X, y):
    """
    训练AdaBoost分类器
    """
    base_tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        random_state=RANDOM_SEED
    )

    try:
        clf = AdaBoostClassifier(
            estimator=base_tree,
            n_estimators=80,
            learning_rate=0.8,
            random_state=RANDOM_SEED
        )
    except TypeError:
        clf = AdaBoostClassifier(
            base_estimator=base_tree,
            n_estimators=80,
            learning_rate=0.8,
            random_state=RANDOM_SEED
        )

    clf.fit(X, y)
    return clf


def evaluate_on_val(clf):
    """
    在验证集上做简单分类评估
    """
    cfg = load_yaml()
    val_img_dir = DATASET_ROOT / cfg["val"]
    val_label_dir = val_img_dir.parent / "labels"

    img_paths = get_image_paths(val_img_dir)
    img_paths = img_paths[:80]

    y_true, y_pred = [], []

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        label_path = val_label_dir / (img_path.stem + ".txt")
        gt_objects = parse_yolo_seg_label(label_path, w, h)

        candidates = generate_kmeans_candidates(img)

        for cand in candidates:
            feat = extract_features(img, cand)
            if feat is None:
                continue

            true_label = assign_label(cand["bbox"], gt_objects)
            pred_label = clf.predict([feat])[0]

            y_true.append(true_label)
            y_pred.append(pred_label)

    print("\n========== AdaBoost验证集分类结果 ==========")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, DESK_ID, CHAIR_ID],
        target_names=["background", "desk", "chair"],
        digits=3,
        zero_division=0
    ))


def infer_one_image(clf, img_path):
    """
    对单张图像进行推理并可视化。
    加入概率阈值和NMS去重，避免识别过多。
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")

    start = time.time()

    candidates = generate_kmeans_candidates(img, debug_name="kmeans_candidates.jpg")

    desk_boxes, desk_scores = [], []
    chair_boxes, chair_scores = [], []

    for cand in candidates:
        feat = extract_features(img, cand)
        if feat is None:
            continue

        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba([feat])[0]
            classes = clf.classes_
            prob_dict = {cls: proba[i] for i, cls in enumerate(classes)}

            desk_prob = prob_dict.get(DESK_ID, 0.0)
            chair_prob = prob_dict.get(CHAIR_ID, 0.0)

            x1, y1, x2, y2 = cand["bbox"]
            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh
            aspect = bw / max(bh, 1)

            # 二次过滤异常区域
            if area < MIN_AREA:
                continue
            if area > img.shape[0] * img.shape[1] * MAX_AREA_RATIO:
                continue
            if aspect < 0.4 or aspect > 6.5:
                continue

            if desk_prob >= PROB_THRESH and desk_prob >= chair_prob:
                desk_boxes.append([x1, y1, x2, y2])
                desk_scores.append(desk_prob)

            elif chair_prob >= PROB_THRESH and chair_prob > desk_prob:
                chair_boxes.append([x1, y1, x2, y2])
                chair_scores.append(chair_prob)

        else:
            pred = clf.predict([feat])[0]
            if pred not in [DESK_ID, CHAIR_ID]:
                continue

            x1, y1, x2, y2 = cand["bbox"]

            if pred == DESK_ID:
                desk_boxes.append([x1, y1, x2, y2])
                desk_scores.append(1.0)
            elif pred == CHAIR_ID:
                chair_boxes.append([x1, y1, x2, y2])
                chair_scores.append(1.0)

    # NMS 去重
    desk_keep = nms_boxes(desk_boxes, desk_scores, NMS_THRESH)
    chair_keep = nms_boxes(chair_boxes, chair_scores, NMS_THRESH)

    result = img.copy()

    for i in desk_keep:
        x1, y1, x2, y2 = map(int, desk_boxes[i])
        score = desk_scores[i]
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result, f"Desk(Ada) {score:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    for i in chair_keep:
        x1, y1, x2, y2 = map(int, chair_boxes[i])
        score = chair_scores[i]
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(result, f"Chair(Ada) {score:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

    elapsed = (time.time() - start) * 1000

    out_path = OUT_DIR / "kmeans_adaboost_result.jpg"
    cv2.imwrite(str(out_path), result)

    print("\n========== 单图推理结果 ==========")
    print(f"图像: {img_path}")
    print(f"候选区域数量: {len(candidates)}")
    print(f"NMS前桌子数量: {len(desk_boxes)}")
    print(f"NMS前椅子数量: {len(chair_boxes)}")
    print(f"NMS后桌子数量: {len(desk_keep)}")
    print(f"NMS后椅子数量: {len(chair_keep)}")
    print(f"处理时间: {elapsed:.2f} ms")
    print(f"结果保存: {out_path.resolve()}")

    return {
        "candidate_count": len(candidates),
        "desk_before_nms": len(desk_boxes),
        "chair_before_nms": len(chair_boxes),
        "desk_count": len(desk_keep),
        "chair_count": len(chair_keep),
        "time_ms": elapsed
    }


def save_summary(result):
    csv_path = OUT_DIR / "kmeans_adaboost_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "方法",
            "候选区域数量",
            "NMS前桌子数量",
            "NMS前椅子数量",
            "NMS后桌子数量",
            "NMS后椅子数量",
            "处理时间/ms",
            "N_CLUSTERS",
            "MIN_AREA",
            "MAX_AREA_RATIO",
            "PROB_THRESH",
            "NMS_THRESH"
        ])
        writer.writerow([
            "K-means + AdaBoost",
            result["candidate_count"],
            result["desk_before_nms"],
            result["chair_before_nms"],
            result["desk_count"],
            result["chair_count"],
            f"{result['time_ms']:.2f}",
            N_CLUSTERS,
            MIN_AREA,
            MAX_AREA_RATIO,
            PROB_THRESH,
            NMS_THRESH
        ])

    print(f"汇总结果保存: {csv_path.resolve()}")


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"找不到data.yaml: {DATA_YAML}")

    if not TEST_IMG.exists():
        raise FileNotFoundError(f"找不到测试图片: {TEST_IMG}")

    print("开始构建 K-means 候选区域 + AdaBoost 训练数据...")
    X, y = build_train_data()

    if len(y) == 0:
        raise RuntimeError("训练样本为空，请检查数据集路径或候选区域生成参数。")

    print("\n========== 训练样本统计 ==========")
    print(f"训练样本总数: {len(y)}")
    unique, counts = np.unique(y, return_counts=True)
    print(f"类别分布: {dict(zip(unique, counts))}")

    print("\n开始训练 AdaBoost 分类器...")
    clf = train_adaboost(X, y)

    evaluate_on_val(clf)

    result = infer_one_image(clf, TEST_IMG)

    save_summary(result)

    print("\n全部完成。")


if __name__ == "__main__":
    main()