import cv2
import numpy as np
import time
import csv
from pathlib import Path
from ultralytics import YOLO

# ================= 配置区 =================
IMG_PATH = r"D:\desktop\classroom\classroom_empty1.jpg"
MODEL_PATH = r"D:\desktop\classroom\classroomtrain\runs\segment\runs\classroom_seg_v2\weights\best.pt"

# 根据你的data.yaml和代码设置
DESK_IDS = [1]     # class_1：桌子
CHAIR_IDS = [2]    # class_2：椅子

OUT_DIR = Path("compare_results")
OUT_DIR.mkdir(exist_ok=True)

# YOLO参数分析
CONF_LIST = [0.10, 0.15, 0.25, 0.35]
# ==========================================


def traditional_hsv_seg(img):
    """
    传统图像处理方法：
    HSV阈值分割 + 形态学处理 + 轮廓提取
    主要用于提取浅绿色/浅青色桌面区域
    """
    start = time.time()

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 该阈值主要针对浅绿色/浅青色课桌，可根据图像微调
    lower = np.array([35, 5, 100])
    upper = np.array([105, 120, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # 限制检测区域，去掉图像上方墙面、黑板等区域
    roi_mask = np.zeros_like(mask)
    roi_mask[int(h * 0.18):, :] = 255
    mask = cv2.bitwise_and(mask, roi_mask)

    # 形态学处理，去除噪声并填补空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_mask = np.zeros_like(mask)
    boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1500:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / max(bh, 1)

        # 根据桌面形状过滤异常区域
        if aspect < 0.6 or aspect > 6.0:
            continue

        boxes.append([x, y, x + bw, y + bh])
        cv2.drawContours(result_mask, [cnt], -1, 255, -1)

    # 可视化
    overlay = img.copy()
    green = np.zeros_like(img)
    green[:, :, 1] = 255
    overlay[result_mask > 0] = cv2.addWeighted(
        overlay, 0.4, green, 0.6, 0
    )[result_mask > 0]

    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overlay, "HSV", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    elapsed = (time.time() - start) * 1000

    return {
        "mask": result_mask,
        "overlay": overlay,
        "boxes": boxes,
        "count": len(boxes),
        "time_ms": elapsed
    }


def yolo_seg(img, model, conf):
    """
    YOLOv8-Seg桌椅实例分割
    """
    start = time.time()

    results = model(img, conf=conf, iou=0.5, device=0, verbose=False)
    r = results[0]

    h, w = img.shape[:2]
    desk_mask = np.zeros((h, w), dtype=np.uint8)
    chair_mask = np.zeros((h, w), dtype=np.uint8)

    desk_boxes = []
    chair_boxes = []

    if r.boxes is not None:
        for i in range(len(r.boxes.cls)):
            cls_id = int(r.boxes.cls[i])
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].cpu().numpy())

            if cls_id in DESK_IDS:
                desk_boxes.append([x1, y1, x2, y2])
                if r.masks is not None:
                    pts = r.masks.xy[i].astype(np.int32)
                    cv2.fillPoly(desk_mask, [pts], 255)

            elif cls_id in CHAIR_IDS:
                chair_boxes.append([x1, y1, x2, y2])
                if r.masks is not None:
                    pts = r.masks.xy[i].astype(np.int32)
                    cv2.fillPoly(chair_mask, [pts], 255)

    # 可视化桌子
    overlay = img.copy()
    green = np.zeros_like(img)
    green[:, :, 1] = 255
    overlay[desk_mask > 0] = cv2.addWeighted(
        overlay, 0.4, green, 0.6, 0
    )[desk_mask > 0]

    for x1, y1, x2, y2 in desk_boxes:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overlay, f"YOLO {conf}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    elapsed = (time.time() - start) * 1000

    return {
        "desk_mask": desk_mask,
        "chair_mask": chair_mask,
        "overlay": overlay,
        "desk_boxes": desk_boxes,
        "chair_boxes": chair_boxes,
        "desk_count": len(desk_boxes),
        "chair_count": len(chair_boxes),
        "time_ms": elapsed
    }


def mask_iou(mask1, mask2):
    """
    计算两个mask之间的IoU。
    注意：这里不是和人工标注GT比较，只用于观察传统方法与YOLO结果的一致性。
    """
    m1 = mask1 > 0
    m2 = mask2 > 0

    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()

    if union == 0:
        return 0.0

    return inter / union


def main():
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {IMG_PATH}")

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"模型不存在: {MODEL_PATH}")

    print("正在运行传统HSV分割方法...")
    trad = traditional_hsv_seg(img)

    cv2.imwrite(str(OUT_DIR / "traditional_hsv_overlay.jpg"), trad["overlay"])
    cv2.imwrite(str(OUT_DIR / "traditional_hsv_mask.jpg"), trad["mask"])

    print(f"传统方法检测桌子区域数量: {trad['count']}")
    print(f"传统方法耗时: {trad['time_ms']:.2f} ms")

    print("\n正在加载YOLOv8-Seg模型...")
    model = YOLO(MODEL_PATH)

    rows = []
    rows.append([
        "HSV阈值+轮廓提取",
        "-",
        trad["count"],
        "-",
        f"{trad['time_ms']:.2f}",
        "-"
    ])

    for conf in CONF_LIST:
        print(f"\n正在运行YOLOv8-Seg, conf={conf}...")
        yolo = yolo_seg(img, model, conf)

        iou = mask_iou(trad["mask"], yolo["desk_mask"])

        out_name = f"yolo_seg_conf_{conf:.2f}.jpg"
        cv2.imwrite(str(OUT_DIR / out_name), yolo["overlay"])
        cv2.imwrite(str(OUT_DIR / f"yolo_mask_conf_{conf:.2f}.jpg"), yolo["desk_mask"])

        print(f"YOLO桌子数量: {yolo['desk_count']}")
        print(f"YOLO椅子数量: {yolo['chair_count']}")
        print(f"YOLO耗时: {yolo['time_ms']:.2f} ms")
        print(f"与传统方法mask IoU: {iou:.3f}")

        rows.append([
            "YOLOv8-Seg",
            conf,
            yolo["desk_count"],
            yolo["chair_count"],
            f"{yolo['time_ms']:.2f}",
            f"{iou:.3f}"
        ])

    # 保存CSV结果
    csv_path = OUT_DIR / "compare_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "方法",
            "置信度阈值",
            "桌子检测数量",
            "椅子检测数量",
            "处理时间/ms",
            "与传统方法Mask IoU"
        ])
        writer.writerows(rows)

    print(f"\n实验完成，结果已保存到: {OUT_DIR.resolve()}")
    print(f"CSV结果文件: {csv_path.resolve()}")


if __name__ == "__main__":
    main()