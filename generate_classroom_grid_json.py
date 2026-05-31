import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sklearn.cluster import DBSCAN


# =========================================================
# 路径配置
# =========================================================
SEG_MODEL_PATH = r"D:\desktop\classroom\classroomtrain\runs\segment\runs\classroom_seg_v2\weights\best.pt"

# 支持图片或视频
IMAGE_OR_VIDEO_PATH = r"D:\desktop\classroom\Desktop 2026.04.29 - 16.36.16.01.mp4"

# 如果输入是视频，使用第几帧作为建模帧
FRAME_INDEX = 0

OUT_JSON = Path(r"D:\desktop\classroom\classroom_grid_dynamic.json")
OUT_VIS = Path(r"D:\desktop\classroom\classroom_grid_vis.jpg")


# =========================================================
# 参数配置
# =========================================================
CONF_THRESH = 0.20

# 课桌类别编号
# 如果检测不到课桌，尝试改为 0
DESK_CLASS_ID = 1

# DBSCAN 行聚类参数
MIN_EPS = 20
EPS_RATIO = 0.35


# =========================================================
# 读取图片或视频帧
# =========================================================
def load_image_or_video_frame(path, frame_index=0):
    path = str(path)
    suffix = Path(path).suffix.lower()

    # 图片格式
    if suffix in [".jpg", ".jpeg", ".png", ".bmp"]:
        img = cv2.imread(path)

        if img is None:
            raise FileNotFoundError(f"无法读取图片: {path}")

        return img

    # 视频格式
    if suffix in [".mp4", ".avi", ".mov", ".mkv"]:
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_index >= total_frames:
            cap.release()
            raise ValueError(
                f"FRAME_INDEX 超出视频总帧数。当前 FRAME_INDEX={frame_index}, 视频总帧数={total_frames}"
            )

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"无法读取视频第 {frame_index} 帧: {path}")

        return frame

    raise ValueError(f"不支持的文件格式: {suffix}")


# =========================================================
# YOLOv8-Seg 检测课桌
# =========================================================
def detect_desks(model, img):
    result = model(img, conf=CONF_THRESH, verbose=False)[0]

    boxes = []

    if result.boxes is None or len(result.boxes) == 0:
        return boxes

    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()

    for box, c, s in zip(xyxy, cls, conf):
        if DESK_CLASS_ID is not None and int(c) != DESK_CLASS_ID:
            continue

        x1, y1, x2, y2 = box

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([
            int(x1),
            int(y1),
            int(x2),
            int(y2),
            float(s)
        ])

    return boxes


# =========================================================
# 根据课桌框生成座位网格
# =========================================================
def generate_grid(boxes):
    if len(boxes) == 0:
        return []

    centers = []

    for b in boxes:
        x1, y1, x2, y2, conf = b
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append([cx, cy])

    centers = np.array(centers)

    y_values = centers[:, 1].reshape(-1, 1)

    y_std = float(np.std(centers[:, 1]))
    eps = max(MIN_EPS, y_std * EPS_RATIO)

    clustering = DBSCAN(
        eps=eps,
        min_samples=1
    ).fit(y_values)

    labels = clustering.labels_

    rows = {}

    for idx, label in enumerate(labels):
        rows.setdefault(label, []).append(idx)

    sorted_rows = sorted(
        rows.items(),
        key=lambda item: np.mean([centers[i][1] for i in item[1]])
    )

    seats = []

    for row_idx, (_, indices) in enumerate(sorted_rows, start=1):
        indices_sorted = sorted(indices, key=lambda i: centers[i][0])

        for col_idx, box_idx in enumerate(indices_sorted, start=1):
            x1, y1, x2, y2, conf = boxes[box_idx]
            cx, cy = centers[box_idx]

            seat = {
                "seat_id": f"R{row_idx}C{col_idx}",
                "row": row_idx,
                "col": col_idx,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center": [int(cx), int(cy)],
                "conf": round(float(conf), 4)
            }

            seats.append(seat)

    return seats


# =========================================================
# 可视化座位网格
# =========================================================
def draw_grid(img, seats):
    vis = img.copy()

    for seat in seats:
        x1, y1, x2, y2 = seat["bbox"]
        cx, cy = seat["center"]
        seat_id = seat["seat_id"]

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)

        cv2.putText(
            vis,
            seat_id,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    cv2.putText(
        vis,
        f"Seats: {len(seats)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2
    )

    return vis


# =========================================================
# 主函数
# =========================================================
def main():
    print("读取建模图像/视频帧...")
    img = load_image_or_video_frame(
        IMAGE_OR_VIDEO_PATH,
        frame_index=FRAME_INDEX
    )

    h, w = img.shape[:2]

    print(f"图像尺寸: {w} x {h}")
    print(f"建模帧编号: {FRAME_INDEX}")

    print("加载 YOLOv8-Seg 课桌分割模型...")
    model = YOLO(SEG_MODEL_PATH)

    print("检测课桌...")
    boxes = detect_desks(model, img)

    print(f"检测到课桌框数量: {len(boxes)}")

    if len(boxes) == 0:
        print("\n未检测到课桌。")
        print("建议检查：")
        print("1. DESK_CLASS_ID 是否正确，可尝试改为 0")
        print("2. CONF_THRESH 是否过高，可尝试改为 0.10")
        print("3. FRAME_INDEX 对应画面是否包含清晰课桌")
        return

    print("生成座位网格...")
    seats = generate_grid(boxes)

    data = {
        "source": str(IMAGE_OR_VIDEO_PATH),
        "frame_index": FRAME_INDEX,
        "image_width": w,
        "image_height": h,
        "seat_count": len(seats),
        "seats": seats
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    vis = draw_grid(img, seats)
    cv2.imwrite(str(OUT_VIS), vis)

    print(f"\n座位网格 JSON 已保存: {OUT_JSON.resolve()}")
    print(f"座位网格可视化图已保存: {OUT_VIS.resolve()}")
    print(f"座位数量: {len(seats)}")


if __name__ == "__main__":
    main()