# step2_run_attendance.py
import cv2, json, torch, numpy as np
from collections import deque
from ultralytics import YOLO

# ================= 配置区 =================
DET_MODEL = r"D:\desktop\classroom\dataset\runs\detect\train3\weights\best.pt"
VIDEO_SRC = r"D:\desktop\classroom\Desktop 2026.04.29 - 16.36.16.01.mp4.mp4"
GRID_JSON = "D:\desktop\classroom\classroom_grid.json"  # 👈 刚生成的新JSON

CONF_THRESH = 0.05  # 极限低阈值，抓取后排（杂波交给网格过滤）
UP_SHIFT_RATIO = 1.5  # 桌面向上延伸 1.5 倍高度作为“合法吃人区”
SMOOTH_WIN = 8  # 防闪烁平滑


# ==========================================

def main():
    print("📊 加载刚才绝对对齐的网格配置...")
    with open(GRID_JSON, "r", encoding="utf-8") as f:
        grid_cfg = json.load(f)

    model = YOLO(DET_MODEL)
    cap = cv2.VideoCapture(VIDEO_SRC)

    # 🏗️ 构建松弛 ROI (只要人体框碰到这个无形区域就算有效)
    rois = []
    for s in grid_cfg["seats"]:
        x1, y1, x2, y2 = s["bbox"]
        new_y1 = max(0, int(y1 - (y2 - y1) * UP_SHIFT_RATIO))
        rois.append([x1, new_y1, x2, y2])

    count_buf = deque(maxlen=SMOOTH_WIN)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print("🚀 开始动态防漏检考勤推理...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. 暴力召回 (低阈值 + 大图推理，后排无处遁形)
        results = model(frame, conf=CONF_THRESH, imgsz=1280, augment=True, verbose=False, stream=True)

        # 2. 空间滤波 (滤除墙上、过道的低分误检)
        valid_boxes = []
        for res in results:
            if res.boxes is None: continue
            for box in res.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                # 检查是否与任意一个合法 ROI 有交集
                for rx1, ry1, rx2, ry2 in rois:
                    if not (bx2 < rx1 or bx1 > rx2 or by2 < ry1 or by1 > ry2):
                        valid_boxes.append([bx1, by1, bx2, by2])
                        break  # 命中一个就保留

        # 3. 平滑渲染
        count_buf.append(len(valid_boxes))
        stable_count = int(round(np.mean(count_buf)))

        # 🎨 画有效的学生 (清爽绿框)
        for x1, y1, x2, y2 in valid_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "STU", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 🌟 状态大屏
        cv2.rectangle(frame, (15, 15), (320, 65), (0, 0, 0), -1)
        cv2.putText(frame, f"实到: {stable_count} / {grid_cfg['total_seats']}",
                    (25, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Smart Attendance System", frame)
        if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()