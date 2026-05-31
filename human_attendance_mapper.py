# count_people_video.py
import cv2
import numpy as np
import torch
from collections import deque
from ultralytics import YOLO
from pathlib import Path

# ================= 配置区 =================
MODEL_PATH = r"D:\desktop\classroom\dataset2\runs\detect\runs\detect\student_STU_yolov8s\weights\best.pt"
VIDEO_SRC  = r"D:\desktop\classroom\Desktop 2026.04.29 - 16.36.16.01.mp4"  # ✅ 已替换为你的视频路径
CONF_THRESH = 0.8
SMOOTH_WIN  = 8  # 滑动窗口大小（防人数瞬间跳变）
# ==========================================

def main():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"❌ 模型不存在: {MODEL_PATH}")
    if not Path(VIDEO_SRC).exists():
        raise FileNotFoundError(f"❌ 视频文件不存在: {VIDEO_SRC}")

    print("📦 加载人体检测模型...")
    model = YOLO(MODEL_PATH)
    device = 0 if torch.cuda.is_available() else "cpu"

    print("📹 打开视频流...")
    cap = cv2.VideoCapture(VIDEO_SRC)
    if not cap.isOpened():
        raise RuntimeError("❌ 无法打开视频文件，请检查格式/路径")

    # 获取视频原始帧率（用于控制播放速度，可选）
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    count_buffer = deque(maxlen=SMOOTH_WIN)
    print(f"✅ 视频识别已启动 | 原始帧率: {fps:.1f} FPS | 按 'q' 退出或等播放完\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("📼 视频播放结束")
                break

            # 1. YOLOv8 推理
            # 替换 model(frame, ...) 这一行
            results = model(
                frame,
                conf=0.20,
                imgsz=1920,
                augment=True,
                iou=0.40,     # 👈 【核心修复】NMS 重叠过滤阈值。只要两个框重叠度超过 40%，直接融合成一个！
                agnostic_nms=True, # 👈 强制合并不同尺度的重叠框
                verbose=False,
                stream=True
            )
            boxes = []
            for res in results:
                if res.boxes is not None:
                    for box in res.boxes:
                        boxes.append(box.xyxy[0].cpu().numpy().astype(int).tolist())

            # 2. 人数统计与平滑滤波
            raw_count = len(boxes)
            count_buffer.append(raw_count)
            stable_count = int(round(np.mean(count_buffer)))

            # 3. 画面绘制
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # 🌟 左上角高亮人数统计
            txt = f"当前人数: {stable_count}"
            cv2.rectangle(frame, (15, 15), (240, 55), (0, 0, 0), -1)
            cv2.putText(frame, txt, (22, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("Video People Counter", frame)
            # 控制播放速度接近原视频帧率
            if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("⏹ 识别已退出")

if __name__ == "__main__":
    main()