# rebuild_grid_from_video.py
import cv2, numpy as np, json
from pathlib import Path
from ultralytics import YOLO
from sklearn.cluster import DBSCAN

# ================= 配置区 =================
# 1. 你的测试视频（必须是固定机位）
VIDEO_SRC = r"D:\desktop\classroom\Desktop 2026.04.29 - 16.36.16.01.mp4"
# 2. 你训练好的 课桌椅分割模型
SEG_MODEL = r"D:\desktop\classroom\classroomtrain\runs\segment\runs\classroom_seg_v2\weights\best.pt"

# 分割参数 (如果视频里有人挡住了桌子，稍微降低 conf 能让模型“脑补”出桌子)
CONF_THRESH = 0.05


# ==========================================

def main():
    if not Path(VIDEO_SRC).exists(): raise FileNotFoundError("❌ 视频不存在")
    if not Path(SEG_MODEL).exists(): raise FileNotFoundError("❌ 分割模型不存在")

    print("📹 正在从视频抽取基准帧...")
    cap = cv2.VideoCapture(VIDEO_SRC)
    ret, frame = cap.read()
    cap.release()
    if not ret: raise RuntimeError("❌ 无法读取视频首帧")

    cv2.imwrite("video_base_frame.jpg", frame)
    print("✅ 基准帧已保存，开始实例分割建模...")

    model = YOLO(SEG_MODEL)
    h, w = frame.shape[:2]
    # 使用大分辨率保证分割精度
    results = model(frame, conf=CONF_THRESH, iou=0.45, imgsz=1280, device=0, verbose=False)

    if results[0].boxes is None:
        raise ValueError("⚠️ 视频首帧中未检测到课桌，无法建模！")

    # 提取课桌 (class_1 是桌子) 以及对应的多边形掩码
    desks = []
    desk_masks = []

    for i in range(len(results[0].boxes.cls)):
        if int(results[0].boxes.cls[i]) == 1:
            x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[i].cpu().numpy())
            desks.append({"box": [x1, y1, x2, y2]})

            # 收集掩码多边形坐标
            if results[0].masks is not None:
                desk_masks.append(results[0].masks.xy[i])

    if len(desks) < 4:
        raise ValueError(f"⚠️ 仅检测到 {len(desks)} 张课桌，建模失败。请检查分割模型或调低 CONF_THRESH")

    # 简易网格聚类逻辑 (JSON 依然正常生成)
    centers = np.array([[(d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2] for d in desks])
    eps_y = max(20, int(np.std(centers[:, 1]) * 0.6))
    row_labels = DBSCAN(eps=eps_y, min_samples=1).fit(centers[:, 1:2]).labels_

    row_groups = {}
    for idx, lbl in enumerate(row_labels):
        row_groups.setdefault(lbl, []).append(idx)
    row_avgs = {lbl: np.mean(centers[indices, 1]) for lbl, indices in row_groups.items()}
    sorted_labels = sorted(row_avgs.keys(), key=lambda k: row_avgs[k])

    grid_seats = []
    seat_id = 1
    for r_idx, lbl in enumerate(sorted_labels):
        indices = row_groups[lbl]
        indices.sort(key=lambda i: centers[i, 0])
        for c_idx, idx in enumerate(indices):
            d = desks[idx]
            cx, cy = centers[idx]
            grid_seats.append({
                "id": seat_id, "row": r_idx + 1, "col": c_idx + 1,
                "bbox": d["box"], "center": [int(cx), int(cy)],
                "grid_id": f"R{r_idx + 1}C{c_idx + 1}"
            })
            seat_id += 1

    # 导出专属于该视频的 JSON (考勤系统照常使用)
    cfg = {"image_size": [w, h], "total_seats": len(grid_seats), "seats": grid_seats}
    with open("classroom_grid.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # ================= 🎨 绘制可视化确认图 (纯净版) =================

    # 绘制半透明青色桌面掩码
    overlay = frame.copy()
    for poly in desk_masks:
        pts = poly.astype(np.int32)
        # OpenCV 中青色的 BGR 值为 (255, 255, 0)
        cv2.fillPoly(overlay, [pts], (255, 255, 0))

        # 将青色图层以 40% 的不透明度融合到原图上
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # 保存干干净净只有青色掩码的图片
    cv2.imwrite("video_grid_aligned.jpg", frame)

    print("🎉 视频专属网格 JSON 已生成！(classroom_grid.json)")
    print("📁 请查看 video_grid_aligned.jpg，获得最纯粹的桌面分割展示图！")


if __name__ == "__main__":
    main()