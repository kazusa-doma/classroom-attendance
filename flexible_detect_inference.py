# flexible_detect_inference.py
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

# ================= 配置区 =================
MODEL_PATH = r"D:\desktop\classroom\dataset\runs\detect\train3\weights\best.pt"
INPUT_PATH = r"D:\desktop\classroom\dataset\test\images\508_jpg.rf.d693866a684a97101687a1364a267882.jpg"
CONF_THRESH = 0.04


# ==========================================

def process_image(img_path, model, device):
    img = cv2.imread(img_path)
    if img is None: raise ValueError("❌ 无法读取图片")

    results = model(img, conf=CONF_THRESH, device=device, verbose=False)
    res_img = results[0].plot()

    # ✅ 修复：使用 stem + suffix 安全拼接，避免 with_suffix 报错
    p = Path(img_path)
    out_path = str(p.parent / f"{p.stem}_result{p.suffix}")
    cv2.imwrite(out_path, res_img)
    print(f"✅ 图片识别完成 | 已保存: {out_path}")
    cv2.imshow("Image Result", res_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(vid_path, model, device):
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened(): raise ValueError("❌ 无法读取视频")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ✅ 修复：视频输出路径同理安全拼接
    p = Path(vid_path)
    out_path = str(p.parent / f"{p.stem}_result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    print("🎬 开始视频推理... (播放窗口中按 'q' 提前退出)")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        results = model(frame, conf=CONF_THRESH, device=device, verbose=False)
        res_frame = results[0].plot()
        out.write(res_frame)
        cv2.imshow("Video Detection", res_frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ 视频识别完成 | 已保存: {out_path}")


def main():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"❌ 模型不存在: {MODEL_PATH}")
    if not Path(INPUT_PATH).exists():
        raise FileNotFoundError(f"❌ 输入文件不存在: {INPUT_PATH}")

    print("📦 加载 YOLOv8 检测模型...")
    model = YOLO(MODEL_PATH)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 运行设备: {'GPU (RTX 3060)' if device == 0 else 'CPU'} | 输入: {INPUT_PATH}")

    ext = Path(INPUT_PATH).suffix.lower()
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
        process_image(INPUT_PATH, model, device)
    elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
        process_video(INPUT_PATH, model, device)
    else:
        raise ValueError("⚠️ 不支持的格式，请使用图片(.jpg/.png)或视频(.mp4/.avi)")


if __name__ == "__main__":
    main()