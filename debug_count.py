# debug_detect_model.py
import cv2
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = r"D:\desktop\classroom\dataset2\runs\detect\runs\detect\student_STU_yolov8s\weights\best.pt"
TEST_IMG = r"D:\desktop\classroom\dataset\test\images\508_jpg.rf.d693866a684a97101687a1364a267882.jpg"


def main():
    if not Path(MODEL_PATH).exists(): raise FileNotFoundError("❌ 模型不存在")
    if not Path(TEST_IMG).exists():   raise FileNotFoundError("❌ 测试图不存在")

    print("📦 加载模型...")
    model = YOLO(MODEL_PATH)
    print(f"🔑 类别映射: {model.names}")

    # ✅ 安全读取训练配置（兼容 dict / Namespace）
    args = model.args
    cfg_str = args if isinstance(args, dict) else vars(args)
    print(f"📐 训练配置: imgsz={cfg_str.get('imgsz', 640)}, nc={cfg_str.get('nc', 1)}")

    # 🔍 测试极低阈值，强制输出所有潜在预测
    results = model(TEST_IMG, conf=0.01, iou=0.7, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print("\n🚨 诊断结论: 模型输出全为空")
        print("   原因1: 训练未收敛 → 检查 runs/detect/train3/results.png 中 mAP50(B) 是否 <0.1")
        print("   原因2: 标签损坏 → 检查 train/labels/*.txt 是否为空或坐标未归一化")
        print("   原因3: 场景分布差异极大 → 训练集为近景，测试图为高位俯拍")
        return

    print(f"\n✅ 模型在 conf=0.01 下输出 {len(boxes)} 个框")
    for i, box in enumerate(boxes):
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        print(f"   📦 框{i + 1}: 类别={cls}({model.names[cls]}) | 置信度={conf:.3f} | 坐标=[{x1},{y1},{x2},{y2}]")

    max_conf = max(float(b.conf[0]) for b in boxes)
    if max_conf < 0.20:
        print("\n💡 结论: 模型能识别但置信度极低。建议: 推理时 conf=0.15，或补充当前视角难例微调")
    else:
        print("\n💡 结论: 模型完全有效！仅因原脚本 conf=0.35 过滤过严。将阈值降至 0.25 即可正常框出")


if __name__ == "__main__":
    main()