import sys
import cv2
import numpy as np
import torch
import json
import time
from collections import deque
from ultralytics import YOLO
from pathlib import Path
from sklearn.cluster import DBSCAN

# PyQt5 核心组件导入
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QFileDialog, QSlider, QVBoxLayout,
                             QHBoxLayout, QGroupBox, QFormLayout, QTextEdit, QMessageBox)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

# ================= 核心配置区 =================
BASE_DIR = Path(r"D:\desktop\classroom")


def find_model(pattern):
    """自动在指定目录下寻找最新的模型权重"""
    found = list(BASE_DIR.rglob(pattern))
    if not found: return None
    found.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(found[0])

#DET_MODEL_PATH = find_model("**/student_STU_v1/**/best.pt")
DET_MODEL_PATH = find_model("**/student_STU_yolov8s/**/best.pt")
SEG_MODEL_PATH = find_model("**/classroom_seg_v2/**/best.pt")

if not DET_MODEL_PATH: DET_MODEL_PATH = "yolov8n.pt"
if not SEG_MODEL_PATH: SEG_MODEL_PATH = "yolov8n-seg.pt"

print(f"✅ 检测模型路径确认: {DET_MODEL_PATH}")
print(f"✅ 分割模型路径确认: {SEG_MODEL_PATH}")

GRID_JSON_PATH = "classroom_grid_dynamic.json"

# 算法核心参数
CONF_DET = 0.15
CONF_SEG = 0.25
UP_SHIFT_RATIO = 1.5
SMOOTH_WIN = 8


# ==============================================

class ClassroomSystemPyQt(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 智慧教室综合考勤系统 v4.2 ")
        self.setGeometry(100, 100, 1200, 750)
        self.setMinimumSize(1000, 650)

        # 全局状态变量
        self.det_model = None
        self.seg_model = None
        self.cap = None
        self.writer = None  # 高效 H.264 视频压缩写入器
        self.last_frame_time = 0  # 原速时钟锁计数器
        self.output_path = "attendance_output.mp4"
        self.is_running = False
        self.is_previewing = False
        self.video_path = None
        self.grid_cfg = None
        self.rois = []
        self.count_buffer = deque(maxlen=SMOOTH_WIN)
        self.device = 0 if torch.cuda.is_available() else "cpu"

        # 🌟 核心状态标志：当前画面是否已建立网格
        self.has_grid_for_current_src = False

        # 初始化 UI 布局
        self.setup_ui()
        self.log_message(f"🖥️ 系统初始化 | 硬件加速: {'GPU (CUDA)' if self.device == 0 else 'CPU'}")

        # 使用 QTimer 替代 Tkinter 的 .after() 进行视频流高频驱动
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.process_video_frame)

        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.preview_camera_frame)

    def setup_ui(self):
        # 创建主窗口中央控制核心
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --------- 左侧面板：媒体控制、超参数及看板 ---------
        left_panel = QVBoxLayout()
        main_layout.addLayout(left_panel, stretch=1)

        # A. 媒体源控制组
        ctrl_group = QGroupBox("1. 输入源调度管理")
        ctrl_layout = QVBoxLayout()
        self.btn_video = QPushButton("📁 选择本地视频")
        self.btn_video.setStyleSheet("height: 32px; font-weight: bold;")
        self.btn_video.clicked.connect(self.load_video)

        self.btn_camera = QPushButton("📷 预览摄像头流")
        self.btn_camera.setStyleSheet("height: 32px; font-weight: bold; background-color: #0066CC; color: white;")
        self.btn_camera.clicked.connect(self.toggle_camera_preview)

        self.lbl_video = QLabel("未选择视频流")
        self.lbl_video.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_video.setAlignment(Qt.AlignCenter)

        ctrl_layout.addWidget(self.btn_video)
        ctrl_layout.addWidget(self.btn_camera)
        ctrl_layout.addWidget(self.lbl_video)
        ctrl_group.setLayout(ctrl_layout)
        left_panel.addWidget(ctrl_group)

        # B. 空间建模组
        model_group = QGroupBox("2. 数字化教室建模")
        model_layout = QVBoxLayout()
        self.btn_model = QPushButton("🏗️ 一键教室建模识别")
        self.btn_model.setStyleSheet("height: 35px; font-weight: bold; background-color: #D35B00; color: white;")
        self.btn_model.clicked.connect(self.build_classroom_grid)
        model_layout.addWidget(self.btn_model)
        model_group.setLayout(model_layout)
        left_panel.addWidget(model_group)

        # C. 算法动态超参数组
        param_group = QGroupBox("3. 动态超参数微调")
        param_layout = QFormLayout()
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(1, 50)
        self.slider_conf.setValue(int(CONF_DET * 100))
        self.slider_conf.valueChanged.connect(self.update_conf)
        self.lbl_conf = QLabel(f"检测阈值 (Conf): {CONF_DET:.2f}")
        param_layout.addRow(self.lbl_conf)
        param_layout.addRow(self.slider_conf)
        param_group.setLayout(param_layout)
        left_panel.addWidget(param_group)

        # D. 启动考勤控制
        self.btn_start = QPushButton("▶ 4. 启动实时人数统计")
        self.btn_start.setStyleSheet(
            "height: 45px; font-weight: bold; background-color: green; color: white; font-size: 14px;")
        self.btn_start.clicked.connect(self.toggle_attendance)
        left_panel.addWidget(self.btn_start)
        left_panel.addStretch()

        # --------- 右侧面板：高清视频看板与系统日志 ---------
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel, stretch=4)

        # 高清显示 Label
        self.video_label = QLabel("欢迎使用\n请先选择视频或开启摄像头")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: #16161a; color: #a0a0aa; font-size: 22px; font-weight: bold; border-radius: 8px;")
        right_panel.addWidget(self.video_label, stretch=5)

        # 终端文本日志框
        self.log_textbox = QTextEdit()
        self.log_textbox.setReadOnly(True)
        self.log_textbox.setStyleSheet("background-color: #1e1e24; color: #00ffcc; font-family: 'Consolas';")
        right_panel.addWidget(self.log_textbox, stretch=1)

    def log_message(self, msg):
        self.log_textbox.append(msg)
        self.log_textbox.ensureCursorVisible()

    def update_conf(self, value):
        conf_float = value / 100.0
        self.lbl_conf.setText(f"检测阈值 (Conf): {conf_float:.2f}")

    def load_video(self):
        if self.is_previewing: self.toggle_camera_preview()
        path, _ = QFileDialog.getOpenFileName(self, "选择本地视频文件", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.video_path = path
            self.has_grid_for_current_src = False  # 更换媒体源，作废历史静态网格
            self.lbl_video.setText(Path(path).name)
            self.log_message(f"🎬 已成功加载视频资源: {Path(path).name}")

    def toggle_camera_preview(self):
        if self.is_running:
            QMessageBox.warning(self, "提示", "请先停止当前的实时人数统计！")
            return
        if not self.is_previewing:
            self.video_path = 0
            self.has_grid_for_current_src = False
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "错误", "底层硬件驱动异常：无法打开本地摄像头！")
                return
            self.is_previewing = True
            self.btn_camera.setText("⏹ 停止摄像头预览")
            self.btn_camera.setStyleSheet("height: 32px; font-weight: bold; background-color: red; color: white;")
            self.lbl_video.setText("视频流: 本地摄像头")
            self.log_message("📷 已开启摄像头实时预览，可进行全局无差别统计或抽取基准帧建模。")
            self.preview_timer.start(30)  # 30ms 轮询频率
        else:
            self.preview_timer.stop()
            self.is_previewing = False
            self.btn_camera.setText("📷 预览摄像头流")
            self.btn_camera.setStyleSheet("height: 32px; font-weight: bold; background-color: #0066CC; color: white;")
            if self.cap: self.cap.release()
            self.display_blank_screen("Camera Preview Stopped")
            self.log_message("⏹ 已停止摄像头预览。")

    def preview_camera_frame(self):
        if not self.is_previewing or not self.cap.isOpened(): return
        ret, frame = self.cap.read()
        if ret:
            h, w = frame.shape[:2]
            cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
            cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 0), 1)
            cv2.putText(frame, "Preview Mode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.display_image_on_ui(frame)

    def display_image_on_ui(self, cv_img):
        # 确保将 OpenCV 默认的 BGR 图像色彩空间转换为 PyQt 能够识别的 RGB 空间
        frame_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w

        # 将底层矩阵打包转换为 QPixmap 纹理形式刷新前端
        qt_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)

    def display_blank_screen(self, text="System Stopped"):
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_image, text, (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self.display_image_on_ui(blank_image)

    def build_classroom_grid(self):
        if self.video_path is None:
            QMessageBox.warning(self, "提示", "当前未捕获任何活跃的视频流，请选择视频或开启相机预览！")
            return

        self.log_message("⏳ 正在抽取空间网格拓扑建模基准帧...")
        if self.is_previewing and self.cap.isOpened():
            ret, frame = self.cap.read()
            self.toggle_camera_preview()
        else:
            temp_cap = cv2.VideoCapture(self.video_path)
            ret, frame = temp_cap.read()
            temp_cap.release()

        if not ret:
            QMessageBox.critical(self, "错误", "无法解码当前视讯帧，基线网格拦截失败！")
            return

        if self.seg_model is None:
            self.log_message("📦 正在异步调入底层实例分割算法底座...")
            self.seg_model = YOLO(SEG_MODEL_PATH)

        self.log_message("🧠 正在进行静态像素级分割与重投影网格计算...")
        results = self.seg_model(frame, conf=CONF_SEG, iou=0.45, imgsz=1280, device=self.device, verbose=False)

        if results[0].boxes is None or len(results[0].boxes) == 0:
            self.log_message("❌ 建模失败：当前教室画面内未捕获到符合特征的课桌几何实体。")
            return

        desks, desk_masks = [], []
        for i in range(len(results[0].boxes.cls)):
            if int(results[0].boxes.cls[i]) == 1:
                x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[i].cpu().numpy())
                desks.append({"box": [x1, y1, x2, y2]})
                if results[0].masks is not None:
                    desk_masks.append(results[0].masks.xy[i])

        if len(desks) < 2:
            self.log_message(f"❌ 空间拓扑聚类失败：捕获到的课桌数量过少，不满足行-列解析基准。")
            return

        centers = np.array([[(d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2] for d in desks])
        eps_y = max(20, int(np.std(centers[:, 1]) * 0.6))
        row_labels = DBSCAN(eps=eps_y, min_samples=1).fit(centers[:, 1:2]).labels_

        row_groups = {}
        for idx, lbl in enumerate(row_labels): row_groups.setdefault(lbl, []).append(idx)
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
                    "bbox": d["box"], "center": [int(cx), int(cy)], "grid_id": f"R{r_idx + 1}C{c_idx + 1}"
                })
                seat_id += 1

        h, w = frame.shape[:2]
        self.grid_cfg = {"image_size": [w, h], "total_seats": len(grid_seats), "seats": grid_seats}
        with open(GRID_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(self.grid_cfg, f, indent=2, ensure_ascii=False)

        self.has_grid_for_current_src = True
        self.log_message(
            f"✅ 静态数字孪生拓扑建模成功！(锁定了全景共 {len(grid_seats)} 张物理课桌) 后续统计将启用【防干扰网格过滤】机制。")

        vis_frame = frame.copy()
        overlay = frame.copy()
        for poly in desk_masks:
            pts = poly.astype(np.int32)
            cv2.fillPoly(overlay, [pts], (255, 255, 0))
        cv2.addWeighted(overlay, 0.4, vis_frame, 0.6, 0, vis_frame)
        cv2.putText(vis_frame, "Grid Generated: Filter Mode Activated", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
        self.display_image_on_ui(vis_frame)

    def toggle_attendance(self):
        if self.is_previewing:
            QMessageBox.warning(self, "提示", "请先关闭摄像头实时预览流！")
            return

        if not self.is_running:
            if self.video_path is None:
                QMessageBox.warning(self, "提示", "多媒体流未载入，请导入离线视频文件或连通相机！")
                return

            # 🌟 动态判决双核心工作流分支
            if self.has_grid_for_current_src and Path(GRID_JSON_PATH).exists():
                with open(GRID_JSON_PATH, "r", encoding="utf-8") as f:
                    self.grid_cfg = json.load(f)
                self.rois = []
                for s in self.grid_cfg["seats"]:
                    x, y, w, h = s["bbox"]
                    new_y = max(0, int(y - h * UP_SHIFT_RATIO))
                    self.rois.append([x, new_y, x + w, y + h])
                self.log_message("🚀 成功激活动态推理引擎 -> 模式方案: [网格空间交并比重叠阻断反干扰]")
            else:
                self.rois = []
                self.log_message("🚀 成功激活动态推理引擎 -> 模式方案: [全画幅全类别无差别深度检测]")

            if self.det_model is None:
                self.log_message("📦 正在载入轻量化人体目标检测神经网络...")
                self.det_model = YOLO(DET_MODEL_PATH)

            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "错误", "无法正常建立解码通道，视讯流连通阻断！")
                return

            # 🆕 完美注入 H.264 视频流低体积高保真压缩写入引擎，彻底防崩溃体积增长
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 采用高动态压缩率的 mp4v 编码器
            self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))

            self.is_running = True
            self.btn_start.setText("⏹ 停止实时人数统计")
            self.btn_start.setStyleSheet(
                "height: 45px; font-weight: bold; background-color: red; color: white; font-size: 14px;")
            self.count_buffer.clear()
            self.last_frame_time = time.time()
            self.play_timer.start(5)  # 以高频时钟高保真驱动，在内部通过时钟锁精细平滑控制帧率
        else:
            self.safely_release_stream()
            self.display_blank_screen("Statistics Stopped")
            self.log_message("⏹ 视频检测流已由操作员主动拦截断开。")

    def safely_release_stream(self):
        """核心解构机制：释放多媒体句柄，完成 H.264 尾部封包落盘保护"""
        self.play_timer.stop()
        self.is_running = False
        self.btn_start.setText("▶ 3. 启动实时人数统计")
        self.btn_start.setStyleSheet(
            "height: 45px; font-weight: bold; background-color: green; color: white; font-size: 14px;")
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.log_message(
                f"📁 【工程归档】带有 HUD 可视化渲染的高清低体积视频已保存在: {Path(self.output_path).resolve()}")

    def process_video_frame(self):
        if not self.is_running or not self.cap.isOpened(): return

        # ⏱️ 差分时钟同步锁：当导入离线视频时，限制播放帧率不超过标准人类物理速度，杜绝倍速快进
        if self.video_path != 0:
            target_delay = 1.0 / 25.0  # 强制限制输出最大帧率 25 FPS
            current_time = time.time()
            if current_time - self.last_frame_time < target_delay:
                return  # 时钟锁生效，跳过本次循环，保持与视频源原速播放同步
            self.last_frame_time = current_time

        ret, frame = self.cap.read()
        if not ret:
            self.safely_release_stream()
            self.log_message("📼 离线视频文件读取至末尾，流播放自然关断。")
            return

        current_conf = self.slider_conf.value() / 100.0

        # 执行前向目标推理
        results = self.det_model(frame, conf=current_conf, iou=0.40, imgsz=1280, augment=True,
                                 agnostic_nms=True, verbose=False, stream=True, device=self.device)

        raw_boxes = []
        for res in results:
            if res.boxes is not None:
                for box in res.boxes:
                    raw_boxes.append(box.xyxy[0].cpu().numpy().astype(int).tolist())

        # 核心判定：进行自适应双机制拓扑碰撞匹配
        valid_boxes = []
        if self.has_grid_for_current_src and len(self.rois) > 0:
            for bx1, by1, bx2, by2 in raw_boxes:
                is_valid = False
                for rx1, ry1, rx2, ry2 in self.rois:
                    if not (bx2 < rx1 or bx1 > rx2 or by2 < ry1 or by1 > ry2):
                        is_valid = True
                        break
                if is_valid: valid_boxes.append([bx1, by1, bx2, by2])
        else:
            valid_boxes = raw_boxes

        self.count_buffer.append(len(valid_boxes))
        stable_count = int(round(np.mean(self.count_buffer)))

        out_frame = frame.copy()

        for x1, y1, x2, y2 in valid_boxes:
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out_frame, "STU", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 现代化高科技HUD归档图层设计
        panel_x, panel_y = 20, 20
        panel_w, panel_h = 320, 80
        radius = 15
        bg_color = (25, 25, 25)
        border_color = (0, 215, 255)
        num_color = (0, 255, 0)

        overlay = out_frame.copy()
        cv2.rectangle(overlay, (panel_x + radius, panel_y), (panel_x + panel_w - radius, panel_y + panel_h), bg_color,
                      -1)
        cv2.rectangle(overlay, (panel_x, panel_y + radius), (panel_x + panel_w, panel_y + panel_h - radius), bg_color,
                      -1)
        cv2.circle(overlay, (panel_x + radius, panel_y + radius), radius, bg_color, -1)
        cv2.circle(overlay, (panel_x + panel_w - radius, panel_y + radius), radius, bg_color, -1)
        cv2.circle(overlay, (panel_x + radius, panel_y + panel_h - radius), radius, bg_color, -1)
        cv2.circle(overlay, (panel_x + panel_w - radius, panel_y + panel_h - radius), radius, bg_color, -1)

        alpha = 0.65
        cv2.addWeighted(overlay, alpha, out_frame, 1 - alpha, 0, out_frame)
        cv2.line(out_frame, (panel_x + 5, panel_y + 15), (panel_x + 5, panel_y + panel_h - 15), border_color, 4)

        title_txt = "Live Counter"
        mode_txt = "[Filtered]" if self.has_grid_for_current_src else "[Global]"
        val_txt = f"{stable_count}"

        cv2.putText(out_frame, title_txt, (panel_x + 20, panel_y + 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(out_frame, title_txt, (panel_x + 20, panel_y + 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, border_color, 1)
        cv2.putText(out_frame, mode_txt, (panel_x + 220, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200),
                    1)
        cv2.putText(out_frame, val_txt, (panel_x + 20, panel_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 4)
        cv2.putText(out_frame, val_txt, (panel_x + 20, panel_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.3, num_color, 2)

        # 🆕 将渲染完毕的带 HUD 的帧流式丢入 H.264 编码器，完成超低磁盘占用存储
        if self.writer is not None:
            self.writer.write(out_frame)

        self.display_image_on_ui(out_frame)

    def closeEvent(self, event):
        """安全析构重写：当用户直接关闭 PyQt 窗口时，强制切断流并 safe 保存视频"""
        self.safely_release_stream()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ClassroomSystemPyQt()
    window.show()
    sys.exit(app.exec_())