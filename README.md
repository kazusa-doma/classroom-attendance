# AI 智慧教室综合考勤系统

基于 YOLOv8 实例分割与目标检测的教室人数自动统计系统，支持视频文件与实时摄像头输入。

## 功能特性

- **教室空间建模**：使用 YOLOv8-seg 实例分割检测课桌，结合 DBSCAN 聚类自动生成座位网格拓扑
- **学生人数统计**：YOLOv8 目标检测识别学生，配合网格 ROI 过滤机制实现精准人数统计
- **双模式运行**：支持全画幅无差别检测与网格空间交并比过滤两种模式
- **实时可视化**：PyQt5 图形界面，实时 HUD 渲染人数看板
- **视频归档**：H.264 编码输出带标注的统计结果视频

## 项目结构

```
.
├── main-system.py                      # 主系统：PyQt5 GUI 完整考勤流程
├── generate_classroom_grid_json.py     # 教室建模工具：生成座位网格 JSON
├── human_attendance_mapper.py          # 独立人数统计脚本（命令行版）
├── step2_run_attendance.py.py          # 考勤运行步骤脚本
├── compare_kmeans_adaboost.py          # 对比实验：K-Means + AdaBoost
├── compare_traditional_yolo_seg.py     # 对比实验：传统方法 vs YOLO 分割
├── flexible_detect_inference.py        # 灵活检测推理脚本
├── test_grid_alignment.py              # 网格对齐测试
├── debug_count.py                      # 调试用人数统计
├── data.yaml                           # 数据集配置
├── classroom_grid.json                 # 静态教室网格配置
├── classroom_grid_dynamic.json         # 动态生成的教室网格
├── mapping_matrix.json                 # 映射矩阵
├── classroomtrain/                     # 教室分割模型训练输出
│   └── runs/segment/runs/classroom_seg_v2/weights/
│       └── best.pt                     # 课桌分割模型权重（YOLOv8-seg）
└── dataset2/                           # 学生检测模型训练输出
    └── runs/detect/runs/detect/student_STU_v1/weights/
        └── best.pt                     # 学生检测模型权重（YOLOv8）
```

## 环境依赖

- Python 3.8+
- PyTorch（CUDA 可选，自动检测）
- ultralytics（YOLOv8）
- PyQt5
- OpenCV（cv2）
- scikit-learn
- NumPy

```bash
pip install torch ultralytics PyQt5 opencv-python scikit-learn numpy
```

## 使用方式

### 1. 启动主系统

```bash
python main-system.py
```

操作流程：
1. 选择本地视频或开启摄像头预览
2. 点击「一键教室建模识别」生成座位网格
3. 调整检测阈值滑块（可选）
4. 点击「启动实时人数统计」开始考勤
5. 统计结果视频自动保存为 `attendance_output.mp4`

### 2. 命令行独立人数统计

修改 `human_attendance_mapper.py` 中的视频路径后：

```bash
python human_attendance_mapper.py
```

### 3. 单独生成教室网格

```bash
python generate_classroom_grid_json.py
```

## 模型说明

| 模型 | 用途 | 路径 |
|------|------|------|
| classroom_seg_v2 | 课桌实例分割（YOLOv8-seg） | `classroomtrain/runs/segment/runs/classroom_seg_v2/weights/best.pt` |
| student_STU_v1 | 学生目标检测（YOLOv8） | `dataset2/runs/detect/runs/detect/student_STU_v1/weights/best.pt` |

## 核心算法

1. **空间建模**：YOLOv8-seg 分割课桌 → 提取包围盒中心点 → DBSCAN 按 Y 轴聚类分行 → 按 X 轴排序分列 → 生成座位网格
2. **人数统计**：YOLOv8 检测人体 → 网格 ROI 交并比过滤（可选）→ 滑动窗口平滑滤波 → 稳定输出人数
3. **帧率控制**：差分时钟同步锁，保证离线视频按原速播放不倍速
