# FoundationPose复现细节

## 1.nerf无头渲染训练

```
pip uninstall -y PyOpenGL-accelerate # 这个包是 PyOpenGL 的 C 扩展加速模块，版本或 ABI 一旦和 NumPy/Python 不匹配就容易段错误

# Headless/离屏渲染常用组合（放在同一个终端会话里）
unset DISPLAY
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0
export MESA_GL_VERSION_OVERRIDE=3.3

# 默认会训练数据集下的每个物体，每个物体1000轮；总共是13个物体，13000轮
python bundlesdf/run_nerf.py \
  --ref_view_dir /root/autodl-tmp/FoundationPose/demo_data/ref_views \
  --dataset linemod
  
# 最终的结果
obj_00xx/nerf下保存的是中间的训练结果和可视化
obj_00x/model下的model.obj是最终生成的三维模型
```

## 2.构建自己的物体

数据集准备：直接参考linemod的ref-views文件夹

生成后面推理可以使用的obj文件：直接使用foundationpose/bundlesdf/run_nerf.py（主要是生成model）

测试数据集格式参考linemod_test_all

<img src="/Users/wangrunze/Desktop/FoundationPose/截屏2025-10-21 17.55.54.png" style="zoom:50%;" />

## 3.model-based的数据需求

**脚本运行前需要的数据：**

（最关键的是CAD）

```
demo_data/mustard0/
├── mesh/textured_simple.obj    # 3D模型，foundationpose对于CAD的单位要求是米！
├── rgb/                        # RGB图像
│   ├── 0000.png
│   └── ...
├── depth/                      # 深度图（uint16, 毫米）
│   ├── 0000.png
│   └── ...
├── masks/                      # 物体mask
│   └── 0000.png               # 只需第一帧！
└── cam_K.txt                  # 相机内参
```

<img src="/Users/wangrunze/Desktop/FoundationPose/截屏2025-10-21 21.42.22.png" style="zoom:50%;" />

**脚本运行后输出的数据：**

```
debug/
├── ob_in_cam/              # 每帧的姿态估计结果
│   ├── 0000.txt           # 4×4 变换矩阵
│   ├── 0001.txt
│   └── ...
├── track_vis/              # 可视化结果 (if debug>=2)
│   ├── 0000.png           # 带3D框和坐标轴的可视化
│   ├── 0001.png
│   └── ...
├── model_tf.obj            # 变换后的模型 (if debug>=3)
└── scene_complete.ply      # 场景点云 (if debug>=3)
```

## 4.mode-free的数据需求

**脚本运行前需要的数据：**

训练nerf阶段（最关键的是相机外参）

```

ref_views_16/                           # 参考视角根目录，最好16个视角
├── ob_0000001/                         # 物体1
│   ├── rgb/                            # ✅ RGB图像
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ... (多个视角)
│   ├── depth_enhanced/                 # ✅ 增强的深度图
│   │   ├── 000000.png                 # 16位PNG, 单位: 毫米
│   │   ├── 000001.png
│   │   └── ...
│   ├── mask/                           # ✅ 物体分割mask
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   ├── mask_refined/                   # ⚠️  精细化的mask (LINEMOD需要)
│   │   ├── 000000.png
│   │   └── ...
│   ├── cam_in_ob/                      # ✅ 相机到物体的变换矩阵：1.COLMAP/SfM、2.BundleSDF、3.机械臂末端姿态
│   │   ├── 000000.txt                 # 4×4矩阵
│   │   ├── 000001.txt
│   │   └── ...
│   ├── K.txt                           # ✅ 相机内参 (3×3)
│   ├── select_frames.yml               # ✅ 选择的帧信息 (可以是空文件)


# NeRF训练完输出结果
ob_0000001/
├── model/
│   ├── model.obj          # 最终3D模型（带纹理）
│   ├── model.mtl          # 材质文件
│   └── model.png          # 纹理贴图
└── nerf/                  # 训练中间文件
    ├── rgb_*.png
    ├── pcd_normalized.ply
    ├── model_latest.pth
    ├── step_*_mesh_*.obj
    └── ...

```

推理阶段（--use_reconstructed_mesh 0 来控制使用NeRF还是原始CAD）

<img src="/Users/wangrunze/Desktop/FoundationPose/截屏2025-10-21 22.10.38.png" style="zoom:50%;" />

**脚本运行后输出的数据：**

```
debug/                                  # 默认输出目录 (可通过--debug_dir指定)
└── ycbv_res.yml                       # ✅ 主要输出：所有姿态估计结果
```

