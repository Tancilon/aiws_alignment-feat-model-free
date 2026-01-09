# 此文件用于测试自己的数据集 - 每帧都进行register而不是追踪
from estimater import *
from datareader import *
import os
import logging
import trimesh
import numpy as np
import cv2
import imageio
import open3d as o3d
from datetime import datetime


def main():
  code_dir = os.path.dirname(os.path.realpath(__file__))

  # 直接定义参数
  mesh_file = "datasets/test_photo/mesh/G90_norm_m.obj"
  test_scene_dir = "/home/dq/mnt/localgit/aiws_alignment-feat-model-free/datasets/test_photo"

  # 初始姿态估计的优化迭代次数,默认为 5 次
  est_refine_iter = 5
  # 调试级别,控制可视化输出 默认为 1 级
  debug = 2

  # 调试信息存储目录
  # debug_dir = f'{code_dir}/my_data_debug6'
  # 获取本地当前时间
  now = datetime.now()
  # 按 年月日时 生成文件夹名
  folder_name = now.strftime("%Y%m%d_%H")
  # 调试信息存储目录
  debug_dir = f'{code_dir}/my_data_debug_{folder_name}'

  set_logging_format()  # 设置日志格式
  set_seed(0)

  mesh = trimesh.load(mesh_file)  # 读取 3D 物体网格文件

  print(type(mesh))  # 确认是 Trimesh 还是 Scene
  # 如果 mesh 是一个 Scene,提取所有子模型
  if isinstance(mesh, trimesh.Scene):
    print("检测到多个子模型:")
    for name, submesh in mesh.geometry.items():
      print(f"名称: {name}, 顶点数: {len(submesh.vertices)}")

  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  # 计算 3D 物体的定向包围盒,返回 to_origin 变换矩阵和 extents 长宽高信息
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  print('物体大小', extents)

  # 计算物体的3D包围盒 表示 3D 物体的边界
  bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
  print("Number of vertices:", len(mesh.vertices))
  print("Number of faces:", len(mesh.faces))

  # 初始化姿态估计器
  scorer = ScorePredictor()  # 评分模型,可能用于评估姿态估计的质量
  refiner = PoseRefinePredictor()  # 姿态优化模型,用于优化初始姿态估计结果
  glctx = dr.RasterizeCudaContext()  # 可能是一个 CUDA 计算环境,用于高效渲染

  # 创建姿态估计器 est,传入 3D 物体模型、评分器、优化器、调试参数等
  est = FoundationPose(
    model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
    scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx
  )
  logging.info("Estimator initialization done")

  # 用户定义的数据读取类,应该用于读取 test_scene_dir 目录中的 RGB + 深度数据
  reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=480, zfar=np.inf)

  # 遍历所有帧,每一帧都进行register
  for i in range(len(reader.color_files)):
    logging.info(f'Processing frame i:{i}')

    color = reader.get_color(i)  # 读取第 i 帧的 RGB 图像
    depth = reader.get_depth(i)  # 读取第 i 帧的深度图
    mask = reader.get_mask(i).astype(bool)  # 读取第 i 帧的 mask

    # 每一帧都进行初始姿态估计
    pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)

    # 只有当 debug 级别 大于等于 3 时,才会执行下面的代码
    if debug >= 3 and i == 0:  # 只在第一帧保存点云,避免重复保存
      m = mesh.copy()
      m.apply_transform(pose)
      m.export(f'{debug_dir}/model_tf.obj')
      xyz_map = depth2xyzmap(depth, reader.K)
      valid = depth >= 0.001
      pcd = toOpen3dCloud(xyz_map[valid], color[valid])
      o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)

    # 保存物体在相机坐标系下的姿态矩阵
    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4, 4))

    # 在 RGB 图像上绘制 3D 包围盒和坐标轴,并显示出来
    if debug >= 1:
      center_pose = pose @ np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0,
                          is_input_rgb=True)
      # cv2.imshow('1', vis[..., ::-1])
      # cv2.waitKey(1)

    # 如果 debug >= 2,保存可视化的跟踪结果图片
    if debug >= 2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)


if __name__ == '__main__':
  main()
