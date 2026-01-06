# 此文件用于测试自己的数据集
# 识别时修改裁剪比例 如果将裁剪比例设置大 则包围盒聚焦于物块的上方
# 如果将裁剪比例设置小 则包围盒聚焦于物块的前方 且方向旋转对应不起来
from estimater import *
from datareader import *
import os
import logging
import trimesh
import numpy as np
import cv2
import imageio
import open3d as o3d
 
 
def main():
  code_dir = os.path.dirname(os.path.realpath(__file__))
 
  # 直接定义参数
  # 要处理的 3D 物体的 .obj 文件路径 包含默认路径/demo_data/mustard0/mesh/textured_simple.obj 另一个文件夹是 kinect_driller_seq
  # # 理论上这两个数据集都可以跑通 但是内存溢出了 应该如何修改呢
  # mesh_file = f'{code_dir}/demo_data/my_data/mesh/textured_simple.obj'
  # # 测试场景的文件夹路径，包含 RGB + 深度图像数据 /demo_data/mustard0
  # test_scene_dir = f'{code_dir}/demo_data/my_data'
  # 姿态跟踪的结果保存在 my_data_debug/ob_in_cam 文件夹下
  mesh_file = "/root/autodl-tmp/FoundationPose/demo_data/diban_test/mesh/G90.obj"  # 或者 untitled
  test_scene_dir = "/root/autodl-tmp/FoundationPose/demo_data/diban_test"
 
  # 初始姿态估计的优化迭代次数，默认为 5 次
  est_refine_iter = 5
  # 姿态跟踪的优化迭代次数，默认为 2 次
  track_refine_iter = 2
  # 调试级别，控制可视化输出 默认为 1 级
  debug = 3
 
  # 调试信息存储目录
  debug_dir = f'{code_dir}/my_data_debug3'
 
  set_logging_format()  # 设置日志格式
  set_seed(0)
 
  mesh = trimesh.load(mesh_file)  # 读取 3D 物体网格文件
 
  # 实验证明 网格拼接对于姿态识别分割没有作用
  # # 拆分成多个子网格
  # sub_meshes = mesh.split(only_watertight=False)
  # # 计算所有子网格的 AABB
  # for i, sub_mesh in enumerate(sub_meshes):
  #   print(f"Sub-mesh {i + 1} AABB size:", sub_mesh.bounding_box.extents)
  # full_mesh = trimesh.util.concatenate(sub_meshes)
  # # to_origin, extents = trimesh.bounds.oriented_bounds(full_mesh)
 
  # 如果扩大mesh 姿态检测为错误 物体为站立状态 改变这个因数物体姿态不会发生变化
  scale_factor = 1  # 放大 10%
 
  print(type(mesh))  # 确认是 Trimesh 还是 Scene
  # 如果 mesh 是一个 Scene，提取所有子模型
  if isinstance(mesh, trimesh.Scene):
    print("检测到多个子模型：")
    for name, submesh in mesh.geometry.items():
      print(f"名称: {name}, 顶点数: {len(submesh.vertices)}")
  # if isinstance(mesh, trimesh.Scene):
  #   mesh = list(mesh.geometry.values())[0]  # 获取场景中的第一个Trimesh对象
 
  # if isinstance(mesh, trimesh.Scene):
  #   # 按体积排序，选择最小的
  #   smallest_mesh = min(mesh.geometry.values(), key=lambda m: m.bounding_box.volume)
  #   mesh = smallest_mesh
 
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')
 
  # 计算 3D 物体的定向包围盒，返回 to_origin 变换矩阵和 extents 长宽高信息
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  # extents *= scale_factor
  print('物体大小',extents)
 
  # 计算物体的3D包围盒 表示 3D 物体的边界
  bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
  print("Number of vertices:", len(mesh.vertices))
  print("Number of faces:", len(mesh.faces))
  # mesh.fill_holes()  # 填补孔洞
  # mesh.remove_unreferenced_vertices()  # 移除未被面引用的顶点
  # mesh.remove_degenerate_faces()  # 移除坏的三角面
  # print("修正后 Number of vertices:", len(mesh.vertices))
  # print("修正后 Number of faces:", len(mesh.faces))
 
  # 初始化姿态估计器
  scorer = ScorePredictor()  # 评分模型，可能用于评估姿态估计的质量
  refiner = PoseRefinePredictor()  # 姿态优化模型，用于优化初始姿态估计结果
  glctx = dr.RasterizeCudaContext()  # 可能是一个 CUDA 计算环境，用于高效渲染
 
  # 创建姿态估计器 est，传入 3D 物体模型、评分器、优化器、调试参数等
  est = FoundationPose(
    model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
    scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx
  )
  logging.info("Estimator initialization done")
 
  # 用户定义的数据读取类，应该用于读取 test_scene_dir 目录中的 RGB + 深度数据
  reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=480, zfar=np.inf) 
  # debug: shorter_side缩放输入图片，同时也会缩放内参
 
  # 实时视频处理
  # 遍历 test_scene_dir 目录中的所有 RGB 帧，并读取对应的深度图
  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
 
    color = reader.get_color(i)  # 读取第 i 帧的 RGB 图像
    depth = reader.get_depth(i)  # 读取第 i 帧的深度图
 
    if i == 0:
      # 从数据集中获取mask数据
      mask = reader.get_mask(0).astype(bool)
      # 进行 初始姿态估计，输入相机内参 (K)、RGB 图像、深度图和物体掩码 并进行 est_refine_iter 轮优化
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)
 
      # 只有当 debug 级别 大于等于 3 时，才会执行下面的代码
      if debug >= 3:  # debug为1级 最基本的可视化 debug为2级 保存中间结果track_vis中的图像 debug为3级 更详细的可视化，例如导出变换后的 3D 物体模型和场景点云
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth >= 0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      # 进行姿态跟踪，从前一帧的姿态开始，优化track_refine_iter轮
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=track_refine_iter)
 
    # 保存物体在相机坐标系下的姿态矩阵
    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4, 4))
 
    # 在 RGB 图像上绘制 3D 包围盒和坐标轴，并显示出来
    if debug >= 1:
      center_pose = pose @ np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0,
                          is_input_rgb=True)
      # cv2.imshow('1', vis[..., ::-1])
      # cv2.waitKey(1)
 
    # 如果 debug >= 2，保存可视化的跟踪结果图片
    if debug >= 2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)
 
 
if __name__ == '__main__':
  main()
 
 
 
 
 
 
 
