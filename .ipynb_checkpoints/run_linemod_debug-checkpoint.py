# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
 
 
from Utils import *
import json,uuid,joblib,os,sys
import scipy.spatial as spatial
from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat
import itertools
from datareader import *
from estimater import *
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/mycpp/build')
import yaml
import re
 
 
def get_mask(reader, i_frame, ob_id, detect_type):
  if detect_type=='box':
    mask = reader.get_mask(i_frame, ob_id)
    H,W = mask.shape[:2]
    vs,us = np.where(mask>0)
    umin = us.min()
    umax = us.max()
    vmin = vs.min()
    vmax = vs.max()
    valid = np.zeros((H,W), dtype=bool)
    valid[vmin:vmax,umin:umax] = 1
  elif detect_type=='mask':
    mask = reader.get_mask(i_frame, ob_id)
    if mask is None:
      return None
    valid = mask>0
  elif detect_type=='detected':
    mask = cv2.imread(reader.color_files[i_frame].replace('rgb','mask_cosypose'), -1)
    valid = mask==ob_id
  else:
    raise RuntimeError
  return valid
 
 
 
def run_pose_estimation_worker(reader, i_frames, est:FoundationPose=None, debug=0, ob_id=None, device='cuda:0'):
  torch.cuda.set_device(device)
  est.to_device(device)
  est.glctx = dr.RasterizeCudaContext(device=device)
 
  result = NestDict()
 
  for i, i_frame in enumerate(i_frames):
    logging.info(f"{i}/{len(i_frames)}, i_frame:{i_frame}, ob_id:{ob_id}")
    print("\n### ", f"{i}/{len(i_frames)}, i_frame:{i_frame}, ob_id:{ob_id}")
    video_id = reader.get_video_id()
    color = reader.get_color(i_frame)
    depth = reader.get_depth(i_frame)
    id_str = reader.id_strs[i_frame]
    H,W = color.shape[:2]
 
    debug_dir =est.debug_dir
 
    ob_mask = get_mask(reader, i_frame, ob_id, detect_type=detect_type)
    if ob_mask is None:
      logging.info("ob_mask not found, skip")
      result[video_id][id_str][ob_id] = np.eye(4)
      return result
 
    est.gt_pose = reader.get_gt_pose(i_frame, ob_id)
 
    pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=ob_mask, ob_id=ob_id)
    logging.info(f"pose:\n{pose}")
 
    if debug>=3:
      m = est.mesh_ori.copy()
      tmp = m.copy()
      tmp.apply_transform(pose)
      tmp.export(f'{debug_dir}/model_tf.obj')
 
    result[video_id][id_str][ob_id] = pose
 
  return result, pose
 
 
def run_pose_estimation():
  print(">>> 正在加载 Warp...", flush=True) # 强制刷新 flush=True
  wp.force_load(device='cuda')
  print(">>> Warp 加载完成，准备初始化 Reader...", flush=True)
  print(f">>> 开始初始化 LinemodReader，路径: {opt.linemod_dir}", flush=True)
  reader_tmp = LinemodReader(opt.linemod_dir, split=None)
  print(">>> LinemodReader 初始化完成！", flush=True)
  print("## opt.linemod_dir:", opt.linemod_dir)
 
  debug = opt.debug
  use_reconstructed_mesh = opt.use_reconstructed_mesh
  debug_dir = opt.debug_dir
 
  res = NestDict()
  glctx = dr.RasterizeCudaContext()
  mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)).to_mesh()
  est = FoundationPose(model_pts=mesh_tmp.vertices.copy(), model_normals=mesh_tmp.vertex_normals.copy(), symmetry_tfs=None, mesh=mesh_tmp, scorer=None, refiner=None, glctx=glctx, debug_dir=debug_dir, debug=debug)
 
  # ob_id
  match = re.search(r'\d+$', opt.linemod_dir)
  if match:
      last_number = match.group()
      ob_id = int(last_number)
  else:
      print("No digits found at the end of the string")
      
  # for ob_id in reader_tmp.ob_ids:
  if ob_id:
    if use_reconstructed_mesh:
      print("## ob_id:", ob_id)
      print("## opt.linemod_dir:", opt.linemod_dir)
      print("## opt.ref_view_dir:", opt.ref_view_dir)
      mesh = reader_tmp.get_reconstructed_mesh(ref_view_dir=opt.ref_view_dir)
    else:
      mesh = reader_tmp.get_gt_mesh(ob_id)
    # symmetry_tfs = reader_tmp.symmetry_tfs[ob_id]  # !!!!!!!!!!!!!!!!
 
    args = []
 
    reader = LinemodReader(opt.linemod_dir, split=None)
    video_id = reader.get_video_id()
    # est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), symmetry_tfs=symmetry_tfs, mesh=mesh)  # raw
    est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), mesh=mesh) # !!!!!!!!!!!!!!!!
 
    print("### len(reader.color_files):", len(reader.color_files))
    for i in range(len(reader.color_files)):
      args.append((reader, [i], est, debug, ob_id, "cuda:0"))
 
    # vis Data
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    os.makedirs(f'{opt.linemod_dir}/track_vis', exist_ok=True)
 
    outs = []
    i = 0
    for arg in args[:200]:
      print("### num:", i)
      out, pose = run_pose_estimation_worker(*arg)
      outs.append(out)
      center_pose = pose@np.linalg.inv(to_origin)
      img_color = reader.get_color(i)
      vis = draw_posed_3d_box(reader.K, img=img_color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(img_color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      imageio.imwrite(f'{opt.linemod_dir}/track_vis/{reader.id_strs[i]}.png', vis)
      i = i + 1
 
    for out in outs:
      for video_id in out:
        for id_str in out[video_id]:
          for ob_id in out[video_id][id_str]:
            res[video_id][id_str][ob_id] = out[video_id][id_str][ob_id]
 
  with open(f'{opt.debug_dir}/linemod_res.yml','w') as ff:
    yaml.safe_dump(make_yaml_dumpable(res), ff)
    print("Save linemod_res.yml OK !!!")
 
 
if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--linemod_dir', type=str, default="/root/autodl-tmp/lm_test_all/test/000015", help="linemod root dir") # lm_test_all  lm_test
  parser.add_argument('--use_reconstructed_mesh', type=int, default=1)
  parser.add_argument('--ref_view_dir', type=str, default="/root/autodl-tmp/FoundationPose/demo_data/ref_views/ob_0000015")
  parser.add_argument('--debug', type=int, default=0)
  parser.add_argument('--debug_dir', type=str, default=f'/root/autodl-tmp/lm_test_all/debug') # lm_test_all  lm_test
  opt = parser.parse_args()
  set_seed(0)
 
  detect_type = 'mask'   # mask / box / detected
  run_pose_estimation()
 
