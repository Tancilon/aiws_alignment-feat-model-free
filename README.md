# FoundationPose基于工件数据运行的全流程

## ------Model-Based------

1. 模型准备：下载官网提供的模型权重，放在“FoundationPose/weights/2023-10-28-18-33-37”的固定路径下，因为“ predict_score.py / predict_pose_refine.py”把两个Predictor的路径给写死了；当然，也可以自己设置一个存放路径，然后通过构造函数/修改源码的方式修改模型的权重路径 (run_linemod.py --> estimater.py --> predict_score.py / predict_pose_refine.py；总共需要加载ScorePredictor - 姿态评分器、PoseRefinePredictor - 姿态精细化器两个估计器的权重)

   <img src="./teasers/model.png" style="zoom:50%;" />

   

2. 数据准备：针对每个工件，用工件的名字建立一个文件夹，文件夹下建立rgb/depth/masks/mesh四个子文件夹+1个相机内参cam_K.txt

   <img src="./teasers/data.png" style="zoom:50%;" />

   - rgb：存储原始分辨率的彩色rgb图片，png格式，命名没有要求

   - depth：存储与彩色rgb对齐的深度图，（uint16, 毫米, png），命名与rgb保持一致。从原始exr格式转换到上述格式的脚本：

     ```
     # 批量转换，输出到单独文件夹
     cd depth_utils
     python exr2png.py /path/to/exr_folder --out_dir /path/to/output_dir
     ```

   - masks：存储每张图片的工件mask，（单通道，0/255，png），命名与rgb保持一致。生成脚本在自动分割项目中

   - mesh：工件的CAD模型。注意：这里的cad模型不是纯白模、没有纹理的stl格式，而是有纹理贴图的obj格式（随机生成一个纯白纹理即可，foundationpose主要基于几何进行pose预测）；同时要保证模型的单位是m。生成纹理的脚本：

     ```
     # 利用blender软件后台的python解释器，需要下载blender软件
     cd depth_utils
     # 如果原始的stl单位是m，用下面的脚本：
     /Applications/Blender.app/Contents/MacOS/Blender -b --python auto_texture.py
     # 如果原始的stl单位是mm，用下面的脚本：
     /Applications/Blender.app/Contents/MacOS/Blender -b --python auto_texture_mm2m.py
     ```

     **注意：**修改obj、mtl、png文件的名字时，需要同时修改文件内部的引用名字！

     <img src="./teasers/mesh.png" style="zoom:50%;" />

   - cam_K.txt：rgb相机的内参数（这里默认深度图已经对齐到rgb，所以统一使用rgb相机的内参数）。从原始tuyang相机的txt文件转换为标准3x3内参文件的脚本：

     ```
     cd depth_utils
     python convert_camK.py
     ```

3. Dataset/Dataloader准备：直接使用官方estimater.py脚本中提供的YcbineoatReader。注意：这里的shorter_side默认是为None，这里需要设置为480或者更小的数值，否则会OOM（YcbineoatReader会自动根据缩放比例对内参进行缩放）

   ```
   # run_demo_weld_register_single.py: L64
   reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=480, zfar=np.inf)
   ```

4. 推理脚本：FoundationPose提供两种推理模式，一个是“est.register“，一个是“est.track_one”。其中register一般是对一段视频中的第一帧进行处理，需要对应的mask，精度较高，但是处理速度较慢；track_one是对一段视频中后续的帧进行追踪，不需要对应的mask，速度非常快，但是精度不如register。对于本项目的焊接场景，直接使用register即可：

   ```
   # 推理脚本的逻辑很简单，继承foundationpose官方代码的只有两行
   # from estimater import *
   # from datareader import *
   
   # 对每一帧都进行register的推理脚本
   # 需要更改：mesh_file、test_scene_dir、debug（debug等级1～3，这里默认使用2，可以输出可视化结果）、debug_dir（预测结果的保存路径）
   python run_demo_weld_register_single.py
   
   # 对第一帧进行register，后续帧进行track的推理脚本
   # 需要更改：mesh_file、test_scene_dir、debug（debug等级1～3，这里默认使用2，可以输出可视化结果）、debug_dir（预测结果的保存路径）
   python run_demo_weld_track.py
   
   ```

   

## ------Model-Free------

**所谓model-free，就是在没有手工制作的CAD条件下，通过rgbd图片进行三维重建（bundlesdf/sam3d/vggt），得到代理三维模型（xxx.obj）。三维重建的质量不要求特别高，但是尺度要对的上，foundationpose同时基于纹理和几何进行预测，对于形状具有较强的鲁棒性。**

1. 基于多视角图片的纹理模型生成：利用bundlesdf进行obj文件的生成，具体细节查看bundlesdf分支（允许物体旋转+相机固定）
2. 直接使用model-based的推理脚本：将生成obj、mtl、png文件直接放进原来model-based所需的mesh文件夹下，就可以用model-based的reader进行数据的加载、图片的缩放和模型的推理
3. 或者使用model-free的推理脚本：也可以使用run_linemod_debug.py，但是需要把参考数据(model.obj)和测试数据(rgbd图片)全部整理成linemod所需的格式，才能使用linemod的datareader。实测下来特别麻烦，而且效果还不好
4. **效果最好的方式：**用bundlesdf生成带有纹理+真实尺度的obj模型，然后使用model-based的run_demo_weld_register_single.py进行逐帧的register推理
