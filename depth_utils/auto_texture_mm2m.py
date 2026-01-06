import bpy
import os

# =============== 配置路径（改成你自己的） ===============
stl_path = "/Users/wangrunze/Desktop/工件数据/工件数据-自采/方管/F120/正常环境/1正面0度俯角/F120.stl"
output_dir = "/Users/wangrunze/Desktop/工件数据/工件数据-自采/方管/F120/正常环境/1正面0度俯角/mesh"
os.makedirs(output_dir, exist_ok=True)

obj_path = os.path.join(output_dir, "F120.obj")
tex_path = os.path.join(output_dir, "white_texture.png")

# =============== 参数：mm -> m 缩放系数 ===============
MM_TO_M = 0.001

# =============== 清空场景 & 导入 STL ===============
bpy.ops.wm.read_factory_settings(use_empty=True)

# （可选）设置场景单位显示为米；不影响 OBJ 内容，只是 UI 显示更直观
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = 1.0  # 1 Blender Unit = 1 m（显示层面）

bpy.ops.import_mesh.stl(filepath=stl_path)

# STL 导入后通常会选中导入物体
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

# =============== 将数值从 mm 转为 m（缩放 0.001 并应用） ===============
obj.scale = (MM_TO_M, MM_TO_M, MM_TO_M)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# =============== 自动 UV 展开 ===============
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project()
bpy.ops.object.mode_set(mode='OBJECT')

# =============== 创建纯白纹理（不依赖 PIL） ===============
tex_size = 256
image = bpy.data.images.new("white_texture", width=tex_size, height=tex_size, alpha=True)

# 填充纯白 RGBA
image.pixels = [1.0, 1.0, 1.0, 1.0] * (tex_size * tex_size)

image.filepath_raw = tex_path
image.file_format = 'PNG'
image.save()

# =============== 创建材质并绑定纹理 ===============
mat = bpy.data.materials.new("WhiteMat")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links

# 找到或创建 Principled BSDF
bsdf = None
for n in nodes:
    if n.type == 'BSDF_PRINCIPLED':
        bsdf = n
        break
if bsdf is None:
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")

# 贴图节点
tex_node = nodes.new("ShaderNodeTexImage")
tex_node.image = image

# 连接 Base Color
links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

# 赋材质
if obj.data.materials:
    obj.data.materials[0] = mat
else:
    obj.data.materials.append(mat)

# =============== 导出 OBJ（Blender 4.0 新接口） ===============
# 注意：我们已 Apply Scale，所以这里不需要再额外 export scale
bpy.ops.wm.obj_export(
    filepath=obj_path,
    export_animation=False,
    export_uv=True,
    export_normals=True,
    export_colors=False,
    export_materials=True,
    export_pbr_extensions=False,
    path_mode='RELATIVE',            # MTL 里写相对路径，通常更稳
    export_triangulated_mesh=True,
    export_selected_objects=False,
)

print("DONE! (mm->m) OBJ / MTL / PNG saved to:", output_dir)
