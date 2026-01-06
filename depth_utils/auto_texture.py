import bpy
import os

# =============== 配置路径（改成你自己的） ===============
stl_path = "/Users/wangrunze/Desktop/工件数据/test_photo/diban.stl"
output_dir = "/Users/wangrunze/Desktop/工件数据/test_photo/mesh"
os.makedirs(output_dir, exist_ok=True)

obj_path = os.path.join(output_dir, "G90.obj")
tex_path = os.path.join(output_dir, "white_texture.png")

# =============== 清空场景 & 导入 STL ===============
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_mesh.stl(filepath=stl_path)
obj = bpy.context.selected_objects[0]

# =============== 自动 UV 展开 ===============
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project()
bpy.ops.object.mode_set(mode='OBJECT')

# =============== 创建纯白纹理（不依赖 PIL） ===============
tex_size = 256
image = bpy.data.images.new("white_texture", width=tex_size, height=tex_size)

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

tex_node = nodes.new("ShaderNodeTexImage")
tex_node.image = image

links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

# 赋材质
if obj.data.materials:
    obj.data.materials[0] = mat
else:
    obj.data.materials.append(mat)

# =============== 导出 OBJ（Blender 4.0 新接口） ===============
bpy.ops.wm.obj_export(
    filepath=obj_path,
    export_animation=False,
    export_uv=True,
    export_normals=True,
    export_colors=False,
    export_materials=True,
    export_pbr_extensions=False,
    path_mode='RELATIVE',
    export_triangulated_mesh=True,
    export_selected_objects=False,
)

print("DONE! OBJ / MTL / PNG saved to:", output_dir)
