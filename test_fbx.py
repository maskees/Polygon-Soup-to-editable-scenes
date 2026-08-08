import aspose.threed as a3d

scene = a3d.Scene()
tmp = a3d.Scene.from_file(r"d:\3d model\data\output\subject_002\intermediate\stage3_partition\part_000.obj")

# Find the first node with geometry
mesh_node = None
for node in tmp.root_node.child_nodes:
    if node.entity is not None:
        mesh_node = node
        break

if mesh_node is not None:
    # Set material
    mat = a3d.shading.LambertMaterial()
    mat.diffuse_color = a3d.utilities.Vector3(1.0, 0.0, 0.0)
    mesh_node.material = mat
    scene.root_node.add_child_node(mesh_node)

scene.save(r"d:\3d model\test.fbx", a3d.FileFormat.FBX7700_BINARY)
print("Saved custom FBX")
