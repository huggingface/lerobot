from urdfpy import URDF
import trimesh, os

u = URDF.load("/home/steven/research/low_cost_robot/simulation/low_cost_robot_6dof/low-cost-arm.urdf")

bad = []
for link in u.links:
    for i, vis in enumerate(link.visuals):
        if vis.geometry.mesh is None:
            bad.append((link.name, i, "no mesh object"))
            continue
        # Resolve the file path the same way urdfpy would
        mesh_files = vis.geometry.mesh.files
        for f in mesh_files:
            try:
                m = trimesh.load(f, force="mesh", skip_materials=True)
                if (getattr(m, "vertices", None) is None or len(m.vertices) == 0 or
                    getattr(m, "faces", None) is None or len(m.faces) == 0):
                    bad.append((link.name, i, f"empty mesh: {f}"))
            except Exception as e:
                bad.append((link.name, i, f"failed load: {f} :: {e}"))
print("BAD VISUALS:", bad)

