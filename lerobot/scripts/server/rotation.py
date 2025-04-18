from scipy.spatial.transform import Rotation as R
import numpy as np
# === Convert Euler angles to Quaternion ===
# Euler format: (roll, pitch, yaw) in radians
euler_angles = [0, 0.7071, 0]  # example: yaw of 90 degrees
r = R.from_euler('xyz', euler_angles)
quat = r.as_quat()  # [x, y, z, w]
print("Quaternion [x y z w]:", quat)

# === Convert Quaternion back to Euler ===
# Quaternion format: [x, y, z, w]
r2 = R.from_quat(quat)
euler = r2.as_euler('xyz')
print("Euler angles [rad]:", euler)


# Define X and Y axes from xyaxes
x_axis = np.array([1.000, -0.002, 0.000])
y_axis = np.array([0.001, 0.364, 0.931])

# Normalize X and Y to ensure they are unit vectors
x_axis /= np.linalg.norm(x_axis)
y_axis /= np.linalg.norm(y_axis)

# Compute Z axis as cross product of X and Y
z_axis = np.cross(x_axis, y_axis)

# Recompute Y to ensure orthogonality (optional but more robust)
y_axis = np.cross(z_axis, x_axis)

# Form rotation matrix (each column is an axis vector)
rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

# Convert rotation matrix to quaternion [x y z w]
r = R.from_matrix(rotation_matrix)
quat_xyzw = r.as_quat()

# Convert to MuJoCo format [w x y z]
quat_mujoco = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

print("MuJoCo quaternion [w x y z]:", quat_mujoco)

