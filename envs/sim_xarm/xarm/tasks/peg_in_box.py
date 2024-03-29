import numpy as np

from xarm import Base


class PegInBox(Base):
    def __init__(self):
        super().__init__("peg_in_box")

    def _reset_sim(self):
        self._act_magnitude = 0
        super()._reset_sim()
        for _ in range(10):
            self._apply_action(np.array([0, 0, 0, 1], dtype=np.float32))
            self.sim.step()

    @property
    def box(self):
        return self.sim.data.get_site_xpos("box_site")

    def is_success(self):
        return np.linalg.norm(self.obj - self.box) <= 0.05

    def get_reward(self):
        dist_xy = np.linalg.norm(self.obj[:2] - self.box[:2])
        dist_xyz = np.linalg.norm(self.obj - self.box)
        return float(dist_xy <= 0.045) * (2 - 6 * dist_xyz) - 0.2 * np.square(self._act_magnitude) - dist_xy

    def _get_obs(self):
        eef_velp = self.sim.data.get_site_xvelp("grasp") * self.dt
        gripper_angle = self.sim.data.get_joint_qpos("right_outer_knuckle_joint")
        eef, box = self.eef - self.center_of_table, self.box - self.center_of_table

        obj = self.obj - self.center_of_table
        obj_rot = self.sim.data.get_joint_qpos("object_joint0")[-4:]
        obj_velp = self.sim.data.get_site_xvelp("object_site") * self.dt
        obj_velr = self.sim.data.get_site_xvelr("object_site") * self.dt

        obs = np.concatenate(
            [
                eef,
                eef_velp,
                box,
                obj,
                obj_rot,
                obj_velp,
                obj_velr,
                eef - box,
                eef - obj,
                obj - box,
                np.array(
                    [
                        np.linalg.norm(eef - box),
                        np.linalg.norm(eef - obj),
                        np.linalg.norm(obj - box),
                        gripper_angle,
                    ]
                ),
            ],
            axis=0,
        )
        return {"observation": obs, "state": eef, "achieved_goal": eef, "desired_goal": box}

    def _sample_goal(self):
        # Gripper
        gripper_pos = np.array([1.280, 0.295, 0.9]) + self.np_random.uniform(-0.05, 0.05, size=3)
        super()._set_gripper(gripper_pos, self.gripper_rotation)

        # Object
        object_pos = gripper_pos - np.array([0, 0, 0.06]) + self.np_random.uniform(-0.005, 0.005, size=3)
        object_qpos = self.sim.data.get_joint_qpos("object_joint0")
        object_qpos[:3] = object_pos
        self.sim.data.set_joint_qpos("object_joint0", object_qpos)

        # Box
        box_pos = np.array([1.61, 0.18, 0.58])
        box_pos[:2] += self.np_random.uniform(-0.11, 0.11, size=2)
        box_qpos = self.sim.data.get_joint_qpos("box_joint0")
        box_qpos[:3] = box_pos
        self.sim.data.set_joint_qpos("box_joint0", box_qpos)

        return self.box

    def step(self, action):
        self._act_magnitude = np.linalg.norm(action[:3])
        return super().step(action)
