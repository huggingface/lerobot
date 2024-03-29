import numpy as np

from xarm import Base


class Lift(Base):
    def __init__(self):
        self._z_threshold = 0.15
        super().__init__("lift")

    @property
    def z_target(self):
        return self._init_z + self._z_threshold

    def is_success(self):
        return self.obj[2] >= self.z_target

    def get_reward(self):
        reach_dist = np.linalg.norm(self.obj - self.eef)
        reach_dist_xy = np.linalg.norm(self.obj[:-1] - self.eef[:-1])
        pick_completed = self.obj[2] >= (self.z_target - 0.01)
        obj_dropped = (self.obj[2] < (self._init_z + 0.005)) and (reach_dist > 0.02)

        # Reach
        if reach_dist < 0.05:
            reach_reward = -reach_dist + max(self._action[-1], 0) / 50
        elif reach_dist_xy < 0.05:
            reach_reward = -reach_dist
        else:
            z_bonus = np.linalg.norm(np.linalg.norm(self.obj[-1] - self.eef[-1]))
            reach_reward = -reach_dist - 2 * z_bonus

        # Pick
        if pick_completed and not obj_dropped:
            pick_reward = self.z_target
        elif (reach_dist < 0.1) and (self.obj[2] > (self._init_z + 0.005)):
            pick_reward = min(self.z_target, self.obj[2])
        else:
            pick_reward = 0

        return reach_reward / 100 + pick_reward

    def _get_obs(self):
        eef_velp = self._utils.get_site_xvelp(self.model, self.data, "grasp") * self.dt
        gripper_angle = self._utils.get_joint_qpos(self.model, self.data, "right_outer_knuckle_joint")
        eef = self.eef - self.center_of_table

        obj = self.obj - self.center_of_table
        obj_rot = self._utils.get_joint_qpos(self.model, self.data, "object_joint0")[-4:]
        obj_velp = self._utils.get_site_xvelp(self.model, self.data, "object_site") * self.dt
        obj_velr = self._utils.get_site_xvelr(self.model, self.data, "object_site") * self.dt

        obs = np.concatenate(
            [
                eef,
                eef_velp,
                obj,
                obj_rot,
                obj_velp,
                obj_velr,
                eef - obj,
                np.array(
                    [
                        np.linalg.norm(eef - obj),
                        np.linalg.norm(eef[:-1] - obj[:-1]),
                        self.z_target,
                        self.z_target - obj[-1],
                        self.z_target - eef[-1],
                    ]
                ),
                gripper_angle,
            ],
            axis=0,
        )
        return {"observation": obs, "state": eef, "achieved_goal": eef, "desired_goal": eef}

    def _sample_goal(self):
        # Gripper
        gripper_pos = np.array([1.280, 0.295, 0.735]) + self.np_random.uniform(-0.05, 0.05, size=3)
        super()._set_gripper(gripper_pos, self.gripper_rotation)

        # Object
        object_pos = self.center_of_table - np.array([0.15, 0.10, 0.07])
        object_pos[0] += self.np_random.uniform(-0.05, 0.05, size=1)
        object_pos[1] += self.np_random.uniform(-0.05, 0.05, size=1)
        object_qpos = self._utils.get_joint_qpos(self.model, self.data, "object_joint0")
        object_qpos[:3] = object_pos
        self._utils.set_joint_qpos(self.model, self.data, "object_joint0", object_qpos)
        self._init_z = object_pos[2]

        # Goal
        return object_pos + np.array([0, 0, self._z_threshold])

    def reset(self):
        self._action = np.zeros(4)
        return super().reset()

    def step(self, action):
        self._action = action.copy()
        return super().step(action)
