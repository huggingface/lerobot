import numpy as np

from xarm import Base


class Push(Base):
    def __init__(self):
        super().__init__("push")

    def _reset_sim(self):
        self._act_magnitude = 0
        super()._reset_sim()

    def is_success(self):
        return np.linalg.norm(self.obj - self.goal) <= 0.05

    def get_reward(self):
        dist = np.linalg.norm(self.obj - self.goal)
        penalty = self._act_magnitude**2
        return -(dist + 0.15 * penalty)

    def _get_obs(self):
        eef_velp = self.sim.data.get_site_xvelp("grasp") * self.dt
        gripper_angle = self.sim.data.get_joint_qpos("right_outer_knuckle_joint")
        eef, goal = self.eef - self.center_of_table, self.goal - self.center_of_table

        obj = self.obj - self.center_of_table
        obj_rot = self.sim.data.get_joint_qpos("object_joint0")[-4:]
        obj_velp = self.sim.data.get_site_xvelp("object_site") * self.dt
        obj_velr = self.sim.data.get_site_xvelr("object_site") * self.dt

        obs = np.concatenate(
            [
                eef,
                eef_velp,
                goal,
                obj,
                obj_rot,
                obj_velp,
                obj_velr,
                eef - goal,
                eef - obj,
                obj - goal,
                np.array(
                    [
                        np.linalg.norm(eef - goal),
                        np.linalg.norm(eef - obj),
                        np.linalg.norm(obj - goal),
                        gripper_angle,
                    ]
                ),
            ],
            axis=0,
        )
        return {"observation": obs, "state": eef, "achieved_goal": eef, "desired_goal": goal}

    def _sample_goal(self):
        # Gripper
        gripper_pos = np.array([1.280, 0.295, 0.735]) + self.np_random.uniform(-0.05, 0.05, size=3)
        super()._set_gripper(gripper_pos, self.gripper_rotation)

        # Object
        object_pos = self.center_of_table - np.array([0.25, 0, 0.07])
        object_pos[0] += self.np_random.uniform(-0.08, 0.08, size=1)
        object_pos[1] += self.np_random.uniform(-0.08, 0.08, size=1)
        object_qpos = self.sim.data.get_joint_qpos("object_joint0")
        object_qpos[:3] = object_pos
        self.sim.data.set_joint_qpos("object_joint0", object_qpos)

        # Goal
        self.goal = np.array([1.600, 0.200, 0.545])
        self.goal[:2] += self.np_random.uniform(-0.1, 0.1, size=2)
        self.sim.model.site_pos[self.sim.model.site_name2id("target0")] = self.goal
        return self.goal

    def step(self, action):
        self._act_magnitude = np.linalg.norm(action[:3])
        return super().step(action)
