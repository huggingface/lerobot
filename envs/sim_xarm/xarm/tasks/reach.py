import numpy as np

from xarm import Base


class Reach(Base):
    def __init__(self):
        super().__init__("reach")

    def _reset_sim(self):
        self._act_magnitude = 0
        super()._reset_sim()

    def is_success(self):
        return np.linalg.norm(self.eef - self.goal) <= 0.05

    def get_reward(self):
        dist = np.linalg.norm(self.eef - self.goal)
        penalty = self._act_magnitude**2
        return -(dist + 0.15 * penalty)

    def _get_obs(self):
        eef_velp = self.sim.data.get_site_xvelp("grasp") * self.dt
        gripper_angle = self.sim.data.get_joint_qpos("right_outer_knuckle_joint")
        eef, goal = self.eef - self.center_of_table, self.goal - self.center_of_table
        obs = np.concatenate(
            [eef, eef_velp, goal, eef - goal, np.array([np.linalg.norm(eef - goal), gripper_angle])], axis=0
        )
        return {"observation": obs, "state": eef, "achieved_goal": eef, "desired_goal": goal}

    def _sample_goal(self):
        # Gripper
        gripper_pos = np.array([1.280, 0.295, 0.735]) + self.np_random.uniform(-0.05, 0.05, size=3)
        super()._set_gripper(gripper_pos, self.gripper_rotation)

        # Goal
        self.goal = np.array([1.550, 0.287, 0.580])
        self.goal[:2] += self.np_random.uniform(-0.125, 0.125, size=2)
        self.sim.model.site_pos[self.sim.model.site_name2id("target0")] = self.goal
        return self.goal

    def step(self, action):
        self._act_magnitude = np.linalg.norm(action[:3])
        return super().step(action)
