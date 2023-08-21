import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from sklearn import datasets
from copy import copy

SCALE = 0.9

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, "reacher.xml", 2)

    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        # reward_dist = -np.linalg.norm(vec)
        # reward_ctrl = -np.square(a).sum()
        # reward = reward_dist + reward_ctrl
        if np.linalg.norm(vec) < 0.02:
            reward = 1
        else:
            reward = 0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        # return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        return ob, reward, done, dict()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    # def reset_model(self):
    #     qpos = (
    #         self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
    #         + self.init_qpos
    #     )
    #     while True:
    #         self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
    #         if np.linalg.norm(self.goal) < 0.2:
    #             break
    #     qpos[-2:] = self.goal
    #     qvel = self.init_qvel + self.np_random.uniform(
    #         low=-0.005, high=0.005, size=self.model.nv
    #     )
    #     qvel[-2:] = 0
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()

    def set_env_type(self, env_type, inter_context=None):
        assert env_type in ['target', 'interpolation']
        self.env_type = env_type
        self.inter_context = inter_context
        self.init_fingertip = []
        self.goals = []

    def reset_model(self):
        qpos = (
            self.np_random.uniform(low=-0.2, high=0.2, size=self.model.nq)
            + self.init_qpos
        )
        # qpos = (
        #     np.array([self.np_random.uniform(low=-np.pi, high=np.pi, size=1)[0],
        #      self.np_random.uniform(low=-np.pi/1.5-0.2, high=-np.pi/1.5+0.2, size=1)[0],
        #      0,
        #      0])
        #     + self.init_qpos
        # )
        while True:
            if self.env_type == 'target':
                self.goal, _ = datasets.make_s_curve(1, noise=0.1)
                self.goal = self.goal[0, (0, 2)]*SCALE
                # self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
            elif self.env_type == 'interpolation':
                random_idx = np.random.choice(self.inter_context.shape[0])
                self.goal = self.inter_context[random_idx, :] + 0.01*np.random.randn(2, )
                # mengdi: self.goal = context
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        self.init_fingertip += [copy(self.get_body_com("fingertip")[:2])]
        self.goals += [copy(self.goal)]
        return self._get_obs()
    
    def save_init_fingertip(self):
        np.save('init_fingertip', self.init_fingertip)
    
    def save_goals(self):
        np.save('goals', self.init_fingertip)

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:2],
                self.get_body_com("fingertip") - self.get_body_com("target"),
            ]
        )
