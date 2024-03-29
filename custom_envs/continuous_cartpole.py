"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R

Continuous version by Ian Danforth
"""

import gym
from gym import logger, register, spaces
from gym.utils import seeding
import numpy as np
import torch

# max_episode_steps = 500
max_episode_steps = 200

register(
    id='ContinuousCartPole-v0',
    entry_point='custom_envs.continuous_cartpole:ContinuousCartPole',
    max_episode_steps=max_episode_steps
)

class ContinuousCartPole(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.state_dim = 4
        self.action_dim = 1

        self.four_thirds = 4.0/3.0

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        # self.force_mag = 30.0
        self.force_mag = 1.0
        self.tau = 0.02  # seconds between state updates
        self.min_action = -30.0
        self.max_action = 30.0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * np.pi / 360 # np.pi / 15
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        # high = np.array([
        #     self.x_threshold * 2,
        #     np.finfo(np.float32).max,
        #     self.theta_threshold_radians * 2,
        #     np.finfo(np.float32).max
        # ], dtype=np.float32)
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float64).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float64).max
        ], dtype=np.float64)

        # self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        # self.action_space = spaces.Box(
        #     low=self.min_action,
        #     high=self.max_action,
        #     shape=(1,),
        #     dtype=np.float32
        # )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,),
            dtype=np.float64
        )

        # LQR state and action matrices
        self.continuous_A = np.array([
            [0, 1, 0, 0],
            [0, 0, -self.gravity * self.masspole / (self.total_mass * (-self.masspole / self.total_mass + self.four_thirds)), 0],
            [0, 0, 0, 1],
            [0, 0, self.gravity / (self.length * (-self.masspole / (self.masscart + self.masscart) + self.four_thirds)), 0]
        ])
        self.continuous_B = np.array([
            [0],
            [(self.masspole / (-self.masspole / self.total_mass + self.four_thirds) + 1) / self.total_mass],
            [0],
            [-1 / (self.length * (-self.masspole / self.total_mass + self.four_thirds))]
        ])

        # Define cost/reward values
        self.Q = np.array([[10, 0,  0, 0],
                           [ 0, 1,  0, 0],
                           [ 0, 0, 10, 0],
                           [ 0, 0,  0, 1]])
        self.R = np.array([[0.1]])

        self.reference_point = np.zeros(self.state_dim)

        # Misc
        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def cost_fn(self, state, action):
        _state = state - self.reference_point

        cost = _state @ self.Q @ _state.T + action @ self.R @ action.T

        return cost

    def vectorized_cost_fn(self, states, actions):
        _states = (states - self.reference_point).T
        mat = torch.diag(_states.T @ self.Q @ _states).unsqueeze(-1) + torch.pow(actions.T, 2) * self.R

        return mat.T

    def reward_fn(self, state, action):
        return -self.cost_fn(state, action)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, force):
        x, x_dot, theta, theta_dot = self.state
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
            (self.length * (self.four_thirds - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return np.array([x, x_dot, theta, theta_dot])

    def step(self, action):
        action = action[0]
        # assert self.action_space.contains(action), \
        #     "%r (%s) invalid" % (action, type(action))

        force = self.force_mag * action
        self.state = self.stepPhysics(force)
        self.step_count += 1
        # x, x_dot, theta, theta_dot = self.state
        # done = x < -self.x_threshold \
        #     or x > self.x_threshold \
        #     or theta < -self.theta_threshold_radians \
        #     or theta > self.theta_threshold_radians \
        #     or self.step_count >= max_episode_steps
        # done = bool(done)
        done = self.step_count >= max_episode_steps

#         if not done:
#             reward = 1.0
#         elif self.steps_beyond_done is None:
#             # Pole just fell!
#             self.steps_beyond_done = 0
#             reward = 1.0
#         else:
#             if self.steps_beyond_done == 0:
#                 logger.warn("""
# You are calling 'step()' even though this environment has already returned
# done = True. You should always call 'reset()' once you receive 'done = True'
# Any further steps are undefined behavior.
#                 """)
#             self.steps_beyond_done += 1
#             reward = 0.0

        # Replace regular +1 reward with negative LQR cost function
        reward = self.reward_fn(
            np.array(self.state),
            np.array([action])
        )

        return self.state, reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        self.step_count = 0
        return self.state

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen-polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def close(self):
        if self.viewer:
            self.viewer.close()