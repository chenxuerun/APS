import numpy as np
import gym
import networkx as nx
from math import sqrt
from tf_agents.environments import gym_wrapper
from tf_agents.environments import wrappers
from tf_agents.environments import tf_py_environment
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter, FuncFormatter

WALLS = {
    'empty':
        np.zeros((10, 10), dtype=int),
    'FourRooms':
        np.array([
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]),
    'Maze11x11':
        np.array([
                  [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                  [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
}

def resize_walls(walls, factor):
    (height, width) = walls.shape
    row_indices = np.array([i for i in range(height) for _ in range(factor)])
    col_indices = np.array([i for i in range(width) for _ in range(factor)])
    walls = walls[row_indices]
    walls = walls[:, col_indices]
    assert walls.shape == (factor * height, factor * width)
    return walls

class PointEnv(gym.Env):
    # 单次活动范围 方差
    def __init__(self, walls=None, resize_factor=1,
                                action_range=1.0, action_noise=0.0):
        if resize_factor > 1:
            self._walls = resize_walls(WALLS[walls], resize_factor)
        else:
            self._walls = WALLS[walls]
        self._apsp = self._compute_apsp(self._walls)
        (height, width) = self._walls.shape
        self._height = height
        self._width = width
        self._action_noise = action_noise
        self.action_space = gym.spaces.Box(
            low=np.array([-action_range, -action_range]),
            high=np.array([action_range, action_range]),
            dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([self._height, self._width]),
            dtype=np.float32)
        # 这个值可以描述环境的难度
        self._k = sqrt(self._height * self._width) / action_range 
        self.reset()

    def reset(self):
        self.state = self._sample_empty_state()
        return self.state.copy()

    # def reset(self):
    #     self.state = np.array([22,22], dtype=np.float)
    #     return self.state.copy()

    def _sample_empty_state(self):
        candidate_states = np.where(self._walls == 0)
        num_candidate_states = len(candidate_states[0])
        state_index = np.random.choice(num_candidate_states)
        state = np.array([candidate_states[0][state_index],
                                candidate_states[1][state_index]], dtype=np.float)
        state += np.random.uniform(size=2)
        assert not self._is_blocked(state)
        return state
    
    def _is_blocked(self, state):
        if not self.observation_space.contains(state):
            return True
        (i, j) = self._discretize_state(state)
        return self._walls[i, j] == 1

    def _discretize_state(self, state, resolution=1.0):
        (i, j) = np.floor(resolution * state).astype(np.int)
        if i == self._height:
            i -= 1
        if j == self._width:
            j -= 1
        return (i, j)

    def _compute_apsp(self, walls): # 外部定义的距离
        (height, width) = walls.shape
        g = nx.Graph()
        # Add all the nodes
        for i in range(height):
            for j in range(width):
                if walls[i, j] == 0:
                    g.add_node((i, j))

        # Add all the edges
        for i in range(height):
            for j in range(width):
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == dj == 0: continue  # Don't add self loops
                        if i + di < 0 or i + di > height - 1: continue  # No cell here
                        if j + dj < 0 or j + dj > width - 1: continue  # No cell here
                        if walls[i, j] == 1: continue  # Don't add edges to walls
                        if walls[i + di, j + dj] == 1: continue  # Don't add edges to walls
                        g.add_edge((i, j), (i + di, j + dj))

        # dist[i, j, k, l] is path from (i, j) -> (k, l)
        dist = np.full((height, width, height, width), np.float('inf'))
        for ((i1, j1), dist_dict) in nx.shortest_path_length(g):
            for ((i2, j2), d) in dist_dict.items():
                dist[i1, j1, i2, j2] = d
        return dist

    def _get_distance(self, obs, goal):
        (i1, j1) = self._discretize_state(obs)
        (i2, j2) = self._discretize_state(goal)
        return self._apsp[i1, j1, i2, j2]

    def step(self, action): 
        if self._action_noise > 0:
            action += np.random.normal(0, self._action_noise)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action)

        num_substeps = 10
        dt = 1.0 / num_substeps
        num_axis = len(action)
        for _ in range(num_substeps):
            for axis in range(num_axis):
                new_state = self.state.copy()
                new_state[axis] += dt * action[axis]
                if not self._is_blocked(new_state):
                    self.state = new_state
        
        done = False
        rew = -1.0
        return self.state.copy(), rew, done, {}

    @property
    def walls(self):
        return self._walls

    @property
    def apsp(self):
        return self._apsp

# gym.Wrapper 和 没有继承自PyEnvironmentBaseWrapper
class GoalConditionedPointWrapper(gym.Wrapper):
    def __init__(self, env, prob_constraint=0.8,
                                min_dist=0, max_dist=5, threshold_dist=1.0):
        self._threshold_dist = threshold_dist
        self._prob_constraint = prob_constraint
        self._min_dist = min_dist
        self._max_dist = max_dist
        super(GoalConditionedPointWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Dict({
            'observation': env.observation_space,
            'goal': env.observation_space})
    
    def _normalize_obs(self, obs):
        return np.array([
            obs[0] / float(self.env._height),
            obs[1] / float(self.env._width)
        ])

    # def reset(self):
    #     obs = self.env.reset()
    #     self._goal = np.array([22,32], dtype=np.float)
    #     return {'observation': self._normalize_obs(obs),
    #             'goal': self._normalize_obs(self._goal)}

    def reset(self):
        goal = None
        count = 0
        while goal is None:
            obs = self.env.reset()
            (obs, goal) = self._sample_goal(obs)
            count += 1
            if count > 1000:
                print('WARNING: Unable to find goal within constraints.')
        self._goal = goal
        return {'observation': self._normalize_obs(obs),
                        'goal': self._normalize_obs(self._goal)}

    def step(self, action):
        obs, _, _, _ = self.env.step(action)
        rew = -1.0
        done = self._is_done(obs, self._goal)
        return {'observation': self._normalize_obs(obs),
                        'goal': self._normalize_obs(self._goal)}, rew, done, {}

    def _is_done(self, obs, goal):
        return np.linalg.norm(obs - goal) < self._threshold_dist

    def _sample_goal(self, obs):
        if np.random.random() < self._prob_constraint:
            return self._sample_goal_constrained(obs, self._min_dist, self._max_dist)
        else:
            return self._sample_goal_unconstrained(obs)

    def set_sample_goal_args(self, prob_constraint=None,
                                                            min_dist=None, max_dist=None):
        assert prob_constraint is not None
        assert min_dist is not None
        assert max_dist is not None
        assert min_dist >= 0
        assert max_dist >= min_dist
        self._prob_constraint = prob_constraint
        self._min_dist = min_dist
        self._max_dist = max_dist

    # obs是一个格子里任何一点，找到离这个格子一定距离的另一个格子，
    # 在另一个格子里找一点。
    def _sample_goal_constrained(self, obs, min_dist, max_dist):
        (i, j) = self.env._discretize_state(obs)
        mask = np.logical_and(self.env._apsp[i, j] >= min_dist,
                                                        self.env._apsp[i, j] <= max_dist)
        mask = np.logical_and(mask, self.env._walls == 0)
        candidate_states = np.where(mask)
        num_candidate_states = len(candidate_states[0])
        if num_candidate_states == 0:
            return (obs, None)
        goal_index = np.random.choice(num_candidate_states)
        goal = np.array([candidate_states[0][goal_index],
                                            candidate_states[1][goal_index]], dtype=np.float)
        goal += np.random.uniform(size=2)
        # goal += np.array([0.5, 0.5]) # 这样取的目标，距离差会稳定一些
        dist_to_goal = self.env._get_distance(obs, goal)
        assert min_dist <= dist_to_goal <= max_dist
        assert not self.env._is_blocked(goal)
        return (obs, goal)

    def _sample_goal_unconstrained(self, obs):
        return (obs, self.env._sample_empty_state())

    @property
    def max_goal_dist(self):
        apsp = self.env._apsp
        return np.max(apsp[np.isfinite(apsp)])

class NonTerminatingTimeLimit(wrappers.PyEnvironmentBaseWrapper):
    def __init__(self, env, duration):
        super(NonTerminatingTimeLimit, self).__init__(env)
        self._duration = duration
        self._step_count = None

    def _reset(self):
        self._step_count = 0
        return self._env.reset()

    @property
    def duration(self):
        return self._duration

    def _step(self, action):
        if self._step_count is None:
            return self.reset()
        ts = self._env.step(action)
        self._step_count += 1
        if self._step_count >= self._duration or ts.is_last():
            self._step_count = None
        return ts

def env_load_fn(environment_name, max_episode_steps = None,
                                    resize_factor = 1, action_range = 1.0, action_noise=0.0,
                                    threshold_dist = 1.0, terminate_on_timeout = False):
    gym_env = PointEnv(walls=environment_name, resize_factor=resize_factor,
                                    action_range=action_range, action_noise=action_noise)
    gym_env = GoalConditionedPointWrapper(gym_env, threshold_dist=threshold_dist)
    env = gym_wrapper.GymWrapper(gym_env, discount=1.0, auto_reset=True)

    if max_episode_steps > 0:
        if terminate_on_timeout: # test 超时的那个ts改成last
            env = wrappers.TimeLimit(env, max_episode_steps)
        else: # train 超时不会把那个ts改成last
            env = NonTerminatingTimeLimit(env, max_episode_steps)

    # tf_env.pyenv.envs[0].gym    <GoalConditionedPointWrapper<PointEnv instance>>
    # tf_env.pyenv.envs                   [<__main__.NonTerminatingTimeLimit at 0x7f9329d18080>]
    # tf_env.pyenv                              <tf_agents.environments.batched_py_environment.BatchedPyEnvironment at 0x7f9329d180f0>
    return tf_py_environment.TFPyEnvironment(env)

def plot_walls(walls):
    (height, width) = walls.shape
    ax = plt.gca()
    for (i, j) in zip(*np.where(walls)):
        x = np.array([j, j+1]) / float(width)
        y0 = np.array([height-i-1, height-i-1]) / float(height)
        y1 = np.array([height-i, height-i]) / float(height)
        ax.fill_between(x, y0, y1, color='grey')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    set_ticks(ax, width, height)

def set_ticks(ax, width, height):
    ax.xaxis.set_major_locator(MultipleLocator(5/width))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(round(width * x))))
    ax.xaxis.set_minor_locator(MultipleLocator(1/width))
    ax.xaxis.set_minor_formatter(NullFormatter())
    
    ax.yaxis.set_major_locator(MultipleLocator(5/height))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(round(height * x))))
    ax.yaxis.set_minor_locator(MultipleLocator(1/height))
    ax.yaxis.set_minor_formatter(NullFormatter())

def plot_envs():
    plt.figure(figsize=(12, 7))
    for index, (name, walls) in enumerate(WALLS.items()):
        plt.subplot(3, 6, index + 1)
        plt.title(name)
        plot_walls(walls)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.suptitle('Navigation Environments', fontsize=20)
    plt.show()

if __name__ == '__main__' :
    plot_walls(WALLS['FourRooms'])
    plt.show()