import numpy as np
import matplotlib.pyplot as plt
from env import plot_walls
import time

render = False

def prepare_rollout(eval_tf_env):
    eval_tf_env.pyenv.envs[0]._duration = 100
    difficulty = 0.4
    max_goal_dist = eval_tf_env.pyenv.envs[0].gym.max_goal_dist
    eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
        prob_constraint=1.0,
        min_dist=max(0, max_goal_dist * (difficulty - 0.05)),
        max_dist=max_goal_dist * (difficulty + 0.05))

def plot_rollouts(eval_tf_env, agent):
    plt.figure(figsize=(6, 6))
    for col_index in range(1):
        plt.subplot(1, 1, col_index + 1)
        plot_walls(eval_tf_env.pyenv.envs[0].env.walls)
        obs_vec, goal, _ = get_rollout(eval_tf_env, agent.policy)
        obs_vec = change_axis(obs_vec)
        goal = change_axis(goal)

        plt.plot(obs_vec[:, 0], obs_vec[:, 1], 'b-o', alpha=0.3)
        plt.scatter([obs_vec[0, 0]], [obs_vec[0, 1]], marker='+',
                color='red', s=200, label='start')
        plt.scatter([obs_vec[-1, 0]], [obs_vec[-1, 1]], marker='+',
                color='green', s=200, label='end')
        plt.scatter([goal[0]], [goal[1]], marker='*',
                color='green', s=200, label='goal')
        if col_index == 0:
            plt.legend(loc='lower left', bbox_to_anchor=(0.0, 1), ncol=3, fontsize=16)
    plt.show()

def get_rollout(tf_env, policy, seed=None):
    np.random.seed(seed)
    obs_vec = []
    waypoint_vec = []
    ts = tf_env.reset()
    goal = ts.observation['goal'].numpy()[0]
    for _ in range(tf_env.pyenv.envs[0]._duration):
        obs_vec.append(ts.observation['observation'].numpy()[0])
        action = policy.action(ts)
        waypoint_vec.append(ts.observation['goal'].numpy()[0]) # 暂时不用
        ts = tf_env.step(action)
        if ts.is_last():
            break
    obs_vec.append(ts.observation['observation'].numpy()[0])
    obs_vec = np.array(obs_vec)
    waypoint_vec = np.array(waypoint_vec)
    return obs_vec, goal, waypoint_vec

def fill_map(eval_tf_env, amount):
    eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
        prob_constraint=0.0, min_dist=0, max_dist=np.inf)
    points = []
    for _ in range(amount):
        ts = eval_tf_env.reset()
        points.append(ts.observation['observation'].numpy()[0])
    points = np.array(points)
    return points

def plot_map(points, eval_tf_env):
    points = change_axis(points)
    plt.figure(figsize=(6, 6))
    plot_walls(eval_tf_env.pyenv.envs[0].env.walls)
    plt.scatter(*points.T)
    plt.show()

def change_axis(points):
    changed_points = np.zeros_like(points)
    if len(points.shape) == 1:
        changed_points[1] = 1 - points[0]
        changed_points[0] = points[1]
    else:
        changed_points[:, 0] = points[:, 1]
        changed_points[:, 1] = 1 - points[:, 0]
    return changed_points

def plot_pd(pdist):
    plt.figure(figsize=(6, 3))
    pdist[pdist>20] = 20
    plt.hist(pdist.flatten(), bins=range(21))
    plt.xlabel('predicted distance')
    plt.ylabel('number of (s, g) pairs')
    plt.show()

def plot_construct_graph(eval_tf_env, pdist, points):
    points = change_axis(points)
    cutoff = 5
    edges_to_display = 8
    plt.figure(figsize=(6, 6))
    plot_walls(eval_tf_env.pyenv.envs[0].env.walls)
    plt.scatter(*points.T)
    for i, s_i in enumerate(points):
        for count, j in enumerate(np.argsort(pdist[i])):
            if count < edges_to_display and pdist[i, j] < cutoff:
                s_j = points[j]
                plt.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c='k', alpha=0.5)
    plt.show()

def compute_pairwise_distance(agent, points):
    pairwise_distance = agent._get_pairwise_dist(points, aggregate='min').numpy()
    return pairwise_distance

def plot_search_path(eval_tf_env, search_policy):
    difficulty = 0.4 #@param {min:0, max: 1, step: 0.1, type:"slider"}
    max_goal_dist = eval_tf_env.pyenv.envs[0].gym.max_goal_dist
    eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
        prob_constraint=1.0,
        min_dist=max(0, max_goal_dist * (difficulty - 0.05)),
        max_dist=max_goal_dist * (difficulty + 0.05))
    ts = eval_tf_env.reset()
    start = ts.observation['observation'].numpy()[0]
    goal = ts.observation['goal'].numpy()[0]
    search_policy.action(ts)

    plt.figure(figsize=(6, 6))
    plot_walls(eval_tf_env.pyenv.envs[0].env.walls)

    waypoint_vec = [start]
    for waypoint_index in search_policy._waypoint_indexes:
        waypoint_vec.append(search_policy._active_set[waypoint_index])
    waypoint_vec.append(goal)
    waypoint_vec = np.array(waypoint_vec)

    plt.scatter([start[0]], [start[1]], marker='+',
                color='red', s=200, label='start')
    plt.scatter([goal[0]], [goal[1]], marker='*',
                color='green', s=200, label='goal')
    plt.plot(waypoint_vec[:, 0], waypoint_vec[:, 1], 'y-s', alpha=0.3, label='waypoint')
    plt.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.15), ncol=4, fontsize=16)
    plt.show()

def rollout_once(eval_tf_env, search_policy):
    ts = eval_tf_env.reset()
    goal = ts.observation['goal'].numpy()[0]
    start = ts.observation['observation'].numpy()[0]
    if render:
        obs_vec = [] #
    for _ in range(eval_tf_env.pyenv.envs[0]._duration):
        if ts.is_last():
            return 'success'
        if render:
            obs_vec.append(ts.observation['observation'].numpy()[0]) #
        try:
            action = search_policy.action(ts)
        except:
            # raise
            return 'no way'
        ts = eval_tf_env.step(action)
    #
    if render:
        obs_vec = np.array(obs_vec)
        obs_vec = change_axis(obs_vec)
        changed_goal = change_axis(goal)
        plt.figure(figsize=(6,6))
        plot_walls(eval_tf_env.pyenv.envs[0].env.walls)
        plt.plot(obs_vec[:, 0], obs_vec[:, 1], 'b-o', alpha=0.3)
        plt.scatter([obs_vec[0, 0]], [obs_vec[0, 1]], marker='+',
                    color='red', s=200, label='start')
        plt.scatter([obs_vec[-1, 0]], [obs_vec[-1, 1]], marker='+',
                    color='green', s=200, label='end')
        plt.scatter([changed_goal[0]], [changed_goal[1]], marker='*',
                    color='green', s=200, label='goal')

        waypoint_vec = [start]
        for waypoint_index in search_policy._waypoint_indexes:
            waypoint_vec.append(search_policy._active_set[waypoint_index])
        waypoint_vec.append(goal)
        waypoint_vec = np.array(waypoint_vec)
        waypoint_vec = change_axis(waypoint_vec)

        plt.plot(waypoint_vec[:, 0], waypoint_vec[:, 1], 'y-s', alpha=0.3, label='waypoint')
        plt.legend(loc='lower center', bbox_to_anchor=(0, -0.2), ncol=2, fontsize=16)
        plt.show()
    #
    return 'cannot reach'

def rollout_with_search(eval_tf_env, search_policy):
    eval_tf_env.pyenv.envs[0]._duration = 300
    seed = np.random.randint(0, 1000000)
    difficulty = 0.9
    max_goal_dist = eval_tf_env.pyenv.envs[0].gym.max_goal_dist
    eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
        prob_constraint=1.0, min_dist=max(0, max_goal_dist * (difficulty - 0.05)),
        max_dist=max_goal_dist * (difficulty + 0.05))
    
    plt.figure(figsize=(12, 5))

    for col_index in range(2):
        use_search = (col_index == 1)
        np.random.seed(seed)
        ts = eval_tf_env.reset()
        goal = ts.observation['goal'].numpy()[0]
        start = ts.observation['observation'].numpy()[0]
        obs_vec = []
        for _ in range(eval_tf_env.pyenv.envs[0]._duration):
            if ts.is_last():
                break
            obs_vec.append(ts.observation['observation'].numpy()[0])
            if use_search:
                action = search_policy.action(ts)
            else:
                action = search_policy._agent.policy.action(ts)
            ts = eval_tf_env.step(action)
        obs_vec = np.array(obs_vec)
        obs_vec = change_axis(obs_vec)
        changed_goal = change_axis(goal)

        title = 'no search' if col_index == 0 else 'search'
        plt.subplot(1, 2, col_index + 1)
        plot_walls(eval_tf_env.pyenv.envs[0].env.walls)
        plt.plot(obs_vec[:, 0], obs_vec[: ,1], 'b-o', alpha=0.3)
        plt.scatter([obs_vec[0, 0]], [obs_vec[0, 1]],
            marker='+', color='red', s=200, label='start')
        plt.scatter([obs_vec[-1, 0]], [obs_vec[-1, 1]],
            marker='+', color='green', s=200, label='end')
        plt.scatter([changed_goal[0]], [changed_goal[1]],
            marker='*', color='green', s=200, label='goal')

        plt.title(title, fontsize=24)
        if use_search and search_policy._open_loop==True:
            waypoint_vec = [start]
            for waypoint_index in search_policy._waypoint_indexes:
                waypoint_vec.append(search_policy._active_set[waypoint_index])
            waypoint_vec.append(goal)
            waypoint_vec = np.array(waypoint_vec)
            waypoint_vec = change_axis(waypoint_vec)

            plt.plot(waypoint_vec[:, 0], waypoint_vec[:, 1], 'y-s', alpha=0.3, label='waypoint')
            plt.legend(loc='lower left', bbox_to_anchor=(-0.8, -0.15), ncol=4, fontsize=16)
    plt.show()