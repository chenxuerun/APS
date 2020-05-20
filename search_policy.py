from tf_agents.policies import tf_policy
from plan_util import fill_map
from util import outer_dist
import tensorflow as tf
import networkx as nx
import numpy as np
import scipy
import math

class SearchPolicy(tf_policy.Base):
    def __init__(self, agent, eval_tf_env, model_path=None, num_points=1000,
                                open_loop=False, edge_len_threshold=5.0, threshold_dist=3.0):
        self._agent = agent
        self._eval_tf_env = eval_tf_env
        self._open_loop = open_loop
        self._combine_ensemble_method = 'min'
        # 小于值认为到达waypoint，如果太小，由于Q中心很大，会导致waypoint不变
        self._threshold_dist = threshold_dist

        self._train_iter = tf.Variable(0, dtype=tf.int32)
        self._edge_len_threshold = tf.Variable(edge_len_threshold, dtype=tf.float32)
        self._num_points = tf.Variable(num_points, dtype=tf.float32)
        
        self._standard_learn_param()
        self._learn = True

        mix_dict = self.model_variable
        self.load_model(model_path, mix_dict)

        points = fill_map(eval_tf_env, int(self._num_points.numpy()))
        self._change_active_set(points)
        super(SearchPolicy, self).__init__(
            agent.policy.time_step_spec, agent.policy.action_spec)

    def _standard_learn_param(self):
        self._rho = 2 # 模式搜索乘数
        self._delta_edge_len_threshold = 0.1
        self._k1 = 1
        self._c1 = 4
        self._n1 = 3
        self._i = 3.0
        self._k2 = 1
        self._c2 = 4
        self._n2 = 3
        self._d = 1.0
        self._tth = 0.1
        self._gamma = 0.9
        self._cth = 0.05

    def _learn_param(self):
        self._rou = 2 # 模式搜索乘数
        self._delta_edge_len_threshold = 0.1
        self._k1 = 1
        self._ps1 = 0
        self._delta_num_points_increase = 10.0
        self._k2 = 1
        self._ps2 = 0
        self._delta_num_points_decrease = 0.1
        self._end_threshold = 0.05
        self._gamma = 1.0
        self._change_threshold = 0.05

    def load_model(self, path, model_variable):
        self._manager = None
        if path:
            self._ckpt = tf.train.Checkpoint(**model_variable)
            self._manager = tf.train.CheckpointManager(self._ckpt, path, max_to_keep=1)
            self._ckpt.restore(self._manager.latest_checkpoint)
            if self._manager.latest_checkpoint:
                print("load sp,  {}.".format(self._manager.latest_checkpoint))

    def save_model(self):
        if self._manager:
            save_path = self._manager.save()
            print("save sp,  iter {}: {}".format(int(self._ckpt.train_iter), save_path))
    
    @property
    def model_variable(self):
        mix_dict = {}
        mix_dict['train_iter'] = self._train_iter
        mix_dict['edge_len_threshold'] = self._edge_len_threshold
        mix_dict['num_points'] = self._num_points
        return mix_dict

    def _build_graph(self):
        g = nx.DiGraph()
        for i, s_i in enumerate(self._active_set):
            for j, s_j in enumerate(self._active_set):
                length = self._pairwise_dist[i, j]
                if length < self._edge_len_threshold:
                    g.add_edge(i, j, weight=length)
        self._graph = g
    
    def _change_active_set(self, points):
        pairwise_dist = self._agent._get_pairwise_dist(
            points,  aggregate=self._combine_ensemble_method)
        pairwise_dist = self._mask(pairwise_dist)
        way_dist = scipy.sparse.csgraph.floyd_warshall(pairwise_dist, directed=True)
        way_dist = way_dist.astype(np.float32)

        self._active_set = points
        self._pairwise_dist = pairwise_dist
        self._way_dist = way_dist
        self._build_graph()

    def _mask(self, pairwise_dist):
        mask = (pairwise_dist > self._edge_len_threshold)
        return tf.compat.v1.where(mask, tf.fill(pairwise_dist.shape, np.inf), pairwise_dist)

    def _get_waypoint(self, time_step):
        obs_tensor = time_step.observation['observation']
        goal_tensor = time_step.observation['goal']
        obs_to_active_set_dist = self._agent._get_pairwise_dist(
            obs_tensor, self._active_set, aggregate=self._combine_ensemble_method) # B x A
        obs_to_active_set_dist = self._mask(obs_to_active_set_dist)
        active_set_to_goal_dist = self._agent._get_pairwise_dist(
            self._active_set, goal_tensor, aggregate=self._combine_ensemble_method) # A x B
        active_set_to_goal_dist = self._mask(active_set_to_goal_dist)

        # The search_dist tensor should be (B x A x A)
        search_dist = sum([
            tf.expand_dims(obs_to_active_set_dist, 2),
            tf.expand_dims(self._way_dist, 0),
            tf.expand_dims(tf.transpose(a=active_set_to_goal_dist), axis=1)])

        # We assume a batch size of 1.
        assert obs_tensor.shape[0] == 1
        min_search_dist = tf.reduce_min(input_tensor=search_dist, axis=[1, 2])[0]
        waypoint_index = tf.argmin(
            input=tf.reduce_min(input_tensor=search_dist, axis=[2]), axis=1)[0]
        waypoint = self._active_set[waypoint_index]

        return waypoint, min_search_dist
    
    def _action(self, time_step, policy_state=(), seed=None):
        goal = time_step.observation['goal']
        dist_to_goal = self._agent._get_dist_to_goal(time_step)[0].numpy()

        if self._open_loop:
            if time_step.is_first():
                self._waypoint_indexes, self._waypoint_to_goal_dists = self._get_path(time_step)
                self._waypoint_counter = 0
            waypoint = self._active_set[self._waypoint_indexes[self._waypoint_counter]]
            time_step.observation['goal'] = waypoint[None]

            dist_to_waypoint = self._agent._get_dist_to_goal(time_step)[0].numpy()

            # trick: 弥补分布式rl的值函数缺陷
            another_dist = outer_dist(self._eval_tf_env, time_step)

            if (dist_to_waypoint < self._threshold_dist or 
                            another_dist < self._eval_tf_env.pyenv.envs[0].gym._threshold_dist):
                self._waypoint_counter = min(self._waypoint_counter + 1,
                                                                                len(self._waypoint_indexes) - 1)
                waypoint = self._active_set[self._waypoint_indexes[self._waypoint_counter]]
                time_step.observation['goal'] = waypoint[None]
                dist_to_waypoint = self._agent._get_dist_to_goal(time_step)[0].numpy()

            dist_to_goal_via_wypt = dist_to_waypoint + \
                self._waypoint_to_goal_dists[self._waypoint_counter]

        else:
            waypoint, dist_to_goal_via_wypt = self._get_waypoint(time_step)
            dist_to_goal_via_wypt = dist_to_goal_via_wypt.numpy()

        if (dist_to_goal_via_wypt < dist_to_goal) or \
            (dist_to_goal > self._edge_len_threshold):
            time_step.observation['goal'] = tf.convert_to_tensor(waypoint[None])
        else:
            time_step.observation['goal'] = goal

        return self._agent.policy.action(time_step, policy_state, seed)

    def _get_path(self, time_step):
        start_to_active_set = self._agent._get_pairwise_dist(
            time_step.observation['observation'], self._active_set,
            aggregate='min').numpy().flatten()
        start_to_active_set = self._mask(start_to_active_set).numpy().flatten()
        active_set_to_goal = self._agent._get_pairwise_dist(
            self._active_set, time_step.observation['goal'],
            aggregate='min').numpy().flatten()
        active_set_to_goal = self._mask(active_set_to_goal).numpy().flatten()
        
        g2 = self._graph.copy()
        for i, (dist_from_start, dist_to_goal) in \
            enumerate(zip(start_to_active_set, active_set_to_goal)):
            if (dist_from_start < self._edge_len_threshold):
                g2.add_edge('start', i, weight=dist_from_start)
            if (dist_to_goal < self._edge_len_threshold):
                g2.add_edge(i, 'goal', weight=dist_to_goal)
        path = nx.shortest_path(g2, 'start', 'goal')
        edge_lengths = []
        for (i, j) in zip(path[:-1], path[1:]):
            edge_lengths.append(g2[i][j]['weight'])
        wypt_to_goal_dist = np.cumsum(edge_lengths[::-1])[::-1]
        waypoint_index = list(path)[1:-1]
        return waypoint_index, wypt_to_goal_dist[1:] # 去掉start, 返回的是waypoint数组下标