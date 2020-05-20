import tensorflow as tf
from tf_agents.agents import tf_agent
from actor_critic import GoalConditionedActorNetwork
from actor_critic import GoalConditionedCriticNetwork
from tf_agents.utils import common
from tf_agents.policies import actor_policy
from tf_agents.policies import ou_noise_policy
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step
import networkx as nx
import numpy as np

class UvfAgent(tf_agent.TFAgent):
    def __init__(self, time_step_spec, action_spec,
                                load_model_path=None, save_model_path=None,
                                ou_stddev=0.0, ou_damping=1.0,
                                target_update_tau=0.05, target_update_period=5,
                                max_episode_steps=None,
                                ensemble_size=3, combine_ensemble_method='min',
                                distance_type='distributional'):
        tf.Module.__init__(self, name='UvfAgent')
        assert max_episode_steps is not None
        self._max_episode_steps = max_episode_steps
        self._ensemble_size = ensemble_size
        self._distance_type = distance_type

        self._actor_network = GoalConditionedActorNetwork(
            time_step_spec.observation, action_spec)
        self._target_actor_network = self._actor_network.copy(
            name='TargetActorNetwork')

        critic_net_input_specs = (time_step_spec.observation, action_spec)
        critic_network = GoalConditionedCriticNetwork(critic_net_input_specs,
            output_dim=max_episode_steps if distance_type=='distributional' else None)
        self._critic_network_list = []
        self._target_critic_network_list = []
        for ensemble_index in range(self._ensemble_size):
            self._critic_network_list.append(
                critic_network.copy(name='CriticNetwork%d' % ensemble_index))
            self._target_critic_network_list.append(
                critic_network.copy(name='TargetCriticNetwork%d' % ensemble_index))

        net_list = [
            self._actor_network, self._target_actor_network
            ] + self._critic_network_list + self._target_critic_network_list
        for net in net_list:
            net.create_variables()

        self._actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)
        self._critic_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)

        self._train_iter = tf.Variable(0)
        mix_dict = self.model_variable
        self.load_model(load_model_path, save_model_path, mix_dict)

        self._ou_stddev = ou_stddev
        self._ou_damping = ou_damping
        self._target_update_tau = target_update_tau
        self._target_update_period = target_update_period

        self._update_target = self._get_target_updater(
            target_update_tau, target_update_period)

        policy = actor_policy.ActorPolicy(
            time_step_spec=time_step_spec, action_spec=action_spec,
            actor_network=self._actor_network, clip=True)
        collect_policy = actor_policy.ActorPolicy(
            time_step_spec=time_step_spec, action_spec=action_spec,
            actor_network=self._actor_network, clip=False)
        # noise x = (1-damping)*x + N(0,std)
        collect_policy = ou_noise_policy.OUNoisePolicy(
            collect_policy, ou_stddev=self._ou_stddev,
            ou_damping=self._ou_damping, clip=True)
        
        super(UvfAgent, self).__init__(time_step_spec,
            action_spec, policy, collect_policy, train_sequence_length=2)

    def iter_plus(self):
        self._train_iter.assign_add(1)

    @property
    def model_variable(self):
        mix_dict = {}
        mix_dict['train_iter'] = self._train_iter
        mix_dict['actor_optimizer'] = self._actor_optimizer
        mix_dict['critic_optimizer'] = self._critic_optimizer
        mix_dict['actor_network'] = self._actor_network
        mix_dict['target_actor_network'] = self._target_actor_network
        for ensemble_index in range(self._ensemble_size):
            mix_dict[
                'critic_network%d' % ensemble_index
                ] = self._critic_network_list[ensemble_index]
            mix_dict[
                'target_critic_network%d' % ensemble_index
                ] = self._target_critic_network_list[ensemble_index]
        return mix_dict

    def load_model(self, load_model_path, save_model_path, model_variable):
        if load_model_path or save_model_path:
            self._ckpt = tf.train.Checkpoint(**model_variable)
            if load_model_path:
                self._load_manager = tf.train.CheckpointManager(
                    self._ckpt, load_model_path, max_to_keep=1)
                self._ckpt.restore(self._load_manager.latest_checkpoint)
                if self._load_manager.latest_checkpoint:
                    print("load agent,  {}.".format(self._load_manager.latest_checkpoint))
                else:
                    print("Initializing from scratch.")
            if save_model_path:
                self._save_manager = tf.train.CheckpointManager(
                    self._ckpt, save_model_path, max_to_keep=1)

    def save_model(self):
        if self._save_manager:
            save_path = self._save_manager.save()
            # train_iter是checkpoint对象初始化时传进去的dict参数，是历史训练总数
            print("save agent,  iter {}: {}".format(int(self._ckpt.train_iter), save_path))

    def _get_target_updater(self, tau=1.0, period=1):
        with tf.compat.v1.name_scope('get_target_updater'):
            def update():
                critic_update_list = []
                for ensemble_index in range(self._ensemble_size):
                    critic_update = common.soft_variables_update(
                        self._critic_network_list[ensemble_index].variables,
                        self._target_critic_network_list[ensemble_index].variables, tau)
                    critic_update_list.append(critic_update)
                actor_update = common.soft_variables_update(
                    self._actor_network.variables,
                    self._target_actor_network.variables, tau)
                return tf.group(critic_update_list + [actor_update])
            return common.Periodically(update, period, 'periodic_update_targets')

    def _train(self, experience, weights=None):
        del weights
        time_steps, actions, next_time_steps = self._experience_to_transitions(experience)

        critic_vars = []
        for ensemble_index in range(self._ensemble_size):
            critic_net = self._critic_network_list[ensemble_index]
            critic_vars.extend(critic_net.variables)
        
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert critic_vars
            tape.watch(critic_vars)
            critic_loss = self.critic_loss(time_steps, actions, next_time_steps)
        tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan')
        critic_grads = tape.gradient(critic_loss, critic_vars)
        self._apply_gradients(critic_grads, critic_vars, self._critic_optimizer)

        actor_vars = self._actor_network.variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert actor_vars, 'No actor variables to optimize.'
            tape.watch(actor_vars)
            actor_loss = self.actor_loss(time_steps)
        tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
        actor_grads = tape.gradient(actor_loss, actor_vars)
        self._apply_gradients(actor_grads, actor_vars, self._actor_optimizer)

        self.train_step_counter.assign_add(1)
        self._update_target()
        total_loss = actor_loss + critic_loss
        return tf_agent.LossInfo(total_loss, (actor_loss, critic_loss))

    def actor_loss(self, time_steps):
        with tf.compat.v1.name_scope('actor_loss'):
            actions, _ = self._actor_network(time_steps.observation, time_steps.step_type)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(actions)
                avg_expected_q_values = self._get_state_values(
                    time_steps, actions, aggregate='mean')
                actions = tf.nest.flatten(actions)
            dqdas = tape.gradient([avg_expected_q_values], actions)
            
            actor_losses = []
            for dqda, action in zip(dqdas, actions): # 其实就只有一个
                loss = common.element_wise_squared_loss(
                    tf.stop_gradient(dqda + action), action)
                loss = tf.reduce_sum(input_tensor=loss, axis=1)
                loss = tf.reduce_mean(input_tensor=loss)
                actor_losses.append(loss)
            
            actor_loss = tf.add_n(actor_losses)

            with tf.compat.v1.name_scope('Losses/'):
                tf.compat.v2.summary.scalar(
                    name='actor_loss', data=actor_loss, step=self.train_step_counter)
        
        return actor_loss

    def _get_state_values(self, time_steps, actions=None, aggregate='mean'):
        # 集合所有网络，用阈值限制
        with tf.compat.v1.name_scope('state_values'):
            expected_q_values = self._get_expected_q_values(time_steps, actions)
            if aggregate is not None:
                if aggregate == 'mean':
                    expected_q_values = tf.reduce_mean(expected_q_values, axis=0)
                elif aggregate == 'min':
                    expected_q_values = tf.reduce_min(expected_q_values, axis=0)
                else:
                    raise ValueError('Unknown method for combining ensemble: %s' % aggregate)
            
            if self._distance_type != 'distributional':
                min_q_val = -1.0 * self._max_episode_steps
                max_q_val = 0.0
                expected_q_values = tf.maximum(expected_q_values, min_q_val)
                expected_q_values = tf.minimum(expected_q_values, max_q_val)
            
            return expected_q_values

    def _get_expected_q_values(self, time_steps, actions=None):
        # 对集群输出求期望
        if actions is None:
            actions, _ = self._actor_network(
                time_steps.observation, time_steps.step_type)
        q_values_list = self._get_critic_output(
            self._critic_network_list, time_steps, actions)
        expected_q_values_list = []
        for q_values in q_values_list:
            if self._distance_type == 'distributional':
                # 概率和距离加权，得到平均距离，输出为负数
                q_probs = tf.nn.softmax(q_values, axis=1)
                batch_size = q_probs.shape[0]
                bin_range = tf.range(1, self._max_episode_steps + 1, dtype=tf.float32)
                neg_bin_range = -1.0 * bin_range
                tiled_bin_range = tf.tile(tf.expand_dims(neg_bin_range, 0), [batch_size, 1])
                assert q_probs.shape == tiled_bin_range.shape
                expected_q_values = tf.reduce_sum(q_probs * tiled_bin_range, axis=1)
                expected_q_values_list.append(expected_q_values)
            elif self._distance_type == 'original':
                expected_q_values_list.append(q_values)
            elif self._distance_type == 'sigmoid':
                # 原本输出的是logits
                q_values = -1 - tf.exp(q_values)
                expected_q_values_list.append(q_values)

        return tf.stack(expected_q_values_list)

    def _apply_gradients(self, gradients, variables, optimizer):
        grads_and_vars = tuple(zip(gradients, variables))
        optimizer.apply_gradients(grads_and_vars)
    
    def critic_loss(self, time_steps, actions, next_time_steps):
        with tf.compat.v1.name_scope('critic_loss'):
            target_actions, _ = self._target_actor_network(
                next_time_steps.observation, next_time_steps.step_type)
            batch_size = target_actions.shape[0]

            critic_loss_list = []
            q_values_list = self._get_critic_output(
                self._critic_network_list, time_steps, actions)
            target_q_values_list = self._get_critic_output(
                self._target_critic_network_list, next_time_steps, target_actions)
            assert len(target_q_values_list) == self._ensemble_size

            for ensemble_index in range(self._ensemble_size):
                target_q_values = target_q_values_list[ensemble_index]

                if self._distance_type == 'distributional':
                    target_q_probs = tf.nn.softmax(target_q_values, axis=1)
                    one_hot = tf.one_hot(
                        tf.zeros(batch_size, dtype=tf.int32), self._max_episode_steps)
                    col_1 = tf.zeros((batch_size, 1))
                    col_middle = target_q_probs[:, :-2]
                    col_last = tf.reduce_sum(
                        input_tensor=target_q_probs[:, -2:], axis=1, keepdims=True)
                    shifted_target_q_probs = tf.concat([col_1, col_middle, col_last], axis=1)
                    assert one_hot.shape == shifted_target_q_probs.shape
                    td_targets = tf.compat.v1.where(
                        next_time_steps.is_last(), one_hot, shifted_target_q_probs)
                    td_targets = tf.stop_gradient(td_targets)
                elif self._distance_type == 'sigmoid':
                    ones = tf.ones(shape=(batch_size,), dtype=tf.float32)
                    sigmoids = tf.nn.sigmoid(target_q_values)
                    original_td_targets = sigmoids / (sigmoids + 1)
                    assert ones.shape == original_td_targets.shape
                    td_targets = tf.compat.v1.where(
                        next_time_steps.is_last(), ones, original_td_targets)
                    td_targets = tf.stop_gradient(td_targets)
                elif self._distance_type == 'original':
                    ones = - tf.ones(shape=(batch_size,))
                    original_td_targets = (next_time_steps.reward + 
                        next_time_steps.discount * target_q_values)
                    # 我的新修改
                    assert ones.shape == original_td_targets.shape
                    td_targets = tf.compat.v1.where(
                        next_time_steps.is_last(), ones, original_td_targets)
                    td_targets = tf.stop_gradient(td_targets)

                q_values = q_values_list[ensemble_index]

                if self._distance_type == 'distributional':
                    critic_loss = tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf.stop_gradient(td_targets), logits=q_values)
                elif self._distance_type == 'sigmoid':
                    critic_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.stop_gradient(td_targets), logits=q_values)
                elif self._distance_type == 'original':
                    critic_loss = common.element_wise_huber_loss(td_targets, q_values)

                critic_loss = tf.reduce_mean(input_tensor=critic_loss)
                critic_loss_list.append(critic_loss)

            critic_loss = tf.reduce_mean(input_tensor=critic_loss_list)
            return critic_loss

    def _get_critic_output(self, critic_net_list, time_steps, actions=None):
        # 基本函数，获得列表输出
        q_values_list = []
        critic_net_input = (time_steps.observation, actions)
        for critic_index in range(self._ensemble_size):
            critic_net = critic_net_list[critic_index]
            q_values, _ = critic_net(critic_net_input, time_steps.step_type)
            q_values_list.append(q_values)
        return q_values_list

    def _experience_to_transitions(self, experience):
        transitions = trajectory.to_transition(experience) # [s a s'] 1分3
        transitions = tf.nest.map_structure(lambda x: tf.squeeze(x, [1]), transitions)
        time_steps, policy_steps, next_time_steps = transitions
        actions = policy_steps.action
        return time_steps, actions, next_time_steps

    def _get_dist_to_goal(self, time_steps, aggregate='mean'):
        # 到目标状态的距离 一列
        q_values = self._get_state_values(time_steps, aggregate=aggregate)
        return -1.0 * q_values

    def _get_pairwise_dist(self, obs_tensor, goal_tensor=None, aggregate='mean'):
        assert aggregate is not None, 'aggregate is None'
        if goal_tensor is None:
            goal_tensor = obs_tensor
        dist_matrix = []
        for obs_index in range(obs_tensor.shape[0]):
            obs = obs_tensor[obs_index]
            obs_repeat_tensor = tf.ones_like(goal_tensor) * tf.expand_dims(obs, 0)
            obs_goal_tensor = {'observation': obs_repeat_tensor, 'goal': goal_tensor}
            time_steps = time_step.transition(obs_goal_tensor, reward=0, discount=1.0)
            dist = self._get_dist_to_goal(time_steps, aggregate=aggregate)
            dist_matrix.append(dist)
        pairwise_dist = tf.stack(dist_matrix)
        # if aggregate is None:
        #     pairwise_dist = tf.transpose(a=pairwise_dist, perm=[1, 0, 2])

        return pairwise_dist