import tensorflow as tf
import numpy as np
import random
import time
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import  tf_metrics
from tf_agents.drivers import dynamic_step_driver
from tf_agents.utils import common
from tf_agents.eval import metric_utils
from plan_util import rollout_once, fill_map

def train_eval(tf_agent, tf_env, eval_tf_env, num_iterations=2e6,
                                initial_collect_steps=1000, batch_size=64, 
                                num_eval_episodes=100, eval_interval=10000,
                                log_inverval=1000, random_seed=0):
    # tf.compat.v1.logging.info('random_seed = %d' % random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    tf.compat.v1.set_random_seed(random_seed)

    max_episode_steps = tf_env.pyenv.envs[0]._duration
    global_step = tf.compat.v1.train.get_or_create_global_step()
    # initial_collect_steps 正好就是buffer_size
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec, batch_size=tf_env.batch_size, 
        max_length=int(num_iterations/2)) # batch=1
    
    eval_metrics = [tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes)]

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env, collect_policy, 
        observers=[replay_buffer.add_batch], num_steps=initial_collect_steps)

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env, collect_policy,
        observers=[replay_buffer.add_batch], num_steps=1)

    initial_collect_driver.run = common.function(initial_collect_driver.run)
    collect_driver.run = common.function(collect_driver.run)
    tf_agent.train = common.function(tf_agent.train)

    initial_collect_driver.run()

    time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)

    dataset = replay_buffer.as_dataset(num_parallel_calls=3,
        sample_batch_size=batch_size, num_steps=2).prefetch(3)
    iterator = iter(dataset)
    
    for _ in range(num_iterations):
        time_step, policy_state = collect_driver.run(
            time_step=time_step, policy_state=policy_state)
        experience, _ = next(iterator)
        train_loss = tf_agent.train(experience)

        tf_agent.iter_plus()

        if global_step.numpy() % eval_interval == 0:
            tf_agent.save_model()
            tf.compat.v1.logging.info('step = %d' % global_step.numpy())

            for dist in [2, 5, 10]:
                tf.compat.v1.logging.info('\t dist = %d' % dist)
                eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
                    prob_constraint=1.0, min_dist=dist, max_dist=dist)
                results = metric_utils.eager_compute(eval_metrics, eval_tf_env, eval_policy,
                    num_episodes=num_eval_episodes,
                    train_step=global_step, summary_prefix='Metrics')
                for (key, value) in results.items():
                    tf.compat.v1.logging.info('\t\t %s = %.2f', key, value.numpy())
                pred_dist = []
                for _ in range(num_eval_episodes):
                    ts = eval_tf_env.reset()
                    dist_to_goal = tf_agent._get_dist_to_goal(ts)[0]
                    pred_dist.append(dist_to_goal.numpy())
                tf.compat.v1.logging.info('\t\t predicted_dist = %.1f (%.1f)' %
                        (np.mean(pred_dist), np.std(pred_dist)))
    return train_loss

def train_sp_once(search_policy, eval_tf_env, num_iterations=10):
    success = 0 # 如果一切顺利，希望尽量减少路径点
    no_way = 0 # 总是找不到路径，应该增大建边阈值，增加路径点
    cannot_reach = 0 # 找到路径但总到不了，应该减小建边阈值

    start = time.clock()
    for _ in range(num_iterations):
        result = rollout_once(eval_tf_env, search_policy) # 13, cpu8秒
        if result == 'success':
            success += 1
        elif result == 'no way':
            no_way += 1
        elif result == 'cannot reach':
            cannot_reach += 1
    rollout_time = time.clock() - start
    print('rollout_num: %d, average_time: %.1f' % (
        num_iterations, rollout_time/num_iterations))

    if search_policy._learn:
        no_problem = True

        if (no_way/num_iterations) > search_policy._cth:
            no_problem = False
            # 确定系数
            if search_policy._c1 > 0:
                search_policy._k1 *= search_policy._rho
            elif search_policy._c1 == 0:
                search_policy._k1 = 1
            if search_policy._c2 > 0:
                search_policy._k2 *= search_policy._rho
            elif search_policy._c2 ==0:
                search_policy._k2 =1

            search_policy._edge_len_threshold.assign_add(
                search_policy._k1 * search_policy._delta_edge_len_threshold)

            search_policy._num_points.assign_add(search_policy._k2 * search_policy._i)

            search_policy._c1 = 5
            search_policy._c2 = 5

            if search_policy._n2 == 0:
                search_policy._d *= search_policy._gamma # 用于终止训练
            print('no way: %.2f' % (no_way/num_iterations))

        if (cannot_reach/num_iterations) > search_policy._cth:
            no_problem = False
            if search_policy._n1 > 1 and search_policy._c1 == 1:
                search_policy._n1 -= 1
                search_policy._edge_len_threshold.assign_sub(
                    search_policy._k1 * search_policy._delta_edge_len_threshold)
            else:
                search_policy._edge_len_threshold.assign_sub(
                    search_policy._delta_edge_len_threshold)
            if search_policy._c1 > 0:
                search_policy._c1 -= 1
            print('cannot reach: %.2f' % (cannot_reach/num_iterations))

        if no_problem:
            if search_policy._n2 > 0 and search_policy._c2 == 1:
                search_policy._n2 -= 1
                search_policy._num_points.assign_sub(search_policy._k2 * search_policy._i)
            else:
                search_policy._num_points.assign_sub(search_policy._d)
            if search_policy._c1 > 0:
                search_policy._c1 -= 1
            if search_policy._c2 > 0:
                search_policy._c2 -= 1

        start = time.clock()
        points = fill_map(eval_tf_env, int(search_policy._num_points.numpy()))
        search_policy._change_active_set(points)
        rt = time.clock() - start
        search_policy.save_model()
        print('change_active_set_time: %.1f \n' % rt)
    
    return (search_policy._d < search_policy._tth) # 用于终止训练

def train_sp(search_policy, eval_tf_env, times=1):
    # 一次训练会训练很多轮，一轮训练会跑很多个episode并改一次参数

    eval_tf_env.pyenv.envs[0]._duration = 400
    max_goal_dist = eval_tf_env.pyenv.envs[0].gym.max_goal_dist
    eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
        prob_constraint=1.0, min_dist=max_goal_dist * 0.8, max_dist=max_goal_dist)

    points_nums = []
    edge_len_thresholds = []

    for _ in range(times):
        print('num_points: %.1f, edge_len_threshold: %.1f' \
            % (search_policy._num_points.numpy(),
                    search_policy._edge_len_threshold.numpy()))
        points_nums.append(search_policy._num_points.numpy())
        edge_len_thresholds.append(search_policy._edge_len_threshold.numpy())
        search_policy._train_iter.assign_add(1)
        end = train_sp_once(search_policy, eval_tf_env, num_iterations=20)
        if end:
            break

    points_nums.append(search_policy._num_points.numpy())
    edge_len_thresholds.append(search_policy._edge_len_threshold.numpy())
    points_nums = np.array(points_nums)
    edge_len_thresholds = np.array(edge_len_thresholds)
    c = np.stack([points_nums, edge_len_thresholds], axis=1)
    np.savetxt('plan_param.csv', c, fmt='%.1f', delimiter=',')

    print('sp training is over................')
    print('num_points: %.1f, edge_len_threshold: %.1f' \
            % (search_policy._num_points.numpy(),
                    search_policy._edge_len_threshold.numpy()))

def eval_sp(search_policy, eval_tf_env, num_map, num_task):
    eval_tf_env.pyenv.envs[0]._duration = 400
    max_goal_dist = eval_tf_env.pyenv.envs[0].gym.max_goal_dist
    eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
        prob_constraint=1.0, min_dist=max_goal_dist * 0.9, max_dist=max_goal_dist *0.9)

    s = time.clock()

    success = 0
    total_time = 0.0
    for _ in range(num_map):
        points = fill_map(eval_tf_env, int(search_policy._num_points.numpy()))
        search_policy._change_active_set(points)
        for _ in range(num_task):
            start = time.clock()
            result = rollout_once(eval_tf_env, search_policy)
            rt = time.clock() - start
            if result == 'success':
                success += 1
                total_time += rt

    print('num_points: %.1f, edge_len_th: %.1f, success_rate: %.2f, avg_time: %.1f' %\
        (search_policy._num_points.numpy(), search_policy._edge_len_threshold.numpy(),\
             success/(num_map * num_task), total_time/success))

    r = time.clock() - s
    print('rt: %d' % r)

def eval(eval_tf_env, tf_agent):
    global_step = tf.compat.v1.train.get_or_create_global_step()
    eval_policy = tf_agent.policy
    num_eval_episodes=100
    eval_metrics = [tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes)]
    for dist in [2, 5, 10]:
        tf.compat.v1.logging.info('\t dist = %d' % dist)
        eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
            prob_constraint=1.0, min_dist=dist, max_dist=dist)
        results = metric_utils.eager_compute(eval_metrics, eval_tf_env, eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step, summary_prefix='Metrics')
        for (key, value) in results.items():
            tf.compat.v1.logging.info('\t\t %s = %.2f', key, value.numpy())
        pred_dist = []
        for _ in range(num_eval_episodes):
            ts = eval_tf_env.reset()
            dist_to_goal = tf_agent._get_dist_to_goal(ts)[0]
            pred_dist.append(dist_to_goal.numpy())
        tf.compat.v1.logging.info('\t\t predicted_dist = %.1f (%.1f)' %
                (np.mean(pred_dist), np.std(pred_dist)))