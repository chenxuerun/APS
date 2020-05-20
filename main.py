import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
# import pretty_errors
from env import env_load_fn
from agent import UvfAgent
from util import q_image
import train
import plan_util
import search_policy
import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
tf.compat.v1.enable_v2_behavior()
tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

resize_factor = 5                                 # 环境放大
ou_stddev = 1.0                                  # agent探索行为，标准差，外部距离
ou_damping = 1.0                             # agent探索行为
action_range = 1.0                             # 上下左右活动范围，外部距离
action_noise = 0.3                             # 环境噪声，标准差，外部距离
threshold_dist = 1.0                          # agent到达目标阈值，外部距离
edge_len_threshold = 1.0               # agent建边阈值，内部距离
num_points = 1.0                               # agent保存的节点数

path = './ckpts'
# env_name = 'Maze11x11'
env_name = 'FourRooms'
distance_type = 'distributional'
max_episode_steps = 20
# load_model_path = os.path.join(path, env_name,  'test')
load_model_path = os.path.join(
    path, env_name, distance_type+'-'+str(max_episode_steps), 'noise-'+str(action_noise))
save_model_path = os.path.join(
    path, env_name, distance_type+'-'+str(max_episode_steps), 'noise-'+str(action_noise))
# save_model_path = os.path.join(path, env_name,  'test')
# plan_model_path = os.path.join(path, env_name, 'plan')
plan_model_path = None

tf_env = env_load_fn(environment_name=env_name, resize_factor=resize_factor,
    max_episode_steps=max_episode_steps, terminate_on_timeout=False,
    action_range=action_range, action_noise=action_noise, threshold_dist=threshold_dist)
eval_tf_env = env_load_fn(environment_name=env_name, resize_factor=resize_factor,
    max_episode_steps=max_episode_steps, terminate_on_timeout=True,
    action_range=action_range, action_noise=action_noise, threshold_dist=threshold_dist)

agent = UvfAgent(tf_env.time_step_spec(), tf_env.action_spec(),
    load_model_path=load_model_path, save_model_path=save_model_path,
    ou_stddev=ou_stddev, ou_damping=ou_damping,
    max_episode_steps=max_episode_steps,
    distance_type=distance_type, ensemble_size=3)

train.eval(eval_tf_env, agent)

# tf_env.pyenv.envs[0]._duration = 300
# eval_tf_env.pyenv.envs[0]._duration = 300

# train.train_eval(agent, tf_env, eval_tf_env, initial_collect_steps=1000,
#     eval_interval=10000, num_eval_episodes=10, num_iterations=20000)

# plan_util.plot_rollouts(eval_tf_env, agent)

# sp = search_policy.SearchPolicy(agent, eval_tf_env,
#     model_path=plan_model_path, num_points=num_points, open_loop=True,
#     edge_len_threshold=edge_len_threshold, threshold_dist=2.0)

# train.train_sp(sp, eval_tf_env, 500)

# train.eval_sp(sp, eval_tf_env, num_map=40, num_task=5)

# plan_util.rollout_with_search(eval_tf_env, sp)