import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math
from env import plot_walls
import matplotlib.ticker as ticker

def q_image(tf_env, agent, state): # state np.array
    state = np.array(state)
    env = tf_env.pyenv.envs[0].gym.env
    assert not env._is_blocked(state), '这个点被阻塞'
    fig, ax = plt.subplots(figsize=(7,6))
    
    (height, width) = env.walls.shape
    goal_states = []
    shifted_goal_states = []
    
    k=2 # 采样密集程度
    x=np.linspace(0, 1, num=k*width+1)
    y=np.linspace(0, 1, num=k*height+1)
    X, Y=np.meshgrid(x, y) # plt坐标
    sh_X=1-Y
    sh_Y=X
    sh_X=np.expand_dims(sh_X,axis=2)
    sh_Y=np.expand_dims(sh_Y,axis=2)
    goal_states=np.concatenate([sh_X,sh_Y],axis=2) # env 坐标
    shifted_goal_states = goal_states.reshape(-1,2)
    
    dists = agent._get_pairwise_dist(tf.convert_to_tensor(state[None]),
        shifted_goal_states, aggregate='min'
        ).numpy().reshape(goal_states.shape[0],goal_states.shape[1])

    cs = plt.contourf(X, Y, dists, 12, alpha=1, cmap='PuBu') # 登高线
    plot_walls(env.walls)
    fig.colorbar(cs, ax=ax)
    plt.show()

def outer_dist(eval_tf_env, time_step):
    height = eval_tf_env.pyenv.envs[0].gym.env._height
    width = eval_tf_env.pyenv.envs[0].gym.env._width
    g_0 = time_step.observation['goal'][0][0] * height
    g_1 = time_step.observation['goal'][0][1] * width
    s_0 = time_step.observation['observation'][0][0] * height
    s_1 = time_step.observation['observation'][0][1] * width
    dist = math.sqrt((s_0 - g_0)**2 + (s_1 - g_1)**2)
    return dist

def plot_plan_param(filename='plan_param.csv'):
    data = np.loadtxt(filename, dtype=np.float64, delimiter=',')
    plt.figure(figsize=(12,12))
    plt.subplot(2, 1, 1)    
    plt.plot(data[:,0])
    plt.tick_params(labelsize=18)
    plt.ylabel('number of waypoints',{'size':24})
    plt.subplot(2, 1, 2)
    plt.plot(data[:,1])
    plt.tick_params(labelsize=18)
    plt.ylabel('maximum edge length',{'size':24})
    plt.xlabel('training iteration',{'size':24})
    plt.show()

def plot_comparison_nw(filename='num_waypoints.csv'):
    data = np.loadtxt(filename, dtype=np.float64, delimiter=',')
    x=np.arange(len(data))
    fig = plt.figure(figsize=(12, 12))
    ax1 = plt.subplot(2, 1, 1)
    ax1.xaxis.set_major_locator(ticker.FixedLocator(data[:, 0]))
    plt.ylabel('success rate', {'size': 24})
    ax1.tick_params(labelsize=18)
    ax1.plot(data[:, 0], data[:, 1])
    ax2 = plt.subplot(2, 1, 2)
    ax2.xaxis.set_major_locator(ticker.FixedLocator(data[:, 0]))
    plt.ylabel('average task time (sec)', {'size': 24})
    plt.xlabel('number of waypoints', {'size': 24})
    ax2.tick_params(labelsize=18)
    ax2.plot(data[:, 0], data[:, 2])
    ax1.legend(labels=['maxium edge length = 5.0'], loc='lower right', fontsize='xx-large')
    ax2.legend(labels=['maxium edge length = 5.0'], loc='lower right', fontsize='xx-large')
    plt.show()

def plot_comparison_me(filename='maximum_edge_length.csv'):
    data = np.loadtxt(filename, dtype=np.float64, delimiter=',')
    x=np.arange(len(data))
    fig = plt.figure(figsize=(12, 12))
    ax1 = plt.subplot(2, 1, 1)
    ax1.xaxis.set_major_locator(ticker.FixedLocator(data[:, 0]))
    plt.ylabel('success rate', {'size': 24})
    ax1.tick_params(labelsize=18)
    ax1.plot(data[:, 0], data[:, 1])
    ax2 = plt.subplot(2, 1, 2)
    ax2.xaxis.set_major_locator(ticker.FixedLocator(data[:, 0]))
    plt.ylabel('task time (sec)', {'size': 24})
    plt.xlabel('maximum edge length', {'size': 24})
    ax2.tick_params(labelsize=18)
    ax2.plot(data[:, 0], data[:, 2])
    ax1.legend(labels=['number of waypoints = 400'], loc='lower right', fontsize='xx-large')
    ax2.legend(labels=['number of waypoints = 400'], loc='upper right', fontsize='xx-large')
    plt.show()