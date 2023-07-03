# ********************************************* NASA *****************************************************

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 13:03:23 2021

@author: ammar@scch
"""

############################################################################################################
#                                           IMPORTING LIBRARIES
# ##########################################################################################################

import itertools

import time
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import gym
import random
import sklearn.preprocessing
import sklearn.pipeline
# import torch
import random
import warnings
import copy
# import seaborn as sns
import tensorflow as tf
import keras

# from IPython.display import clear_output
# from gym import spaces
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines.deepq.policies import LnMlpPolicy, LnCnnPolicy
# from stable_baselines import DQN
# from stable_baselines.common.env_checker import check_env
# from IPython.display import display
# from sklearn.linear_model import LinearRegression, SGDRegressor
# from sklearn.kernel_approximation import RBFSampler
# from sklearn.decomposition import PCA
# from lib import plotting
# from collections import deque
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=DeprecationWarning)
scaler = StandardScaler()

############################################################################################################
#                                           DATASET PREPARATION
# ##########################################################################################################

dir_path = os.getcwd()
df = pd.read_csv(dir_path + '/CMAPSSData/train_FD001.txt', sep=" ", header=None, skipinitialspace=True).dropna(axis=1)
df = df.rename(columns={0: 'unit', 1: 'cycle', 2: 'W1', 3: 'W2', 4: 'W3'})
df_A = df[df.columns[[0, 1]]]
df_W = df[df.columns[[2, 3, 4]]]
df_S = df[df.columns[list(range(5, 26))]]
df_X = pd.concat([df_W, df_S], axis=1)

'''Standardization'''
df_X = scaler.fit_transform(df_X)

'''train_test split'''
engine_unit = 1

'''##
# %% ENGINE UNIT SPECIFIC DATA
engine_unit = 1
engine_df_A = df_A[df_A['unit'] == engine_unit]
engine_df_X = df_X.iloc[engine_df_A.index[0]:engine_df_A.index[-1] + 1]
engine_df_W = df_W.iloc[engine_df_A.index[0]:engine_df_A.index[-1] + 1]

##
# %% NORMALIZE DATA
X = scaler.fit_transform(engine_df_X)
# X = (((engine_df_X - engine_df_X.mean()) / engine_df_X.std()).fillna(0))
# X = ((engine_df_X - engine_df_X.min()) / (engine_df_X.max() - engine_df_X.min())).fillna(0)).values'''

'''
##
# %% READ RUL & APPEND

# df_RUL = pd.read_csv(dir_path + '/CMAPSSData/RUL_FD001.txt', sep=" ", header=None, skipinitialspace=True).dropna(axis=1)
# df_RUL.columns = ['RUL']
# df_z_scaled_RUL = df_z_scaled.join(df_RUL, 1)

##
# %% REGRESSION TO GET "RUL distribution"

# x = df_z_scaled_RUL.iloc[:,list(range(5, 26))]
# y = df_RUL

##
# %% DIMENSIONALITY REDUCTION TO GET "HEALTH INDICATOR"

sensor_data = df_z_scaled.iloc[:, list(range(5, 26))].dropna(axis=1)
pca = PCA(n_components=1)
principalComponents = (1 - pca.fit_transform(sensor_data))

pdf = pd.DataFrame(data=principalComponents, columns=['health indicator'])
pdf_normalized = (pdf - pdf.min()) / (pdf.max() - pdf.min()) * 100

df_scaled_principal = df_z_scaled.join(pdf_normalized, 1)
df_scaled_principal = df_scaled_principal.rename(columns={0: 'engine unit', 1: 'cycle'})


##
# %% VISUALIZATION
engine_unit = 76
engine_df = df_scaled_principal[df_scaled_principal['engine unit'] == engine_unit]
# engine_df.plot.line('cycle', 'health indicator')
# plt.show()

HI = np.array(engine_df['health indicator'])[0:191].astype(np.float32)
# plt.plot(HI)
# plt.show()
'''

############################################################################################################
# **********************************************************************************************************
#                                           ENVIRONMENT MODELING
# **********************************************************************************************************
# ##########################################################################################################

reward_replace = -300
reward_hold = 1
reward_failure = -500
policy = {}
policy_test = {}

'''
class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(100)
        self.observation_space = gym.spaces.Box(low=np.array([0]), high=np.array([100]))
        self.reward = 0
        self.cycle = 0
        self.state = np.array([np.array(HI[self.cycle]).astype(np.float32)])
        self.done = False

    def step(self, action):
        if self.cycle > failure_state:
            self.state = np.array([np.array(HI[self.cycle]).astype(np.float32)])
            self.reward = reward_failure
            self.done = True
            print("|cycle reached failure state|:", self.cycle, "reward:", self.reward, '\n')
        elif self.cycle <= failure_state:
            if action == 0:
                print("|hold|:", self.cycle)
                if self.cycle == failure_state:
                    self.state = np.array([np.array(HI[self.cycle]).astype(np.float32)])
                    self.reward = reward_failure
                    self.done = True
                    print("|cycle reached failure state|:", self.cycle, "reward:", self.reward, '\n')
                else:
                    self.cycle += 1
                    if HI[self.cycle] > T:
                        self.reward = reward_hold
                        self.state = np.array([np.array(HI[self.cycle]).astype(np.float32)])
                        self.done = False
                        print("|system running|", "health:", HI[self.cycle], "reward:", self.reward, '\n')
                    elif HI[self.cycle] <= T:
                        self.reward = reward_failure
                        self.state = np.array([np.array(HI[self.cycle]).astype(np.float32)])
                        self.done = True
                        print("|system failed|", "health:", HI[self.cycle], "reward:", self.reward, '\n')
            elif action == 1:
                print("|replace|:", self.cycle, "health:", HI[self.cycle])
                self.reward = reward_replace / (self.cycle + z)
                self.cycle = 0
                self.state = np.array([np.array(HI[self.cycle]).astype(np.float32)])
                self.done = True
                print("reward:", self.reward, '\n')
        info = {}
        return self.state, self.reward, self.done, info

    def reset(self):
        self.cycle = 0
        self.state = np.array([np.array(HI[self.cycle]).astype(np.float32)])
        self.done = False
        return self.state
'''


class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, is_training=True, verbose=False):
        self.action_space = gym.spaces.Discrete(2)
        # self.observation_space = gym.spaces.Discrete(100)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(df_X.shape[1],))
        self.reward = 0
        self.cycle = 0
        self.done = False
        self.engine_unit = engine_unit
        self.engine_df_A = df_A[df_A['unit'] == self.engine_unit]
        self.X = df_X[self.engine_df_A.index[0]:self.engine_df_A.index[-1] + 1]
        self.state = self.X[self.cycle]
        self.failure_state = self.engine_df_A['cycle'].max()
        self.train = is_training
        self.verbose = verbose

    def get_next_engine_data(self):
        self.engine_unit += 1
        if self.train:
            if self.engine_unit > int((df_A['unit'].max() * 80 / 100)):
                self.engine_unit = 1
        else:
            if self.engine_unit > df_A['unit'].max():
                self.engine_unit = int((df_A['unit'].max() * 80 / 100) + 1)
        if self.verbose:
            print("********|engine unit|********:", self.engine_unit)
        self.engine_df_A = df_A[df_A['unit'] == self.engine_unit]
        self.X = df_X[self.engine_df_A.index[0]:self.engine_df_A.index[-1] + 1]
        self.failure_state = self.engine_df_A['cycle'].max() - 1
        return self.X

    def step(self, action):
        if action == 0:
            if self.verbose:
                print("|hold|:", self.cycle)
            if self.cycle == self.failure_state:
                self.reward = reward_failure
                self.state = self.X[self.cycle]
                self.done = True
                if self.train:
                    policy[self.engine_unit] = {'unit': self.engine_unit,
                                                'failure_state': self.failure_state,
                                                'replace_state': None}
                else:
                    policy_test[self.engine_unit] = {'unit': self.engine_unit,
                                                     'failure_state': self.failure_state,
                                                     'replace_state': None}
                if self.verbose:
                    print("|cycle reached failure state|:", self.cycle, "reward:", self.reward, '\n')
            else:
                self.reward = reward_hold
                self.cycle += 1
                self.state = self.X[self.cycle]
                self.done = False
                if self.verbose:
                    print("|system running|", "reward:", self.reward, '\n')
        elif action == 1:
            if self.verbose:
                print("|replace|:", self.cycle)
            self.reward = reward_replace / (self.cycle + 0.1)
            self.state = self.X[self.cycle]
            self.done = True
            if self.train:
                policy[self.engine_unit] = {'unit': self.engine_unit,
                                            'failure_state': self.failure_state,
                                            'replace_state': self.cycle}
            else:
                policy_test[self.engine_unit] = {'unit': self.engine_unit,
                                                 'failure_state': self.failure_state,
                                                 'replace_state': self.cycle}
        if self.verbose:
            print("reward:", self.reward, '\n')
        info = {}
        return self.state, self.reward, self.done, info

    def reset(self):
        self.X = self.get_next_engine_data()
        self.cycle = 0
        self.state = self.X[self.cycle]
        self.done = False
        return self.state


env = CustomEnv()
# check_env(env)

############################################################################################################
#                                           Function Approximation
# ##########################################################################################################

'''observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()

featurizer = sklearn.pipeline.FeatureUnion([
    ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
    ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
    ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
    ("rbf4", RBFSampler(gamma=0.5, n_components=100))
])
featurizer.fit(scaler.fit_transform(observation_examples))


class FunctionApproximator:
    def __init__(self):

        self.models = []
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)

    def featurize_state(self, state):

        scaled = scaler.transform([state])
        features = featurizer.transform(scaled)
        return features[0]

    def predict(self, s, a=None):

        state_features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([state_features])[0] for m in self.models])
        else:
            return self.models[a].predict([state_features])[0]

    def update(self, s, a, y):

        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])


def make_epsilon_greedy_policy(estimator, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def q_learning(env, estimator, num_episodes, discount_factor=0.6, epsilon=0.0, epsilon_decay=1.0):
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):

        policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay ** i_episode, env.action_space.n)
        state = env.reset()

        for t in itertools.count():

            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            next_state, reward, end, _ = env.step(action)

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            q_values_next = estimator.predict(next_state)
            td_target = reward + discount_factor * np.max(q_values_next)

            estimator.update(state, action, td_target)

            if i_episode % 10 == 0:
                print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, reward))

            if end:
                break

            state = next_state

    return stats'''

############################################################################################################
#                                       APPROXIMATE (DEEP) Q LEARNING
# ##########################################################################################################

n_actions = env.action_space.n
state_dim = env.observation_space.shape

tf.reset_default_graph()
sess = tf.InteractiveSession()
keras.backend.set_session(sess)

network = keras.models.Sequential()
network.add(keras.layers.InputLayer(state_dim))

# let's create a network for approximate q-learning following guidelines above
network.add(keras.layers.Dense(128, activation='relu'))
network.add(keras.layers.Dense(256, activation='relu'))
network.add(keras.layers.Dense(n_actions, activation='linear'))

network.summary()


def get_action(state, epsilon=0):
    """
    sample actions with epsilon-greedy policy
    recap: with p = epsilon pick random action, else pick action with highest Q(s,a)
    """

    q_values = network.predict(state[None])[0]

    exploration = np.random.random()
    if exploration < epsilon:
        action = np.random.choice(n_actions, 1)[0]
    else:
        action = np.argmax(q_values)
    return action


states_ph = tf.placeholder('float32', shape=(None,) + state_dim)
actions_ph = tf.placeholder('int32', shape=[None])
rewards_ph = tf.placeholder('float32', shape=[None])
next_states_ph = tf.placeholder('float32', shape=(None,) + state_dim)
is_done_ph = tf.placeholder('bool', shape=[None])

# get q-values for all actions in current states
predicted_qvalues = network(states_ph)

# select q-values for chosen actions
predicted_qvalues_for_actions = tf.reduce_sum(predicted_qvalues * tf.one_hot(actions_ph, n_actions), axis=1)

gamma = 0.95

# compute q-values for all actions in next states
predicted_next_qvalues = network(next_states_ph)

# compute V*(next_states) using predicted next q-values
next_state_values = tf.reduce_max(predicted_next_qvalues, axis=1)

# compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
target_qvalues_for_actions = rewards_ph + gamma * next_state_values

# at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
target_qvalues_for_actions = tf.where(is_done_ph, rewards_ph, target_qvalues_for_actions)

# mean squared error loss to minimize
loss = (tf.stop_gradient(target_qvalues_for_actions) - predicted_qvalues_for_actions) ** 2
loss = tf.reduce_mean(loss)

# training function that resembles agent.update(state, action, reward, next_state) from tabular agent
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)


def generate_session(t_max=500, epsilon=0, train=False):
    """play env with approximate q-learning agent and train it at the same time"""
    total_reward = 0
    s = env.reset()

    for t in range(t_max):
        a = get_action(s, epsilon=epsilon)
        next_s, r, done, _ = env.step(a)

        if train:
            sess.run(train_step, {
                states_ph: [s], actions_ph: [a], rewards_ph: [r],
                next_states_ph: [next_s], is_done_ph: [done]
            })

        total_reward += r
        s = next_s
        if done:
            break

    return total_reward


############################################################################################################
#                                              DEEP Q LEARNING
# ##########################################################################################################

'''class ReplayBuffer:
    def __init__(self, batch_size=32, size=1000000):
        """
        batch_size (int): number of data points per batch
        size (int): size of replay buffer.
        """
        self.batch_size = batch_size
        self.memory = deque(maxlen=size)

    def remember(self, s_t, a_t, r_t, s_t_next, d_t):
        """
        s_t (np.ndarray double): state
        a_t (np.ndarray int): action
        r_t (np.ndarray double): reward
        d_t (np.ndarray float): done flag
        s_t_next (np.ndarray double): next state
        """
        self.memory.append((s_t, a_t, r_t, s_t_next, d_t))

    def sample(self):
        """
        random sampling of data from buffer
        """
        # if we don't have enough samples yet
        size = min(self.batch_size, len(self.memory))
        return random.sample(self.memory, size)


class VectorizedEnvWrapper(gym.Wrapper):
    def __init__(self, env, num_envs=1):
        """
        env (gym.Env): to make copies of
        num_envs (int): number of copies
        """
        super().__init__(env)
        self.num_envs = num_envs
        self.envs = [copy.deepcopy(env) for n in range(num_envs)]

    def reset(self):
        """
        Return and reset each environment
        """
        return np.asarray([env.reset() for env in self.envs])

    def step(self, actions):
        """
        Take a step in the environment and return the result.
        actions (np.ndarray int)
        """
        next_states, rewards, dones = [], [], []
        for env, action in zip(self.envs, actions):
            next_state, reward, done, _ = env.step(action)
            if done:
                next_states.append(env.reset())
            else:
                next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
        return np.asarray(next_states), np.asarray(rewards), \
               np.asarray(dones)


class DeepQLearner:
    def __init__(self, env,
                 alpha=0.001, gamma=0.95,
                 epsilon_i=1.0, epsilon_f=0.001, n_epsilon=0.1):
        """
        env (VectorizedEnvWrapper): the vectorized gym.Env
        alpha (float): learning rate
        gamma (float): discount factor
        epsilon_i (float): initial value for epsilon
        epsilon_f (float): final value for epsilon
        n_epsilon (float): proportion of timesteps over which to
                           decay epsilon from epsilon_i to
                           epsilon_f
        """

        self.num_envs = env.num_envs
        self.M = env.action_space.n  # number of actions
        self.N = env.observation_space.shape[0]  # dimensionality of state space

        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.n_epsilon = n_epsilon
        self.epsilon = epsilon_i
        self.gamma = gamma

        self.Q = torch.nn.Sequential(
            torch.nn.Linear(self.N, 24),
            torch.nn.ReLU(),
            torch.nn.Linear(24, 24),
            torch.nn.ReLU(),
            torch.nn.Linear(24, self.M)
        ).double()

        self.Q_ = copy.deepcopy(self.Q)

        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=alpha)

    def synchronize(self):
        """
        Used to make the parameters of Q_ match with Q.
        """
        self.Q_.load_state_dict(self.Q.state_dict())

    def act(self, s_t):
        """
        Epsilon-greedy policy.
        s_t (np.ndarray): the current state.
        """
        s_t = torch.as_tensor(s_t).double()
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.M, size=self.num_envs)
        else:
            with torch.no_grad():
                return np.argmax(self.Q(s_t).numpy(), axis=1)

    def decay_epsilon(self, n):
        """
        Epsilon decay.
        n (int): proportion of training complete
        """
        self.epsilon = max(
            self.epsilon_f,
            self.epsilon_i - (n / self.n_epsilon) * (self.epsilon_i - self.epsilon_f))

    def update(self, s_t, a_t, r_t, s_t_next, d_t):
        """
        Learning step.
        s_t (np.ndarray double): state
        a_t (np.ndarray int): action
        r_t (np.ndarray double): reward
        d_t (np.ndarray float): done flag
        s_t_next (np.ndarray double): next state
        """

        # make sure everything is torch.Tensor and type-compatible with Q
        s_t = torch.as_tensor(s_t).double()
        a_t = torch.as_tensor(a_t).long()
        r_t = torch.as_tensor(r_t).double()
        s_t_next = torch.as_tensor(s_t_next).double()
        d_t = torch.as_tensor(d_t).double()

        # we don't want gradients when calculating the target y
        with torch.no_grad():
            # taking 0th element because torch.max returns both maximum
            # and argmax
            Q_next = torch.max(self.Q_(s_t_next), dim=1)[0]
            target = r_t + (1 - d_t) * self.gamma * Q_next

        # use advanced indexing on the return to get the predicted
        # Q values corresponding to the actions chosen in each environment.
        Q_pred = self.Q(s_t)[range(self.num_envs), a_t]
        loss = torch.mean((target - Q_pred) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train(env, agent, replay_buffer, T=20000, n_theta=100):
    """
    env (VectorizedEnvWrapper): vectorized gym.Env
    agent (DeepQLearner)
    buffer (ReplayBuffer)
    T (int): total number of training timesteps
    batch_size: number of
    """

    # for plotting
    returns = []
    episode_rewards = 0
    mean_return = []
    running_return = []

    s_t = env.reset()
    for t in range(T):
        # synchronize Q and Q_
        if t % n_theta == 5:
            agent.synchronize()

        a_t = agent.act(s_t)
        s_t_next, r_t, d_t = env.step(a_t)

        # store data into replay buffer
        replay_buffer.remember(s_t, a_t, r_t, s_t_next, d_t)
        s_t = s_t_next

        # learn by sampling from replay buffer
        for batch in replay_buffer.sample():
            agent.update(*batch)

        # for plotting
        episode_rewards += r_t
        for i in range(env.num_envs):
            if d_t[i]:
                returns.append(episode_rewards[i])
                running_return.append(episode_rewards[i])
                episode_rewards[i] = 0

        # epsilon decay
        agent.decay_epsilon(t / T)

        if t % 100 == 0:
            mean_return.append(np.mean(returns))
            returns = []
    plot_returns(running_return)
    plt.plot(mean_return)
    plt.show()
    return agent


sns.set()


def plot_returns(returns, window=10):
    """
    Returns (iterable): list of returns over time
    window: window for rolling mean to smooth plotted curve
    """
    sns.lineplot(
        data=pd.DataFrame(returns).rolling(window=window).mean()[window - 1::window]
    )'''

############################################################################################################
# **********************************************************************************************************
#                                               TRAINING
# **********************************************************************************************************
# ##########################################################################################################

# Define and Train the agent
print('###################################################################################################')
print("Training")
print('###################################################################################################', '\n')

# *************************** Tabular Q Learning *****************************

'''q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.01
gamma = 0.9
epsilon = 0.2

all_epochs = []
episode_reward = []
mean_episode_reward = []

for i in range(1, 1000000):
    state = env.reset()

    epochs, reward = 0, 0
    done = False
    total_reward = 0

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values
            # action = np.random.choice(np.where(q_table[int(state)] == q_table[int(state)].max())[0])

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state
        epochs += 1
        total_reward += reward

    episode_reward.append(total_reward)
    if i % 100 == 0:
        print(f"Episode: {i}")
        epsilon *= 0.999
        mean_episode_reward.append(np.mean(episode_reward))
        episode_reward = []

plt.plot(mean_episode_reward)'''

# ************************** Function Approximation **************************

'''estimator = FunctionApproximator()
stats = q_learning(env, estimator, 100, epsilon=0.1)

# plotting.plot_episode_stats(stats, smoothing_window=25)

'''

# ************************** Approximate Q Learning **************************

epsilon = 0.5
total_returns = []
for i in range(2000):
    session_rewards = [generate_session(epsilon=epsilon, train=True) for _ in range(int(df_A['unit'].max() * 80 / 100))]
    print("epoch #{}\tmean reward = {:.3f}\tepsilon = {:.3f}".format(i, np.mean(session_rewards), epsilon))
    total_returns.append(np.mean(session_rewards))
    epsilon *= 0.99
    # Make sure epsilon is always nonzero during training
    if epsilon <= 1e-4:
        break
plt.plot(total_returns)
plt.show()

# ************************** Double Deep Q Learning **************************

'''env = VectorizedEnvWrapper(env, num_envs=32)
agent = DeepQLearner(env, alpha=1e-3, gamma=0.95)
replay_buffer = ReplayBuffer(batch_size=8)
agent = train(env, agent, replay_buffer, T=500000)'''

# ******************** Double Deep Q Learning (library) **********************

'''model = DQN(LnMlpPolicy, env, verbose=1, tensorboard_log="./dqn_nasa_tensorboard/").learn(total_timesteps=10000)
# tensorboard --logdir ./dqn_nasa_tensorboard/
model.save("deepq_nasa")'''

print("Training finished.\n")

############################################################################################################
# **********************************************************************************************************
#                                         EVALUATION
# **********************************************************************************************************
# ##########################################################################################################

print('###################################################################################################')
print("Testing")
print('###################################################################################################', '\n')

# *************************** Tabular Q Learning *****************************

'''"""Evaluate agent's performance after Q-learning"""

total_epochs = 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, reward = 0, 0

    done = False

    while not done:
        action = np.argmax(q_table[int(state)])
        state, reward, done, info = env.step(action)
        epochs += 1
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")'''

# ************************** Function Approximation **************************

'''state = env.reset()
while True:
    q_values = estimator.predict(state)
    best_action = np.argmax(q_values)
    next_state, reward, done, _ = env.step(best_action)
    if done:
        break
    state = next_state'''

# ************************** Approximate Q Learning **************************

'''total_returns = []
for i in range(10):
    session_rewards = [generate_session(epsilon=0, train=False) for _ in range(100)]
    total_returns.append(np.mean(session_rewards))
env.close()
plt.plot(total_returns)
plt.show()'''

'''df_test = pd.read_csv(dir_path + '/CMAPSSData/test_FD001.txt', sep=" ", header=None, skipinitialspace=True).dropna(
    axis=1)
df = df_test.rename(columns={0: 'unit', 1: 'cycle', 2: 'W1', 3: 'W2', 4: 'W3'})
df_A = df[df.columns[[0, 1]]]
df_W = df[df.columns[[2, 3, 4]]]
df_X = df[df.columns[list(range(5, 26))]]'''

engine_unit = int((df_A['unit'].max() * 80 / 100) + 1)
env = CustomEnv(is_training=False)

session_rewards = [generate_session(epsilon=0, train=False) for _ in range(int(df_A['unit'].max() * 20 / 100))]
total_returns_test = np.mean(session_rewards)
print("mean reward = ", total_returns_test)

policy = pd.DataFrame.from_dict(policy).T
policy['remaining_cycles'] = policy['failure_state'] - policy['replace_state']
policy_test = pd.DataFrame.from_dict(policy_test).T
policy_test['remaining_cycles'] = policy_test['failure_state'] - policy_test['replace_state']

np.savetxt(r'./sensor_data_train', policy.values, fmt='%d')
np.savetxt(r'./sensor_data_test', policy_test.values, fmt='%d')

env.close()

# ****************************** Deep Q Learning *****************************

'''env = CustomEnv()
env = VectorizedEnvWrapper(env, num_envs=1)
agent = DeepQLearner(env, alpha=1e-3, gamma=0.95)
episodes = 100
episode_return = []

for _ in range(episodes):
    s_t = env.reset()
    d_t = False
    r_t = 0
    total_return = 0
    while not d_t:
        total_return += r_t
        a_t = agent.act(s_t)
        s_t_next, r_t, d_t = env.step(a_t)
        s_t = s_t_next
        # store data into replay buffer
        replay_buffer.remember(s_t, a_t, r_t, s_t_next, d_t)
        s_t = s_t_next
        # learn by sampling from replay buffer
        for batch in replay_buffer.sample():
            agent.update(*batch)
    episode_return.append(total_return)

plt.plot(episode_return)
'''

# ******************** Double Deep Q Learning (library) **********************

'''env.reset()
for obs in HI:
    obs = np.array([np.array(obs).astype(np.float32)])
    action, _states = model.predict(obs)
    print("action:", action)
    state, reward, done, info = env.step(action)
    print("HI:", state, "G:", reward, "is_done:", done, '\n')
    print('####################################################', '\n')'''

print("done")
