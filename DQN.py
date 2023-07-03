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
import torch
import random
import warnings
import copy
import seaborn as sns
import tensorflow as tf
import tensorflow.keras
import math

from IPython.display import clear_output
from gym import spaces
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import LnMlpPolicy, LnCnnPolicy
from stable_baselines import DQN
from stable_baselines.common.env_checker import check_env
from IPython.display import display
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import PCA
from lib import plotting
from collections import deque
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings("ignore", category=DeprecationWarning)
standard = StandardScaler()
minmax = MinMaxScaler()
dir_path = os.getcwd()

############################################################################################################
# **********************************************************************************************************
#                                              CASE STUDIES
# **********************************************************************************************************
# ##########################################################################################################

############################################################################################################
#                                              NASA-CMAPSS
# ##########################################################################################################

dataset = 'train_FD002.txt'
df_full = pd.read_csv(dir_path + r'/CMAPSSData/' + dataset, sep=" ", header=None, skipinitialspace=True).dropna(axis=1)
df_full = df_full.rename(columns={0: 'unit', 1: 'cycle', 2: 'W1', 3: 'W2', 4: 'W3'})
# df = df_full

# mapping
train_set = [*range(1, 23), *range(24, 32), *range(33, 39), 40, 44, 45, 46, 49, 51, 53,
             *range(55, 61), 62, 63, 64, 66, 67, 69, 70, 71, 74, 78, 81, 88, 94, 97, 102, 103, 105,
             106, 107, 108, 118, 120, 128, 133, 136, 137, 141, 165, 173, 176, 178]
test_set = [185, 188, 192, 194, 197, 208, 212, 214, 217, 219, 225, 231, 234, 238, 244, 252, 253, 256, 258, 260]
combined_set = train_set + test_set

df = pd.DataFrame()
for engines in combined_set:
    df = pd.concat([df, df_full[df_full['unit'] == engines]], ignore_index=True)

zip_iterator = zip(combined_set, list(range(1, 101)))
mapping_dict = dict(zip_iterator)
df["unit"] = df["unit"].map(mapping_dict)

df_A = df[df.columns[[0, 1]]]
df_W = df[df.columns[[2, 3, 4]]]
df_S = df[df.columns[list(range(5, 26))]]
df_X = pd.concat([df_W, df_S], axis=1)

# RUL as sensor reading
# df_A['RUL'] = 0
# for i in range(1, 101):
# df_A['RUL'].loc[df_A['unit'] == i] = df_A[df_A['unit'] == i].cycle.max() - df_A[df_A['unit'] == i].cycle

# Standardization
df_X = standard.fit_transform(df_X)

# train_test split
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
#                                            HYDRAULIC SYSTEM
# ##########################################################################################################
''''''
# PS1 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\PS1.txt').mean(axis=1)
# PS2 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\PS2.txt').mean(axis=1)
# PS3 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\PS3.txt').mean(axis=1)
# PS4 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\PS4.txt').mean(axis=1)
# PS5 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\PS5.txt').mean(axis=1)
# PS6 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\PS6.txt').mean(axis=1)
# EPS1 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\EPS1.txt').mean(axis=1)
# FS1 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\FS1.txt').mean(axis=1)
# FS2 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\FS2.txt').mean(axis=1)
# TS1 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\TS1.txt').mean(axis=1)
# TS2 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\TS2.txt').mean(axis=1)
# TS3 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\TS3.txt').mean(axis=1)
# TS4 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\TS4.txt').mean(axis=1)
# VS1 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\VS1.txt').mean(axis=1)
# CE = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\CE.txt').mean(axis=1)
# CP = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\CP.txt').mean(axis=1)
# SE = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\SE.txt').mean(axis=1)
#
# X = np.array([PS1, PS2, PS3, PS4, PS5, PS6, EPS1, FS1, FS2, TS1, TS2, TS3, TS4, VS1, CE, CP, SE]).T
# np.savetxt(r"C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\sensors.csv", X, delimiter=" ")

# df_X = pd.read_csv(r"C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\sensors.csv", sep=" ", header=None)
# df_X.columns = ['PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'EPS1', 'FS1', 'FS2', 'TS1', 'TS2', 'TS3', 'TS4', 'VS1',
#                 'CE', 'CP', 'SE']
#
# df_Y = pd.read_csv(r"C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\profile.txt", sep="\t", header=None)
# df_Y.columns = ['Cooler_condition', 'Valve_condition', 'Internal_pump_leakage', 'Hydraulic_accumulator', 'stable_flag']

'''fig, ax = plt.subplots()

ax.plot(df_Y['Valve condition'], color='red', label='Valve condition')
ax.tick_params(axis='y', labelcolor='red')
ax.legend(loc="upper right")

ax2 = ax.twinx()
ax2.plot(df_Y['Cooler condition'], color='green', label='Cooler condition')
ax2.tick_params(axis='y', labelcolor='green')
ax2.legend(loc="upper left")

ax3 = ax.twinx()
ax3.plot(df_Y['Internal pump leakage'], color='green', label='Internal pump leakage')
ax3.tick_params(axis='y', labelcolor='green')
ax3.legend(loc="lower right")

ax4 = ax.twinx()
ax4.plot(df_Y['Hydraulic accumulator'], color='blue', label='Hydraulic accumulator')
ax4.tick_params(axis='y', labelcolor='blue')
ax4.legend(loc="lower left")

ax5 = ax.twinx()
ax5.plot(df_Y['stable flag'], color='orange', label='stable flag')
ax5.tick_params(axis='y', labelcolor='orange')
ax5.legend(loc="upper center")

plt.show()'''

############################################################################################################
#                                               YOKOGAWA
# ##########################################################################################################

'''import datetime

df_yoko = pd.read_csv(r'C:/Users/abbas/Desktop/case_studies/Datasets/Yokogawa/CYG-2019-OCT-NOV edited.csv', sep=";")
df_yoko[['Date', 'Time']] = df_yoko['TimeStamp'].str.split(' ', 1, expand=True)
df_yoko['Date'] = pd.to_datetime(df_yoko['Date'], format='%d.%m.%Y')
df_yoko['Time_'] = pd.to_datetime(df_yoko['TimeStamp'])
df_yoko["Month"] = df_yoko["Date"].dt.month
df_yoko["Day"] = df_yoko["Date"].dt.day
df_yoko["Time"] = df_yoko["Time_"].dt.time
df_yoko["Hour"] = df_yoko["Time_"].dt.hour
df_yoko["Minute"] = df_yoko["Time_"].dt.minute

df_yoko["SeverityLevel"] = df_yoko["Severity"] * df_yoko["AlarmLevel"]
df_yoko['Cons'] = df_yoko['TagName'].shift(periods=1)
df_yoko['Cons2'] = df_yoko['TagName'].shift(periods=2)
df_yoko['Cons3'] = df_yoko['TagName'].shift(periods=3)

df_yoko = df_yoko.drop(columns=['SubConditionName', 'TimeStampNS', 'EngUnit', 'StationTimeMilli', 'ItemName', 'Source',
                                'ModeStatName'])
df_yoko_sub = df_yoko[df_yoko['TagName'] != '025TIC010']

df_yoko_sub = df_yoko_sub[['TimeStamp', 'TagName', 'StationName', 'ConditionName', 'Severity', 'AlarmLevel',
                           'SeverityLevel', 'AlarmOff', 'AlarmBlink', 'AckRequired', "Month", "Day", "Time",
                           "Hour", "Minute", 'Cons', 'Cons2', 'Cons3']]

count = df_yoko_sub['Severity'].value_counts()
sort = df_yoko_sub.groupby(["TagName", "Severity"]).size().reset_index(name="Frequency").sort_values(by='Frequency',
                                                                                                     ascending=[False])
df_yoko_sub.groupby(["Month", "Day", "Hour"])['TagName'].nunique().reset_index(name="Frequency").sort_values(by='Month')
event = df_yoko_sub.loc[
    (df_yoko_sub["Month"] == 11) & (df_yoko_sub["Day"] == 5) & (df_yoko_sub["Time"] == datetime.time(15, 16, 3))]

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(df_yoko_sub.corr(), dtype=bool))
sns.heatmap(df_yoko_sub.corr(), mask=mask)
'''

############################################################################################################
# **********************************************************************************************************
#                                       HIDDEN MARKOV MODEL (LIBRARY)
# **********************************************************************************************************
# ##########################################################################################################

'''from hmmlearn import hmm
from random import randint
import pickle

# df_S = df[df.columns[[6, 8, 11, 12, 15, 16, 19]]]
# df_hmm = pd.concat([df_A['cycle'], df_S], axis=1)

df_hmm = minmax.fit_transform(df_S)

df_hmm = pd.DataFrame(df_hmm)
cols_to_drop = df_hmm.nunique()[df_hmm.nunique() == 1].index
df_hmm = df_hmm.drop(cols_to_drop, axis=1)
cols_to_drop = df_hmm.nunique()[df_hmm.nunique() == 2].index
df_hmm = df_hmm.drop(cols_to_drop, axis=1).to_numpy()

lengths = [df[df['unit'] == i].cycle.max() for i in range(1, df_A['unit'].max() + 1)]
# o = df_X[df_A[df_A['unit'] == 1].index[0]:df_A[df_A['unit'] == 1].index[-1] + 1]

num_states = 15
remodel = hmm.GaussianHMM(n_components=num_states,
                          n_iter=500,
                          verbose=True, )
                          # init_params="cm", params="cmt")


transmat = np.zeros((num_states, num_states))
# Left-to-right: each state is connected to itself and its
# direct successor.
for i in range(num_states):
    if i == num_states - 1:
        transmat[i, i] = 1.0
    else:
        transmat[i, i] = transmat[i, i + 1] = 0.5

# Always start in first state
# startprob = np.zeros(num_states)
# startprob[0] = 1.0

# remodel.startprob_ = startprob
# remodel.transmat_ = transmat


remodel.fit(df_hmm, lengths)


# with open("HMM_model.pkl", "wb") as file: pickle.dump(remodel, file)
# with open("filename.pkl", "rb") as file: pickle.load(file)

state_seq = remodel.predict(df_hmm, lengths)
pred = [state_seq[df[df['unit'] == i].index[0]:df[df['unit'] == i].index[-1] + 1] for i in
        range(1, df_A['unit'].max() + 1)]

prob = remodel.predict_proba(df_hmm, lengths)
prob_next_step = remodel.transmat_[state_seq, :]

HMM_out = [prob[df[df['unit'] == i].index[0]:df[df['unit'] == i].index[-1] + 1]
           for i in range(1, df_A['unit'].max() + 1)]
failure_states = [pred[i][-1] for i in range(df_A['unit'].max())]


# RUL Prediction - Monte Carlo Simulation
from sklearn.utils import check_random_state

transmat_cdf = np.cumsum(remodel.transmat_, axis=1)
random_state = check_random_state(remodel.random_state)


predRUL = []
for i in range(df_A[df_A['unit'] == 1]['cycle'].max()):
    RUL = []
    for j in range(100):
        cycle = 0
        pred_obs_seq = [df_hmm[i]]
        pred_state_seq = remodel.predict(pred_obs_seq)
        while pred_state_seq[-1] not in set(failure_states):
            cycle += 1
            prob_next_state = (transmat_cdf[pred_state_seq[-1]] > random_state.rand()).argmax()
            prob_next_obs = remodel._generate_sample_from_state(prob_next_state, random_state)
            pred_obs_seq = np.append(pred_obs_seq, [prob_next_obs], 0)
            pred_state_seq = remodel.predict(pred_obs_seq)
        RUL.append(cycle)
    # noinspection PyTypeChecker
    predRUL.append(round(np.mean(RUL)))

plt.plot(predRUL)
plt.plot(df_A[df_A['unit'] == 1].RUL)
plt.show()


plt.figure(0)
plt.plot(pred[0])
plt.plot(pred[1])
plt.plot(pred[2])
plt.xlabel('# Flights')
plt.ylabel('HMM states')
plt.show()

plt.figure(1)
E = [randint(1, df_A['unit'].max()) for p in range(0, 10)]
for e in E:
    plt.plot(pred[e - 1])
plt.xlabel('# Flights')
plt.ylabel('HMM states')
plt.legend(E, title='engine unit')

plt.figure(2)
plt.scatter(list(range(1, len(failure_states) + 1)), failure_states)
plt.xlabel('Engine units')
plt.ylabel('HMM states')
plt.legend(title='failure states')

plt.figure(3)
pca = PCA(n_components=2).fit_transform(df_hmm)
for class_value in range(num_states):
    # get row indexes for samples with this class
    row_ix = np.where(state_seq == class_value)
    plt.scatter(pca[row_ix, 0], pca[row_ix, 1])
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.legend(list(range(0, num_states)), title='HMM states')

plt.show()

import seaborn as sns
sns.heatmap(df_hmm.corr())'''

'''# Generate samples
X, Y = remodel._generate_sample_from_state(np.array([df_X[0]]))
# Plot the sampled data
plt.plot(X[:, 0], X[:, 1], ".-", label="observations", ms=6,
         mfc="orange", alpha=0.7)
plt.show()'''

'''
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

df_X = PCA(n_components=2).fit_transform(df_X)

gmm = GaussianMixture(n_components=2, n_init=10)
gmm.fit(df_X)

print("using sklearn")
print("best pi : ", gmm.weights_)
print("best mu :", gmm.means_)


# def plot_densities(data, mu, sigma, alpha = 0.5, colors='grey'):
# grid_x, grid_y = np.mgrid[X[:,0].min():X[:,0].max():200j,
#  X[:,1].min():X[:,1].max():200j]
# grid = np.stack([grid_x, grid_y], axis=-1)
# for mu_c, sigma_c in zip(mu, sigma):
# plt.contour(grid_x, grid_y, multivariate_normal(mu_c, sigma_c).pdf(grid), colors=colors, alpha=alpha)


def plot_contours(data, means, covs):
    """visualize the gaussian components over the data"""
    plt.figure()
    plt.plot(data[:, 0], data[:, 1], 'ko')

    delta = 0.025
    k = means.shape[0]
    x = np.arange(-2.5, 10.0, delta)
    y = np.arange(-2.5, 10.0, delta)
    x_grid, y_grid = np.meshgrid(x, y)
    coordinates = np.array([x_grid.ravel(), y_grid.ravel()]).T

    col = ['green', 'red', 'indigo', 'yellow', 'blue']
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        z_grid = multivariate_normal(mean, cov).pdf(coordinates).reshape(x_grid.shape)
        plt.contour(x_grid, y_grid, z_grid, colors=col[i])
    plt.tight_layout()


print("check whether the best one converged: ", gmm.converged_)
print("how many steps to convergence: ", gmm.n_iter_)

plot_contours(df_X, gmm.means_, gmm.covariances_)
plt.xlabel("data[:, 0]", fontsize=12)
plt.ylabel("data[:, 1]", fontsize=12)
plt.show()


def E_step(data, pi, mu, sigma):
    N = data.shape[0]  # number of data-points
    K = pi.shape[0]  # number of clusters, following notation used before in description
    d = mu.shape[1]  # dimension of each data point, think of these as attributes
    zeta = np.zeros((N, K))  # this is basically responsibility which should be equal to posterior.

    for nk in range(K):
        zeta[:, nk] = pi[nk] * multivariate_normal.pdf(data, mean=mu[nk], cov=sigma[nk])
        # calculate responsibility for each cluster
    zeta = zeta / np.sum(zeta, axis=1, keepdims=True)
    # use the sum over all the clusters, thus axis=1. Denominator term.
    # print ("zeta shape: ", zeta.shape)
    return zeta


def M_step(data, zeta):
    N, D = data.shape
    K = zeta.shape[1]  # use the posterior shape calculated in E-step to determine the no. of clusters
    pi = np.zeros(K)
    mu = np.zeros((K, D))
    sigma = np.zeros((K, D, D))

    for ik in range(K):
        n_k = zeta[:, ik].sum()  # we use the definition of N_k
        pi[ik] = n_k / N  # definition of the weights
        elements = np.reshape(zeta[:, ik], (zeta.shape[0], 1))
        # get each columns and reshape it (K, 1) form so that later broadcasting is possible.
        mu[ik, :] = (np.multiply(elements, data)).sum(axis=0) / n_k
        sigma_sum = 0.
        for i in range(N):
            var = data[i] - mu[ik]
            sigma_sum = sigma_sum + zeta[i, ik] * np.outer(var, var)  # outer product creates the covariance matrix
        sigma[ik, :] = sigma_sum / n_k
    return pi, mu, sigma


def elbo(data, zeta, pi, mu, sigma):
    N = data.shape[0]  # no. of data-points
    K = zeta.shape[1]  # no. of clusters
    d = data.shape[1]  # dim. of each object

    l = 0.
    for i in range(N):
        x = data[i]
        for k in range(K):
            pos_dist = zeta[i, k]  # p(z_i=k|x) = zeta_ik
            log_lik = np.log(multivariate_normal.pdf(x, mean=mu[k, :], cov=sigma[k, :, :]) + 1e-20)  # log p(x|z)
            log_q = np.log(zeta[i, k] + 1e-20)  # log q(z) = log p(z_i=k|x)
            log_pz = np.log(pi[k] + 1e-20)  # log p(z_k =1) =\pi _k
            l = (l + np.multiply(pos_dist, log_pz) + np.multiply(pos_dist, log_lik) +
                 np.multiply(pos_dist, -log_q))
    # print ("check loss: ", loss)
    return l


def train_loop(data, K, tolerance=1e-3, max_iter=500, restart=50):
    N, d = data.shape
    elbo_best = -np.inf  # loss set to the lowest value
    pi_best = None
    mu_best = None
    sigma_best = None
    zeta_f = None
    for _ in range(restart):
        pi = np.ones(K) / K  # if 3 clusters then an array of [.33, .33, .33] # the sum of pi's should be one
        # that's why normalized
        mu = np.random.rand(K, d)  # no condition on
        sigma = np.tile(np.eye(d), (K, 1, 1))  # to initialize sigma we first start with ones only at the diagonals
        # the sigmas are postive semi-definite and symmetric
        last_iter_loss = None
        all_losses = []
        try:
            for i in range(max_iter):
                zeta = E_step(data, pi, mu, sigma)
                pi, mu, sigma = M_step(data, zeta)
                l = elbo(data, zeta, pi, mu, sigma)
                if l > elbo_best:
                    elbo_best = l
                    pi_best = pi
                    mu_best = mu
                    sigma_best = sigma
                    zeta_f = zeta
                if last_iter_loss and abs(
                        (l - last_iter_loss) / last_iter_loss) < tolerance:  # insignificant improvement
                    break
                last_iter_loss = l
                all_losses.append(l)
        except np.linalg.LinAlgError:  # avoid the delta function situation
            pass
    return elbo_best, pi_best, mu_best, sigma_best, all_losses, zeta_f


best_loss, pi_best, mu_best, sigma_best, ls_lst, final_posterior = train_loop(df_X, 5)
'''

############################################################################################################
# **********************************************************************************************************
#                                           HIDDEN MARKOV MODEL 1
# **********************************************************************************************************
# ##########################################################################################################

'''engine_df_A = df_A[df_A['unit'] == engine_unit]
X = df_X[engine_df_A.index[0]:engine_df_A.index[-1] + 1]


class ProbabilityVector:
    def __init__(self, probabilities: dict):
        states = probabilities.keys()
        probs = probabilities.values()

        assert len(states) == len(probs)
        "The probabilities must match the states."

        assert len(states) == len(set(states))
        "The states must be unique."

        assert abs(sum(probs) - 1.0) < 1e-12
        "Probabilities must sum up to 1."

        assert len(list(filter(lambda x: 0 <= x <= 1, probs))) == len(probs), \
            "Probabilities must be numbers from [0, 1] interval."

        self.states = sorted(probabilities)
        self.values = np.array(list(map(lambda x:
                                        probabilities[x], self.states))).reshape(1, -1)

    @classmethod
    def initialize(cls, states: list):
        size = len(states)
        rand = np.random.rand(size) / (size ** 2) + 1 / size
        rand /= rand.sum(axis=0)
        return cls(dict(zip(states, rand)))

    @classmethod
    def from_numpy(cls, array: np.ndarray, states: list):
        return cls(dict(zip(states, list(array))))

    @property
    def dict(self):
        return {k: v for k, v in zip(self.states, list(self.values.flatten()))}

    @property
    def df(self):
        return pd.DataFrame(self.values, columns=self.states, index=['probability'])

    def __repr__(self):
        return "P({}) = {}.".format(self.states, self.values)

    def __eq__(self, other):
        if not isinstance(other, ProbabilityVector):
            raise NotImplementedError
        if (self.states == other.states) and (self.values == other.values).all():
            return True
        return False

    def __getitem__(self, state: str) -> float:
        if state not in self.states:
            raise ValueError("Requesting unknown probability state from vector.")
        index = self.states.index(state)
        return float(self.values[0, index])

    def __mul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityVector):
            return self.values * other.values
        elif isinstance(other, (int, float)):
            return self.values * other
        else:
            NotImplementedError

    def __rmul__(self, other) -> np.ndarray:
        return self.__mul__(other)

    def __matmul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityMatrix):
            return self.values @ other.values

    def __truediv__(self, number) -> np.ndarray:
        if not isinstance(number, (int, float)):
            raise NotImplementedError
        x = self.values
        return x / number if number != 0 else x / (number + 1e-12)

    def argmax(self):
        index = self.values.argmax()
        return self.states[index]


class ProbabilityMatrix:
    def __init__(self, prob_vec_dict: dict):
        assert len(prob_vec_dict) > 1, \
            "The numebr of input probability vector must be greater than one."
        assert len(set([str(x.states) for x in prob_vec_dict.values()])) == 1, \
            "All internal states of all the vectors must be indentical."
        assert len(prob_vec_dict.keys()) == len(set(prob_vec_dict.keys())), \
            "All observables must be unique."

        self.states = sorted(prob_vec_dict)
        self.observables = prob_vec_dict[self.states[0]].states
        self.values = np.stack([prob_vec_dict[x].values \
                                for x in self.states]).squeeze()

    @classmethod
    def initialize(cls, states: list, observables: list):
        size = len(states)
        rand = np.random.rand(size, len(observables)) \
               / (size ** 2) + 1 / size
        rand /= rand.sum(axis=1).reshape(-1, 1)
        aggr = [dict(zip(observables, rand[i, :])) for i in range(len(states))]
        pvec = [ProbabilityVector(x) for x in aggr]
        return cls(dict(zip(states, pvec)))

    @classmethod
    def from_numpy(cls, array:
    np.ndarray,
                   states: list,
                   observables: list):
        p_vecs = [ProbabilityVector(dict(zip(observables, x)))
                  for x in array]
        return cls(dict(zip(states, p_vecs)))

    @property
    def dict(self):
        return self.df.to_dict()

    @property
    def df(self):
        return pd.DataFrame(self.values,
                            columns=self.observables, index=self.states)

    def __repr__(self):
        return "PM {} states: {} -> obs: {}.".format(
            self.values.shape, self.states, self.observables)

    def __getitem__(self, observable: str) -> np.ndarray:
        if observable not in self.observables:
            raise ValueError("Requesting unknown probability observable from the matrix.")
        index = self.observables.index(observable)
        return self.values[:, index].reshape(-1, 1)


from itertools import product
from functools import reduce


class HiddenMarkovChain:
    def __init__(self, T, E, pi):
        self.T = T  # transmission matrix A
        self.E = E  # emission matrix B
        self.pi = pi
        self.states = pi.states
        self.observables = E.observables

    def __repr__(self):
        return "HML states: {} -> observables: {}.".format(
            len(self.states), len(self.observables))

    @classmethod
    def initialize(cls, states: list, observables: list):
        T = ProbabilityMatrix.initialize(states, states)
        E = ProbabilityMatrix.initialize(states, observables)
        pi = ProbabilityVector.initialize(states)
        return cls(T, E, pi)

    def _create_all_chains(self, chain_length):
        return list(product(*(self.states,) * chain_length))

    def score(self, observations: list) -> float:
        def mul(x, y): return x * y

        score = 0
        all_chains = self._create_all_chains(len(observations))
        for idx, chain in enumerate(all_chains):
            expanded_chain = list(zip(chain, [self.T.states[0]] + list(chain)))
            expanded_obser = list(zip(observations, chain))

            p_observations = list(map(lambda x: self.E.df.loc[x[1], x[0]], expanded_obser))
            p_hidden_state = list(map(lambda x: self.T.df.loc[x[1], x[0]], expanded_chain))
            p_hidden_state[0] = self.pi[chain[0]]

            score += reduce(mul, p_observations) * reduce(mul, p_hidden_state)
        return score


class HiddenMarkovChain_FP(HiddenMarkovChain):
    def _alphas(self, observations: list) -> np.ndarray:
        alphas = np.zeros((len(observations), len(self.states)))
        alphas[0, :] = self.pi.values * self.E[observations[0]].T
        for t in range(1, len(observations)):
            alphas[t, :] = (alphas[t - 1, :].reshape(1, -1)
                            @ self.T.values) * self.E[observations[t]].T
        return alphas

    def score(self, observations: list) -> float:
        alphas = self._alphas(observations)
        return float(alphas[-1].sum())


class HiddenMarkovChain_Simulation(HiddenMarkovChain):
    def run(self, length: int) -> (list, list):
        assert length >= 0, "The chain needs to be a non-negative number."
        s_history = [0] * (length + 1)
        o_history = [0] * (length + 1)

        prb = self.pi.values
        obs = prb @ self.E.values
        s_history[0] = np.random.choice(self.states, p=prb.flatten())
        o_history[0] = np.random.choice(self.observables, p=obs.flatten())

        for t in range(1, length + 1):
            prb = prb @ self.T.values
            obs = prb @ self.E.values
            s_history[t] = np.random.choice(self.states, p=prb.flatten())
            o_history[t] = np.random.choice(self.observables, p=obs.flatten())

        return o_history, s_history


class HiddenMarkovChain_Uncover(HiddenMarkovChain_Simulation):
    def _alphas(self, observations: list) -> np.ndarray:
        alphas = np.zeros((len(observations), len(self.states)))
        alphas[0, :] = list(self.pi.values()) * self.E[observations[0]].T
        for t in range(1, len(observations)):
            alphas[t, :] = (alphas[t - 1, :].reshape(1, -1) @ self.T.values) \
                           * self.E[observations[t]].T
        return alphas

    def _betas(self, observations: list) -> np.ndarray:
        betas = np.zeros((len(observations), len(self.states)))
        betas[-1, :] = 1
        for t in range(len(observations) - 2, -1, -1):
            betas[t, :] = (self.T.values @ (self.E[observations[t + 1]] * betas[t + 1, :].reshape(-1, 1))).reshape(1,
                                                                                                                   -1)
        return betas

    def uncover(self, observations: list) -> list:
        alphas = self._alphas(observations)
        betas = self._betas(observations)
        maxargs = (alphas * betas).argmax(axis=1)
        return list(map(lambda x: self.states[x], maxargs))


class HiddenMarkovLayer(HiddenMarkovChain_Uncover):
    def _digammas(self, observations: list) -> np.ndarray:
        L, N = len(observations), len(self.states)
        digammas = np.zeros((L - 1, N, N))

        alphas = self._alphas(observations)
        betas = self._betas(observations)
        score = self.score(observations)
        for t in range(L - 1):
            P1 = (alphas[t, :].reshape(-1, 1) * self.T.values)
            P2 = self.E[observations[t + 1]].T * betas[t + 1].reshape(1, -1)
            digammas[t, :, :] = P1 * P2 / score
        return digammas


class HiddenMarkovModel:
    def __init__(self, hml: HiddenMarkovLayer):
        self.layer = hml
        self._score_init = 0
        self.score_history = []

    @classmethod
    def initialize(cls, states: list, observables: list):
        layer = HiddenMarkovLayer.initialize(states, observables)
        return cls(layer)

    def update(self, observations: list) -> float:
        alpha = self.layer._alphas(observations)
        beta = self.layer._betas(observations)
        digamma = self.layer._digammas(observations)
        score = alpha[-1].sum()
        gamma = alpha * beta / score

        L = len(alpha)
        obs_idx = [self.layer.observables.index(x) \
                   for x in observations]
        capture = np.zeros((L, len(self.layer.states), len(self.layer.observables)))
        for t in range(L):
            capture[t, :, obs_idx[t]] = 1.0

        pi = gamma[0]
        T = digamma.sum(axis=0) / gamma[:-1].sum(axis=0).reshape(-1, 1)
        E = (capture * gamma[:, :, np.newaxis]).sum(axis=0) / gamma.sum(axis=0).reshape(-1, 1)

        self.layer.pi = ProbabilityVector.from_numpy(pi, self.layer.states)
        self.layer.T = ProbabilityMatrix.from_numpy(T, self.layer.states, self.layer.states)
        self.layer.E = ProbabilityMatrix.from_numpy(E, self.layer.states, self.layer.observables)

        return score

    def train(self, observations: list, epochs: int, tol=None):
        self._score_init = 0
        self.score_history = (epochs + 1) * [0]
        early_stopping = isinstance(tol, (int, float))

        for epoch in range(1, epochs + 1):
            score = self.update(observations)
            print("Training... epoch = {} out of {}, score = {}.".format(epoch, epochs, score))
            if early_stopping and abs(self._score_init - score) / score < tol:
                print("Early stopping.")
                break
            self._score_init = score
            self.score_history[epoch] = score


# HI = ["0.", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1."]
HI = np.arange(0., 1.1, 0.1)
pi = {0.: 0, 0.1: 0, 0.2: 0, 0.3: 0, 0.4: 0, 0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0, 1.: 1}
model = HiddenMarkovModel.initialize(HI, df_X[0:100, 0])
model.layer.pi = pi
model.train(df_X[0:100, 0], epochs=100)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.semilogy(model.score_history)
ax.set_xlabel('Epoch')
ax.set_ylabel('Score')
ax.set_title('Training history')
plt.grid()
plt.show()'''

############################################################################################################
# **********************************************************************************************************
#                                           HIDDEN MARKOV MODEL 2
# **********************************************************************************************************
# ##########################################################################################################

'''def forward(V, a, b, initial_distribution):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, V[0]]

    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            # Matrix Computation Steps
            #                  ((1x2) . (1x2))      *     (1)
            #                        (1)            *     (1)
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]

    return alpha


def backward(V, a, b):
    beta = np.zeros((V.shape[0], a.shape[0]))

    # setting beta(T) = 1
    beta[V.shape[0] - 1] = np.ones((a.shape[0]))

    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])

    return beta


def baum_welch(V, a, b, initial_distribution, n_iter=100):
    M = a.shape[0]
    T = len(V)

    for n in range(n_iter):
        alpha = forward(V, a, b, initial_distribution)
        beta = backward(V, a, b)

        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
            for i in range(M):
                numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, V == l], axis=1)

        b = np.divide(b, denominator.reshape((-1, 1)))

    return (a, b)


def viterbi(V, a, b, initial_distribution):
    T = V.shape[0]
    M = a.shape[0]

    omega = np.zeros((T, M))
    omega[0, :] = np.log(initial_distribution * b[:, V[0]])

    prev = np.zeros((T - 1, M))

    for t in range(1, T):
        for j in range(M):
            # Same as Forward Probability
            probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, V[t]])

            # This is our most probable state given previous state at time t (1)
            prev[t - 1, j] = np.argmax(probability)

            # This is the probability of the most probable state (2)
            omega[t, j] = np.max(probability)

    # Path Array
    S = np.zeros(T)

    # Find the most probable last hidden state
    last_state = np.argmax(omega[T - 1, :])

    S[0] = last_state

    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1

    # Flip the path array since we were backtracking
    S = np.flip(S, axis=0)

    # Convert numeric values to actual hidden states
    result = []
    for s in S:
        if s == 0:
            result.append("A")
        else:
            result.append("B")

    return result


engine_df_A = df_A[df_A['unit'] == engine_unit]
V = df_X[engine_df_A.index[0]:engine_df_A.index[-1] + 1]


# Transition Probabilities
a = np.ones((30, 30))
a = a / np.sum(a, axis=1)

# Emission Probabilities
b = np.ones((V.shape[0], V.shape[1]))
b = b / np.sum(b, axis=1).reshape((-1, 1))

# Equal Probabilities for the initial distribution
initial_distribution = np.ones((V.shape[1], V.shape[0]))*0.5

a, b = baum_welch(V, a, b, initial_distribution, n_iter=1000)

predicted_observations = np.array((viterbi(V, a, b, initial_distribution)))

print(predicted_observations)'''

############################################################################################################
# **********************************************************************************************************
#                                 INPUT OUTPUT HIDDEN MARKOV MODEL (LIBRARY)
# **********************************************************************************************************
# ##########################################################################################################

import json
from IOHMM import UnSupervisedIOHMM
from IOHMM import OLS, CrossEntropyMNL
from hmmlearn import _hmmc
import logging

logging.getLogger().setLevel(logging.INFO)


def split_X_lengths(X, lengths):
    if lengths is None:
        return [X]
    else:
        cs = np.cumsum(lengths)
        n_samples = len(X)
        if cs[-1] > n_samples:
            raise ValueError("more than {} samples in lengths array {}"
                             .format(n_samples, lengths))
        elif cs[-1] != n_samples:
            warnings.warn(
                "less that {} samples in lengths array {}; support for "
                "silently dropping samples is deprecated and will be removed"
                    .format(n_samples, lengths),
                DeprecationWarning, stacklevel=3)
        return np.split(X, cs)[:-1]


############################################################################################################
#                                              NASA-CMAPSS
# ##########################################################################################################

''' num_states = 5

############################################### Training ###################################################

df[['W1', 'W2', 'W3', *range(5, 26)]] = minmax.fit_transform(pd.concat([df_W, df_S], axis=1))
cols_to_drop = df.nunique()[df.nunique() == 1].index
df = df.drop(cols_to_drop, axis=1)
cols_to_drop = df.nunique()[df.nunique() == 2].index
df = df.drop(cols_to_drop, axis=1)

outputs = [[m] for m in df.head() if type(m) == int]

df_obs = PCA(n_components=19).fit_transform(df[np.array(outputs).squeeze()])
df_obs = pd.DataFrame(df_obs)
df_input = PCA(n_components=1).fit_transform(df[['W1', 'W2']])
df_hmm = df_obs
df_hmm['W'] = df_input

outputs = [[m] for m in df_obs.head() if type(m) == int]

num_states = 15
SHMM = UnSupervisedIOHMM(num_states=num_states)
SHMM.set_models(model_emissions=[OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True)],
                model_transition=CrossEntropyMNL(solver='lbfgs'),
                model_initial=CrossEntropyMNL(solver='lbfgs'))
# We set operating conditions as the input covariate associate with the sensor output
SHMM.set_inputs(covariates_initial=[], covariates_transition=['W'],
                covariates_emissions=[['W']] * len(outputs))
SHMM.set_outputs(outputs)

lengths = [df_A[df_A['unit'] == i].cycle.max() for i in range(1, df_A['unit'].max() + 1)]
split_data = split_X_lengths(df_hmm, lengths)
SHMM.set_data(split_data)

SHMM.train()

json_dict = SHMM.to_json('../models/UnSupervisedIOHMM/')
with open('../models/UnSupervisedIOHMM/model.json', 'w') as outfile:
    json.dump(json_dict, outfile, indent=4, sort_keys=True)

# transmat = np.empty((num_states, num_states))
# for i in range(num_states):
#     transmat = np.concatenate((transmat, np.exp(SHMM.model_transition[i].predict_log_proba(np.array([[]])))))
# transmat = transmat[num_states:]
# startprob = SHMM.model_initial.coef.T'''

############################################### Loading ####################################################

with open('../models/UnSupervisedIOHMM/model.json') as json_file:
    json_dict = json.load(json_file)
SHMM = UnSupervisedIOHMM.from_json(json_dict)

##

lengths = [df_A[df_A['unit'] == i].cycle.max() for i in range(1, df_A['unit'].max() + 1)]
state_sequences = []
for i in range(df_A['unit'].max()):
    for j in range(lengths[i]):
        state_sequences.append(np.argmax(np.exp(SHMM.log_gammas[i])[j]))
pred = [state_sequences[df[df['unit'] == i].index[0]:df[df['unit'] == i].index[-1] + 1] for i in
        range(1, df_A['unit'].max() + 1)]
failure_states = [pred[i][-1] for i in range(df_A['unit'].max())]

HMM_out = [np.exp(SHMM.log_gammas[i]) for i in range(len(SHMM.log_gammas))]

hierarchical = []
h = []
failure = np.unique(failure_states)
for i in range(len(pred)):
    h = []
    for j in range(len(failure)):
        h.append(np.where(np.array(pred[i]) == failure[j]))
    h = np.squeeze(np.sort(np.concatenate(h, 1)))
    hierarchical.append(h)

'''def log_mask_zero(a):
    """
    Compute the log of input probabilities masking divide by zero in log.

    Notes
    -----
    During the M-step of EM-algorithm, very small intermediate start
    or transition probabilities could be normalized to zero, causing a
    *RuntimeWarning: divide by zero encountered in log*.

    This function masks this unharmful warning.
    """
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        return np.log(a)


def _do_viterbi_pass(framelogprob):
    n_samples, n_components = framelogprob.shape
    state_sequence, logprob = _hmmc._viterbi(n_samples, n_components, log_mask_zero(startprob),
                                             log_mask_zero(transmat), framelogprob)
    return logprob, state_sequence


def _decode_viterbi(X):
    framelogprob = SHMM.log_gammas[0]
    return _do_viterbi_pass(framelogprob)


def decode(X, lengths=None):
    decoder = {"viterbi": _decode_viterbi}["viterbi"]
    logprob = 0
    sub_state_sequences = []
    for sub_X in split_X_lengths(X, lengths):
        # XXX decoder works on a single sample at a time!
        sub_logprob, sub_state_sequence = decoder(sub_X)
        logprob += sub_logprob
        sub_state_sequences.append(sub_state_sequence)
    return logprob, np.concatenate(sub_state_sequences)


def predict(X, lengths=None):
    """
    Find most likely state sequence corresponding to ``X``.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix of individual samples.
    lengths : array-like of integers, shape (n_sequences, ), optional
        Lengths of the individual sequences in ``X``. The sum of
        these should be ``n_samples``.

    Returns
    -------
    state_sequence : array, shape (n_samples, )
        Labels for each sample from ``X``.
    """
    _, state_sequence = decode(X, lengths)
    return state_sequence


state_seq = predict(df)'''

############################################################################################################
#                                            HYDRAULIC SYSTEM
# ##########################################################################################################

'''num_states = 5

############################################### Training ###################################################

df_X = minmax.fit_transform(df_X)
df_obs = PCA(n_components=17).fit_transform(df_X)
df_obs = pd.DataFrame(df_obs)
df_hmm = df_obs

outputs = [[m] for m in df_obs.head()]

SHMM = UnSupervisedIOHMM(num_states=num_states)
SHMM.set_models(model_emissions=[OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True)],
                model_transition=CrossEntropyMNL(solver='lbfgs'),
                model_initial=CrossEntropyMNL(solver='lbfgs'))
# We set operating conditions as the input covariate associate with the sensor output
SHMM.set_inputs(covariates_initial=[], covariates_transition=[],
                covariates_emissions=[[]] * len(outputs))
SHMM.set_outputs(outputs)
SHMM.set_data([df_hmm])
SHMM.train()

json_dict = SHMM.to_json('../models/HydraulicUnSupervisedIOHMM/' + str(num_states) + '/')
with open('../models/HydraulicUnSupervisedIOHMM/' + str(num_states) + '/model.json', 'w') as outfile:
    json.dump(json_dict, outfile, indent=4, sort_keys=True)

# transmat = np.empty((num_states, num_states))
# for i in range(num_states):
#     transmat = np.concatenate((transmat, np.exp(SHMM.model_transition[i].predict_log_proba(np.array([[]])))))
# transmat = transmat[num_states:]
# startprob = SHMM.model_initial.coef.T

############################################### Loading ####################################################

with open('../models/HydraulicUnSupervisedIOHMM/' + str(num_states) + '/model.json') as json_file:
    json_dict = json.load(json_file)
SHMM = UnSupervisedIOHMM.from_json(json_dict)

##

state_sequences = np.argmax(np.exp(SHMM.log_gammas[0]), 1)
pred = state_sequences
HMM_out = np.exp(SHMM.log_gammas)

plt.figure(1)
plt.plot(pred)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
plt.show()

plt.figure(2)
pca = PCA(n_components=2).fit_transform(df_X)
for class_value in range(num_states):
    row_ix = np.where(pred == class_value)
    plt.scatter(pca[row_ix, 0], pca[row_ix, 1])
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.legend(list(range(0, num_states)), title='HMM states')
plt.show()

df_Y['HMM_state_pred'] = pred

##
df_cond = df_Y.query("Cooler_condition==3 & \
                      Valve_condition!=73 & \
                      Internal_pump_leakage!=2 & \
                      Hydraulic_accumulator!=90 & \
                      stable_flag!=1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='cooler condition = close to total failure')

##
df_cond = df_Y.query("Cooler_condition!=3 & \
                      Valve_condition==73 & \
                      Internal_pump_leakage!=2 & \
                      Hydraulic_accumulator!=90 & \
                      stable_flag!=1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='Valve condition = close to total failure')

##
df_cond = df_Y.query("Cooler_condition!=3 & \
                      Valve_condition!=73 & \
                      Internal_pump_leakage==2 & \
                      Hydraulic_accumulator!=90 & \
                      stable_flag!=1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='Internal pump leakage = severe leakage')

##
df_cond = df_Y.query("Cooler_condition!=3 & \
                      Valve_condition!=73 & \
                      Internal_pump_leakage!=2 & \
                      Hydraulic_accumulator==90 & \
                      stable_flag!=1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='Hydraulic accumulator = close to total failure')

##
df_cond = df_Y.query("stable_flag==1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='stable flag = static conditions might not have been reached yet')'''

'''def log_mask_zero(a):
    """
    Compute the log of input probabilities masking divide by zero in log.

    Notes
    -----
    During the M-step of EM-algorithm, very small intermediate start
    or transition probabilities could be normalized to zero, causing a
    *RuntimeWarning: divide by zero encountered in log*.

    This function masks this unharmful warning.
    """
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        return np.log(a)


def _do_viterbi_pass(framelogprob):
    n_samples, n_components = framelogprob.shape
    state_sequence, logprob = _hmmc._viterbi(n_samples, n_components, log_mask_zero(startprob),
                                             log_mask_zero(transmat), framelogprob)
    return logprob, state_sequence


def _decode_viterbi(X):
    framelogprob = SHMM.log_gammas[0]
    return _do_viterbi_pass(framelogprob)


def decode(X, lengths=None):
    decoder = {"viterbi": _decode_viterbi}["viterbi"]
    logprob = 0
    sub_state_sequences = []
    for sub_X in split_X_lengths(X, lengths):
        # XXX decoder works on a single sample at a time!
        sub_logprob, sub_state_sequence = decoder(sub_X)
        logprob += sub_logprob
        sub_state_sequences.append(sub_state_sequence)
    return logprob, np.concatenate(sub_state_sequences)


def predict(X, lengths=None):
    """
    Find most likely state sequence corresponding to ``X``.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix of individual samples.
    lengths : array-like of integers, shape (n_sequences, ), optional
        Lengths of the individual sequences in ``X``. The sum of
        these should be ``n_samples``.

    Returns
    -------
    state_sequence : array, shape (n_samples, )
        Labels for each sample from ``X``.
    """
    _, state_sequence = decode(X, lengths)
    return state_sequence


state_seq = predict(df)'''

############################################################################################################
#                                                YOKOGAWA
# ##########################################################################################################

'''num_states = 5

############################################### Training ###################################################

df_X = minmax.fit_transform(df_X)
df_obs = PCA(n_components=17).fit_transform(df_X)
df_obs = pd.DataFrame(df_obs)
df_hmm = df_obs

outputs = [[m] for m in df_obs.head()]

SHMM = UnSupervisedIOHMM(num_states=num_states)
SHMM.set_models(model_emissions=[OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True)],
                model_transition=CrossEntropyMNL(solver='lbfgs'),
                model_initial=CrossEntropyMNL(solver='lbfgs'))
# We set operating conditions as the input covariate associate with the sensor output
SHMM.set_inputs(covariates_initial=[], covariates_transition=[],
                covariates_emissions=[[]] * len(outputs))
SHMM.set_outputs(outputs)
SHMM.set_data([df_hmm])
SHMM.train()

json_dict = SHMM.to_json('../models/HydraulicUnSupervisedIOHMM/' + str(num_states) + '/')
with open('../models/HydraulicUnSupervisedIOHMM/' + str(num_states) + '/model.json', 'w') as outfile:
    json.dump(json_dict, outfile, indent=4, sort_keys=True)

# transmat = np.empty((num_states, num_states))
# for i in range(num_states):
#     transmat = np.concatenate((transmat, np.exp(SHMM.model_transition[i].predict_log_proba(np.array([[]])))))
# transmat = transmat[num_states:]
# startprob = SHMM.model_initial.coef.T

############################################### Loading ####################################################

with open('../models/HydraulicUnSupervisedIOHMM/' + str(num_states) + '/model.json') as json_file:
    json_dict = json.load(json_file)
SHMM = UnSupervisedIOHMM.from_json(json_dict)

##

state_sequences = np.argmax(np.exp(SHMM.log_gammas[0]), 1)
pred = state_sequences
HMM_out = np.exp(SHMM.log_gammas)

plt.figure(1)
plt.plot(pred)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
plt.show()

plt.figure(2)
pca = PCA(n_components=2).fit_transform(df_X)
for class_value in range(num_states):
    row_ix = np.where(pred == class_value)
    plt.scatter(pca[row_ix, 0], pca[row_ix, 1])
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.legend(list(range(0, num_states)), title='HMM states')
plt.show()

df_Y['HMM_state_pred'] = pred

##
df_cond = df_Y.query("Cooler_condition==3 & \
                      Valve_condition!=73 & \
                      Internal_pump_leakage!=2 & \
                      Hydraulic_accumulator!=90 & \
                      stable_flag!=1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='cooler condition = close to total failure')

##
df_cond = df_Y.query("Cooler_condition!=3 & \
                      Valve_condition==73 & \
                      Internal_pump_leakage!=2 & \
                      Hydraulic_accumulator!=90 & \
                      stable_flag!=1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='Valve condition = close to total failure')

##
df_cond = df_Y.query("Cooler_condition!=3 & \
                      Valve_condition!=73 & \
                      Internal_pump_leakage==2 & \
                      Hydraulic_accumulator!=90 & \
                      stable_flag!=1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='Internal pump leakage = severe leakage')

##
df_cond = df_Y.query("Cooler_condition!=3 & \
                      Valve_condition!=73 & \
                      Internal_pump_leakage!=2 & \
                      Hydraulic_accumulator==90 & \
                      stable_flag!=1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='Hydraulic accumulator = close to total failure')

##
df_cond = df_Y.query("stable_flag==1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='stable flag = static conditions might not have been reached yet')'''

'''def log_mask_zero(a):
    """
    Compute the log of input probabilities masking divide by zero in log.

    Notes
    -----
    During the M-step of EM-algorithm, very small intermediate start
    or transition probabilities could be normalized to zero, causing a
    *RuntimeWarning: divide by zero encountered in log*.

    This function masks this unharmful warning.
    """
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        return np.log(a)


def _do_viterbi_pass(framelogprob):
    n_samples, n_components = framelogprob.shape
    state_sequence, logprob = _hmmc._viterbi(n_samples, n_components, log_mask_zero(startprob),
                                             log_mask_zero(transmat), framelogprob)
    return logprob, state_sequence


def _decode_viterbi(X):
    framelogprob = SHMM.log_gammas[0]
    return _do_viterbi_pass(framelogprob)


def decode(X, lengths=None):
    decoder = {"viterbi": _decode_viterbi}["viterbi"]
    logprob = 0
    sub_state_sequences = []
    for sub_X in split_X_lengths(X, lengths):
        # XXX decoder works on a single sample at a time!
        sub_logprob, sub_state_sequence = decoder(sub_X)
        logprob += sub_logprob
        sub_state_sequences.append(sub_state_sequence)
    return logprob, np.concatenate(sub_state_sequences)


def predict(X, lengths=None):
    """
    Find most likely state sequence corresponding to ``X``.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix of individual samples.
    lengths : array-like of integers, shape (n_sequences, ), optional
        Lengths of the individual sequences in ``X``. The sum of
        these should be ``n_samples``.

    Returns
    -------
    state_sequence : array, shape (n_samples, )
        Labels for each sample from ``X``.
    """
    _, state_sequence = decode(X, lengths)
    return state_sequence


state_seq = predict(df)'''

############################################################################################################
# **********************************************************************************************************
#                                         ENVIRONMENT MODELING
# **********************************************************************************************************
# ##########################################################################################################

c_f = -1000
c_r = -100
do_nothing = 0
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

HMM_out.insert(0, [])
for i in range(1, len(HMM_out)):
    HMM_out[i] = np.insert(HMM_out[i], 0, np.zeros((1, 15)), axis=0)

hierarchical.insert(0, [])
for i in range(1, len(hierarchical)):
    hierarchical[i] = np.insert(hierarchical[i], 0, 1000, axis=0)


class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, is_training=True, verbose=False):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(df_X.shape[1],))
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(HMM_out[1].shape[1],))
        self.action_space = gym.spaces.Discrete(2)
        self.reward = 0
        self.cycle = 1
        self.done = False
        self.engine_unit = engine_unit
        self.engine_df_A = df_A[df_A['unit'] == self.engine_unit]
        self.X = df_X[self.engine_df_A.index[0]:self.engine_df_A.index[-1] + 1]
        self.X = HMM_out[self.engine_unit]
        self.state = np.expand_dims(self.X[self.cycle], axis=0)
        self.state = self.X[self.cycle]
        self.failure_state = self.engine_df_A['cycle'].max() - 1
        self.train = is_training
        self.verbose = verbose

    def get_next_engine_data(self):
        self.engine_unit += 1
        if self.train:
            if self.engine_unit > int((df_A['unit'].max() * 0.8)):
                self.engine_unit = 1
        else:
            if self.engine_unit > df_A['unit'].max():
                self.engine_unit = int((df_A['unit'].max() * 0.8) + 1)
        if self.verbose:
            print("********|engine unit|********:", self.engine_unit)
        self.engine_df_A = df_A[df_A['unit'] == self.engine_unit]
        self.X = df_X[self.engine_df_A.index[0]:self.engine_df_A.index[-1] + 1]
        self.X = HMM_out[self.engine_unit]
        self.failure_state = self.engine_df_A['cycle'].max() - 1
        return self.X

    def step(self, action):
        if action == 0:
            if self.verbose:
                print("|hold|:", self.cycle)
            if self.cycle == self.failure_state:
                self.reward = (c_r + c_f) / self.cycle
                self.state = np.expand_dims(self.X[self.cycle], axis=0)
                self.state = self.X[self.cycle]
                self.done = True
                if self.train:
                    policy[self.engine_unit] = {'unit': self.engine_unit,
                                                'failure_state': self.failure_state,
                                                'replace_state': self.cycle}
                else:
                    policy_test[self.engine_unit] = {'unit': self.engine_unit,
                                                     'failure_state': self.failure_state,
                                                     'replace_state': self.cycle,
                                                     'reward': self.reward}
                if self.verbose:
                    print("|cycle reached failure state|:", self.cycle, "reward:", self.reward, '\n')
            else:
                self.reward = do_nothing
                self.cycle += 1
                self.state = np.expand_dims(self.X[self.cycle], axis=0)
                self.state = self.X[self.cycle]
                self.done = False
                if self.verbose:
                    print("|system running|", "reward:", self.reward, '\n')
                '''if self.cycle in hierarchical[self.engine_unit]:
                    self.state = np.expand_dims(self.X[self.cycle], axis=0)
                    # self.state = self.X[self.cycle]
                    self.done = False
                    if self.verbose:
                        print("|system running|", "reward:", self.reward, '\n')
                else:
                    pass'''
        elif action == 1:
            if self.verbose:
                print("|replace|:", self.cycle)
            if self.cycle == self.failure_state:
                self.reward = (c_r + c_f) / self.cycle
            else:
                self.reward = c_r / (self.cycle + 0.1)
            self.state = np.expand_dims(self.X[self.cycle], axis=0)
            self.state = self.X[self.cycle]
            if self.train:
                policy[self.engine_unit] = {'unit': self.engine_unit,
                                            'failure_state': self.failure_state,
                                            'replace_state': self.cycle}
            else:
                policy_test[self.engine_unit] = {'unit': self.engine_unit,
                                                 'failure_state': self.failure_state,
                                                 'replace_state': self.cycle,
                                                 'reward': self.reward}
            self.done = True
        if self.verbose:
            print("reward:", self.reward, '\n')
        info = {}
        return self.state, self.reward, self.done, info

    def reset(self):
        self.X = self.get_next_engine_data()
        self.cycle = 1
        self.state = np.expand_dims(self.X[self.cycle], axis=0)
        self.state = self.X[self.cycle]
        self.done = False
        return self.state


env = CustomEnv()

############################################################################################################
# **********************************************************************************************************
#                                   DEEP REINFORCEMENT LEARNING ARCHITECTURE
# **********************************************************************************************************
# ##########################################################################################################


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
#                                        Approximate (Deep) Q Learning
# ##########################################################################################################

'''n_actions = env.action_space.n
state_dim = env.observation_space.shape

tf.reset_default_graph()
sess = tf.InteractiveSession()
keras.backend.set_session(sess)

############################################## DNN #########################################################

network = keras.models.Sequential()
network.add(keras.layers.InputLayer(state_dim))
# let's create a network for approximate q-learning following guidelines above
network.add(keras.layers.Dense(128, activation='relu'))
network.add(keras.layers.Dense(256, activation='relu'))
network.add(keras.layers.Dense(n_actions, activation='linear'))

############################################## RNN #########################################################

network = keras.models.Sequential()
network.add(keras.layers.LSTM(128, return_sequences=True, input_shape=[None, state_dim[0]]))
network.add(keras.layers.LSTM(256))
network.add(keras.layers.Dense(n_actions))

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


states_ph = tf.placeholder('float32', shape=(None, None) + state_dim)
actions_ph = tf.placeholder('int32', shape=[None])
rewards_ph = tf.placeholder('float32', shape=[None])
next_states_ph = tf.placeholder('float32', shape=(None, None) + state_dim)
is_done_ph = tf.placeholder('bool', shape=[None])

# get q-values for all actions in current states
predicted_qvalues = network(states_ph)

# select q-values for chosen actions
predicted_qvalues_for_actions = tf.reduce_sum(predicted_qvalues * tf.one_hot(actions_ph, n_actions), axis=1)

gamma = 0.95
alpha = 1e-4

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
train_step = tf.train.AdamOptimizer(alpha).minimize(loss)


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

    return total_reward'''

############################################################################################################
#                                           Double Deep Q Learning
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
            torch.nn.Linear(self.N, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.M)
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
        return self.epsilon

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
        epsilon = agent.decay_epsilon(t / T)

        if t % 100 == 0:
            mean_return.append(np.mean(returns))
            print('epoch: ', t, '    mean_return: ', '{:.2f}'.format(round(np.average(returns), 3)), '    epsilon: ',
                  epsilon)

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
#                                       Double Deep Q Learning (PER)
# ##########################################################################################################

import os
import random
import gym
import pylab
import numpy as np
from collections import deque
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Add
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K


class SumTree(object):
    data_pointer = 0

    # Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    def __init__(self, capacity):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema below
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

    # Here we define function that will add our priority score in the sumtree leaf and add the experience in data:
    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  # If we're above the capacity, we go back to first index (we overwrite)
            self.data_pointer = 0

    # Update the leaf priority score and propagate the change through tree
    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        # this method is faster than the recursive loop in the reference code
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    # Here build a function to get a leaf from our tree. So we'll build a function to get the leaf_index,
    # priority value of that leaf and experience associated with that leaf index:
    def get_leaf(self, v):
        parent_index = 0

        # the while loop is faster than the method in the reference code
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node


# Now we finished constructing our SumTree object, next we'll build a memory object.
class Memory(object):  # stored as ( state, action, reward, next_state ) in SumTree
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree
        self.tree = SumTree(capacity)

    # Next, we define a function to store a new experience in our tree.
    # Each new experience will have a score of max_prority (it will be then improved when we use this exp to train our DDQN).
    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this experience will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)  # set the max priority for new priority

    # Now we create sample function, which will be used to pick batch from our tree memory, which will be used to train our model.
    # - First, we sample a minibatch of n size, the range [0, priority_total] into priority ranges.
    # - Then a value is uniformly sampled from each range.
    # - Then we search in the sumtree, for the experience where priority score correspond to sample values are retrieved from.
    def sample(self, n):
        # Create a minibatch array that will contains the minibatch
        minibatch = []

        b_idx = np.empty((n,), dtype=np.int32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n  # priority segment

        for i in range(n):
            # A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # Experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)

            b_idx[i] = index

            minibatch.append([data[0], data[1], data[2], data[3], data[4]])

        return b_idx, minibatch

    # Update the priorities on the tree
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


def OurModel(input_shape, action_space, dueling):
    X_input = Input(input_shape)
    X = X_input

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X)

    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    if dueling:
        state_value = Dense(1, kernel_initializer='he_uniform')(X)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value)

        action_advantage = Dense(action_space, kernel_initializer='he_uniform')(X)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(
            action_advantage)

        X = Add()([state_value, action_advantage])
    else:
        # Output Layer with # of actions: 2 nodes (left, right)
        X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X, name='BenchMark_Maintenance')
    model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
                  metrics=["accuracy"])

    model.summary()
    return model


class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.EPISODES = 4000
        memory_size = 1000
        self.MEMORY = Memory(memory_size)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate

        # EXPLORATION HYPERPARAMETERS for epsilon and epsilon greedy strategy
        self.epsilon = 1.0  # exploration probability at start
        self.epsilon_min = 0.01  # minimum exploration probability
        self.epsilon_decay = 0.0005  # exponential decay rate for exploration prob

        self.batch_size = 32

        # defining model parameters
        self.ddqn = True  # use doudle deep q network
        self.Soft_Update = False  # use soft parameter update
        self.dueling = False  # use dealing netowrk
        self.epsilot_greedy = False  # use epsilon greedy strategy
        self.USE_PER = True

        self.TAU = 0.1  # target network soft update hyperparameter

        self.Save_Path = 'Models_D3QN-PER'
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.scores, self.episodes, self.average = [], [], []

        self.Model_name = os.path.join(self.Save_Path, 'hmm_out.h5')

        # create main model and target model
        self.model = OurModel(input_shape=(self.state_size,), action_space=self.action_size, dueling=self.dueling)
        self.target_model = OurModel(input_shape=(self.state_size,), action_space=self.action_size,
                                     dueling=self.dueling)

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        if not self.Soft_Update and self.ddqn:
            self.target_model.set_weights(self.model.get_weights())
            return
        if self.Soft_Update and self.ddqn:
            q_model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1 - self.TAU) + q_weight * self.TAU
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_model.set_weights(target_model_theta)

    def remember(self, state, action, reward, next_state, done):
        experience = state, action, reward, next_state, done
        if self.USE_PER:
            self.MEMORY.store(experience)
        else:
            self.memory.append((experience))

    def act(self, state, decay_step):
        # EPSILON GREEDY STRATEGY
        if self.epsilot_greedy:
            # Here we'll use an improved version of our epsilon greedy strategy for Q-learning
            explore_probability = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(
                -self.epsilon_decay * decay_step)
        # OLD EPSILON STRATEGY
        else:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= (1 - self.epsilon_decay)
            explore_probability = self.epsilon

        if explore_probability > np.random.rand():
            # Make a random action (exploration)
            return random.randrange(self.action_size), explore_probability
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            # Take the biggest Q value (= the best action)
            return np.argmax(self.model.predict(state)), explore_probability

    def replay(self):
        if self.USE_PER:
            tree_idx, minibatch = self.MEMORY.sample(self.batch_size)
        else:
            minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        # predict Q-values for starting state using the main network
        target = self.model.predict(state)
        target_old = np.array(target)
        # predict best action in ending state using the main network
        target_next = self.model.predict(next_state)
        # predict Q-values for ending state using the target network
        target_val = self.target_model.predict(next_state)

        for i in range(len(minibatch)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                if self.ddqn:  # Double - DQN
                    # current Q Network selects the action
                    # a'_max = argmax_a' Q(s', a')
                    a = np.argmax(target_next[i])
                    # target Q Network evaluates the action
                    # Q_max = Q_target(s', a'_max)
                    target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])
                else:  # Standard - DQN
                    # DQN chooses the max Q value among next actions
                    # selection and evaluation of action is on the target Q Network
                    # Q_max = max_a' Q_target(s', a')
                    target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        if self.USE_PER:
            indices = np.arange(self.batch_size, dtype=np.int32)
            absolute_errors = np.abs(target_old[indices, np.array(action)] - target[indices, np.array(action)])
            # Update priority
            self.MEMORY.batch_update(tree_idx, absolute_errors)

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)

    pylab.figure(figsize=(18, 9))

    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        pylab.plot(self.episodes, self.average, 'r')
        pylab.plot(self.episodes, self.scores, 'b')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Steps', fontsize=18)
        dqn = 'DQN_'
        softupdate = ''
        dueling = ''
        greedy = ''
        PER = ''
        if self.ddqn: dqn = 'DDQN_'
        if self.Soft_Update: softupdate = '_soft'
        if self.dueling: dueling = '_Dueling'
        if self.epsilot_greedy: greedy = '_Greedy'
        if self.USE_PER: PER = '_PER'
        try:
            pylab.savefig(dqn + str(self.env) + softupdate + dueling + greedy + PER + ".png")
        except OSError:
            pass

        return str(self.average[-1])[:5]

    def run(self):
        decay_step = 0
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                # self.env.render()
                decay_step += 1
                action, explore_probability = self.act(state, decay_step)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    # every step update target model
                    self.update_target_model()

                    # every episode, plot the result
                    average = self.PlotModel(i, e)

                    print("episode: {}/{}, replacement @: {}, e: {:.2}, average: {}".format(e, self.EPISODES, i,
                                                                                            explore_probability, average))
                    # print("Saving trained model to", self.Model_name)
                    self.save(self.Model_name)
                    break
                self.replay()
        self.env.close()

    def test(self, env):
        self.load(self.Model_name)
        self.env = env
        for e in range(20):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                # self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, replacement @: {}".format(e, self.EPISODES, i))
                    break


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

'''import time

initial_epsilon = 0.5
epsilon_decay = 0.99
epsilon = initial_epsilon
total_returns = []

start_time = time.time()
for i in range(1000):
    session_rewards = [generate_session(epsilon=epsilon, train=True) for _ in range(int(df_A['unit'].max() * 0.8))]
    print("epoch #{}\tmean reward = {:.3f}\tepsilon = {:.3f}".format(i, np.mean(session_rewards), epsilon))
    total_returns.append(np.mean(session_rewards))
    epsilon *= epsilon_decay
    # Make sure epsilon is always nonzero during training
    if epsilon <= 1e-4:
        break

print("Training Time:")
print("--- %s seconds ---" % (time.time() - start_time))
plt.plot(total_returns)
plt.show()'''

# ************************ Double Deep Q Learning ****************************

'''env = VectorizedEnvWrapper(env, num_envs=64)
agent = DeepQLearner(env, alpha=1e-4, gamma=0.95)
replay_buffer = ReplayBuffer(batch_size=8)
agent = train(env, agent, replay_buffer, T=100000)'''

# ******************** Double Deep Q Learning (library) **********************

'''model = DQN(LnMlpPolicy, env, verbose=1, tensorboard_log="./dqn_nasa_tensorboard/").learn(total_timesteps=10000)
# tensorboard --logdir ./dqn_nasa_tensorboard/
model.save("deepq_nasa")'''

# ********************** Double Deep Q Learning (PER) ************************

agent = DQNAgent(env)
agent.run()

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

'''engine_unit = int((df_A['unit'].max() * 0.8) + 1)
env = CustomEnv(is_training=False)

start_time = time.time()
session_rewards = [generate_session(epsilon=0, train=False) for _ in range(int(df_A['unit'].max() * 0.2))]
print("Inference Time:")
print("--- %s seconds ---" % (time.time() - start_time))

total_returns_test = np.mean(session_rewards)
print("mean reward = ", total_returns_test)

policy = pd.DataFrame.from_dict(policy).T
policy['remaining_cycles'] = policy['failure_state'] - policy['replace_state']
policy_test = pd.DataFrame.from_dict(policy_test).T
policy_test['remaining_cycles'] = policy_test['failure_state'] - policy_test['replace_state']

M = int(df_A['unit'].max() * 0.2)
IMC = (M * c_r) / np.sum(policy_test['failure_state'] - 1)
CMC = (M * (c_r + c_f)) / np.sum(policy_test['failure_state'])
print("Average remaining cycles", np.average(policy_test['remaining_cycles']))

np.savetxt(r'./output/sensor_data_train_HMM' + ' ' + dataset + ' ' + str(IMC) + ' ' + str(CMC) + ' ' + str(alpha) + ' '
           + str(gamma) + ' ' + str(initial_epsilon) + ' ' + str(epsilon_decay) + ' ' + str(c_r) + ' ' + str(c_f) + ' '
           + str(do_nothing) + ' ' + str(network.count_params()) + ' ' + str(num_states) + ' '
           + str(np.round(np.mean(total_returns_test), 2))
           + '.txt', policy.values, fmt='%d')
np.savetxt(r'./output/sensor_data_test_HMM' + ' ' + dataset + ' ' + str(IMC) + ' ' + str(CMC) + ' ' + str(alpha) + ' '
           + str(gamma) + ' ' + str(initial_epsilon) + ' ' + str(epsilon_decay) + ' ' + str(c_r) + ' ' + str(c_f) + ' '
           + str(do_nothing) + ' ' + str(network.count_params()) + ' ' + str(num_states) + ' '
           + str(np.round(np.mean(total_returns_test), 2))
           + '.txt', policy_test.values, fmt='%d')

env.close()'''

# ************************** Double Deep Q Learning **************************

'''engine_unit = int((df_A['unit'].max() * 0.8) + 1)
env = CustomEnv(is_training=False)
env = VectorizedEnvWrapper(env, num_envs=1)
# agent = DeepQLearner(env, alpha=1e-3, gamma=0.95)
episodes = int(df_A['unit'].max() * 0.2)
episode_return = []
s_t = env.reset()

for _ in range(episodes):
    d_t = False
    r_t = 0
    total_return = []
    while not d_t:
        a_t = agent.act(s_t)
        s_t_next, r_t, d_t = env.step(a_t)
        s_t = s_t_next
        total_return.append(r_t)
        # store data into replay buffer
        # replay_buffer.remember(s_t, a_t, r_t, s_t_next, d_t)
        # s_t = s_t_next
        # learn by sampling from replay buffer
        # for batch in replay_buffer.sample():
        # agent.update(*batch)

policy = pd.DataFrame.from_dict(policy).T
policy['remaining_cycles'] = policy['failure_state'] - policy['replace_state']
policy_test = pd.DataFrame.from_dict(policy_test).T
policy_test['remaining_cycles'] = policy_test['failure_state'] - policy_test['replace_state']

M = int(df_A['unit'].max() * 0.2)
IMC = (M * c_r) / np.sum(policy_test['failure_state'] - 1)
CMC = (M * (c_r + c_f)) / np.sum(policy_test['failure_state'])
print("mean reward = ", np.mean(policy_test['reward']))
print("Average remaining cycles", np.average(policy_test['remaining_cycles']))'''

# ******************** Double Deep Q Learning (library) **********************

'''env.reset()
for obs in HI:
    obs = np.array([np.array(obs).astype(np.float32)])
    action, _states = model.predict(obs)
    print("action:", action)
    state, reward, done, info = env.step(action)
    print("HI:", state, "G:", reward, "is_done:", done, '\n')
    print('####################################################', '\n')'''

# ********************** Double Deep Q Learning (PER) ************************

engine_unit = int((df_A['unit'].max() * 0.8) + 1)
env = CustomEnv(is_training=False)
agent.test(env)

policy = pd.DataFrame.from_dict(policy).T
policy['remaining_cycles'] = policy['failure_state'] - policy['replace_state']
policy_test = pd.DataFrame.from_dict(policy_test).T
policy_test['remaining_cycles'] = policy_test['failure_state'] - policy_test['replace_state']

M = int(df_A['unit'].max() * 0.2)
IMC = (M * c_r) / np.sum(policy_test['failure_state'] - 1)
CMC = (M * (c_r + c_f)) / np.sum(policy_test['failure_state'])
print("mean reward = ", np.mean(policy_test['reward']))
print("Average remaining cycles", np.average(policy_test['remaining_cycles']))

env.close()

print("done")
