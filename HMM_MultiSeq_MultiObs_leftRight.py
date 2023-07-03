############################################################################################################
#                                           IMPORTING LIBRARIES
# ##########################################################################################################

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from hmmlearn import hmm


standard = StandardScaler()
minmax = MinMaxScaler()

############################################################################################################
#                                           DATASET PREPARATION
# ##########################################################################################################

dir_path = os.getcwd()
dataset = 'train_FD001.txt'
df = pd.read_csv(dir_path + r'/CMAPSSData/' + dataset, sep=" ", header=None, skipinitialspace=True).dropna(axis=1)
df = df.rename(columns={0: 'unit', 1: 'cycle', 2: 'W1', 3: 'W2', 4: 'W3'})
df_A = df[df.columns[[0, 1]]]
df_S = df[df.columns[list(range(5, 26))]]


############################################################################################################
# **********************************************************************************************************
#                                       HIDDEN MARKOV MODEL (LIBRARY)
# **********************************************************************************************************
# ##########################################################################################################


df_hmm = minmax.fit_transform(df_S)

df_hmm = pd.DataFrame(df_hmm)
cols_to_drop = df_hmm.nunique()[df_hmm.nunique() == 1].index
df_hmm = df_hmm.drop(cols_to_drop, axis=1)
cols_to_drop = df_hmm.nunique()[df_hmm.nunique() == 2].index
df_hmm = df_hmm.drop(cols_to_drop, axis=1).to_numpy()

lengths = [df[df['unit'] == i].cycle.max() for i in range(1, df_A['unit'].max() + 1)]

num_states = 15
remodel = hmm.GaussianHMM(n_components=num_states,
                          n_iter=500,
                          verbose=True,
                          init_params="cm", params="cmt")

transmat = np.zeros((num_states, num_states))

# Left-to-right: each state is connected to itself and its direct successor.
for i in range(num_states):
    if i == num_states - 1:
        transmat[i, i] = 1.0
    else:
        transmat[i, i] = transmat[i, i + 1] = 0.5

# Always start in first state
startprob = np.zeros(num_states)
startprob[0] = 1.0

remodel.startprob_ = startprob
remodel.transmat_ = transmat
remodel = remodel.fit(df_hmm, lengths)

state_seq = remodel.predict(df_hmm)
pred = [state_seq[df[df['unit'] == i].index[0]:df[df['unit'] == i].index[-1] + 1] for i in
        range(1, df_A['unit'].max() + 1)]


plt.figure(0)
plt.plot(pred[0])
plt.xlabel('# Flights')
plt.ylabel('HMM states')

plt.figure(1)
plt.plot(pred[1])
plt.xlabel('# Flights')
plt.ylabel('HMM states')

plt.figure(2)
plt.plot(pred[2])
plt.xlabel('# Flights')
plt.ylabel('HMM states')

plt.show()