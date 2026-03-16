import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from src import deployment,waveforms,utils,core
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pickle
import os
import argparse
import time

# %load_ext autoreload
# %autoreload 2

# [R1] A. Murtada, R. Hu, B. S. M. R. Rao and U. Schroeder, "Widely Distributed
# Radar Imaging: Unmediated ADMM Based Approach," in IEEE Journal of Selected
# Topics in Signal Processing, vol. 17, no. 2, pp. 389-402, March 2023, doi:
# 10.1109/JSTSP.2022.3210766.

# simulation settings 
S = 4                          # num. sensing antenna processing units (APUs)
C = 4                           # num. comm. APUs = resource units (RUs)
M = 4                           # num. antennas per APU 

UList = C*np.arange(1,11)                          # num. real users per RU
Umax = 10 # max. num. devs per comm. APU

K = Umax*C                         # num. subcarriers
N = 1                          # num. OFDM symbols
fc = 6E9                        # central frequency [Hz]
lambdac = constants.c/fc        # wavelength @fc [m]
df = 120E3                      # inter-subcarrier spacing [Hz]
Delta = lambdac/2               # inter-antenna spacing [m]

I = 20                          # num. points (sq. root, discretized space)
delta = ...                     # discretized space resulution
LRoom = 60*4                    # perimeter of the square room [m] 
L = 10                           # number of scatterers

txPow = 1        # power allocation [W]
noisePow2 = 1e-6                      # noise power
bw = 120E3                          # bandwidth [Hz]

# ADMM parameters
mu = .1/18
beta = 100
alpha = .01

# APU positions (OK)
posAPUs, interAPUSpacing = deployment.computeDeploymentAPUs(LRoom,M,Delta,S+C)

# generate K subCarriers with central freq. fc and separation df
subCarriersFreq, subCarrierWavelength = waveforms.subCarriersGen(fc,K,df)

# generate the grid of points (discrete space)
posPoints,radarCrossSection = deployment.computeMeasGrid(I**2,1,LRoom,L,0)

# define the roles of the APUs "1" -> comm. "0" -> sensing
# roleAPUs = np.tile(np.array([0, 0, 1, 0, 0]),C)
roleAPUs = deployment.generateRoleAPUs(S//4,C//4)

# # compute the antenna positions of each APU
# posAnts = deployment.computeAntPositions(posAPUs,M,Delta,LRoom)


# create one Parallel instance (so the pool is reused)
nJobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
taskId = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
parallel = Parallel(n_jobs=nJobs, verbose=5)

print(f"[task {taskId}] Using {nJobs} threads")

cases = [(U, seed) for U in UList for seed in np.arange(1000)]

rawResults = parallel(
    delayed(core.simulation)(
        txPow,Delta,M,LRoom,Umax,U,posAPUs,posPoints,radarCrossSection,
        subCarriersFreq,roleAPUs,beta,mu,alpha,noisePow2,bw,N,seed
    )
    for U, seed in cases
)

results = [
    {
        "U": U,
        "seed": seed,
        "reconstSceneList": output[0],
        "primalResidualList": output[1],
        "rateList": output[2],
    }
    for (U, seed), output in zip(cases, rawResults)
]

filename = f"results/results_{taskId}.pkl"
with open(filename, 'wb') as f:
    pickle.dump({'results': results, 
                 'params': {
                     'C': C,
                     'S': S, 
                     'UList': UList,
                     'Umax': Umax,
                     'K': K,
                     'N': N,
                     'I': I,
                     'fc': fc,
                     'Delta': Delta,
                     'LRoom': LRoom,
                     'L': L,
                     'bw': bw,
                     'mu': mu,
                     'beta': beta,
                     'alpha': alpha,
                     'df': df,
                     'txPow': txPow,
                     'noisePow2': noisePow2,
                     'M': M,
                 }
                 }, f)

# resultsBySeed = []
# for seed in np.arange(1000):
#     reconstSceneList, primalResidualList, rateList = core.simulation(
#         txPow,Delta,M,LRoom,Umax,U,posAPUs,posPoints,radarCrossSection,
#         subCarriersFreq,roleAPUs,beta,mu,alpha,noisePow2,bw,N,seed)
        
#     resultsBySeed.append({
#         "seed": seed,
#         "reconstSceneList": reconstSceneList,
#         "primalResidualList": primalResidualList,
#         "rateList": rateList,
#     })


# zFusedWeighted, zFusedVote, zFusedLikelihood = utils.fusion(reconstSceneList,primalResidualList)

# scene = radarCrossSection.reshape(I,I)
# scene = scene/np.max(scene)

# zFusedWeighted = zFusedWeighted/np.max(zFusedWeighted)
# supportEst = zFusedVote > 0
# zFusedLikelihood = zFusedLikelihood/np.max(zFusedLikelihood)
# supportTruth = scene > 0

# tp = np.sum(supportTruth & supportEst)
# fp = np.sum(~supportTruth & supportEst)
# fn = np.sum(supportTruth & ~supportEst)
# tn = np.sum(~supportTruth & ~supportEst)

# precision = tp/(tp + fp + 1e-12)
# recall = tp/(tp + fn + 1e-12)
# f1 = 2 * precision*recall/(precision + recall + 1e-12)
# iou = tp / (tp + fp + fn + 1e-12)

# precision = tp / (tp + fp + 1e-12)
# print(precision)

# mssim, _ = utils.ssim(scene,reconst)
# print(mssim)

# # Create two subplots and unpack the output array immediately
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
# ax1.imshow(scene, aspect="auto")
# ax2.imshow(zFusedWeighted, aspect="auto")
# ax3.imshow(supportEst, aspect="auto")
# ax4.imshow(zFusedLikelihood, aspect="auto")

# plt.show()
