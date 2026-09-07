# This script reproduces the results of Fig. 4 in the manuscript 

# O. Martínez Rosabal, O. L. A. López,
# "Radio Stripe-Based Distributed ISAC System with Dynamic Sensing-Communication
# Reconfiguration", arXiv preprint arXiv:2604.08982, 2026. (To be published in
# the proceedings of the 2026 IEEE International Symposium on Personal, Indoor
# and Mobile Radio Communications (PIMRC'26).)

# Version: 1.0 
# Last modified: 2026-09-07

# License: This code is licensed under MIT License. See the LICENSE file in the repository root.

# If you use this code in research resulting in a publication, please cite
# the paper above.

import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from src import deployment,waveforms,utils,channels,arrays,optim

# =========================================
# simulation parameters
# =========================================
nAPUs = 8                       # num. APUs
C = np.arange(1,nAPUs)          # num. comm. APUs
S = nAPUs - C                   # num. sensing APUs
M = 8                           # num. antennas per APU 
DList = np.array([6, 22])       # num. users
K = 64                          # num. subcarriers
N = 1                           # num. OFDM symbols
fc = 5.955E9                    # carrier frequency [Hz]
lambdac = constants.c/fc        # wavelength @fc [m]
df = 312.5E3                    # inter-subcarrier spacing [Hz]
Delta = lambdac/2               # inter-antenna spacing [m]
I = 20                          # num. points (sq. root, discretized space)
LRoom = 60*4                    # perimeter of the square room [m] 
L = 10                          # number of scatterers
Ptx = 1                         # total power budget [W]
noisePow2 = 1e-6                # effective noise power
iter = 50                       # ADMM iterations
MCIter = 1000                  # Monte Carlo iterations

# ADMM parameters
mu = .1/18
beta = 100
alpha = .01

# =========================================
# scene reconstruction
# =========================================

# APU positions
posAPUs, interAPUSpacing = deployment.computeDeploymentAPUs(LRoom,M,Delta,nAPUs)

# subcarrier generation
subCarriersFreq, subCarrierWavelength = waveforms.subCarriersGen(fc,df)

# generate the grid of points (discrete space)
posPoints,radarCrossSection = deployment.computeMeasGrid(I**2,1,LRoom,L)

# dict to store results
results = {}

for i in np.arange(0,nAPUs-1):
    # define the roles of the APUs "1" -> comm. "0" -> sensing
    roleAPUs = deployment.generateRoleAPUs(S[i],C[i])

    results[(S[i],C[i])] = {
        "sumRate": np.full((len(DList),MCIter,roleAPUs.shape[0]), np.nan),
        "reconstScene": np.full((len(DList),MCIter,roleAPUs.shape[0],I,I), np.nan),
        "primalResidual": np.full((len(DList),MCIter,roleAPUs.shape[0]), np.nan),
    }

    for roleIdx, role in enumerate(roleAPUs):
        idxCommAPUs = np.flatnonzero(role == 1)

        for DIdx, D in enumerate(DList):
            for seed in np.arange(MCIter):
                # equal power allocation among APUs and devices
                powPerAPUPerDev = Ptx/(C[i]*D)

                # device positions
                posDevices = deployment.computeDevsDeployment(D,LRoom)

                # allocate the first D subcarriers
                freqPerDev = subCarriersFreq[:D]
            
                # devs channel
                HDev = channels.devsChannelCoeffs(posDevices,freqPerDev,posAPUs,LRoom,Delta,M,idxCommAPUs)
            
                # comm. rate under coherent combining
                SNR = (powPerAPUPerDev/noisePow2)*(np.sum(np.linalg.norm(HDev, axis=2), axis=1)**2)
            
                # compute rate
                perUserRate = df*np.log2(1 + SNR)

                results[(S[i],C[i])]["sumRate"][DIdx,seed,roleIdx] = np.sum(perUserRate)
                
                # compute precoders
                precoders = arrays.computePrecoders(posAPUs,posDevices,M,LRoom,Delta,freqPerDev,powPerAPUPerDev,idxCommAPUs)
                    
                # channel matrix
                H, tau = channels.pointsChannelCoeffs(posPoints,freqPerDev,posAPUs,LRoom,Delta,M,role,radarCrossSection)
            
                # transmitted symbols
                qamSymbols = waveforms.gen16QAM(D,freqPerDev.size)
                ofdmSymbols = waveforms.genOFDMSym(precoders,qamSymbols)
                
                # ADMM opt. 
                reconstScene, primalResidual = optim.admmOptim(beta,mu,alpha,role,freqPerDev,posAPUs,
                                                                posPoints,LRoom,Delta,tau,ofdmSymbols,noisePow2,H,iter)

                results[(S[i],C[i])]["reconstScene"][DIdx,seed,roleIdx,:,:] = reconstScene.reshape(I,I)
                results[(S[i],C[i])]["primalResidual"][DIdx,seed,roleIdx] = primalResidual

# =========================================
# Pareto region
# =========================================

# plot settings
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 15,
        "legend.fontsize": 15,
    }
)

zScene = radarCrossSection.reshape(I,I)

DIdx = DList.index(12)

paretoResults = {}

for i in range(nAPUs-1):
    key = (S[i], C[i])

    nConfigs = results[key]["sumRate"].shape[2]

    avgPrecision = np.zeros(nConfigs)
    avgSumRate = np.zeros(nConfigs)

    for configIdx in range(nConfigs):
        for seed in range(MCIter):
            zSceneEst = results[key]["reconstScene"][DIdx,seed,configIdx,:,:]
            primalRes = results[key]["primalResidual"][DIdx,seed,configIdx]

            # compute precision
            avgPrecision[configIdx] += (1/MCIter)*utils.estQuality(zSceneEst,zScene)

            # compute average sum rate
            avgSumRate[configIdx] += (1/MCIter)*results[key]["sumRate"][DIdx,seed,configIdx]

        paretoResults[key] = {
            "avgPrecision": avgPrecision,
            "avgSumRate": avgSumRate,
        }

fig1, ax1 = plt.subplots(num=2, figsize=(8, 6), clear=True)

ax1.scatter(paretoResults[(7, 1)]["sumRate"], paretoResults[(7, 1)]["precision"],
            marker="o", s=40, color="black", label=r"$C=1,\ S=7$")

ax1.scatter(paretoResults[(6, 2)]["sumRate"], paretoResults[(6, 2)]["precision"],
            marker="+", s=50, color="#D95319", label=r"$C=2,\ S=6$")

ax1.scatter(paretoResults[(5, 3)]["sumRate"], paretoResults[(5, 3)]["precision"],
            marker="*", s=50, color="#EDB120", label=r"$C=3,\ S=5$")

ax1.scatter(paretoResults[(4, 4)]["sumRate"], paretoResults[(4, 4)]["precision"],
            marker="x", s=50, color="#7E2F8E", label=r"$C=4,\ S=4$")

ax1.scatter(paretoResults[(3, 5)]["sumRate"], paretoResults[(3, 5)]["precision"],
            marker="^", s=40, color="#77AC30", label=r"$C=5,\ S=3$")

ax1.scatter(paretoResults[(2, 6)]["sumRate"], paretoResults[(2, 6)]["precision"],
            marker="s", s=40, color="#A2142F", label=r"$C=6,\ S=2$")

ax1.scatter(paretoResults[(1, 7)]["sumRate"], paretoResults[(1, 7)]["precision"],
            marker="d", s=40, color="#0072BD", label=r"$C=7,\ S=1$")

ax1.grid(True)
ax1.set_axisbelow(True)
ax1.set_xlabel("average sum rate [Mbit/s]")
ax1.set_ylabel("average precision")
ax1.legend(loc="best")

fig1.tight_layout()

plt.show()