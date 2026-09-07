# This script reproduces the results of Fig. 5 in the manuscript 

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
MList = np.arange(2,18,2)       # num. antennas per APU 
D = 20                          # num. users
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
MCIter = 1000                   # Monte Carlo iterations

# ADMM parameters
mu = .1/18
beta = 100
alpha = .01

# =========================================
# scene reconstruction
# =========================================

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
        "sumRate": np.full((len(MList),MCIter,roleAPUs.shape[0]), np.nan),
        "reconstScene": np.full((len(MList),MCIter,roleAPUs.shape[0],I,I), np.nan),
        "primalResidual": np.full((len(MList),MCIter,roleAPUs.shape[0]), np.nan),
    }

    for roleIdx, role in enumerate(roleAPUs):
        idxCommAPUs = np.flatnonzero(role == 1)

        for MIdx, M in enumerate(MList):
            # APU positions
            posAPUs, interAPUSpacing = deployment.computeDeploymentAPUs(LRoom,M,Delta,nAPUs)

            for seed in np.arange(MCIter):
                # equal power allocation among APUs and devices
                powPerAPUPerDev = Ptx/(C[i]*D)

                # device positions
                posDevices = deployment.computeDevsDeployment(D,LRoom,seed)

                # allocate the first D subcarriers
                freqPerDev = subCarriersFreq[:D]
            
                # devs channel
                HDev = channels.devsChannelCoeffs(posDevices,freqPerDev,posAPUs,LRoom,Delta,M,idxCommAPUs)
            
                # comm. rate under coherent combining
                SNR = (powPerAPUPerDev/noisePow2)*(np.sum(np.linalg.norm(HDev, axis=2), axis=1)**2)
            
                # compute rate
                perUserRate = df*np.log2(1 + SNR)

                results[(S[i],C[i])]["sumRate"][MIdx,seed,roleIdx] = np.sum(perUserRate)
                
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

                results[(S[i],C[i])]["reconstScene"][MIdx,seed,roleIdx,:,:] = reconstScene.reshape(I,I)
                results[(S[i],C[i])]["primalResidual"][MIdx,seed,roleIdx] = primalResidual

# =========================================
# Precision versus number of antennas
# =========================================

# plot settings
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 15,
        "legend.fontsize": 15,
    }
)

avgPrecisionVSAntennas = np.zeros((len(MList), nAPUs-1))

precisionPerSeed = np.zeros((len(MList), MCIter))
zScene = radarCrossSection.reshape(I,I)

for i in np.arange(0,nAPUs-1):
    for MIdx, M in enumerate(MList):
        for seed in np.arange(MCIter):
            sceneToFuse = results[(S[i],C[i])]["reconstScene"][MIdx,seed,:,:,:]
            primalRes = results[(S[i],C[i])]["primalResidual"][MIdx,seed,:]

            # fusing the images 
            zFused = utils.fusion(sceneToFuse,primalRes) 

            # compute precision
            avgPrecisionVSAntennas[MIdx,i] += (1/MCIter)*utils.estQuality(zFused,zScene)

fig1, ax1 = plt.subplots(num=1, figsize=(8, 6), clear=True)

ax1.plot(MList, avgPrecisionVSAntennas[:,0], linestyle="-", marker="o", markersize=6,
         color="black", linewidth=1.5, label=r"$C=1,\ S=7$")
ax1.plot(MList, avgPrecisionVSAntennas[:,1], linestyle="-", marker="+", markersize=6,
         color="#D95319", linewidth=1.5, label=r"$C=2,\ S=6$")
ax1.plot(MList, avgPrecisionVSAntennas[:,2], linestyle="-", marker="*", markersize=6,
         color="#EDB120", linewidth=1.5, label=r"$C=3,\ S=5$")
ax1.plot(MList, avgPrecisionVSAntennas[:,3], linestyle="-", marker="x", markersize=6,
         color="#7E2F8E", linewidth=1.5, label=r"$C=4,\ S=4$")
ax1.plot(MList, avgPrecisionVSAntennas[:,4], linestyle="-", marker="^", markersize=6,
         color="#77AC30", linewidth=1.5, label=r"$C=5,\ S=3$")
ax1.plot(MList, avgPrecisionVSAntennas[:,5], linestyle="-", marker="s", markersize=6,
         color="#A2142F", linewidth=1.5, label=r"$C=6,\ S=2$")
ax1.plot(MList, avgPrecisionVSAntennas[:,6], linestyle="-", marker="d", markersize=6,
         color="#0072BD", linewidth=1.5, label=r"$C=7,\ S=1$")

ax1.grid(True)
ax1.set_axisbelow(True)
ax1.set_xticks(MList)
ax1.set_xlabel(r"$M$")
ax1.set_ylabel("average precision")
ax1.legend(loc="best")

fig1.tight_layout()

plt.show()