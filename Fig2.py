# This script reproduces the results of Fig. 2 in the manuscript 

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
from scipy.io import loadmat
import matplotlib.pyplot as plt
from src import deployment,waveforms,channels,arrays,optim

# =========================================
# simulation parameters
# =========================================
S = 4                          # num. sensing APUs
C = 4                           # num. comm. APUs
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
seed = 1                        # random seed for reproducibility
iter = 50                       # ADMM iterations

# ADMM parameters
mu = .1/18
beta = 100
alpha = .01

# =========================================
# scene reconstruction
# =========================================

# APU positions
posAPUs, interAPUSpacing = deployment.computeDeploymentAPUs(LRoom,M,Delta,S+C)

# subcarrier generation
subCarriersFreq, subCarrierWavelength = waveforms.subCarriersGen(fc,df)

# generate the grid of points (discrete space)
posPoints,radarCrossSection = deployment.computeMeasGrid(I**2,1,LRoom,L)

# define the roles of the APUs "1" -> comm. "0" -> sensing
roleAPUs = deployment.generateRoleAPUs(S,C)

reconstSceneList = []
primalResidualList = []
perUserRateList = []

idxCommAPUs = np.flatnonzero(roleAPUs[0] == 1)

for D in DList:
    # equal power allocation among APUs and devices
    powPerAPUPerDev = Ptx/(C*D)

    # device positions
    posDevs = deployment.computeDevsDeployment(D,LRoom,seed)

    # allocate the first D subcarriers
    freqPerDev = subCarriersFreq[:D]

    # devs channel
    HDev = channels.devsChannelCoeffs(posDevs,freqPerDev,posAPUs,LRoom,Delta,M,idxCommAPUs)

    # comm. rate under coherent combining
    SNR = (powPerAPUPerDev/noisePow2)*(np.sum(np.linalg.norm(HDev, axis=2), axis=1)**2)

    # compute rate
    perUserRate = df*np.log2(1 + SNR)

    # compute precoders
    precoders = arrays.computePrecoders(posAPUs,posDevs,M,LRoom,Delta,freqPerDev,powPerAPUPerDev,idxCommAPUs)

    # channel matrix
    H, tau = channels.pointsChannelCoeffs(posPoints,freqPerDev,posAPUs,LRoom,Delta,M,roleAPUs[0],radarCrossSection)

    # transmitted symbols
    qamSymbols = waveforms.gen16QAM(D,freqPerDev.size)
    ofdmSymbols = waveforms.genOFDMSym(precoders,qamSymbols)
    
    # ADMM opt. 
    reconstScene, primalResidual = optim.admmOptim(beta,mu,alpha,roleAPUs[0],freqPerDev,posAPUs,posPoints,LRoom,Delta,tau,ofdmSymbols,noisePow2,H,iter)

    perUserRateList.append(perUserRate)
    reconstSceneList.append(reconstScene.reshape(I,I))
    primalResidualList.append(primalResidual)

    print(f"[DONE ] seed={seed}, D={D}", flush=True)


# =========================================
# plot results
# =========================================
originalScene = radarCrossSection.reshape(I,I)/np.max(radarCrossSection.reshape(I,I))
sceneU6 = reconstSceneList[0]/np.max(reconstSceneList[0])
sceneU22 = reconstSceneList[1]/np.max(reconstSceneList[1])

fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

scenes = [originalScene, sceneU6, sceneU22]
labels = ["(a)", "(b)", "(c)"]

for ax, scene, label in zip(axes, scenes, labels):

    ax.imshow(
        scene,
        extent=[0, 60, 0, 60],
        origin="lower",
        aspect="equal"
    )

    ax.set_title(label, fontsize=14)

    ax.set_xlabel(r"$x\,[\mathrm{m}]$", fontsize=14)
    ax.set_ylabel(r"$y\,[\mathrm{m}]$", fontsize=14)

    ax.tick_params(labelsize=12)

plt.show()