import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from src import deployment,arrays,channels,optim,waveforms,utils

# %load_ext autoreload
# %autoreload 2

# [R1] A. Murtada, R. Hu, B. S. M. R. Rao and U. Schroeder, "Widely Distributed
# Radar Imaging: Unmediated ADMM Based Approach," in IEEE Journal of Selected
# Topics in Signal Processing, vol. 17, no. 2, pp. 389-402, March 2023, doi:
# 10.1109/JSTSP.2022.3210766.

# simulation settings 
S = 16                          # num. sensing antenna processing units (APUs)
C = 4                           # num. comm. APUs = resource units (RUs)
M = 4                           # num. antennas per APU 

Uc = C*4                          # num. real users per RU
Uv = 0                          # num. virtual users per RU
U = Uc + Uv                     # num. devices per RU
sectorRad = np.array([10, 30])   #
sectorAngle = 15*np.pi/180      # sector angle opening 5 degrees
Umax = 4 # max. num. devs per comm. APU

K = Umax*C                         # num. subcarriers
N = 20                          # num. OFDM symbols
fc = 6E9                        # central frequency [Hz]
lambdac = constants.c/fc        # wavelength @fc [m]
df = 120E3                      # inter-subcarrier spacing [Hz]
Delta = lambdac/2               # inter-antenna spacing [m]

I = 20                          # num. points (sq. root, discretized space)
delta = ...                     # discretized space resulution
LRoom = 60*4                    # perimeter of the square room [m] 
L = 10                           # number of scatterers

txPow = np.ones((U*C,1))        # power allocation vector
sigma2 = 1                      # noise power

# ADMM parameters
mu = .1/18
beta = 100
alpha = .01

# APU positions (OK)
posAPUs, interAPUSpacing = deployment.computeDeploymentAPUs(LRoom,M,Delta,S+C)

# define the roles of the APUs "1" -> comm. "0" -> sensing
roleAPUs = np.tile(np.array([0, 1, 0, 0, 0]),C)

# compute the antenna positions of each APU
posAnts = deployment.computeAntPositions(posAPUs,M,Delta,LRoom)

# device positions
posDevs, idxAPU2Dev = deployment.computeDevsDeployment(posAPUs,U,4,roleAPUs,LRoom)

# generate K subCarriers with central freq. fc and separation df
subCarriersFreq, subCarrierWavelength = waveforms.subCarriersGen(fc,K,df)

# assign Umax subCarriers to each of comm. APU
subCarrierBlockPerAPU = subCarriersFreq.size // C
startSubCarrierPerAPU = np.arange(C) * subCarrierBlockPerAPU

subCarrierFreqAlloc = np.zeros((K,1))
for i in range(C):
    idxSubCarriers = startSubCarrierPerAPU[i] + np.arange(Umax)

    subCarrierFreqAlloc[np.arange(Umax) + Umax*i] = subCarriersFreq[idxSubCarriers]

# compute precoders (not OK)
steeringVectors, precoders = arrays.computePrecoders(posAPUs,posDevs,idxAPU2Dev,M,LRoom,Delta,subCarrierFreqAlloc,roleAPUs)

# generate the grid of points (discrete space)
posPoints,radarCrossSection = deployment.computeMeasGrid(I**2,1,LRoom,L,1)

utils.displayScenario(LRoom,posAPUs,posDevs,posAnts,S+C,posPoints,roleAPUs,radarCrossSection,idxAPU2Dev)

# channel matrix
H, tau = channels.computeChannelCoeffs(posPoints,subCarrierFreqAlloc,Uc,posAPUs,LRoom,Delta,M,roleAPUs,radarCrossSection)

# transmitted symbols
qamSymbols = waveforms.gen16QAM(N,subCarrierFreqAlloc.size,seed=None)
ofdmSymbols = waveforms.genOFDMSym(precoders,txPow,subCarrierFreqAlloc.size,qamSymbols,M,N)

# ADMM opt. loop
radarCrossSectionEst = optim.admmOptim(beta,mu,alpha,roleAPUs,Uc,subCarrierFreqAlloc,posAPUs,posPoints,LRoom,Delta,tau,ofdmSymbols,sigma2,H)

# # Create two subplots and unpack the output array immediately
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(radarCrossSection.reshape((I,I)), aspect="auto")
ax2.imshow(radarCrossSectionEst.reshape((I,I)), aspect="auto")

plt.show()