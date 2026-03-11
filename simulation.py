import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from src import deployment,arrays,channels,optim,waveforms,utils
import matplotlib.pyplot as plt

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

U = C*5                          # num. real users per RU
Umax = 5 # max. num. devs per comm. APU

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

txPow = np.ones(U)        # power allocation vector
noisePow2 = 1e-6                      # noise power
bw = 1                          # bandwidth [Hz]

# ADMM parameters
mu = .1/18
beta = 100
alpha = .01

# APU positions (OK)
posAPUs, interAPUSpacing = deployment.computeDeploymentAPUs(LRoom,M,Delta,S+C)

# define the roles of the APUs "1" -> comm. "0" -> sensing
roleAPUs = np.tile(np.array([0, 0, 1, 0, 0]),C)

# compute the antenna positions of each APU
posAnts = deployment.computeAntPositions(posAPUs,M,Delta,LRoom)

# device positions
posDevs, idxAPU2Dev = deployment.computeDevsDeployment(posAPUs,U,Umax,roleAPUs,LRoom,8)

# generate K subCarriers with central freq. fc and separation df
subCarriersFreq, subCarrierWavelength = waveforms.subCarriersGen(fc,K,df)

# subCarriers are allocated in a continuous manner to each comm. APU
# the number of subCarriers allocated to each APU is Umax
# the bw blocks are assigned to each comm. APU is order of apearance, i.e., 1st
# block to the first APU in the roleAPUs vector and so on.
subCarrierBlockPerAPU = subCarriersFreq.size // C
startSubCarrierPerAPU = np.arange(C)*subCarrierBlockPerAPU

idxCommAPUs = np.flatnonzero(roleAPUs == 1)

subCarrierFreqAlloc = np.zeros(K)
freqPerDev = np.zeros(U)
for i, apu in enumerate(idxCommAPUs):
    # devs served by this APU
    idxUsersThisAPU = np.where(idxAPU2Dev == apu)[0]
    numDevsThisAPU = len(idxUsersThisAPU)

    # subcarriers belonging to this APU
    idxSubCarriers = startSubCarrierPerAPU[i] + np.arange(Umax)
    subCarrierFreqAlloc[np.arange(Umax) + Umax*i] = subCarriersFreq[idxSubCarriers]

    # subcarrier assignment to each user in the APU
    for j, dev in enumerate(idxUsersThisAPU):
        freqPerDev[dev] = subCarriersFreq[idxSubCarriers[j]]

# compute precoders
steeringVectors, precoders = arrays.computePrecoders(posAPUs,posDevs,idxAPU2Dev,M,LRoom,Delta,freqPerDev)

# generate the grid of points (discrete space)
posPoints,radarCrossSection = deployment.computeMeasGrid(I**2,1,LRoom,L,1)

# utils.displayScenario(LRoom,posAPUs,posDevs,posAnts,S+C,posPoints,roleAPUs,radarCrossSection,idxAPU2Dev)

# channel matrix
H, tau = channels.pointsChannelCoeffs(posPoints,freqPerDev,idxAPU2Dev,U,posAPUs,LRoom,Delta,M,roleAPUs,radarCrossSection)

# transmitted symbols
qamSymbols = waveforms.gen16QAM(N,freqPerDev.size,seed=None)
ofdmSymbols = waveforms.genOFDMSym(precoders,txPow,freqPerDev.size,qamSymbols,M,N)

# compute rate
sumRate = 0
perAPURate = np.zeros(C)
perUserRate = np.zeros(U)

# devs channel
HDev = channels.devsChannelCoeffs(posDevs,freqPerDev,posAPUs,LRoom,Delta,M,idxAPU2Dev)

for dev, apu in enumerate(idxAPU2Dev):
    # per use rate
    perUserRate[dev] = bw*np.log2(1 + 10*np.linalg.norm(HDev[dev,:])**2/(noisePow2*bw))

    # per APU rate
    perAPURate[np.where(idxCommAPUs == apu)[0][0]] += perUserRate[dev]

    # sum rate
    sumRate += perUserRate[dev]

print(
    f"Worst per user rate: {np.min(perUserRate):.4f}, "
    f"Worst per APU rate: {np.min(perAPURate):.4f}, "
    f"Sum rate: {sumRate:.4f}"
)

# ADMM opt. loop
radarCrossSectionEst = optim.admmOptim(beta,mu,alpha,roleAPUs,freqPerDev,idxAPU2Dev,posAPUs,posPoints,LRoom,Delta,tau,ofdmSymbols,noisePow2,H,0)

scene = radarCrossSection.reshape(I,I)
scene = scene/np.max(scene)

reconst = radarCrossSectionEst.reshape(I,I)
reconst = reconst/np.max(reconst)

mssim, _ = utils.ssim(scene,reconst)
print(mssim)

# Create two subplots and unpack the output array immediately
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(scene, aspect="auto")
ax2.imshow(reconst, aspect="auto")

plt.show()
