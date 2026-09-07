"""
Functions used for the simulations presented in:

O. Martínez Rosabal and O. L. A. López,
"Radio Stripe-Based Distributed ISAC System with Dynamic Sensing-Communication
Reconfiguration," arXiv:2604.08982, 2026.

Version: 1.0
Last modified: 2026-09-07

License: MIT
See the LICENSE file in the repository root.

If you use this code in research resulting in a publication, please cite
the paper above.
"""

import numpy as np
from scipy import constants
from .arrays import computeSteeringVecs

def softThresholding(inputVec,threshold):
    """
        Implements the soft thresholding operator used in the ADMM optimization algorithm
    
        Args
        -------------
        inputVec : input vector to be thresholded
        threshold : threshold value

        Returns
        -------------
        outputVec : thresholded vector
        """
    return np.sign(inputVec)*np.maximum(np.abs(inputVec) - threshold, 0.0)

def vectSignal(roleAPUs,freqPerDev,posAPUs,posPoints,LRoom,Delta,tau,ofdmSymbols,noisePow2,H,seed=0):
    """
    Vectorization step of the received signal in the form of a compressed
    sensing problem

    Args
    -------------
    roleAPUs : roles of the APUs "1" communication and "0" sensing (S+C,1)
    freqPerDev : subcarriers frequency (D,1), [Hz]
    posAPUs : 2D positions of the APUs (S+C,2), [m]
    posPoints : 2D Cartesian coordinates of the grid points (I**2,2), [m]
    LRoom : perimeter of the room [m]
    Delta : inter-antenna spacing [m]
    tau : bistatic channel delay (S,D,I**2)
    ofdmSymbols : transmitted OFDM symbols (M*D,C)
    noisePow2 : noise power [W]
    H : channel matrix points (S,D,M,M)
    seed : seed for the random number generator (noise)

    Returns
    -------------
    PhiSenAPUs : sensing matrix APUs (M*D,I**2,S)
    YSenAPU : vectorized meas. signal per APU (M*D,S)
    """

    # num. devices 
    D = freqPerDev.size

    # num. points
    numPoints = posPoints.shape[0]

    # num. commun. APUs
    idxCommAPUs = np.flatnonzero(roleAPUs == 1)
    C = idxCommAPUs.size

    # num. sensing APUs
    idxSenAPUs = np.flatnonzero(roleAPUs == 0)
    S = idxSenAPUs.size

    # num. antennas per APU
    CM,_ = ofdmSymbols.shape 
    M = CM // C

    # seed of the random number generator for reproducibility
    rng = np.random.default_rng(seed)

    # memory preallocation
    PhiSenAPUs = np.zeros((M*D,numPoints,S),dtype=complex)
    YSenAPU = np.zeros((M*D,S),dtype=complex)

    for idxSen, sAPU in enumerate(idxSenAPUs):
        posSenAPU = posAPUs[sAPU]

        for idxSubCarr, subCarrier in enumerate(freqPerDev):
            PhiFreq = np.zeros((M, numPoints), dtype=complex)

            xk = ofdmSymbols[:,idxSubCarr]

            for idxComm, cAPU in enumerate(idxCommAPUs):
                # position of the c-th comm. APU
                posCommAPU = posAPUs[cAPU]

                steeringVecComm,steeringVecSen = computeSteeringVecs(posPoints,subCarrier,posSenAPU,posCommAPU,LRoom,M,Delta)

                phase = np.exp(-1j*2*np.pi*subCarrier*tau[idxSen,idxComm,:])   

                # sensing matrix for a given pair of commun. and sensing APUs at a given subcarrier
                PhiFreq += steeringVecSen*(phase*(steeringVecComm.conj().T @ xk[idxComm*M:(idxComm+1)*M]))[None,:]

            PhiSenAPUs[idxSubCarr*M:(idxSubCarr+1)*M,:,idxSen] = PhiFreq 

            # noise vector 
            noise = np.sqrt(noisePow2/2)*(rng.standard_normal((M,)) + 1j*rng.standard_normal((M,)))

            # received signal at a given sensing APU at the subCarrier 
            YFreq = sum(H[idxSen,idxComm,idxSubCarr,:,:] @ xk[idxComm*M:(idxComm+1)*M] for idxComm in range(C)) + noise

            # list of received signal vectors across frequencies at the sensing APU
            YSenAPU[idxSubCarr*M:(idxSubCarr+1)*M,idxSen] = YFreq

    return PhiSenAPUs, YSenAPU

def admmOptim(beta,mu,alpha,roleAPUs,freqPerDev,posAPUs,posPoints,LRoom,Delta,tau,ofdmSymbols,sigma2,H,iter,seed=0):
    """
    Implements the ADMM optimization algorithm using the vectorized received
    signal 

    Args
    -------------
    beta : ADMM penalty parameter
    mu : ADMM penalty parameter
    alpha : ADMM penalty parameter
    roleAPUs : roles of the APUs "1" communication and "0" sensing (S+C,1)
    freqPerDev : subcarriers frequency (D,1), [Hz]
    posAPUs : 2D positions of the APUs (S+C,2), [m]
    posPoints : 2D positions of the grid points (I**2,2), [m]
    LRoom : length of the room (m)
    Delta : distance between antennas (m)
    tau : time delays (S,C,D)
    ofdmSymbols : OFDM symbols (D,)
    sigma2 : noise variance
    H : channel matrix points (S,D,M,M)
    iter : num. of ADMM iterations
    seed : seed for the random number generator (noise)

    Returns
    -------------
    zGNext : estimated radar cross section (I**2,)
    primalRes : primal residual norm
    """
    absTol = 1E-2
    relTol = 1E-2

    # num. points per dimension
    I = int(np.sqrt(posPoints.shape[0]))

    # num. sensing APUs
    S = int(np.sqrt(np.flatnonzero(roleAPUs == 0).size))

    # vectorization step
    PhiSensingAPUs, YSensingAPU = vectSignal(roleAPUs,freqPerDev,posAPUs,posPoints,LRoom,Delta,tau,ofdmSymbols,sigma2,H,seed)

    # memory preallocation
    z = np.zeros((I**2,S))
    zGPrev = np.zeros((I**2,))
    gamma = np.zeros((I**2,S))

    gram = []
    matched = []
    for sensingAPUIdx in np.arange(S):
        phaseCorrection = np.angle(PhiSensingAPUs[:,:,sensingAPUIdx].conj().T @ YSensingAPU[:,sensingAPUIdx])
        PhiSensingAPUsCorrected = PhiSensingAPUs[:,:,sensingAPUIdx] @ np.diag(np.exp(1j*phaseCorrection))

        gram.append(PhiSensingAPUsCorrected.conj().T @ PhiSensingAPUsCorrected)
        matched.append(PhiSensingAPUsCorrected.conj().T @ YSensingAPU[:,sensingAPUIdx])

    for i in np.arange(iter):
        # update of local images
        for sensingAPUIdx in np.arange(S):
            Mx = mu*gram[sensingAPUIdx] + beta*np.eye(I**2)        
            vec = mu*matched[sensingAPUIdx] + beta*zGPrev - gamma[:,sensingAPUIdx]
            z[:,sensingAPUIdx] = np.maximum(np.real(np.linalg.solve(Mx,vec)),0.0)

        # update of global image
        inputVec = np.mean(z,axis=1) + 1/beta*np.mean(gamma,axis=1)
        threshold = alpha/(S*beta)
        zGNext = softThresholding(inputVec,threshold)

        # update of dual variable
        for sensingAPUIdx in np.arange(S):
            gamma[:,sensingAPUIdx] += beta*(z[:,sensingAPUIdx] - zGNext)

        # primal and dual residuals (eqs. 21, 22)
        primalRes = z - zGNext[:,None]
        dualRes = np.sqrt(S)*beta*(zGNext - zGPrev)
       
        # feasibility tolerances (eq. 24)
        primalTol = np.sqrt(S*I**2)*absTol + relTol*np.maximum(
            np.linalg.norm(z, ord='fro'), np.sqrt(S)*np.linalg.norm(zGNext)
            )
        dualTol = np.sqrt(S*I**2)*absTol + np.linalg.norm(gamma, ord='fro')*relTol

        print(i, np.linalg.norm(primalRes,ord="fro"), np.linalg.norm(dualRes))
        zGPrev = zGNext   

    return zGNext, np.linalg.norm(primalRes,ord="fro")