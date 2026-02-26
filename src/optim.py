import numpy as np
from scipy import constants
from .arrays import computeSteeringVecs

def softThresholding(inputVec,threshold):
    return np.sign(inputVec)*np.maximum(np.abs(inputVec) - threshold, 0.0)

def vectSignal(roleAPUs,Uc,subCarrierFreqAlloc,posAPUs,posPoints,LRoom,Delta,tau,ofdmSymbols,sigma2,H):
    """
    Vectorization step of the received signal in the form of a compressed
    sensing problem

    Args
    -------------
    S : num. sensing APUs
    I : sq. root of the num. of points in meas. grid
    PhiSensingAPUs : sensing matrix APUs (M*N*Uc*C,I**2,S) 
    YSensingAPU : vectorized meas. signal per APU (M*N*Uc*C,S)
    beta : opt. param.
    mu : opt. param.
    alpha : opt. param.

    Returns
    -------------
    zGNext : estimated radar cross section (I**2,)
    
    Other 
    -------------
    M : num. ants per APU
    N : num. of OFDM symbs.
    Uc : num. devs per comm. APU
    C : num. comm. APUs
    """

    M,N,_ = ofdmSymbols.shape
    S, _, _, _ = H.shape
    numPoints, _ = posPoints.shape
    I = int(np.sqrt(numPoints))
    C = roleAPUs.sum()
    S = posAPUs.shape[0] - C

    # vectorization step
    commAPUIdx = np.flatnonzero(roleAPUs == 1)
    sensingAPUIdxs = np.flatnonzero(roleAPUs == 0)
    PhiSensingAPUs = np.zeros((M*N*Uc,I**2,S),dtype=complex)
    YSensingAPU = np.zeros((M*N*Uc,S),dtype=complex)
    for sensingAPUIdx in np.arange(S):
        Phi = []
        Y = []
        for subCarrierIdx, subCarrier in enumerate(subCarrierFreqAlloc):

            posCommAPU = posAPUs[commAPUIdx[subCarrierIdx // Uc]]
            posSensingAPU = posAPUs[sensingAPUIdxs[sensingAPUIdx]]

            _,_,katriRaoProd = computeSteeringVecs(posPoints,subCarrier,posSensingAPU,posCommAPU,LRoom,M,Delta)
            # subCarrierIdx = commAPUIdx[subCarrierIdx // Uc]
            Theta = np.diag(np.exp(-1j*2*np.pi*subCarrier*tau[sensingAPUIdx,subCarrierIdx,:]))

            PhiFreq = np.kron(ofdmSymbols[:,:,subCarrierIdx].T,np.eye(M)) @ katriRaoProd @ Theta

            Phi.append(PhiFreq)

            # noise vector 
            noise = np.sqrt(sigma2/2)*(np.random.randn(M,N) + 1j*np.random.randn(M,N))
            
            # received signal at a given sensing APU at the subCarrier 
            YFreq = H[sensingAPUIdx,subCarrierIdx,:,:] @ ofdmSymbols[:,:,subCarrierIdx] + noise

            Y.append(YFreq.ravel(order='F'))

        # list of sensing matrices across frequencies for a given sensing APU
        PhiSensingAPUs[:,:,sensingAPUIdx] = np.vstack(Phi)

        # list of received signal vectors across frequencies at the sensing APU
        YSensingAPU[:,sensingAPUIdx] = np.concatenate(Y)

    return PhiSensingAPUs, YSensingAPU

def admmOptim(beta,mu,alpha,roleAPUs,Uc,subCarrierFreqAlloc,posAPUs,posPoints,LRoom,Delta,tau,ofdmSymbols,sigma2,H):
    """
    Implements the ADMM optimization algorithm using the vectorized received
    signal 

    Args
    -------------
    S : num. sensing APUs
    I : sq. root of the num. of points in meas. grid
    PhiSensingAPUs : sensing matrix APUs (M*N*Uc*C,I**2,S) 
    YSensingAPU : vectorized meas. signal per APU (M*N*Uc*C,S)
    beta : opt. param.
    mu : opt. param.
    alpha : opt. param.

    Returns
    -------------
    zGNext : estimated radar cross section (I**2,)
    
    Other 
    -------------
    M : num. ants per APU
    N : num. of OFDM symbs.
    Uc : num. devs per comm. APU
    C : num. comm. APUs
    """
    S, _, _, _ = H.shape
    numPoints, _ = posPoints.shape
    I = int(np.sqrt(numPoints))
    C = roleAPUs.sum()
    S = posAPUs.shape[0] - C

    # vectorization step
    PhiSensingAPUs, YSensingAPU = vectSignal(roleAPUs,Uc,subCarrierFreqAlloc,posAPUs,posPoints,LRoom,Delta,tau,ofdmSymbols,sigma2,H)

    # ADMM var. initialization 
    z = np.zeros((I**2,S))
    zGPrev = np.zeros((I**2,))
    gamma = np.zeros((I**2,S))

    for i in np.arange(50):
        # update of local images
        for sensingAPUIdx in np.arange(S):
            phaseCorrection = np.angle(PhiSensingAPUs[:,:,sensingAPUIdx].conj().T @ YSensingAPU[:,sensingAPUIdx]) 
            PhiSensingAPUsCorrected = PhiSensingAPUs[:,:,sensingAPUIdx] @ np.diag(np.exp(1j*phaseCorrection))
            Mx = mu*PhiSensingAPUsCorrected.conj().T @ PhiSensingAPUsCorrected + beta*np.eye(I**2)        
            vec = mu*PhiSensingAPUsCorrected.conj().T @ YSensingAPU[:,sensingAPUIdx] + beta*zGPrev - gamma[:,sensingAPUIdx]
            z[:,sensingAPUIdx] = np.maximum(np.real(np.linalg.solve(Mx,vec)),0.0)

        # update of global image
        inputVec = np.mean(z,axis=1) + 1/beta*np.mean(gamma,axis=1)
        threshold = alpha/(S*beta)
        zGNext = softThresholding(inputVec,threshold)

        # update of dual variable
        for sensingAPUIdx in np.arange(S):
            gamma[:,sensingAPUIdx] += beta*(z[:,sensingAPUIdx] - zGNext)

        primalRes = z - zGNext[:,None]
        dualRes = beta*(zGNext - zGPrev)
        # primalTol = 
        # dualTol = 
        print(i, np.linalg.norm(primalRes), np.linalg.norm(dualRes))

        zGPrev = zGNext    

    return zGNext