import numpy as np
from scipy import constants
from .arrays import computeSteeringVecs

def softThresholding(inputVec,threshold):
    return np.sign(inputVec)*np.maximum(np.abs(inputVec) - threshold, 0.0)

def vectSignal(roleAPUs,freqPerDev,idxAPU2Dev,posAPUs,posPoints,LRoom,Delta,tau,ofdmSymbols,noisePow2,H,seed):
    """
    Vectorization step of the received signal in the form of a compressed
    sensing problem

    Args
    -------------
    S : num. sensing APUs
    I : sq. root of the num. of points in meas. grid
    PhiSensingAPUs : sensing matrix APUs (M*N*U,I**2,S) 
    YSensingAPU : vectorized meas. signal per APU (M*N*U,S)
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
    U : num. devs
    C : num. comm. APUs
    """

    M,N,_ = ofdmSymbols.shape
    S, _, _, _ = H.shape
    numPoints, _ = posPoints.shape
    I = int(np.sqrt(numPoints))
    C = roleAPUs.sum()
    S = posAPUs.shape[0] - C
    U = freqPerDev.size

    rng = np.random.default_rng(seed)

    # vectorization step
    sensingAPUIdxs = np.flatnonzero(roleAPUs == 0)
    PhiSensingAPUs = np.zeros((M*N*U,I**2,S),dtype=complex)
    YSensingAPU = np.zeros((M*N*U,S),dtype=complex)
    for sensingAPUIdx in np.arange(S):
        Phi = []
        Y = []
        for dev, subCarrier in enumerate(freqPerDev):

            posCommAPU = posAPUs[idxAPU2Dev[dev]]
            posSensingAPU = posAPUs[sensingAPUIdxs[sensingAPUIdx]]

            _,_,katriRaoProd = computeSteeringVecs(posPoints,subCarrier,posSensingAPU,posCommAPU,LRoom,M,Delta)
            
            Theta = np.diag(np.exp(-1j*2*np.pi*subCarrier*tau[sensingAPUIdx,dev,:]))

            PhiFreq = np.kron(ofdmSymbols[:,:,dev].T,np.eye(M)) @ katriRaoProd @ Theta

            Phi.append(PhiFreq)

            # noise vector 
            noise = np.sqrt(noisePow2/2)*(rng.standard_normal((M,N)) + 1j*rng.standard_normal((M,N)))
            
            # received signal at a given sensing APU at the subCarrier 
            YFreq = H[sensingAPUIdx,dev,:,:] @ ofdmSymbols[:,:,dev] + noise

            Y.append(YFreq.ravel(order='F'))

        # list of sensing matrices across frequencies for a given sensing APU
        PhiSensingAPUs[:,:,sensingAPUIdx] = np.vstack(Phi)

        # list of received signal vectors across frequencies at the sensing APU
        YSensingAPU[:,sensingAPUIdx] = np.concatenate(Y)

    return PhiSensingAPUs, YSensingAPU

def admmOptim(beta,mu,alpha,roleAPUs,freqPerDev,idxAPU2Dev,posAPUs,posPoints,LRoom,Delta,tau,ofdmSymbols,sigma2,H,seed):
    """
    Implements the ADMM optimization algorithm using the vectorized received
    signal 

    Args
    -------------
    S : num. sensing APUs
    I : sq. root of the num. of points in meas. grid
    PhiSensingAPUs : sensing matrix APUs (M*N*U,I**2,S) 
    YSensingAPU : vectorized meas. signal per APU (M*N*U,S)
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
    U : num. devs 
    C : num. comm. APUs
    """
    absTol = 1E-2
    relTol = 1E-2

    S, _, _, _ = H.shape
    I = int(np.sqrt(posPoints.shape[0])) 
    C = roleAPUs.sum()
    S = posAPUs.shape[0] - C

    # vectorization step
    PhiSensingAPUs, YSensingAPU = vectSignal(roleAPUs,freqPerDev,idxAPU2Dev,posAPUs,posPoints,LRoom,Delta,tau,ofdmSymbols,sigma2,H,seed)

    # ADMM var. initialization 
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

    for i in np.arange(50):
        # update of local images
        for sensingAPUIdx in np.arange(S):
            # phaseCorrection = np.angle(PhiSensingAPUs[:,:,sensingAPUIdx].conj().T @ YSensingAPU[:,sensingAPUIdx]) 
            # PhiSensingAPUsCorrected = PhiSensingAPUs[:,:,sensingAPUIdx] @ np.diag(np.exp(1j*phaseCorrection))
            # Mx = mu*PhiSensingAPUsCorrected.conj().T @ PhiSensingAPUsCorrected + beta*np.eye(I**2)        
            # vec = mu*PhiSensingAPUsCorrected.conj().T @ YSensingAPU[:,sensingAPUIdx] + beta*zGPrev - gamma[:,sensingAPUIdx]
            # z[:,sensingAPUIdx] = np.maximum(np.real(np.linalg.solve(Mx,vec)),0.0)
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

        # print(i, "dual variable", np.linalg.norm(z,ord="fro"))
        # primal and dual residuals (eqs. 21, 22)
        primalRes = z - zGNext[:,None]
        dualRes = np.sqrt(S)*beta*(zGNext - zGPrev)
        # print(np.linalg.norm(zGNext,ord=0))
        # feasibility tolerances (eq. 24)
        primalTol = np.sqrt(S*I**2)*absTol + relTol*np.maximum(
            np.linalg.norm(z, ord='fro'), np.sqrt(S)*np.linalg.norm(zGNext)
            )
        dualTol = np.sqrt(S*I**2)*absTol + np.linalg.norm(gamma, ord='fro')*relTol

        # stopping criteria
        # print(i, np.linalg.norm(primalRes), primalTol)
        # print(i, np.linalg.norm(dualRes), dualTol)
        # print(i, primalTol, dualTol)
        print(i, np.linalg.norm(primalRes,ord="fro"), np.linalg.norm(dualRes))
        zGPrev = zGNext   

    return zGNext