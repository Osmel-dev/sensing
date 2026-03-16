import numpy as np
from .deployment import computeDevsDeployment
from .arrays import computePrecoders
from .channels import pointsChannelCoeffs, devsChannelCoeffs
from .waveforms import gen16QAM, genOFDMSym
from .optim import admmOptim

def simulation(txPow,Delta,M,LRoom,Umax,U,posAPUs,posPoints,radarCrossSection,subCarriersFreq,roleAPUs,beta,mu,alpha,noisePow2,bw,N,seed):
    """
    main simulation function
    
    Args
    -------------
    input : description

    Returns
    -------------
    output : description
    """

    reconstSceneList = []
    primalResidualList = []
    rateList = []

    C = roleAPUs[0].sum()
    K = subCarriersFreq.size
    I = int(np.sqrt(posPoints.shape[0]))

    for role in range(roleAPUs.shape[0]):
        # device positions
        posDevs, idxAPU2Dev = computeDevsDeployment(posAPUs,U,Umax,roleAPUs[role],LRoom,seed)

        # subCarriers are allocated in a continuous manner to each comm. APU
        # the number of subCarriers allocated to each APU is Umax
        # the bw blocks are assigned to each comm. APU is order of apearance, i.e., 1st
        # block to the first APU in the roleAPUs vector and so on.
        subCarrierBlockPerAPU = K // C
        startSubCarrierPerAPU = np.arange(C)*subCarrierBlockPerAPU

        idxCommAPUs = np.flatnonzero(roleAPUs[role] == 1)

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
        steeringVectors, precoders = computePrecoders(posAPUs,posDevs,idxAPU2Dev,M,LRoom,Delta,freqPerDev,txPow)

        # utils.displayScenario(LRoom,posAPUs,posDevs,posAnts,S+C,posPoints,roleAPUs,radarCrossSection,idxAPU2Dev)

        # channel matrix
        H, tau = pointsChannelCoeffs(posPoints,freqPerDev,idxAPU2Dev,U,posAPUs,LRoom,Delta,M,roleAPUs[role],radarCrossSection)

        # transmitted symbols
        qamSymbols = gen16QAM(N,freqPerDev.size)
        ofdmSymbols = genOFDMSym(precoders,txPow,freqPerDev.size,qamSymbols,M,N)

        # compute rate
        sumRate = 0
        perAPURate = np.zeros(C)
        perUserRate = np.zeros(U)

        # devs channel
        HDev = devsChannelCoeffs(posDevs,freqPerDev,posAPUs,LRoom,Delta,M,idxAPU2Dev)

        devsPerAPU = np.bincount(idxAPU2Dev, minlength=roleAPUs.size)
        for dev, apu in enumerate(idxAPU2Dev):
            # per use rate
            perUserRate[dev] = bw*np.log2(1 + txPow/devsPerAPU[apu]*np.linalg.norm(HDev[dev,:])**2/(noisePow2*bw))

            # per APU rate
            perAPURate[np.where(idxCommAPUs == apu)[0][0]] += perUserRate[dev]

            # sum rate
            sumRate += perUserRate[dev]

        print(
            f"Worst per user rate: {np.min(perUserRate):.4f}, "
            f"Worst per APU rate: {np.min(perAPURate):.4f}, "
            f"Sum rate: {sumRate:.4f}"
        )

        rateMetrics = {
        "sumRate": sumRate,
        "perAPURate": perAPURate,
        "perUserRate": perUserRate,
        }

        rateList.append(rateMetrics)

        # ADMM opt. loop
        reconstScene, primalResidual = admmOptim(beta,mu,alpha,roleAPUs[role],freqPerDev,idxAPU2Dev,posAPUs,posPoints,LRoom,Delta,tau,ofdmSymbols,noisePow2,H,50)

        reconstSceneList.append(reconstScene.reshape(I,I))
        primalResidualList.append(primalResidual)


    return reconstSceneList, primalResidualList, rateList