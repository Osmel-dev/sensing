import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

def displayScenario(roomPerimeter,positionAPUs,positionDevs,positionAnts,numAPUs,positionPoints,roleAPUs,radarCrossSection,idxAPUDevs):
    """
    Display the APUs deployment
    
    Args
    -------------
    roomPerimeter : perimeter of the room [m]
    positionAPUs : 2D positions of the APUs center (numberOfAPUs x 2)
    numAPUs : total number of APUs
    """

    fig, ax = plt.subplots(figsize=(7,7))
    sideLength = roomPerimeter/4
    ax.plot([0, sideLength, sideLength, 0, 0], [0, 0, sideLength, sideLength, 0], 'k-', linewidth=2)
    
    ax.scatter(positionAPUs[roleAPUs==0,0], positionAPUs[roleAPUs==0,1], c='blue', s=50, label='sensing APU')
    ax.scatter(positionAPUs[roleAPUs==1,0], positionAPUs[roleAPUs==1,1], c='magenta', s=50, label='communication APU')
    
    ax.scatter(positionDevs[:,0], positionDevs[:,1], c='green', s=40, label='Devices', marker='s')

    for t in range(positionAPUs.shape[0]):
        ax.text(positionAPUs[t,0]+.5, positionAPUs[t,1]+.5, str(t), fontsize=9, ha='left', va='bottom')

    for d in range(positionDevs.shape[0]):
        ax.text(positionDevs[d,0]+.5, positionDevs[d,1]+.5, str(idxAPUDevs[d]), fontsize=8, ha='left', va='bottom')

    # nonZeroRadar = radarCrossSection != 0
    # color = np.where(nonZeroRadar, 'red', 'black')
    # ax.scatter(positionPoints[:,0], positionPoints[:,1], c=color, s=10, label='Points', marker='x')

    for i in range(numAPUs):
        ax.scatter(positionAnts[i,:,0], positionAnts[i,:,1], c='black', s=15)

    ax.set_aspect('equal')
    ax.set_title("System model")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True)
    ax.legend(loc="center left",bbox_to_anchor=(1.02, 0.5),borderaxespad=0.0)
    plt.tight_layout()
    plt.show()

def ssim(zTruth,zReconst):
    """
    Computes the structural similarity index between the reconstructed radar
    image and a ground-truth scene

    Args
    -------------
    alpha : opt. param.

    Returns
    -------------
    zGNext : estimated radar cross section (I**2,)
    """

    sigmaGaussFilter = 1.5
    radiusGaussFilter = 5

    K1 = 0.01
    K2 = 0.03 
    
    L = 1

    C1 = (K1*L)**2
    C2 = (K2*L)**2

    # local means
    muTruth = gaussian_filter(zTruth, sigma=sigmaGaussFilter, radius=radiusGaussFilter)
    muReconst = gaussian_filter(zReconst, sigma=sigmaGaussFilter, radius=radiusGaussFilter)

    # local variances
    sigmaTruth = gaussian_filter(zTruth*zTruth, sigma=sigmaGaussFilter, radius=radiusGaussFilter) - muTruth**2
    sigmaReconst = gaussian_filter(zReconst*zReconst, sigma=sigmaGaussFilter, radius=radiusGaussFilter) - muReconst**2

    # local covariance
    sigmaJoint = gaussian_filter(zTruth*zReconst, sigma=sigmaGaussFilter, radius=radiusGaussFilter) - muTruth*muReconst

    # SSIM
    num = (2*muTruth*muReconst + C1)*(2*sigmaJoint + C2)
    den = (muTruth**2 + muReconst**2 + C1)*(sigmaTruth + sigmaReconst + C2)

    ssimMap = num/den
    mssim = np.mean(ssimMap)

    return  mssim, ssimMap

def fusion(zGlobal,primalResiduals):
    """
    Docstring

    Args
    -------------
    param : description

    Returns
    -------------
    output : description
    """

    # convert lists to arrays
    zGlobal = np.stack(zGlobal, axis=2)          
    primalResiduals = np.asarray(primalResiduals, dtype=float)  

    # ADMM residual confidence: compute the weighted sum of the recovered
    # scenes. The weights are compued based on the norm of the primal residual.

    weights = 1.0/(1e-10 + primalResiduals)
    weights /= np.sum(weights)
    zFusedWeighted = np.sum(zGlobal*weights[None, None,], axis=2)

    # voting fusion: combine where each snapshot believes targets exist. Each
    # "cell" or "pixel" is rated against a threshold and its value is turned
    # into a '0' or '1' depending on whether it is above or below the threshold.
    # Then the preprocessed schenes are added.

    # relative threshold with respect to each snapshot peak
    alpha = 0.5 
    thresh = alpha*np.max(zGlobal, axis=(0, 1))

    # binary support map per snapshot
    votes_binary = zGlobal >= thresh[None, None, :]        

    # number of votes per cell
    zFusedVote = np.sum(votes_binary, axis=2)               

    # likelihood fusion: uses a multiplicative fusion. Each snapshot is
    # converted into a normalized nonnegative map. The maps of all snapshots are
    # then multiplied.
    
    eps = 1e-12
    gamma = 5.0

    # probability-like map per snapshot
    probMap = (np.abs(zGlobal)**gamma)
    probMap /= np.sum(probMap, axis=(0, 1), keepdims=True) + eps  
    zFusedLikelihood = np.sum(np.log(probMap + eps), axis=2)     


    return zFusedWeighted, zFusedVote, zFusedLikelihood