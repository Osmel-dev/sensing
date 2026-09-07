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

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

def displayScenario(roomPerimeter,positionAPUs,positionDevs,positionAnts,numAPUs,roleAPUs,idxAPUDevs):
    """
    Display the system model with the room, APUs, devices, and antennas.
    
    Args
    -------------
    roomPerimeter : perimeter of the room [m]
    positionAPUs : 2D positions of the APUs center (numberOfAPUs,2), [m]
    positionDevs : 2D positions of the devices (numberOfDevices,2), [m]
    positionAnts : 2D positions of the APUs antennas (numAPUs,antsPerAPU,2), [m]
    numAPUs : total number of APUs
    roleAPUs : roles of the APUs (1 -> comm., 0 -> sensing)
    idxAPUDevs : indices of the devices associated with each APU (numAPUs,numDevicesPerAPU)
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

def fusion(zGlobal,primalResiduals):
    """
    Fuses the reconstructed images from different APUs assigments using a
    weighted average based on the primal residuals of the ADMM optimization.

    Args
    -------------
    zGlobal : reconstructed images from different APUs assigments (num.
    Configs., I, I) 
    primalResiduals : primal residuals from the ADMM optimization (num. Configs.)

    Returns
    -------------
    zFused : fused scene representation (I, I)
    """

    # max per configuration (axis = spatial dims)
    maxPerConfig = np.max(zGlobal, axis=(1, 2), keepdims=True)

    # normalize
    zGlobalNorm = zGlobal/(maxPerConfig + 1e-10)

    weights = 1.0/(1e-10 + primalResiduals)
    weights /= np.sum(weights)
    zFused = np.sum(zGlobalNorm*weights[:,np.newaxis,np.newaxis], axis=0)

    return zFused

def estQuality(zFused,zScene):
    """
    Computes the precision of the fused scene representation compared to the ground-truth scene.

    Args
    -------------
    zFused : fused scene representation (I, I)
    zScene : ground-truth scene representation (I, I)

    Returns
    -------------
    precision : ratio of true positives to the total number of positive predictions
    """

    zFused = zFused/(np.max(zFused))

    supportFused = zFused >= .85
    supportTrueScene = zScene > 0

    # true positive (tp), false positive (fp)
    tp = np.sum(supportTrueScene & supportFused)
    fp = np.sum(~supportTrueScene & supportFused)
    
    return tp/(tp + fp + 1e-12)