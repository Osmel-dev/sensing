import matplotlib.pyplot as plt
import numpy as np

def displayScenario(roomPerimeter,positionAPUs,positionDevs,positionAnts,numAPUs,positionPoints,roleAPUs,radarCrossSection):
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
    
    ax.scatter(positionDevs[:,0], positionDevs[:,1], c='green', s=25, label='Devices', marker='s')
    nonZeroRadar = radarCrossSection != 0
    color = np.where(nonZeroRadar, 'red', 'black')
    ax.scatter(positionPoints[:,0], positionPoints[:,1], c=color, s=10, label='Points', marker='x')

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