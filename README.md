Radio Stripe-Based Distributed ISAC System with Dynamic Sensing-Communication Reconfiguration
==================

This code package is related to the following scientific article:

O. Martínez Rosabal and O. L. Alcaraz López, "Radio Stripe-Based Distributed ISAC System with Dynamic Sensing-Communication Reconfiguration," accepted for publication in 2026 IEEE International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC'26).

Available at: [https://arxiv.org/pdf/2604.08982](https://arxiv.org/pdf/2604.08982)

## Abstract of Article

Integrated sensing and communications (ISAC) has emerged as an intrinsic service of upcoming 6G wireless systems, enabling the reuse of communication signals for environmental sensing and supporting context-aware network functionalities. Meanwhile, the evolution of the wireless infrastructure toward distributed systems creates new opportunities for collaborative sensing from spatially separated nodes. Motivated by this trend, this work investigates a radio stripe aided ISAC system as a low-complexity implementation of a distributed system. We study the trade-off between achievable sum rate and sensing precision when downlink signals are used for target localization within the service area. By exploiting the architectural homogeneity of the radio stripes transceivers, each unit can be dynamically configured to operate in either communication or sensing mode. We formulate a targets localization problem considering the measurements of multiple sensing-communication configurations. Due to the large number of measurements and the continuity of the search space, we propose discretizing the service area and then solve the estimation problem in batches. The targets are finally estimated using a fusion strategy. Our results show that increasing the number devices and sensing antenna processing units (APUs) boosts sensing precision at the expense of degrading the sum rate. The latter remains constant for a given number of communication APUs regardless of their positions. Moreover, changing the number of antennas reveals a non-monotonic impact on sensing performance due to the trade-off between array gain and illumination uniformity.

## Content of Code Package

The repository contains python scripts and user-defined functions required to reproduce the numerical results in the article. To run the code, we recommend the use of high-performance computing services as these may take hours if not days to complete on a personal computer. 

See each file for further documentation.

## Acknowledgements

This work is partially supported by the Research Council of Finland (Grants 362782 (ECO-LITE), and 369116 (6G Flagship)). The authors also wish to acknowledge CSC - IT Center for Science, Finland, for computational resources.
