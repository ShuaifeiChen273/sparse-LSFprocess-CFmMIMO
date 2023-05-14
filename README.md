## Energy-Efficient Cell-Free Massive MIMO Through Sparse Large-Scale Fading Processing

This is a code package is related to the following scientific article:

Shuaifei Chen, Jiayi Zhang, Emil Björnson, Özlem Tuğfe Demir, and B. Ai, “[Energy-efficient cell-free massive MIMO through sparse large-scale fading processing](https://ieeexplore.ieee.org/document/10113887),” IEEE Transactions on Wireless Communications, To appear, 2023.

The package contains a simulation environment, based on Matlab, that reproduces some of the numerical results and figures in the article. *We encourage you to also perform reproducible research!*

## Abstract of Article

Cell-free massive multiple-input multiple-output (CF mMIMO) systems serve the user equipments (UEs) by geographically distributed access points (APs) by means of joint transmission and reception. 

To limit the power consumption due to fronthaul signaling and processing, each UE should only be served by a subset of the APs, but it is hard to identify that subset. Previous works have tackled this combinatorial problem heuristically. 

In this paper, 

- we propose a sparse distributed processing design for CF mMIMO, where the AP-UE association and long-term signal processing coefficients are jointly optimized. 
- We formulate two sparsity-inducing mean-squared error (MSE) minimization problems and solve them by using efficient proximal approaches with block-coordinate descent. 
- For the downlink, more specifically, we develop a virtually optimized large-scale fading precoding (V-LSFP) scheme using uplink-downlink duality. 

The numerical results show that the proposed sparse processing schemes work well in both uplink and downlink. In particular, they achieve almost the same spectral efficiency as if all APs would serve all UEs, while the energy efficiency is 2-4 times higher thanks to the reduced processing and signaling.

## Content of Code Package

The package generates the simulation results used in Figure 2 and Figure 5. To be specific:

- `simulationFigure2`  and `simulationFigure5`: Main functions;
- `generateSetup_correlatedFading`: Generate one setup with UEs and APs at random locations;
  - `functionRlocalscattering_theta`: Generate the spatial correlation matrix for the local scattering model;
- `functionChannelEstimates`: Generate channel realizations with estimates and estimation error correlation matrices;
- `functionComputeExpectations`: Obtain the expectations for the computation of SE.
- `functionComputeSE`: Compute distributed SE with UatF bound ;
  - `computeSE_SLSFD`: Compute uplink SE with O- LSFD and S-LSFD;
    - `warmRestart`: Perform warm-restart;
      - `proximalNesterov`: Perform proximal algorithm with Nesterov step;
  - `computeSE_PLSFD`: Compute uplink SE with P- LSFD ;
  - `computeSE_VLSFP`: Compute downlink SE with V-LSFP;
  - `computeSE_SLSFP`: Compute downlink SE with S-LSFP;
  - `computeSE_HEUR`: Compute downlink SE with HUER;
  - `computeSE_FPA`: Compute downlink SE with distributed FPA;

- `functionComputeEE`: Compute EE with the proposed power consumption model.

See each file for further documentation.


## License and Referencing

This code package is licensed under the GPLv2 license. If you in any way use this code for research that results in publications, please cite our original article listed above.
