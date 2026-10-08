\# Reproduction Discrepancies



\## Purpose



This document records discrepancies observed while reproducing

the reference study:



"Automatic control of UAVs: new adaptive rules and type-3 fuzzy

stabilizer" (Cai et al., 2024).



The purpose is to clearly distinguish:



1\. parameters and equations explicitly reported in the paper,

2\. numerical results obtained by directly implementing them,

3\. implementation assumptions required because of missing details.



\---



\## 1. Initial conditions



The paper reports the following initial UAV states:



chi\_i(0) = 0.10, i = 1, ..., 8.



It also reports:



zeta\_hat\_i(0) = 0.10, i = 1, ..., 9



h\_hat\_1(0) = 0.10



h\_hat\_2(0) = 0.20



h\_hat\_3(0) = 0.30



m\_hat(0) = 0.65



The implemented reproduction uses these values directly.



\---



\## 2. Initial sliding surfaces



Using Eq. (31), the reported reference values and the reported

initial state give:



s(0) =



\[-1.79439510,

&#x20;-0.74719755,

&#x20;-2.84159265,

&#x20;-5.70000000]



However, the trajectories shown in Fig. 7 appear to start at

substantially different values.



Therefore, Fig. 7 cannot be reproduced exactly from the published

initial conditions and Eq. (31) without additional assumptions.



\---



\## 3. Initial control inputs



Using the reported initial conditions together with Eqs. (33) and

(35), and zero initial fuzzy compensation, the implementation gives:



u1 ≈ 25.01 N



u2 ≈ 1.77



u3 ≈ 1.45



u4 ≈ 4.20



These values differ substantially from the initial control values

shown in Fig. 6, particularly for u1 and u4.



For example, the plotted u1 signal appears to begin close to 9.2 N,

rather than approximately 25 N.



This indicates that at least one simulation parameter, initial

condition, scaling factor, fuzzy output, or implementation detail

used to generate Fig. 6 is not fully specified in the paper.



\---



\## 4. Adaptive-parameter trajectories



The paper states that all zeta\_hat\_i parameters start from 0.10.



However, several trajectories displayed in Fig. 8 visually appear

to start from values substantially different from 0.10.



This creates another discrepancy between the simulation description

and the presented figures.



\---



\## 5. Type-3 FLS configuration



The paper provides the mathematical structure of the Type-3 fuzzy

membership functions, firing strengths, z-slice aggregation, and

adaptive consequent weights.



However, numerical values required for an exact reconstruction are

not fully reported, including:



\- number of membership functions,

\- membership-function centers,

\- membership-function spreads,

\- exact number and values of z-slices,

\- initial fuzzy consequent weights,

\- complete rule-base configuration.



The reproduction therefore uses explicitly documented

implementation-specific values for these quantities.



\---



\## 6. Closed-loop reproduction result



The equation-based implementation remains numerically finite during

the initial simulation interval and initially approaches the desired

trajectory.



At approximately 1 s, the roll angle moves through a region where:



cos(phi) \* cos(theta)



approaches zero.



Because Eq. (33) contains:



1 / (cos(phi) \* cos(theta))



in the altitude controller, the thrust command becomes very large.



In the current diagnostic experiment, the maximum magnitude of u1

reaches approximately 1.67e3 N by 1 s.



No artificial saturation has been applied to the baseline because

doing so would alter the published controller.



\---



\## 7. Reproduction strategy



The project will therefore maintain two clearly separated versions:



\### Published-equation baseline



Implements the equations and explicitly reported numerical parameters

as directly as possible.



Any deviation from the published figures is reported rather than

hidden.



\### Proposed stabilized extension



A separate practical modification will be developed to prevent

singularity-related control growth and improve robustness.



The stabilized version will be treated as an original experimental

extension and will not be presented as an exact implementation of

the reference paper.

