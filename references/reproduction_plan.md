\# Reproduction Plan



\## Reference Paper



Automatic control of UAVs: new adaptive rules and type-3 fuzzy stabilizer



\## Minimum Reproduction Requirement (MRR)



The reproduction study will implement the main control pipeline of the reference paper in a numerical simulation environment.



The implementation will include:



1\. UAV dynamic model

2\. State-space representation

3\. Reference trajectory definition

4\. Tracking error calculation

5\. Sliding surfaces

6\. Adaptive controller

7\. Type-3 Fuzzy Logic System

8\. Fuzzy compensation of modeling errors and disturbances

9\. Numerical simulation

10\. Trajectory tracking plots

11\. Sliding surface plots

12\. Performance evaluation using MSE/RMSE



\## Reproduction Experiment



The primary reproduction experiment will evaluate whether the UAV states can track the reference roll, pitch, yaw, and altitude values used in the reference study.



The reproduced results will be compared with the behavior and numerical results reported in the original paper where possible.



\## Proposed Extension



A robustness and ablation analysis will be conducted.



The complete Type-3 FLS + adaptive SMC controller will be compared against a controller without fuzzy compensation under different disturbance levels.



Example disturbance levels:



\- 0%

\- 5%

\- 10%

\- 20%



\## Evaluation Metrics



\- RMSE

\- Maximum Tracking Error

\- Settling Time

\- Control Effort

\- Execution Time



\## Expected Outputs



\- UAV state trajectories

\- Reference vs. actual trajectory plots

\- Sliding surface plots

\- Control signal plots

\- Performance tables

\- Robustness comparison plots

