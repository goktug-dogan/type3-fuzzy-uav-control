\# Type-3 Fuzzy UAV Control



Reproduction and experimental extension of an adaptive Type-3 Fuzzy Logic System (T3-FLS) based UAV controller.



The project is based on:



> J. Cai, H. Zhang, A. Khadakar, A. Mohammadzadeh, and C. Zhang,  

> "Automatic control of UAVs: new adaptive rules and type-3 fuzzy stabilizer,"  

> Complex \& Intelligent Systems, vol. 10, pp. 7235–7248, 2024.  

> DOI: 10.1007/s40747-024-01434-y



\---



\## Project Overview



The reference study proposes an adaptive UAV controller combining:



\- UAV nonlinear dynamics

\- Sliding Mode Control (SMC)

\- Online adaptive parameter estimation

\- Type-3 Fuzzy Logic System (T3-FLS)

\- Fuzzy compensation for modeling errors and disturbances



The objectives of this project are:



1\. Reproduce the main mathematical components of the reference study.

2\. Implement the UAV dynamics and adaptive controller.

3\. Implement the Type-3 fuzzy inference structure.

4\. Investigate discrepancies between the published equations and figures.

5\. Develop a stabilized tracking extension.

6\. Evaluate robustness under actuator/control-effectiveness uncertainty.

7\. Perform an ablation study with T3-FLS enabled and disabled.



\---



\## Reference UAV Model



The UAV state vector is implemented as:



```text

X = \[

&#x20;   phi,

&#x20;   phi\_dot,

&#x20;   theta,

&#x20;   theta\_dot,

&#x20;   psi,

&#x20;   psi\_dot,

&#x20;   z,

&#x20;   z\_dot

]

```



where:



\- `phi` = roll angle

\- `theta` = pitch angle

\- `psi` = yaw angle

\- `z` = altitude



The four control channels are:



```text

u1 = total thrust / altitude control

u2 = roll control

u3 = pitch control

u4 = yaw control

```



The model parameters are taken from the reference paper where explicitly available.



\---



\## Reference Trajectory



The reference commands used in the simulation are:



```text

phi\_d   = pi / 3

theta\_d = pi / 6

psi\_d   = pi / 2

z\_d     = 3 m

```



The initial UAV states are:



```text

chi\_i(0) = 0.10

```



for all eight states, following the simulation setup reported in the paper.



\---



\## Adaptive Controller



The project implements the adaptive controller structure given in the reference study, including:



```text

zeta\_hat\_1 ... zeta\_hat\_9

h\_hat\_1 ... h\_hat\_3

m\_hat

```



Initial estimates:



```text

zeta\_hat\_i(0) = 0.10



h\_hat\_1(0) = 0.10

h\_hat\_2(0) = 0.20

h\_hat\_3(0) = 0.30



m\_hat(0) = 0.65

```



The reported adaptation parameters are also reproduced:



```text

alpha\_i = 0.05

beta\_i  = 0.05

vartheta = 3 / 7

```



A signed fractional-power implementation is used for terms involving:



```text

s^(3/7)

```



to safely handle negative sliding-surface values.



\---



\## Type-3 Fuzzy Logic System



The Type-3 fuzzy inference implementation follows the mathematical structure of Eqs. (19)–(29) in the reference paper.



Implemented components include:



\- Type-3 membership functions

\- Lower and upper slices

\- Product-based rule firing

\- z-slice aggregation

\- Adaptive consequent weights

\- Four fuzzy compensation outputs



The controller mapping is:



```text

f1 -> altitude

f2 -> roll

f3 -> pitch

f4 -> yaw

```



\### Important Reproduction Limitation



The paper defines the mathematical Type-3 FLS structure but does not fully report several numerical configuration details required for exact reproduction, including:



```text

number of membership functions

membership-function centers

membership-function spreads

exact z-slice configuration

complete fuzzy rule-base configuration

initial fuzzy consequent parameters

```



Therefore, these parameters are explicitly treated as implementation-specific choices in this project.



The current implementation uses three membership functions per input:



```text

Negative

Zero

Positive

```



with nine rules per two-input fuzzy channel.



\---



\## Reproduction Discrepancies



During implementation, several differences were observed between the numerical values obtained directly from the published equations and the trajectories displayed in the paper.



Examples include:



\- Initial sliding-surface values

\- Initial controller outputs

\- Adaptive parameter trajectories

\- Unspecified Type-3 fuzzy parameters



For example, applying the published initial conditions directly to the implemented equations gives approximately:



```text

Initial sliding surfaces:



\[-1.7944, -0.7472, -2.8416, -5.7000]

```



and initial control values approximately:



```text

u1 = 25.01

u2 = 1.77

u3 = 1.45

u4 = 4.20

```



These do not fully agree with the values visually shown in the paper figures.



A detailed record is available in:



```text

references/reproduction\_discrepancies.md

```



For this reason, the project does not claim exact numerical reproduction of every published figure.



\---



\## Proposed Extension



A separate stabilized tracking extension was developed without modifying the original equation-based baseline.



The proposed version introduces:



\- Tracking-error integral sliding surface

\- Denominator regularization

\- Control-input bounds

\- Adaptive-parameter projection

\- Robustness experiments

\- Type-3 fuzzy ablation experiments



The proposed sliding surface is:



```text

s = e\_dot + alpha \* e + k \* integral(e)

```



instead of the baseline form:



```text

s = e\_dot + alpha \* e + k \* integral(s)

```



This modification directly integrates tracking error and improves long-term reference tracking in the implemented system.



The proposed extension is not presented as part of the original paper.



\---



\## Proposed Controller Results



A 10-second simulation with:



```text

dt = 0.001 s

```



produced the following final outputs:



```text

phi   = 1.060372

theta = 0.540540

psi   = 1.582792

z     = 2.943021

```



Reference values:



```text

phi\_d   = 1.047198

theta\_d = 0.523599

psi\_d   = 1.570796

z\_d     = 3.000000

```



Final errors:



```text

phi   =  0.013175

theta =  0.016941

psi   =  0.011996

z     = -0.056979

```



\---



\## Tracking Metrics



Full 10-second simulation MSE:



| Signal | MSE | RMSE |

|---|---:|---:|

| Roll (`phi`) | 0.030634 | 0.175024 |

| Pitch (`theta`) | 0.011008 | 0.104919 |

| Yaw (`psi`) | 0.144001 | 0.379475 |

| Altitude (`z`) | 0.985692 | 0.992820 |



Steady-state metrics are also calculated for:



```text

5 s <= t <= 10 s

```



| Signal | Steady-State MSE | Steady-State RMSE |

|---|---:|---:|

| Roll (`phi`) | 0.000612 | 0.024735 |

| Pitch (`theta`) | 0.004210 | 0.064881 |

| Yaw (`psi`) | 0.049742 | 0.223029 |

| Altitude (`z`) | 0.126642 | 0.355868 |



\---



\## Comparison with Published MSE



The reference paper reports the following MSE values for its proposed controller:



| Signal | Paper-reported MSE | This project's full 10 s MSE |

|---|---:|---:|

| Roll | 0.0284 | 0.0306 |

| Pitch | 0.0069 | 0.0110 |

| Yaw | 0.0612 | 0.1440 |

| Altitude | 0.2236 | 0.9857 |



The steady-state MSE of the proposed extension is lower than the paper-reported values for all four channels.



However, these values must \*\*not\*\* be interpreted as a direct performance superiority claim because the paper does not fully specify the exact time interval and numerical evaluation protocol used for Table 2.



The comparison is therefore presented as an experimental reference rather than an exact benchmark.



\---



\## Robustness Experiment



The proposed controller is evaluated under time-varying control-effectiveness uncertainty.



The tested uncertainty levels are:



```text

0%

5%

10%

20%

```



Each control input is multiplied by a time-varying factor representing actuator/control-effectiveness variation.



The experiment compares:



```text

T3-FLS enabled

vs.

T3-FLS disabled

```



for every uncertainty level.



This results in eight experiments.



\---



\## Ablation Results



The ablation experiments show that the role of T3-FLS is channel-dependent in the current reproduction.



For the altitude channel, enabling T3-FLS substantially improves steady-state robustness.



Example at 20% uncertainty:



```text

T3-FLS enabled:

Altitude steady-state RMSE = 0.338895



T3-FLS disabled:

Altitude steady-state RMSE = 0.684222

```



However, roll, pitch, and yaw tracking in the current implementation can perform better without fuzzy compensation.



This result is not hidden or adjusted.



One likely explanation is that the reference paper does not provide the complete numerical configuration of the fuzzy system, so the implemented T3-FLS is not guaranteed to reproduce the authors' original tuning.



The experiments therefore demonstrate both the benefit and computational/control-effort trade-offs of the implemented fuzzy compensation.



\---



\## Control Effort and Computational Cost



The ablation experiments also measure:



```text

integral squared control effort

execution time

```



The current Type-3 implementation requires substantially more computation than the controller without fuzzy compensation.



The experiments showed that T3-FLS generally increased computational cost and commanded control effort while improving altitude robustness.



Therefore, the implemented T3-FLS exhibits a performance-cost trade-off rather than providing uniform improvement across all state channels.



\---



\## Project Structure



```text

type3-fuzzy-uav-control/

│

├── src/

│   ├── \_\_init\_\_.py

│   ├── uav\_model.py

│   ├── reference\_trajectory.py

│   ├── tracking.py

│   ├── adaptive\_controller.py

│   ├── controller.py

│   ├── type3\_fuzzy.py

│   ├── fuzzy\_config.py

│   ├── fuzzy\_system.py

│   ├── fuzzy\_compensator.py

│   ├── stabilized\_controller.py

│   └── proposed\_tracking.py

│

├── experiments/

│   ├── baseline/

│   │   ├── \_\_init\_\_.py

│   │   └── stability\_check.py

│   │

│   └── proposed/

│       ├── \_\_init\_\_.py

│       ├── stabilized\_simulation.py

│       ├── proposed\_stability\_check.py

│       ├── generate\_results.py

│       ├── compare\_with\_paper.py

│       └── robustness\_ablation.py

│

├── results/

│   ├── figures/

│   └── tables/

│

├── references/

│   ├── README.md

│   ├── reproduction\_plan.md

│   └── reproduction\_discrepancies.md

│

├── report/

├── presentation/

│

├── run\_all.py

├── requirements.txt

├── .gitignore

└── README.md

```



\---



\## Installation



Clone the repository:



```bash

git clone https://github.com/goktug-dogan/type3-fuzzy-uav-control.git

```



Enter the project directory:



```bash

cd type3-fuzzy-uav-control

```



Create a virtual environment:



```bash

python -m venv .venv

```



Activate it in Git Bash on Windows:



```bash

source .venv/Scripts/activate

```



Install dependencies:



```bash

pip install -r requirements.txt

```



\---



\## Quick Validation



Run:



```bash

python run\_all.py

```



This checks the main modules and the proposed closed-loop simulation.



A successful run should finish with:



```text

PROJECT VALIDATION COMPLETED



Modules passed: 10/10



All selected project checks PASSED.

```



The current validated quick run completes the following modules successfully:



```text

src.uav\_model

src.tracking

src.adaptive\_controller

src.controller

src.type3\_fuzzy

src.fuzzy\_system

src.fuzzy\_compensator

src.stabilized\_controller

src.proposed\_tracking

experiments.proposed.stabilized\_simulation

```



\---



\## Full Experiment Run



To execute the complete validation and experimental pipeline:



```bash

python run\_all.py --full

```



This includes:



```text

model validation

adaptive-controller validation

Type-3 fuzzy validation

baseline diagnostic

proposed-controller stability test

result generation

paper comparison

robustness experiments

T3-FLS ablation

```



Generated outputs are saved under:



```text

results/figures/

results/tables/

```



\---



\## Individual Experiments



Baseline equation-based diagnostic:



```bash

python -m experiments.baseline.stability\_check

```



Proposed long-term stability diagnostic:



```bash

python -m experiments.proposed.proposed\_stability\_check

```



Proposed 10-second simulation:



```bash

python -m experiments.proposed.stabilized\_simulation

```



Generate metrics and figures:



```bash

python -m experiments.proposed.generate\_results

```



Compare with paper-reported MSE:



```bash

python -m experiments.proposed.compare\_with\_paper

```



Run robustness and fuzzy ablation:



```bash

python -m experiments.proposed.robustness\_ablation

```



\---



\## Generated Results



Typical generated figures include:



```text

roll\_tracking.png

pitch\_tracking.png

yaw\_tracking.png

altitude\_tracking.png

control\_inputs.png

sliding\_surfaces.png

mass\_estimate.png

robustness\_roll\_phi.png

robustness\_pitch\_theta.png

robustness\_yaw\_psi.png

robustness\_altitude\_z.png

robustness\_control\_effort.png

```



Generated tables include:



```text

proposed\_time\_series.csv

proposed\_metrics.csv

proposed\_metrics.json

paper\_vs\_proposed.csv

robustness\_ablation.csv

```



\---



\## Main Experimental Findings



The main findings obtained from the current implementation are:



1\. Direct implementation of the reported equations initially moves the UAV toward the desired trajectory but encounters instability near the attitude-related altitude-control singularity.



2\. The term



```text

1 / (cos(phi) \* cos(theta))

```



can produce very large altitude-control inputs when the attitude approaches configurations where the denominator becomes small.



3\. The proposed tracking-error integral surface substantially improves long-term tracking.



4\. The stabilized proposed controller remains numerically finite during the 10-second simulation.



5\. At the end of the 10-second simulation, all four controlled outputs are close to their reference values.



6\. The Type-3 fuzzy subsystem provides a clear robustness benefit in the altitude channel in the implemented experiments.



7\. The Type-3 fuzzy subsystem does not uniformly improve the attitude channels in the current configuration.



8\. T3-FLS increases computational cost and control effort.



9\. Missing fuzzy-system configuration parameters in the original study prevent exact numerical reproduction of the authors' reported fuzzy-controller behavior.



\---



\## Limitations



The main limitations of this reproduction are:



1\. The reference paper does not provide the complete numerical T3-FLS configuration.



2\. Several values inferred directly from the published equations do not fully agree with the displayed figures.



3\. The numerical integration currently uses a fixed-step Euler implementation.



4\. Rotor-speed dynamics are not explicitly simulated.



5\. The gyroscopic rotor term is currently evaluated with:



```text

omega\_bar = 0

```



6\. The proposed stabilization parameters are implementation-specific.



7\. Robustness experiments use simulated time-varying actuator uncertainty rather than hardware flight tests.



8\. The current fuzzy system uses implementation-specific membership-function parameters, normalization ranges, slice values, and rule configuration.



9\. The reported steady-state comparison should not be interpreted as an exact benchmark against the paper because the original MSE evaluation protocol is not fully specified.



10\. Results represent a reproducible numerical study rather than an exact reconstruction of the authors' original simulation environment.



\---



\## Reproducibility



All reported project results are generated directly from the code in this repository.



No manually fabricated experiment values are used.



The experiment scripts save numerical results to CSV/JSON files and figures to the `results/` directory.



The repository commit history is maintained incrementally to document the development process.



The primary validation commands are:



```bash

python run\_all.py

```



and:



```bash

python run\_all.py --full

```



\---



\## Generative AI Usage



Generative AI tools were used during development for:



\- Code structure suggestions

\- Debugging assistance

\- Documentation support

\- Interpretation of mathematical expressions

\- Experiment organization



All generated code, equations, assumptions, experiment configurations, numerical results, and references are reviewed and verified before inclusion in the project.



Generative AI-generated numerical experiment results are not used.



All reported experimental values are produced by executing the project code.



\---



\## Reference



J. Cai, H. Zhang, A. Khadakar, A. Mohammadzadeh, and C. Zhang,  

"Automatic control of UAVs: new adaptive rules and type-3 fuzzy stabilizer,"  

Complex \& Intelligent Systems, vol. 10, pp. 7235–7248, 2024.  

doi: 10.1007/s40747-024-01434-y

