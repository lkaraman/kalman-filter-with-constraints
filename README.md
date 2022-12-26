# Constraint Kalman filter based on probability density truncation

Classifies longitudinal behavior of vehicles based on the probabilistic
output from the constrained Kalman filter.

The algorithm works as following:
 - vehicle's longitudinal and lateral position is fed to CACV (Constant Acceleration Constant Velocity)
Kalman filter
 - resulting estimated states are smoothed with [RTS smoother](https://pubmed.ncbi.nlm.nih.gov/22163819/)
 - new estimations and covariance matrices are used in probability truncation


For usage please download [NGSIM dataset](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm) (csv measurements on US101) and
place it in _data_ folder as **measurements.csv**

Based on the [MATLAB algorithm](https://academic.csuohio.edu/simond/kalmanconstrained/) written by Dan Simon