# Balance precision (quantization) thresholds

Task: keep N-link pendulum upright (theta=0). Success = all |theta_i|<0.5 rad
for 60 s across 8 seeds (init theta ~ U(-5e-4,5e-4)). Precision model: angle
measurements rounded to grid `dtheta`, pivot-velocity command rounded to grid
`dv`. Threshold = largest grid step still succeeding (log-bisection bracket;
headline number is the geometric mean of the bracket).

Controllers: **FD** = discrete LQR + finite-difference velocity estimate;
**Kalman** = discrete LQR + steady-state Kalman filter (knows dtheta/dv).
LQR weights tuned per N over r in {0.01,0.1,1}, q_theta in {1,10,100}.

## Headline (g=9.81, dt=0.01): best controller per N

| N | best ctrl | max dtheta (rad) | max dv (m/s) | LQR (q_theta, r) |
|---|-----------|------------------|--------------|------------------|
| 1 | FD | 0.641 | 0.91 | (100, 0.01) |
| 2 | FD | 0.087 | 0.987 | (100, 0.01) |
| 3 | FD | 0.0139 | 0.795 | (100, 0.01) |
| 4 | FD | 0.00254 | 0.695 | (100, 0.01) |
| 5 | FD | 0.000463 | 0.591 | (10, 0.01) |

## Both controllers (g=9.81, dt=0.01)

| N | FD max dtheta | KF max dtheta | FD max dv | KF max dv |
|---|---------------|---------------|-----------|-----------|
| 1 | 0.641 | 0.545 | 0.91 | 0.91 |
| 2 | 0.087 | 0.0682 | 0.987 | 0.676 |
| 3 | 0.0139 | 0.00926 | 0.795 | 0.56 |
| 4 | 0.00254 | 0.00247 | 0.695 | 0.354 |
| 5 | 0.000463 | 0.000658 | 0.591 | 0.416 |

## Joint trade-off (best controller, g=9.81, dt=0.01)

dv threshold when dtheta is held at a fraction of its own max. 'inf' means
dv was never the binding constraint in [1e-6, 1.0].

| N | ctrl | dtheta=maxdth/10 -> max dv | dtheta=maxdth/3 -> max dv |
|---|------|----------------------------|---------------------------|
| 1 | FD | >=1 (not binding) | >=1 (not binding) |
| 2 | FD | >=1 (not binding) | 0.96 |
| 3 | FD | >=1 (not binding) | 0.96 |
| 4 | FD | 0.774 | 0.839 |
| 5 | FD | 0.607 | 0.575 |

## dt scaling (best controller per N, g=9.81)

max dtheta and max dv at dt in {0.002, 0.005, 0.01, 0.02}.

| N | ctrl | metric | dt=0.002 | dt=0.005 | dt=0.01 | dt=0.02 | scaling |
|---|------|--------|-------|-------|-------|-------|---------|
| 1 | FD | max_dtheta | 0.862 | 0.733 | 0.641 | 0.56 | ~dt^-0.19 |
| 1 | FD | max_dv | 0.153 | 0.405 | 0.91 | 1 | ~dt^0.81 |
| 2 | FD | max_dtheta | 0.111 | 0.0918 | 0.087 | 0.074 | ~dt^-0.18 |
| 2 | FD | max_dv | 0.195 | 0.463 | 0.987 | 1 | ~dt^0.71 |
| 3 | FD | max_dtheta | 0.0132 | 0.0139 | 0.0139 | 0.0115 | ~dt^-0.06 |
| 3 | FD | max_dv | 0.175 | 0.373 | 0.795 | 1 | ~dt^0.76 |
| 4 | FD | max_dtheta | 0.00275 | 0.00268 | 0.00254 | 0.00184 | ~dt^-0.18 |
| 4 | FD | max_dv | 0.195 | 0.416 | 0.695 | 0.817 | ~dt^0.62 |
| 5 | FD | max_dtheta | 0.000624 | 0.000591 | 0.000463 | - | ~dt^-0.18 |
| 5 | FD | max_dv | 0.185 | 0.318 | 0.591 | - | ~dt^0.72 |

## g scaling (best controller, dt=0.01, N in {1,3,5})

| N | ctrl | metric | g=4.905 | g=9.81 | g=19.62 | scaling |
|---|------|--------|-------|-------|-------|---------|
| 1 | FD | max_dtheta | 0.607 | 0.641 | 0.658 | ~g^0.06 |
| 1 | FD | max_dv | 0.91 | 0.91 | 1 | ~g^0.07 |
| 3 | FD | max_dtheta | 0.0132 | 0.0139 | 0.0155 | ~g^0.12 |
| 3 | FD | max_dv | 0.575 | 0.795 | 1 | ~g^0.40 |
| 5 | FD | max_dtheta | 0.000463 | 0.000463 | 0.000373 | ~g^-0.16 |
| 5 | FD | max_dv | 0.427 | 0.591 | 0.676 | ~g^0.33 |

