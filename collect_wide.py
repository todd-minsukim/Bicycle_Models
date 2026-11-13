import numpy as np
from bicycle_models import VehicleParams
from collect_data import collect, save

p = VehicleParams()
speeds = np.linspace(1.5, 4.5, 13)     # 1.5 ~ 4.5 m/s, 0.25 간격
deltas = np.linspace(0.0, 0.40, 41)    # 0 ~ 0.40 rad, 0.01 간격 (delta_max = 0.42)

for tire in ["linear", "pacejka"]:
    data = collect(tire, speeds, deltas, p, ay_limit=1.0)
    save(data, p, f"data_{tire}_wide.npz")
