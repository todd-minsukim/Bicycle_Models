import time
import numpy as np

from bicycle_models import VehicleParams, DynamicBicycle, G
from paths import make_oval
from run_experiments import run, metrics, train_correctors, to_array

t0 = time.time()
p = VehicleParams()
path = make_oval()

speeds = [2.0, 2.5, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8]
seeds = [0, 1, 2]
kmax = np.max(np.abs(path.kappa))
ay_ratio = np.array([v ** 2 * kmax / (p.mu * G) for v in speeds])

model = DynamicBicycle(p, tire="pacejka")
trained = {s: train_correctors("pacejka", s) for s in seeds}

# mode -> (use_slip, slip_linear_pred)
modes = {"yaw": (False, False), "slip": (True, False), "pred": (True, True)}

# ---------- 1) 부호 확인용: 보정 없는 PP ----------
# pred가 slip과 비슷하고 yaw보다 작으면 부호가 맞게 들어간 것
print("PP only, arc RMS [cm]")
for j, v in enumerate(speeds):
    row = [metrics(run(model, path, v, None, u, slip_pred=pr), path, v)["arc_rms"] * 100
           for u, pr in modes.values()]
    print(f"  a_y/mug={ay_ratio[j]:.2f}   yaw {row[0]:6.2f}   slip {row[1]:6.2f}   pred {row[2]:6.2f}")

# ---------- 2) 해석적 vs MLP ----------
res = {}
for mode in ["slip", "pred"]:
    u, pr = modes[mode]
    for name in ["Formula", "MLP"]:
        print(f"[{mode}] {name} ...  ({time.time() - t0:.0f} s)")
        res[(mode, name)] = [[metrics(run(model, path, v, trained[s][name], u, slip_pred=pr), path, v)
                              for v in speeds] for s in seeds]

save = {"speeds": np.array(speeds), "ay_ratio": ay_ratio}
for mode in ["slip", "pred"]:
    f = to_array(res[(mode, "Formula")], "arc_rms") * 100
    m = to_array(res[(mode, "MLP")], "arc_rms") * 100
    print(f"\n===== E2 / heading={mode} =====")
    print("  a_y/mug   Formula [cm]    MLP [cm]    MLP vs Formula [%]")
    for j, r in enumerate(ay_ratio):
        print(f"   {r:4.2f}    {f[:, j].mean():6.2f}       {m[:, j].mean():6.2f}       "
              f"{100 * (1 - m[:, j].mean() / f[:, j].mean()):6.1f}")
    for name in ["Formula", "MLP"]:
        for key in ["arc_rms", "arc_mean"]:
            save[f"E2_{mode}_{name}_{key}"] = to_array(res[(mode, name)], key)

np.savez("pred_heading_results.npz", **save)
print(f"\nsaved pred_heading_results.npz  ({time.time() - t0:.0f} s)")
