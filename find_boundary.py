import time
import numpy as np
from bicycle_models import VehicleParams, DynamicBicycle, G
from paths import make_oval, ARC
from run_experiments import run, metrics, train_correctors, to_array


def sat_ratio(log, path, v, p, skip_laps=1):
    """원호 구간에서 조향각이 delta_max에 걸린 비율"""
    after = log["t"] > skip_laps * path.length / v
    on_arc = after & (path.seg[log["idx"]] == ARC)
    return float(np.mean(np.abs(log["delta"][on_arc]) >= p.delta_max - 1e-6))


if __name__ == "__main__":
    t0 = time.time()
    p = VehicleParams()
    path = make_oval()
    kmax = np.max(np.abs(path.kappa))

    ratios = np.array([0.66, 0.70, 0.74, 0.77, 0.80, 0.83, 0.86, 0.89, 0.92, 0.95])
    speeds = np.sqrt(ratios * p.mu * G / kmax)          # a_y/μg → 속도
    seeds = [0, 1, 2, 3, 4]
    names = ["Formula", "LUT", "MLP"]
    print("speeds [m/s]:", np.round(speeds, 3))

    res = {}
    for exp, tire in [("E1", "linear"), ("E2", "pacejka")]:
        model = DynamicBicycle(p, tire=tire)
        for seed in seeds:
            c = train_correctors(tire, seed)
            for name in names:
                print(f"[{exp}] {name} seed {seed}  ({time.time() - t0:.0f} s)")
                row = []
                for v in speeds:
                    log = run(model, path, v, c[name], True)
                    m = metrics(log, path, v)
                    m["sat"] = sat_ratio(log, path, v, p)
                    row.append(m)
                res.setdefault((exp, name), []).append(row)

    # 시드별 감소율 (MLP vs Formula)
    gain = {}
    for exp in ["E1", "E2"]:
        f = to_array(res[(exp, "Formula")], "arc_rms")
        m = to_array(res[(exp, "MLP")], "arc_rms")
        gain[exp] = 100 * (1 - m / f)                   # (시드 × 속도)

    print("\n a_y/μg |   E1 gain [%]   |   E2 gain [%]   | E2−E1 [%p]")
    for j, r in enumerate(ratios):
        e1, e2 = gain["E1"][:, j], gain["E2"][:, j]
        print(f"  {r:4.2f}  | {e1.mean():6.1f} ± {e1.std(ddof=1):4.1f} "
              f"| {e2.mean():6.1f} ± {e2.std(ddof=1):4.1f} | {e2.mean() - e1.mean():6.1f}")

    print("\n원호 RMS [cm] (평균) / 조향 포화 비율 [%] (평균)")
    for exp in ["E1", "E2"]:
        for name in names:
            a = to_array(res[(exp, name)], "arc_rms").mean(axis=0) * 100
            s = to_array(res[(exp, name)], "sat").mean(axis=0) * 100
            print(f"  {exp} {name:8s} RMS:", " ".join(f"{x:6.2f}" for x in a))
            print(f"  {exp} {name:8s} sat:", " ".join(f"{x:6.1f}" for x in s))

    save = {"ratios": ratios, "speeds": speeds,
            "gain_E1": gain["E1"], "gain_E2": gain["E2"]}
    for e in ["E1", "E2"]:
        for n in names:
            for key in ["arc_rms", "arc_mean", "steer_rate", "sat"]:
                save[f"{e}_{n}_{key}"] = to_array(res[(e, n)], key)
    np.savez("boundary_results.npz", **save)
    print(f"\nsaved boundary_results.npz  (총 {time.time() - t0:.0f} s)")
