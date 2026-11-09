import time
import numpy as np
import matplotlib.pyplot as plt

from bicycle_models import VehicleParams, KinematicBicycle, DynamicBicycle, simulate, G
from paths import make_oval, ARC
from pure_pursuit import PurePursuit
from correctors import FormulaCorrection, LUTCorrection, MLPCorrection

QUICK = False     # True로 하면 속도 3개, 반복 1회만 (코드 확인용, 1~2분)


def run(model, path, v, corrector=None, n_laps=3, dt=0.002):
    """한 조건으로 오벌 n_laps 바퀴 주행 → 기록 반환"""
    pp = PurePursuit(path, model.p, speed=v, corrector=corrector)
    s0 = model.initial_state(x=model.p.lr, y=0.0, psi=0.0)
    T = n_laps * path.length / v
    simulate(model, s0, pp, T, dt)
    return pp.get_log()


def rms(a):
    return float(np.sqrt(np.mean(a ** 2)))


def metrics(log, path, v, skip_laps=1):
    """첫 바퀴 제외 지표 3개"""
    after = log["t"] > skip_laps * path.length / v
    on_arc = after & (path.seg[log["idx"]] == ARC)
    e = log["e_y"]
    steer_rate = np.diff(log["delta"]) / np.diff(log["t"])      # 조향 변화율 [rad/s]
    return {
        "arc_rms": rms(e[on_arc]),
        "arc_max": float(np.max(np.abs(e[on_arc]))),
        "steer_rate": rms(steer_rate[after[1:]]),
    }


def train_correctors(tire, seed, train_ratio=0.8):
    """seed마다 학습 데이터 80%를 다르게 뽑아 보정기 3종 학습"""
    d = np.load(f"data_{tire}.npz")
    v, k, dd = d["v"], d["kappa"], d["dd"]
    rng = np.random.default_rng(seed)
    tr = rng.random(len(v)) < train_ratio
    return {
        "Formula": FormulaCorrection().fit(v[tr], k[tr], dd[tr]),
        "LUT": LUTCorrection().fit(v[tr], k[tr], dd[tr]),
        "MLP": MLPCorrection(seed=seed).fit(v[tr], k[tr], dd[tr]),
    }


def to_array(runs, key):
    """runs[seed][speed] = dict  →  배열 (seed 수, 속도 수)"""
    return np.array([[m[key] for m in row] for row in runs])


if __name__ == "__main__":
    t_start = time.time()
    p = VehicleParams()
    path = make_oval()

    speeds = [2.0, 2.5, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8]
    seeds = [0, 1, 2]
    if QUICK:
        speeds, seeds = [2.5, 3.2, 3.8], [0]

    kmax = np.max(np.abs(path.kappa))
    ay_ratio = np.array([v ** 2 * kmax / (p.mu * G) for v in speeds])
    ctrl_names = ["PP", "Formula", "LUT", "MLP"]

    # 참고선: 기구학 모델 (동역학 효과 없음)
    print("Kinematic 참고선 ...")
    kin = np.array([metrics(run(KinematicBicycle(p), path, v), path, v)["arc_rms"]
                    for v in speeds])

    results = {}      # results[(실험, 제어기)] = [[seed0 속도별 dict], [seed1 ...], ...]
    for exp, tire in [("E1", "linear"), ("E2", "pacejka")]:
        model = DynamicBicycle(p, tire=tire)

        # 보정 없는 PP: 학습이 없어서 결과가 seed와 무관 → 한 번만
        print(f"\n[{exp}] {tire} / PP ...")
        results[(exp, "PP")] = [[metrics(run(model, path, v), path, v) for v in speeds]]

        for name in ["Formula", "LUT", "MLP"]:
            results[(exp, name)] = []
        for seed in seeds:
            print(f"[{exp}] {tire} / seed {seed}: 보정기 학습 ...")
            correctors = train_correctors(tire, seed)
            for name, c in correctors.items():
                print(f"[{exp}] {tire} / seed {seed} / {name} 주행 ...  "
                      f"({time.time() - t_start:.0f} s 경과)")
                results[(exp, name)].append(
                    [metrics(run(model, path, v, c), path, v) for v in speeds])

    # ---------------- 결과 표 ----------------
    save_dict = {"speeds": np.array(speeds), "ay_ratio": ay_ratio, "kinematic": kin}
    for exp in ["E1", "E2"]:
        print(f"\n===== {exp} : 원호 구간 횡오차 RMS [cm]  (평균 ± 표준편차) =====")
        print("  a_y/μg |" + "".join(f"{n:>16s}" for n in ctrl_names) + "   | Kinematic")
        arr = {n: to_array(results[(exp, n)], "arc_rms") * 100 for n in ctrl_names}
        for j, r in enumerate(ay_ratio):
            cells = "".join(f"{arr[n][:, j].mean():9.2f} ± {arr[n][:, j].std():4.2f}"
                            for n in ctrl_names)
            print(f"   {r:4.2f}  |{cells}   | {kin[j]*100:6.2f}")

        print(f"\n----- {exp} : PP 대비 원호 RMS 감소율 [%] -----")
        for j, r in enumerate(ay_ratio):
            base = arr["PP"][0, j]
            cells = "".join(f"{n}: {100 * (1 - arr[n][:, j].mean() / base):6.1f}   "
                            for n in ["Formula", "LUT", "MLP"])
            print(f"   a_y/μg={r:4.2f}   {cells}")

        print(f"\n----- {exp} : 조향 변화율 RMS [rad/s] (평균) -----")
        for j, r in enumerate(ay_ratio):
            cells = "".join(f"{n}: {to_array(results[(exp, n)], 'steer_rate')[:, j].mean():6.3f}   "
                            for n in ctrl_names)
            print(f"   a_y/μg={r:4.2f}   {cells}")

        for n in ctrl_names:
            for key in ["arc_rms", "arc_max", "steer_rate"]:
                save_dict[f"{exp}_{n}_{key}"] = to_array(results[(exp, n)], key)

    np.savez("experiment_results.npz", **save_dict)
    print("\nsaved experiment_results.npz")

    # ---------------- 그림 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    styles = {"PP": ("tab:gray", "o"), "Formula": ("k", "s"),
              "LUT": ("tab:orange", "^"), "MLP": ("tab:blue", "D")}
    for ax, exp, title in zip(axes, ["E1", "E2"],
                              ["E1: linear tire", "E2: Pacejka tire"]):
        ax.plot(ay_ratio, kin * 100, "--", color="lightgray", label="Kinematic (ref.)")
        for n in ctrl_names:
            a = to_array(results[(exp, n)], "arc_rms") * 100
            color, marker = styles[n]
            label = "PP (no corr.)" if n == "PP" else f"PP + {n}"
            ax.errorbar(ay_ratio, a.mean(axis=0), yerr=a.std(axis=0),
                        color=color, marker=marker, capsize=3, label=label)
        ax.set_title(title)
        ax.set_xlabel(r"$a_y / \mu g$")
        ax.grid(alpha=0.3); ax.legend()
    axes[0].set_ylabel("lateral error RMS on arcs [cm]")
    plt.tight_layout()
    plt.savefig("experiments_E1_E2.png", dpi=150)
    print(f"saved experiments_E1_E2.png  (총 {time.time() - t_start:.0f} s)")
    plt.show()
