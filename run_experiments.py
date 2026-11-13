import time
import numpy as np
import matplotlib.pyplot as plt

from bicycle_models import VehicleParams, KinematicBicycle, DynamicBicycle, simulate, G
from paths import make_oval, ARC
from pure_pursuit import PurePursuit
from correctors import FormulaCorrection, LUTCorrection, MLPCorrection

QUICK = False     # True: 속도 3개, 반복 1회 (코드 확인용, 3~5분)


def run(model, path, v, corrector=None, use_slip=False, n_laps=3, dt=0.002):
    """한 조건으로 오벌 n_laps 바퀴 주행 → 기록 반환"""
    pp = PurePursuit(path, model.p, speed=v, corrector=corrector, use_slip=use_slip)
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
    steer_rate = np.diff(log["delta"]) / np.diff(log["t"])
    
    # 부호 있는 횡오차: 코너 바깥 +, 안쪽 −
    # e_y는 경로 왼쪽이 +라서, 좌회전(κ>0)이면 왼쪽 = 안쪽 → 부호를 뒤집음
    k_sign = np.sign(path.kappa[log["idx"]])
    e_out = -k_sign * e
    
    return {
        "arc_rms": rms(e[on_arc]),
        "arc_max": float(np.max(np.abs(e[on_arc]))),
        "arc_mean": float(np.mean(e_out[on_arc])),
        "steer_rate": rms(steer_rate[after[1:]]),
    }


def train_correctors(tire, seed, train_ratio=0.8, suffix=""):
    """seed마다 학습 데이터 80%를 다르게 뽑아 보정기 3종 학습"""
    d = np.load(f"data_{tire}{suffix}.npz")
    v, k, dd = d["v"], d["kappa"], d["dd"]
    rng = np.random.default_rng(seed)
    tr = rng.random(len(v)) < train_ratio
    formula = FormulaCorrection()
    formula.K = VehicleParams().K_us
    return {
        "Formula": FormulaCorrection().fit(v[tr], k[tr], dd[tr]),
        "LUT": LUTCorrection().fit(v[tr], k[tr], dd[tr]),
        "MLP": MLPCorrection(seed=seed).fit(v[tr], k[tr], dd[tr]),
    }


def to_array(runs, key):
    result = []
    for row in runs:
        values = []
        for m in row:
            values.append(m[key])
        result.append(values)
    return np.array(result)


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
    exps = [("E1", "linear"), ("E2", "pacejka")]
    modes = ["yaw", "slip"]
    ctrl_names = ["PP", "Formula", "LUT", "MLP"]

    # 참고선: 기구학 모델
    print("Kinematic 참고선 ...")
    kin = np.array([metrics(run(KinematicBicycle(p), path, v), path, v)["arc_rms"]
                    for v in speeds])

    # ---------------- 주행 ----------------
    results = {}      # results[(실험, 헤딩, 제어기)] = [[seed0 속도별 dict], ...]
    for exp, tire in exps:
        model = DynamicBicycle(p, tire=tire)

        # 보정기는 seed마다 한 번 학습해서 yaw/slip 둘 다에 사용
        trained = {seed: train_correctors(tire, seed) for seed in seeds}

        for mode in modes:
            slip = (mode == "slip")
            print(f"\n[{exp}/{mode}] PP ...  ({time.time() - t_start:.0f} s)")
            results[(exp, mode, "PP")] = [
                [metrics(run(model, path, v, None, slip), path, v) for v in speeds]]

            for name in ["Formula", "LUT", "MLP"]:
                results[(exp, mode, name)] = []
                for seed in seeds:
                    print(f"[{exp}/{mode}] {name} seed {seed} ...  "
                          f"({time.time() - t_start:.0f} s)")
                    c = trained[seed][name]
                    results[(exp, mode, name)].append(
                        [metrics(run(model, path, v, c, slip), path, v) for v in speeds])

    # ---------------- 표 ----------------
    save_dict = {"speeds": np.array(speeds), "ay_ratio": ay_ratio, "kinematic": kin}
    for exp, _ in exps:
        for mode in modes:
            print(f"\n===== {exp} / heading={mode} : 원호 횡오차 RMS [cm] (평균 ± 표준편차) =====")
            print("  a_y/μg |" + "".join(f"{n:>16s}" for n in ctrl_names) + "   | Kinematic")
            arr = {n: to_array(results[(exp, mode, n)], "arc_rms") * 100 for n in ctrl_names}
            for j, r in enumerate(ay_ratio):
                cells = "".join(f"{arr[n][:, j].mean():9.2f} ± {arr[n][:, j].std():4.2f}"
                                for n in ctrl_names)
                print(f"   {r:4.2f}  |{cells}   | {kin[j]*100:6.2f}")

            print(f"  --- MLP vs Formula 감소율 [%] (양수 = MLP가 나음) ---")
            for j, r in enumerate(ay_ratio):
                f = arr["Formula"][:, j].mean()
                m = arr["MLP"][:, j].mean()
                print(f"   a_y/μg={r:4.2f}   {100 * (1 - m / f):6.1f}")

            print(f"  --- 조향 변화율 RMS [rad/s] (평균) ---")
            for j, r in enumerate(ay_ratio):
                cells = "".join(
                    f"{n}: {to_array(results[(exp, mode, n)], 'steer_rate')[:, j].mean():6.3f}   "
                    for n in ctrl_names)
                print(f"   a_y/μg={r:4.2f}   {cells}")

            for n in ctrl_names:
                for key in ["arc_rms", "arc_max", "steer_rate"]:
                    save_dict[f"{exp}_{mode}_{n}_{key}"] = to_array(results[(exp, mode, n)], key)

    np.savez("experiment_results.npz", **save_dict)
    print("\nsaved experiment_results.npz")

    # ---------------- 그림 1: 오차 분해 (최고 속도) ----------------
    j = len(speeds) - 1
    bars = [("yaw", "PP", "PP"),
            ("yaw", "Formula", "+Formula"),
            ("yaw", "MLP", "+MLP"),
            ("slip", "PP", "+slip"),
            ("slip", "Formula", "+slip\n+Formula"),
            ("slip", "MLP", "+slip\n+MLP")]
    x = np.arange(len(bars))
    w = 0.38
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    for k, (exp, tire) in enumerate(exps):
        means, stds = [], []
        for mode, n, _ in bars:
            a = to_array(results[(exp, mode, n)], "arc_rms")[:, j] * 100
            means.append(a.mean()); stds.append(a.std())
        ax1.bar(x + (k - 0.5) * w, means, w, yerr=stds, capsize=3,
                label=f"{exp}: {tire} tire")
    ax1.axhline(kin[j] * 100, color="gray", ls="--", lw=1, label="Kinematic (ref.)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([b[2] for b in bars])
    ax1.set_ylabel("lateral error RMS on arcs [cm]")
    ax1.set_title(f"Error decomposition at $a_y/\\mu g$ = {ay_ratio[j]:.2f}")
    ax1.grid(alpha=0.3, axis="y"); ax1.legend()
    plt.tight_layout()
    plt.savefig("fig1_decomposition.png", dpi=150)

    # ---------------- 그림 2,3: E1, E2 ----------------
    styles = {"PP": ("tab:gray", "o"), "Formula": ("k", "s"),
              "LUT": ("tab:orange", "^"), "MLP": ("tab:blue", "D")}
    fig_names = {"E1": "E1_linear.png", "E2": "E2_pacejka.png"}

    for exp, tire in exps:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for n in ctrl_names:
            a = to_array(results[(exp, "slip", n)], "arc_rms") * 100
            color, marker = styles[n]
            label = "PP + slip" if n == "PP" else f"PP + slip + {n}"
            ax.errorbar(ay_ratio, a.mean(axis=0), yerr=a.std(axis=0),
                        color=color, marker=marker, capsize=3, label=label)
        ax.set_title(f"{exp}: {tire} tire (slip-aware heading)")
        ax.set_xlabel(r"$a_y / \mu g$")
        ax.set_ylabel("lateral error RMS on arcs [cm]")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_names[exp], dpi=300)

    print(f"saved decomposition.png, {', '.join(fig_names.values())}")
    plt.show()
