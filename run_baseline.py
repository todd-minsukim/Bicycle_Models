import numpy as np
import matplotlib.pyplot as plt

from bicycle_models import VehicleParams, KinematicBicycle, DynamicBicycle, simulate, G
from paths import make_oval, ARC
from pure_pursuit import PurePursuit


def run(model, path, v, n_laps=3, dt=0.002):
    """한 모델, 한 속도로 트랙을 n_laps 바퀴 주행"""
    pp = PurePursuit(path, model.p, speed=v)
    s0 = model.initial_state(x=model.p.lr, y=0.0, psi=0.0)   # 뒷축이 (0, 0)에 오도록
    T = n_laps * path.length / v                              # n_laps 바퀴 도는 시간
    t, S, U = simulate(model, s0, pp, T, dt)
    return S, pp.get_log()


def rms(a):
    return float(np.sqrt(np.mean(a ** 2)))


def evaluate(log, path, v, p, skip_laps=1):
    """첫 바퀴는 버리고 계산
    반환: 원호 구간 RMS, 전체 RMS, 최대 |e_y|, 조향 포화 비율"""
    after = log["t"] > skip_laps * path.length / v           # 첫 바퀴 이후
    on_arc = after & (path.seg[log["idx"]] == ARC)            # 그중 원호 구간
    e = log["e_y"]
    sat = np.mean(np.abs(log["delta"][after]) >= p.delta_max - 1e-9)
    return rms(e[on_arc]), rms(e[after]), float(np.max(np.abs(e[after]))), float(sat)


if __name__ == "__main__":
    p = VehicleParams()
    path = make_oval()

    models = {
        "Kinematic": KinematicBicycle(p),
        "Dynamic (linear)": DynamicBicycle(p, tire="linear"),
        "Dynamic (Pacejka)": DynamicBicycle(p, tire="pacejka"),
    }

    # ---------------- (A) 동작 확인 ----------------
    S, log = run(models["Kinematic"], path, v=2.0)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(np.append(path.x, path.x[0]), np.append(path.y, path.y[0]),
               "k--", lw=1, label="reference")
    ax[0].plot(S[:, 0], S[:, 1], lw=1, label="vehicle (CG)")
    ax[0].axis("equal"); ax[0].grid(alpha=0.3); ax[0].legend()
    ax[0].set_title("Sanity check: Kinematic, v = 2.0 m/s, 3 laps")
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]")

    ax[1].plot(log["t"], log["e_y"] * 100, lw=1)
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("e_y [cm]")
    ax[1].set_title("Lateral error over time")
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("sanity_check.png", dpi=150)
    print("saved sanity_check.png\n")

    # ---------------- (B) 속도 스윕 ----------------
    speeds = [2.0, 2.5, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8]
    kmax = np.max(np.abs(path.kappa))
    ay_ratio = np.array([v ** 2 * kmax / (p.mu * G) for v in speeds])

    results = {}
    for name, model in models.items():
        arc_rms = []
        for v in speeds:
            _, log = run(model, path, v)
            r_arc, r_all, e_max, sat = evaluate(log, path, v, p)
            arc_rms.append(r_arc)
            print(f"{name:18s} v={v:.1f} (a_y/μg={v**2*kmax/(p.mu*G):.2f})  "
                  f"arc RMS={r_arc*100:6.2f} cm  all RMS={r_all*100:6.2f} cm  "
                  f"max={e_max*100:6.2f} cm  sat={sat*100:4.1f}%")
        results[name] = np.array(arc_rms)
        print()

    # 체크포인트: 동역학 때문에 늘어난 오차 (동역학 − 기구학)
    print("=== 동역학 때문에 늘어난 원호 구간 RMS [cm] (= 보정기가 줄일 수 있는 몫) ===")
    for v, r, d_lin, d_pac in zip(speeds, ay_ratio,
                                  (results["Dynamic (linear)"] - results["Kinematic"]) * 100,
                                  (results["Dynamic (Pacejka)"] - results["Kinematic"]) * 100):
        print(f"v={v:.1f}  a_y/μg={r:.2f}   linear: {d_lin:+6.2f}   Pacejka: {d_pac:+6.2f}")

    plt.figure(figsize=(7, 4))
    for name, vals in results.items():
        plt.plot(ay_ratio, vals * 100, "o-", label=name)
    plt.xlabel(r"$a_y / \mu g$")
    plt.ylabel("lateral error RMS on arcs [cm]")
    plt.title("Baseline PP (no correction)")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig("baseline_rms.png", dpi=150)
    print("\nsaved baseline_rms.png")
    plt.show()
