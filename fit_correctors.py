import numpy as np
import matplotlib.pyplot as plt

from bicycle_models import VehicleParams, G
from correctors import FormulaCorrection, PolyCorrection, LUTCorrection, MLPCorrection


def load(tire):
    d = np.load(f"data_{tire}.npz")
    return d["v"], d["kappa"], d["dd"], d["is_test"]


def fit_all(tire):
    """한 타이어 데이터로 보정기 3종 학습 (학습용 80%만 사용)"""
    v, k, dd, is_test = load(tire)
    tr = ~is_test
    print(f"[{tire}] 학습 {tr.sum()}개")
    formula = FormulaCorrection()
    formula.K = VehicleParams().K_us
    return [formula,
            PolyCorrection().fit(v[tr], k[tr], dd[tr]),
            LUTCorrection().fit(v[tr], k[tr], dd[tr]),
            MLPCorrection().fit(v[tr], k[tr], dd[tr])]


if __name__ == "__main__":
    p = VehicleParams()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

    for ax, tire in zip(axes, ["linear", "pacejka"]):
        correctors = fit_all(tire)
        v, k, dd, is_test = load(tire)

        # 테스트 데이터(학습에 안 쓴 20%)에서 오차
        print(f"[{tire}] 테스트 RMSE (Δδ):")
        for c in correctors:
            pred = np.array([c(vi, ki) for vi, ki in zip(v[is_test], k[is_test])])
            rmse = np.sqrt(np.mean((pred - dd[is_test]) ** 2))
            print(f"    {c.name:8s} {rmse*1000:6.2f} mrad")
            

        # 그림: 왼쪽 선회 데이터 점 + 각 보정기 곡선 (속도 3개)
        left = k > 0
        ax.scatter(v[left] ** 2 * k[left], dd[left] * 1000, s=6, color="lightgray", label="data")
        for vv, ls in [(2.0, ":"), (3.0, "--"), (3.8, "-")]:
            kk = np.linspace(0.01, p.mu * G / vv ** 2, 60)
            for c, color in zip(correctors, ["k", "tab:green", "tab:orange", "tab:blue"]):
                y = [c(vv, ki) * 1000 for ki in kk]
                ax.plot(vv ** 2 * kk, y, ls, color=color,
                        label=f"{c.name}" if vv == 3.8 else None)
        ax.set_title(f"{tire} tire  (lines: v = 2.0 / 3.0 / 3.8 m/s)")
        ax.set_xlabel(r"$a_y$ [m/s$^2$]")
        ax.grid(alpha=0.3); ax.legend()

    axes[0].set_ylabel(r"$\Delta\delta$ [mrad]")
    plt.tight_layout()
    plt.savefig("correctors.png", dpi=150)
    print("saved correctors.png")
    plt.show()
