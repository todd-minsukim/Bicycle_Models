import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

THRESH = 10.0

d = np.load("boundary_results.npz")
r = d["ratios"]

# ---- 그림 A: E2 세 보정기 (회색 PP 제외) ----
styles = {"Formula": ("k", "s"), "Poly": ("tab:green", "v"),
          "LUT": ("tab:orange", "^"), "MLP": ("tab:blue", "D")}
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for n, (c, mk) in styles.items():
    a = d[f"E2_{n}_arc_rms"] * 100
    ax.errorbar(r, a.mean(0), yerr=a.std(0, ddof=1), color=c, marker=mk,
                capsize=3, label=f"PP + slip + {n}")
ax.set_xlabel(r"$a_y / \mu g$")
ax.set_ylabel("lateral error RMS on arcs [cm]")
ax.set_title("E2: Pacejka tire (slip-aware heading)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("fig_E2_correctors_zoom.png", dpi=300)

# ---- 감소율과 유의성 ----
f = d["E2_Formula_arc_rms"]
m = d["E2_MLP_arc_rms"]
gain = d["gain_E2"]
g_mean, g_std = gain.mean(0), gain.std(0, ddof=1)
p = np.array([ttest_rel(f[:, j], m[:, j], alternative="greater").pvalue
              for j in range(len(r))])
sat = d["E2_MLP_sat"].mean(0) * 100

print(" a_y/μg | gain [%]       | p-value | sat [%] | dominant")
for j in range(len(r)):
    dom = g_mean[j] >= THRESH and p[j] < 0.05
    print(f"  {r[j]:4.2f}  | {g_mean[j]:5.1f} ± {g_std[j]:4.1f} | {p[j]:.3f}   "
          f"| {sat[j]:5.1f}   | {'O' if dom else '-'}")

# 기준선을 처음 넘는 지점 (선형 보간)
def crossing(r, g, thresh=THRESH):
    for j in range(1, len(r)):
        if g[j - 1] < thresh <= g[j]:
            return r[j - 1] + (thresh - g[j - 1]) * (r[j] - r[j - 1]) / (g[j] - g[j - 1])
    return None

cross = crossing(r, g_mean)
d08 = np.load("boundary_results_mu08.npz")
g08 = d08["gain_E2"]
cross08 = crossing(d08["ratios"], g08.mean(0))
print(f"10% 교차점: mu=1.0 -> {cross}, mu=0.8 -> {cross08}")

# ---- 그림 B: 감소율 + 기준선 + 교차점 ----
fig, ax = plt.subplots(figsize=(6.5, 4.5))
g1 = d["gain_E1"]
ax.errorbar(r, g1.mean(0), yerr=g1.std(0, ddof=1), color="tab:gray", marker="o",
            capsize=3, label="E1 (linear)")
ax.errorbar(r, g_mean, yerr=g_std, color="tab:blue", marker="D",
            capsize=3, label=r"E2 (Pacejka, $\mu$ = 1.0)")
ax.errorbar(d08["ratios"], g08.mean(0), yerr=g08.std(0, ddof=1), color="tab:purple",
            marker="s", capsize=3, label=r"E2 (Pacejka, $\mu$ = 0.8)")
ax.axhline(THRESH, color="k", ls="--", lw=1, label=f"{THRESH:.0f}% threshold")
ax.axhline(0, color="k", lw=0.5)
if cross is not None:
    ax.axvline(cross, color="tab:red", ls=":", lw=1.5)
    ax.annotate(f"{cross:.2f}", (cross, THRESH), xytext=(5, 5),
                textcoords="offset points", color="tab:red")
ax.set_xlabel(r"$a_y / \mu g$")
ax.set_ylabel("MLP error reduction vs. analytical [%]")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("fig5_dominance.png", dpi=300)
e = np.load("experiment_results.npz")
ay = e["ay_ratio"]
styles5 = {"PP": ("tab:gray", "o"), "Formula": ("k", "s"), "Poly": ("tab:green", "v"),
           "LUT": ("tab:orange", "^"), "MLP": ("tab:blue", "D")}
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (exp, title) in zip(axes, [("E1", "(a) E1: linear tire"),
                                    ("E2", "(b) E2: Pacejka tire")]):
    for n, (c, mk) in styles5.items():
        a = e[f"{exp}_slip_{n}_arc_rms"] * 100
        label = "PP + slip" if n == "PP" else f"PP + slip + {n}"
        ax.errorbar(ay, a.mean(0), yerr=a.std(0), color=c, marker=mk,
                    capsize=3, label=label)
    ax.set_title(title)
    ax.set_xlabel(r"$a_y / \mu g$")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("lateral error RMS on arcs [cm]")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig("fig3_E1_E2.png", dpi=300)
plt.show()
