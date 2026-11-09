import numpy as np
from dataclasses import dataclass
 
G = 9.81
 
# 차량 파라미터 (F1TENTH 공개값을 참고해 1/10 스케일 수준 근사)

@dataclass
class VehicleParams:
    m: float = 3.47          # 질량 [kg]
    Iz: float = 0.04712      # 요 관성모멘트 [kg m^2]
    lf: float = 0.15875      # CG ~ 앞축 [m]
    lr: float = 0.17145      # CG ~ 뒷축 [m]
    Caf: float = 85.0        # 앞 코너링 강성 [N/rad]
    Car: float = 95.0        # 뒤 코너링 강성 [N/rad]
    mu: float = 1.0          # 노면 마찰계수
    delta_max: float = 0.42  # 최대 조향각 [rad]
 
    @property
    def L(self):
        return self.lf + self.lr
 
    @property
    def Fzf(self):  # 앞축 수직하중 [N]
        return self.m * G * self.lr / self.L
 
    @property
    def Fzr(self):  # 뒷축 수직하중 [N]
        return self.m * G * self.lf / self.L
 
    @property
    def K_us(self):  # 언더스티어 계수 [rad/(m/s^2)]
        return self.m / self.L * (self.lr / self.Caf - self.lf / self.Car)
 

# 타이어 모델: alpha [rad] -> Fy [N]

class LinearTire:
 
    def __init__(self, C_alpha):
        self.C_alpha = C_alpha
 
    def __call__(self, alpha):
        return self.C_alpha * alpha
 
 
class PacejkaTire:
    """Magic Formula: F_y = D sin(C atan(B a - E(B a - atan(B a))))
 
    def __init__(self, C_alpha, Fz, mu, C=1.3, E=0.0):
        self.D = mu * Fz
        self.C = C
        self.E = E
        self.B = C_alpha / (C * self.D)
 
    def __call__(self, alpha):
        Ba = self.B * alpha
        return self.D * np.sin(self.C * np.arctan(Ba - self.E * (Ba - np.arctan(Ba))))
 
 

# 적분기

def rk4(f, s, u, dt):
    k1 = f(s, u)
    k2 = f(s + 0.5 * dt * k1, u)
    k3 = f(s + 0.5 * dt * k2, u)
    k4 = f(s + dt * k3, u)
    return s + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
 
 

# Kinematic Bicylce Model
#   상태 s = [x, y, psi],  입력 u = [delta, v]
#   roll without slipping

class KinematicBicycle:
    state_names = ["x", "y", "psi"]
 
    def __init__(self, p: VehicleParams):
        self.p = p
 
    def deriv(self, s, u):
        _, _, psi = s
        delta, v = u
        beta = np.arctan(self.p.lr * np.tan(delta) / self.p.L)  # CM 슬립각
        return np.array([
            v * np.cos(psi + beta),
            v * np.sin(psi + beta),
            v * np.cos(beta) * np.tan(delta) / self.p.L,
        ])
 
    def step(self, s, u, dt):
        return rk4(self.deriv, s, u, dt)
 
    def yaw_rate(self, s, u):
        return self.deriv(s, u)[2]
 
    def initial_state(self, x=0.0, y=0.0, psi=0.0):
        return np.array([x, y, psi], dtype=float)
 
 

# Dynamic Bicycle Model
#   상태 s = [x, y, psi, vy, r],  입력 u = [delta, vx]
#   vx 일정 가정 (종방향 동역학 생략)

class DynamicBicycle:
    state_names = ["x", "y", "psi", "vy", "r"]
 
    def __init__(self, p: VehicleParams, tire="linear", pacejka_C=1.3, pacejka_E=0.0):
        self.p = p
        self.tire_type = tire
        if tire == "linear":
            self.tire_f = LinearTire(p.Caf)
            self.tire_r = LinearTire(p.Car)
        elif tire == "pacejka":
            self.tire_f = PacejkaTire(p.Caf, p.Fzf, p.mu, pacejka_C, pacejka_E)
            self.tire_r = PacejkaTire(p.Car, p.Fzr, p.mu, pacejka_C, pacejka_E)
        else:
            raise ValueError("tire must be 'linear' or 'pacejka'")
 
    def slip_angles(self, s, u):
        _, _, _, vy, r = s
        delta, vx = u
        af = delta - np.arctan2(vy + self.p.lf * r, vx)
        ar = -np.arctan2(vy - self.p.lr * r, vx)
        return af, ar
 
    def deriv(self, s, u):
        _, _, psi, vy, r = s
        delta, vx = u
        if vx < 0.3:
            raise ValueError("동역학 모델은 저속(vx<0.3 m/s)에서 수치적으로 불안정합니다.")
        p = self.p
        af, ar = self.slip_angles(s, u)
        Fyf = self.tire_f(af)
        Fyr = self.tire_r(ar)
        vy_dot = (Fyf * np.cos(delta) + Fyr) / p.m - vx * r
        r_dot = (p.lf * Fyf * np.cos(delta) - p.lr * Fyr) / p.Iz
        return np.array([
            vx * np.cos(psi) - vy * np.sin(psi),
            vx * np.sin(psi) + vy * np.cos(psi),
            r,
            vy_dot,
            r_dot,
        ])
 
    def step(self, s, u, dt):
        return rk4(self.deriv, s, u, dt)
 
    def yaw_rate(self, s, u):
        return s[4]
 
    def initial_state(self, x=0.0, y=0.0, psi=0.0, vy=0.0, r=0.0):
        return np.array([x, y, psi, vy, r], dtype=float)
 
 

# Simulation

def simulate(model, s0, control, T, dt=0.002):
    n = int(round(T / dt))
    S = np.zeros((n + 1, len(s0)))
    U = np.zeros((n + 1, 2))
    t = np.arange(n + 1) * dt
    S[0] = s0
    for k in range(n):
        delta, v = control(t[k], S[k])
        delta = np.clip(delta, -model.p.delta_max, model.p.delta_max)
        U[k] = (delta, v)
        S[k + 1] = model.step(S[k], U[k], dt)
    U[-1] = U[-2]
    return t, S, U
 
 
def steady_state_curvature(model, delta, v, T=3.0, dt=0.002, tol=1e-3):
    s0 = model.initial_state()
    t, S, U = simulate(model, s0, lambda t, s: (delta, v), T, dt)
    r_hist = np.array([model.yaw_rate(S[k], U[k]) for k in range(len(t))])
    tail = r_hist[t > T - 0.5]
    if np.ptp(tail) > tol * max(1.0, abs(tail.mean())):
        return None
    return tail.mean() / v
 
 
# Validation

if __name__ == "__main__":
    import matplotlib.pyplot as plt
 
    p = VehicleParams()
    kin = KinematicBicycle(p)
    dyn_lin = DynamicBicycle(p, tire="linear")
    dyn_pac = DynamicBicycle(p, tire="pacejka")
 
    print(f"L = {p.L:.4f} m, Fzf = {p.Fzf:.2f} N, Fzr = {p.Fzr:.2f} N")
    print(f"K_us = {p.K_us:.5f} rad/(m/s^2)  ({'언더스티어' if p.K_us > 0 else '오버스티어'})")
 
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
 
    # (1) 저속 비교
    delta, v, T = 0.2, 0.5, 15.0
    for model, name, ls in [(kin, "Kinematic", "-"), (dyn_lin, "Dynamic (linear)", "--"),
                            (dyn_pac, "Dynamic (Pacejka)", ":")]:
        _, S, _ = simulate(model, model.initial_state(), lambda t, s: (delta, v), T)
        ax[0].plot(S[:, 0], S[:, 1], ls, label=name)
    ax[0].set_title(f"(1) Low speed  v={v} m/s, δ={delta} rad")
    ax[0].axis("equal"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]")
 
    # (2) 중속 비교: 같은 조향각에서 회전반경
    delta, v, T = 0.15, 4.0, 4.0
    for model, name, ls in [(kin, "Kinematic", "-"), (dyn_lin, "Dynamic (linear)", "--"),
                            (dyn_pac, "Dynamic (Pacejka)", ":")]:
        _, S, U = simulate(model, model.initial_state(), lambda t, s: (delta, v), T)
        kappa = model.yaw_rate(S[-1], U[-1]) / v
        ax[1].plot(S[:, 0], S[:, 1], ls, label=f"{name}  R={1/kappa:.2f} m")
    ax[1].set_title(f"(2) Mid speed  v={v} m/s, δ={delta} rad")
    ax[1].axis("equal"); ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[1].set_xlabel("x [m]"); ax[1].set_ylabel("y [m]")
 
    # (3) 정상상태 선회: δ - Lκ vs a_y
    speeds = [1.5, 2.5, 3.5, 4.5, 5.5]
    deltas = np.linspace(0.01, 0.20, 18)  # 소각 범위: 공식이 소각 근사로 유도됐으므로
    ay_max = 0
    for model, name, marker in [(dyn_lin, "Dynamic (linear)", "o"), (dyn_pac, "Dynamic (Pacejka)", "x")]:
        ays, ds = [], []
        for v in speeds:
            for d in deltas:
                k = steady_state_curvature(model, d, v)
                if k is None:
                    continue
                ays.append(v ** 2 * k)
                ds.append(d - p.L * k)
        ax[2].scatter(ays, ds, marker=marker, s=18, label=name)
        ay_max = max(ay_max, max(ays))
    a = np.linspace(0, ay_max, 50)
    ax[2].plot(a, p.K_us * a, "k-", lw=1.5, label=r"formula  $K_{us}\,a_y$")
    ax[2].axvline(p.mu * G, color="gray", ls="--", lw=1, label=r"$\mu g$")
    ax[2].set_title(r"(3) Steady-state:  $\delta - L\kappa$ vs $a_y$")
    ax[2].set_xlabel(r"$a_y = v^2\kappa$ [m/s$^2$]")
    ax[2].set_ylabel(r"$\delta - L\kappa$ [rad]  (extra steer)")
    ax[2].legend(); ax[2].grid(alpha=0.3)
 
    plt.tight_layout()
    plt.show()
