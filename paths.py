import numpy as np
from dataclasses import dataclass

G = 9.81
STRAIGHT, CLOTHOID, ARC = 0, 1, 2 


@dataclass
class Path:
    x: np.ndarray        # [m]
    y: np.ndarray        # [m]
    psi: np.ndarray      # 접선 헤딩 [rad], [-pi, pi)
    kappa: np.ndarray    # 곡률 [1/m], 좌회전 +
    s: np.ndarray        # 호 길이 [m]
    seg: np.ndarray      # 구간 라벨 (STRAIGHT / CLOTHOID / ARC)
    length: float        # 전체 길이 [m]
    closed: bool
    name: str = ""
    closure_err: float = 0.0   # 닫힌 경로의 시작/끝 위치 오차 [m] (검증용)

def _build(segments, ds, closed, name):
    Ls = np.array([sg[0] for sg in segments], float)
    K0 = np.array([sg[1] for sg in segments], float)
    K1 = np.array([sg[2] for sg in segments], float)
    LAB = np.array([sg[3] for sg in segments], int)
    if np.any(Ls <= 0):
        raise ValueError("구간 길이는 양수여야 합니다.")

    b = np.concatenate([[0.0], np.cumsum(Ls)])          # 구간 경계의 호 길이
    total = b[-1]
    psi_b = np.concatenate([[0.0], np.cumsum(0.5 * (K0 + K1) * Ls)])  # 구간 시작 헤딩

    def eval_profile(s):
        i = np.clip(np.searchsorted(b, s, side="right") - 1, 0, len(Ls) - 1)
        u = s - b[i]
        kap = K0[i] + (K1[i] - K0[i]) * u / Ls[i]
        psi = psi_b[i] + K0[i] * u + 0.5 * (K1[i] - K0[i]) * u**2 / Ls[i]   # ∫κ ds (정확)
        return kap, psi, LAB[i]

    n = int(round(total / ds))
    sub = 20                                           # 더 세밀한 위치 적분을 위해
    sf = np.linspace(0.0, total, n * sub + 1)
    _, psif, _ = eval_profile(sf)
    h = sf[1] - sf[0]
    xf = np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(psif[1:]) + np.cos(psif[:-1])) * h)])
    yf = np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(psif[1:]) + np.sin(psif[:-1])) * h)])

    idx = np.arange(0, n * sub + 1, sub)
    s, x, y = sf[idx], xf[idx], yf[idx]
    kappa, psi, seg = eval_profile(s)

    closure = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    if closed:                                         # 마지막 점 = 첫 점이므로 제거
        s, x, y, kappa, psi, seg = s[:-1], x[:-1], y[:-1], kappa[:-1], psi[:-1], seg[:-1]

    psi = (psi + np.pi) % (2 * np.pi) - np.pi
    return Path(x, y, psi, kappa, s, seg, total, closed, name, closure)
  

def make_oval(R=2.0, straight=6.0, transition=0.5, ds=0.02):
    
    k = 1.0 / R
    arc = np.pi * R - transition       # 원호길이
    if arc <= 0:
        raise ValueError("transition이 너무 깁니다 (transition < πR 이어야 함).")

    def corner():
        segs = []
        if transition > 0:
            segs.append((transition, 0.0, k, CLOTHOID))
        segs.append((arc, k, k, ARC))
        if transition > 0:
            segs.append((transition, k, 0.0, CLOTHOID))
        return segs

    segments = ([(straight / 2, 0.0, 0.0, STRAIGHT)] + corner()
                + [(straight, 0.0, 0.0, STRAIGHT)] + corner()
                + [(straight / 2, 0.0, 0.0, STRAIGHT)])
    return _build(segments, ds, closed=True, name=f"oval (R={R} m)")

