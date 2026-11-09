import math
import numpy as np


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi  # 각도 범위를 -pi에서 pi 범위로 설정


class PurePursuit:
    def __init__(self, path, params, speed, k_ld=0.3, ld_min=0.3, search_window=40):
        # 파라미터
        self.path = path              # paths.py의 Path
        self.p = params               # VehicleParams (L, lr, delta_max 사용)
        self.v = speed                # 전진 속도 [m/s] (일정)
        self.k_ld = k_ld              # Ld = k_ld * v + ld_min
        self.ld_min = ld_min
        self.window = search_window   # 최근접 점 검색 범위 [점 개수]

        # 상태
        self.N = len(path.x)
        self.nearest_idx = 0

        # 기록
        self.log = {"t": [], "e_y": [], "e_psi": [], "kappa_cmd": [],
                    "delta": [], "idx": [], "target": []}

    # ---------- 보조 ----------
    def wrap_idx(self, k):
        if self.path.closed:
            return k % self.N   # 닫힌 경로면 넘어감
        return min(k, self.N - 1)  # 열린 경로면 멈춤

    def rear_axle(self, s):     
        x, y, yaw = s[0], s[1], s[2]
        xr = x - self.p.lr * math.cos(yaw)
        yr = y - self.p.lr * math.sin(yaw)
        return xr, yr, yaw   # 뒤축 기준으로 재계산

    # ---------- 계산 ----------
    def find_nearest(self, xr, yr):
        best_idx = self.nearest_idx
        best_dist = float("inf")
        for k in range(self.nearest_idx, self.nearest_idx + self.window):
            i = self.wrap_idx(k)
            d = math.hypot(self.path.x[i] - xr, self.path.y[i] - yr)
            if d < best_dist:
                best_dist = d
                best_idx = i
        self.nearest_idx = best_idx
        return best_idx

    def find_target(self, start_idx, xr, yr, Ld):
        for k in range(start_idx, start_idx + self.N):       # 최대 한 바퀴까지만 탐색
            i = self.wrap_idx(k)
            if math.hypot(self.path.x[i] - xr, self.path.y[i] - yr) >= Ld:
                return i
        return self.wrap_idx(start_idx + self.N - 1)         # 못 찾으면 마지막 점

    def tracking_error(self, i, xr, yr, yaw):
        """최근접점 i 기준 횡오차(경로 왼쪽 +)와 헤딩오차"""
        dx = xr - self.path.x[i]
        dy = yr - self.path.y[i]
        psi_ref = self.path.psi[i]
        e_y = -math.sin(psi_ref) * dx + math.cos(psi_ref) * dy
        e_psi = wrap(yaw - psi_ref)
        return e_y, e_psi
      
    def __call__(self, t, s):
        xr, yr, yaw = self.rear_axle(s)

        nearest = self.find_nearest(xr, yr)
        Ld = self.k_ld * self.v + self.ld_min
        target = self.find_target(nearest, xr, yr, Ld)
        tx, ty = self.path.x[target], self.path.y[target]

        # 목표점 → 차량 좌표계
        dx = tx - xr
        dy = ty - yr
        y_r = -math.sin(yaw) * dx + math.cos(yaw) * dy

        # 곡률 → 조향각
        L_t = math.hypot(dx, dy)
        if L_t < 1e-6:
            kappa = 0.0
        else:
            kappa = 2.0 * y_r / (L_t * L_t)        # = 2 sin(alpha) / L_t
        delta = math.atan(self.p.L * kappa)
        delta = max(-self.p.delta_max, min(self.p.delta_max, delta))

        # 기록
        e_y, e_psi = self.tracking_error(nearest, xr, yr, yaw)
        self.log["t"].append(t)
        self.log["e_y"].append(e_y)
        self.log["e_psi"].append(e_psi)
        self.log["kappa_cmd"].append(kappa)
        self.log["delta"].append(delta)
        self.log["idx"].append(nearest)
        self.log["target"].append(target)

        return delta, self.v

    def get_log(self):
        return {k: np.array(v) for k, v in self.log.items()}
