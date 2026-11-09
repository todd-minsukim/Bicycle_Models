"""
pure_pursuit.py
시뮬레이터용 Pure Pursuit (ROS2 pp_tracker와 같은 알고리즘) + 보정기 연결

ROS2 버전과 달라진 점
  1. ROS2 노드/토픽/타이머 제거 → simulate()가 pp(t, s)를 직접 호출
  2. 출력: 각속도 w  →  조향각 delta = atan(L * kappa)  (자전거 모델)
  3. 기준점: 무게중심 상태에서 뒷축 위치를 계산해서 사용
  4. lookahead: 고정값 → Ld = k_ld * v + ld_min (속도 비례)
  5. 닫힌 경로면 인덱스가 끝에서 0으로 돌아감 (여러 바퀴 주행)
  6. e_y, e_psi, kappa_cmd 등 매 스텝 기록
  7. 조향 제한: max_w → delta_max
  8. 보정기(corrector) 연결: delta = atan(L * kappa) + corrector(v, kappa)
     corrector=None이면 보정 없는 베이스라인
"""
import math
import numpy as np


def wrap(a):
    """각도를 [-pi, pi) 범위로"""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class PurePursuit:
    def __init__(self, path, params, speed, k_ld=0.3, ld_min=0.3, search_window=40,
                 corrector=None):
        # 파라미터
        self.path = path              # paths.py의 Path
        self.p = params               # VehicleParams (L, lr, delta_max 사용)
        self.v = speed                # 전진 속도 [m/s] (일정)
        self.k_ld = k_ld              # Ld = k_ld * v + ld_min
        self.ld_min = ld_min
        self.window = search_window   # 최근접 점 검색 범위 [점 개수]
        self.corrector = corrector    # None이면 보정 없음 (베이스라인)

        # 상태
        self.N = len(path.x)
        self.nearest_idx = 0

        # 기록
        self.log = {"t": [], "e_y": [], "e_psi": [], "kappa_cmd": [],
                    "delta": [], "dd": [], "idx": [], "target": []}

    # ---------- 보조 ----------
    def wrap_idx(self, k):
        """닫힌 경로면 끝에서 0으로 돌아가고, 열린 경로면 마지막 점에서 멈춤"""
        if self.path.closed:
            return k % self.N
        return min(k, self.N - 1)

    def rear_axle(self, s):
        """상태 s(무게중심 기준) → 뒷축 위치와 헤딩
        기구학 [x, y, psi], 동역학 [x, y, psi, vy, r] 둘 다 앞 3개가 x, y, psi"""
        x, y, yaw = s[0], s[1], s[2]
        xr = x - self.p.lr * math.cos(yaw)
        yr = y - self.p.lr * math.sin(yaw)
        return xr, yr, yaw

    # ---------- 계산 ----------
    def find_nearest(self, xr, yr):
        """뒷축에서 가장 가까운 경로점 (직전 위치부터 앞으로 window개만 탐색)"""
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
        """nearest부터 앞으로 가며, 뒷축에서 Ld 이상 떨어진 첫 점 = lookahead 점"""
        for k in range(start_idx, start_idx + self.N):       # 최대 한 바퀴까지만 탐색
            i = self.wrap_idx(k)
            if math.hypot(self.path.x[i] - xr, self.path.y[i] - yr) >= Ld:
                return i
        return self.wrap_idx(start_idx + self.N - 1)         # 못 찾으면 가장 먼 점

    def tracking_error(self, i, xr, yr, yaw):
        """최근접점 i 기준 횡오차(경로 왼쪽 +)와 헤딩오차"""
        dx = xr - self.path.x[i]
        dy = yr - self.path.y[i]
        psi_ref = self.path.psi[i]
        e_y = -math.sin(psi_ref) * dx + math.cos(psi_ref) * dy
        e_psi = wrap(yaw - psi_ref)
        return e_y, e_psi

    # ---------- 메인 (simulate가 매 스텝 호출) ----------
    def __call__(self, t, s):
        xr, yr, yaw = self.rear_axle(s)

        nearest = self.find_nearest(xr, yr)
        Ld = self.k_ld * self.v + self.ld_min
        target = self.find_target(nearest, xr, yr, Ld)
        tx, ty = self.path.x[target], self.path.y[target]

        # 목표점 → 차량 좌표계 (옆 성분)
        dx = tx - xr
        dy = ty - yr
        y_r = -math.sin(yaw) * dx + math.cos(yaw) * dy

        # 곡률 명령
        L_t = math.hypot(dx, dy)
        if L_t < 1e-6:
            kappa = 0.0
        else:
            kappa = 2.0 * y_r / (L_t * L_t)        # = 2 sin(alpha) / L_t

        # 조향각 = 기구학 PP 조향 + 보정
        delta_kin = math.atan(self.p.L * kappa)
        dd = self.corrector(self.v, kappa) if self.corrector else 0.0
        delta = delta_kin + dd
        delta = max(-self.p.delta_max, min(self.p.delta_max, delta))

        # 기록
        e_y, e_psi = self.tracking_error(nearest, xr, yr, yaw)
        self.log["t"].append(t)
        self.log["e_y"].append(e_y)
        self.log["e_psi"].append(e_psi)
        self.log["kappa_cmd"].append(kappa)
        self.log["delta"].append(delta)
        self.log["dd"].append(dd)
        self.log["idx"].append(nearest)
        self.log["target"].append(target)

        return delta, self.v

    def get_log(self):
        return {k: np.array(v) for k, v in self.log.items()}
