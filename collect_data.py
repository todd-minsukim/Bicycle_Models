import numpy as np
import matplotlib.pyplot as plt

from bicycle_models import VehicleParams, DynamicBicycle, steady_state_curvature, G


def collect(tire, speeds, deltas, p, ay_limit=0.8, noise_std=0.005, seed=0):
    rng = np.random.default_rng(seed)        # 노이즈용 난수 (seed 고정 → 매번 같은 결과)
    model = DynamicBicycle(p, tire=tire)
    rows = []
    n_skip_unsteady, n_skip_ay = 0, 0

    for v in speeds:
        for d in deltas:
            k = steady_state_curvature(model, d, v)    # δ, v 고정으로 주행 → 정상상태 κ
            if k is None:                              # 정상상태에 도달 못 함
                n_skip_unsteady += 1
                continue
            if v ** 2 * k > ay_limit * p.mu * G:       # 횡가속도 한계 초과
                n_skip_ay += 1
                continue
            k_meas = k + rng.normal(0.0, noise_std)    # 측정 노이즈
            rows.append((v, k_meas, d))

    data = np.array(rows)                              # 열: v, κ, δ
    mirror = data * np.array([1.0, -1.0, -1.0])        # 우회전: κ, δ 부호 반전
    data = np.vstack([data, mirror])

    print(f"[{tire}] 수집 {len(rows)}개 (+대칭 {len(rows)}개), "
          f"제외: 비정상상태 {n_skip_unsteady}, a_y 초과 {n_skip_ay}")
    return data


def save(data, p, filename, test_ratio=0.2, seed=0):
    v, kappa, delta = data[:, 0], data[:, 1], data[:, 2]
    dd = delta - np.arctan(p.L * kappa)                # 보정기가 배울 목표: Δδ

    rng = np.random.default_rng(seed)
    is_test = rng.random(len(v)) < test_ratio          # 20%는 테스트용으로 떼어 둠

    np.savez(filename, v=v, kappa=kappa, delta=delta, dd=dd, is_test=is_test)
    print(f"  saved {filename}: train {np.sum(~is_test)}, test {np.sum(is_test)}")


if __name__ == "__main__":
    p = VehicleParams()
    speeds = np.linspace(1.5, 4.0, 11)     # 1.5, 1.75, ..., 4.0
    deltas = np.linspace(0.0, 0.30, 31)    # 0.00, 0.01, ..., 0.30

    fig, ax = plt.subplots(figsize=(7, 4))
    for tire, marker in [("linear", "o"), ("pacejka", "x")]:
        data = collect(tire, speeds, deltas, p)
        save(data, p, f"data_{tire}.npz")
