"""
correctors.py
보정기 3종. 모두 같은 형태로 사용:
    c = XxxCorrection().fit(v, kappa, dd)     # 학습
    c(v, kappa) -> Δδ [rad]                   # 사용
"""
import numpy as np


class FormulaCorrection:
    """해석적 보정: Δδ = K · v² · κ  (언더스티어 공식 K_us·a_y 꼴)
    K는 시뮬레이터 참값이 아니라 데이터에서 최소제곱으로 피팅 (공정한 비교)"""
    name = "Formula"

    def fit(self, v, kappa, dd):
        a = v ** 2 * kappa                           # a_y
        self.K = float(np.sum(a * dd) / np.sum(a * a))   # 원점 지나는 직선의 최소제곱 기울기
        return self

    def __call__(self, v, kappa):
        return self.K * v * v * kappa


class LUTCorrection:
    """룩업테이블 보정: 데이터 점들 사이를 선형 보간.
    데이터 범위 밖이면 가장 가까운 점 값을 사용"""
    name = "LUT"

    def fit(self, v, kappa, dd):
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        # v와 κ의 단위가 달라서 범위를 비슷하게 맞춘 뒤 보간
        self.scale = np.array([1.0 / np.ptp(v), 1.0 / np.ptp(kappa)])
        X = np.stack([v, kappa], axis=1) * self.scale
        self.lin = LinearNDInterpolator(X, dd)
        self.near = NearestNDInterpolator(X, dd)
        return self

    def __call__(self, v, kappa):
        x = np.array([[v, kappa]]) * self.scale
        y = self.lin(x)[0]
        if np.isnan(y):                  # 데이터 범위 밖
            y = self.near(x)[0]
        return float(y)


class MLPCorrection:
    """신경망 보정: 입력 (v, κ) → 은닉층 2개(각 32개, tanh) → Δδ
    학습은 PyTorch, 시뮬레이션 중 계산은 numpy (빠르게)"""
    name = "MLP"

    def __init__(self, hidden=32, epochs=3000, lr=1e-2, seed=0):
        self.hidden, self.epochs, self.lr, self.seed = hidden, epochs, lr, seed

    def fit(self, v, kappa, dd):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)

        # 입력/출력 정규화 (평균 0, 표준편차 1 근처로 맞추면 학습이 잘 됨)
        X = np.stack([v, kappa], axis=1)
        self.x_mean, self.x_std = X.mean(axis=0), X.std(axis=0)
        self.y_std = dd.std()
        Xt = torch.tensor((X - self.x_mean) / self.x_std, dtype=torch.float32)
        Yt = torch.tensor(dd / self.y_std, dtype=torch.float32).unsqueeze(1)

        net = nn.Sequential(
            nn.Linear(2, self.hidden), nn.Tanh(),
            nn.Linear(self.hidden, self.hidden), nn.Tanh(),
            nn.Linear(self.hidden, 1),
        )
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)

        for ep in range(self.epochs):
            opt.zero_grad()
            loss = ((net(Xt) - Yt) ** 2).mean()      # 평균제곱오차
            loss.backward()
            opt.step()
            if ep % 1000 == 0 or ep == self.epochs - 1:
                print(f"    MLP epoch {ep:5d}  loss {loss.item():.5f}")

        # 학습된 가중치를 numpy로 꺼내 둠
        self.layers = [(m.weight.detach().numpy(), m.bias.detach().numpy())
                       for m in net if isinstance(m, nn.Linear)]
        return self

    def __call__(self, v, kappa):
        h = (np.array([v, kappa]) - self.x_mean) / self.x_std
        for i, (W, b) in enumerate(self.layers):
            h = W @ h + b
            if i < len(self.layers) - 1:          # 마지막 층만 tanh 없음
                h = np.tanh(h)
        return float(h[0] * self.y_std)
