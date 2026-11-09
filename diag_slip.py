import numpy as np
from bicycle_models import VehicleParams, DynamicBicycle, simulate, G
from paths import make_oval
from pure_pursuit import PurePursuit
from run_experiments import metrics, train_correctors

p = VehicleParams()
path = make_oval()
speeds = [2.0, 2.8, 3.4, 3.8]
kmax = np.max(np.abs(path.kappa))

for tire in ["linear", "pacejka"]:
    model = DynamicBicycle(p, tire=tire)
    c = train_correctors(tire, seed=0)
    print(f"\n[{tire}] 원호 RMS [cm]")
    print("  a_y/μg   PP    PP+Formula   PP+slip   PP+slip+Formula   PP+slip+MLP")
    for v in speeds:
        row = []
        for corr, slip in [(None, False), (c["Formula"], False),
                           (None, True), (c["Formula"], True), (c["MLP"], True)]:
            pp = PurePursuit(path, p, speed=v, corrector=corr, use_slip=slip)
            simulate(model, model.initial_state(x=p.lr), pp, 3 * path.length / v, 0.002)
            row.append(metrics(pp.get_log(), path, v)["arc_rms"] * 100)
        print(f"  {v**2*kmax/(p.mu*G):.2f}  " + "  ".join(f"{r:9.2f}" for r in row))
