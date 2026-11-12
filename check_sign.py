import numpy as np
from bicycle_models import VehicleParams, KinematicBicycle, DynamicBicycle
from paths import make_oval
from run_experiments import run, metrics, train_correctors

p = VehicleParams()
path = make_oval()
v = 3.8

m = metrics(run(KinematicBicycle(p), path, v), path, v)
print(f"Kinematic                 mean {m['arc_mean']*100:+6.2f} cm   RMS {m['arc_rms']*100:5.2f} cm")

for exp, tire in [("E1", "linear"), ("E2", "pacejka")]:
    model = DynamicBicycle(p, tire=tire)
    c = train_correctors(tire, seed=0)
    for mode, name in [("yaw", None), ("yaw", "Formula"), ("slip", None),
                       ("slip", "Formula"), ("slip", "MLP")]:
        corr = c[name] if name else None
        m = metrics(run(model, path, v, corr, mode == "slip"), path, v)
        label = f"{exp} {mode:4s} {name or 'PP'}"
        print(f"{label:25s} mean {m['arc_mean']*100:+6.2f} cm   RMS {m['arc_rms']*100:5.2f} cm")
