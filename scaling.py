import pandas as pd
import os

grid_dir = r"C:\Users\mosta\Flexibility-Quantification"
#grid_ids = [176, 177, 1056, 1690, 1811, 2534]
grid_ids = [176]

def scale_residual_load(ts_load, ts_feedin, ts_reference, energy_ev=0, energy_hp=0):
    energy_load = ts_load.sum()
    energy_feedin = ts_feedin.sum()
    energy_reference = ts_reference.sum()
    scaling_factor = (energy_load + energy_ev + energy_hp -
                      energy_feedin)/energy_reference
    print(energy_load)
    print(energy_feedin)
    print(energy_reference)
    print(scaling_factor)
    print(ts_load.shape)
    print(ts_feedin.shape)
    print(ts_reference.shape)
    new_residual_load = ts_load - ts_feedin - \
                        scaling_factor * ts_reference
    if abs(new_residual_load.sum()) > 1e-4:
        raise ValueError("New residual load should be balanced. Please check.")
    if scaling_factor < 0:
        new_load = ts_load - scaling_factor * ts_reference
    else:
        new_load = ts_load
    return new_residual_load, new_load


def scale_residual_load_v2(ts_load, ts_feedin, ts_reference, energy_ev=0, energy_hp=0):
    energy_load = ts_load.sum()
    energy_feedin = ts_feedin.sum()
    energy_reference = ts_reference.sum()
    scaling_factor = (energy_load + energy_ev + energy_hp -
                      energy_feedin)
    if scaling_factor < 0:
        new_residual_load = ts_load - (1 + scaling_factor/energy_feedin) * ts_feedin
    else:
        new_residual_load = ts_load - ts_feedin - \
                            scaling_factor/energy_reference * ts_reference
    if abs(new_residual_load.sum()) > 1e-4:
        raise ValueError("New residual load should be balanced. Please check.")
    return new_residual_load


if __name__ == "__main__":
    ts_reference = pd.read_csv("data/vres_reference_ego100.csv", index_col=0,
                               parse_dates=True)
    for grid_id in grid_ids:
        ts_loads = pd.read_csv(os.path.join(grid_dir, str(grid_id), "load.csv"),
                               index_col=0, parse_dates=True)
        ts_generators = pd.read_csv(os.path.join(grid_dir, str(grid_id), "generation.csv"),
                                    index_col=0, parse_dates=True)
        new_res_load, new_load = scale_residual_load(ts_loads.sum(axis=1), ts_generators.sum(axis=1),
                                           ts_reference.sum(axis=1))
    print("SUCCESS")