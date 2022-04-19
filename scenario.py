import pandas as pd
import os
import pyomo.environ as pm
import matplotlib.pyplot as plt

from scaling import scale_residual_load
from storage_equivalent import add_storage_equivalents, minimize_energy

grid_dir = r"H:\Grids_IYCE"
grid_ids = [176, 177, 1056, 1690, 1811, 2534]
scenario = "reference"


if __name__ == "__main__":
    solver = "gurobi"
    time_increment = pd.to_timedelta('1h')
    ts_reference = pd.read_csv("data/vres_reference_ego100.csv", index_col=0,
                               parse_dates=True)
    #ts_reference.loc[:, "solar"] = 0
    shifted_energy_df = pd.DataFrame()
    shifted_energy_rel_df = pd.DataFrame()
    for grid_id in grid_ids:
        ts_loads = pd.read_csv(os.path.join(grid_dir, str(grid_id), "load.csv"),
                               index_col=0, parse_dates=True)
        ts_generators = pd.read_csv(os.path.join(grid_dir, str(grid_id), "generation.csv"),
                                    index_col=0, parse_dates=True)
        #ts_generators.loc[:, "solar"] = 0
        new_res_load, new_load = scale_residual_load(ts_loads.sum(axis=1), ts_generators.sum(axis=1),
                                           ts_reference.sum(axis=1))
        print("Percentage of energy stored in grid {}: {}".format(
            grid_id, new_res_load.abs().sum()/2/new_load.sum()))
        model = pm.ConcreteModel()
        model.time_set = pm.RangeSet(0, len(new_res_load) - 1)
        model.time_non_zero = model.time_set - [model.time_set.at(1)]
        model.time_increment = time_increment
        model.weighting = [1, 7, 30, 365]
        model = add_storage_equivalents(model, new_res_load)
        model.objective = pm.Objective(rule=minimize_energy,
                                       sense=pm.minimize,
                                       doc='Define objective function')
        opt = pm.SolverFactory(solver)
        results = opt.solve(model, tee=True)
        charging = pd.Series(model.charging.extract_values()).unstack()
        energy_levels = pd.Series(model.energy_levels.extract_values()).unstack()
        caps = pd.Series(model.caps_pos.extract_values()) + pd.Series(model.caps_neg.extract_values())
        caps_neg = pd.Series(model.caps_neg.extract_values())
        relative_energy_levels = (energy_levels.T + caps_neg).divide(caps)
        abs_charging = pd.Series(model.abs_charging.extract_values()).unstack()
        total_demand = new_load.sum()
        shifted_energy_rel_df[grid_id] = (abs_charging.sum(axis=1) / 2)/ total_demand * 100
        shifted_energy_df[grid_id] = (abs_charging.sum(axis=1) / 2)
        plt.show()
    shifted_energy_df.to_csv("results/storage_equivalents_{}.csv".format(scenario))
    shifted_energy_rel_df.T.plot.bar(stacked=True)
    plt.title("Load and wind")
    plt.savefig("results/Storage_{}.png".format(scenario))
    plt.show()
    print("SUCCESS")