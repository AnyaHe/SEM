import pandas as pd
import os
import pyomo.environ as pm
import matplotlib.pyplot as plt

from scaling import scale_residual_load
from storage_equivalent import add_storage_equivalents, minimize_energy
from ev_model import add_ev_model
from heat_pump_model import scale_heat_demand_to_grid, add_heat_pump_model

grid_dir = r"C:\Users\aheider\Documents\Grids\Grids_IYCE"
grid_ids = [176, 177, 1056, 1690, 1811, 2534]
scenario = "Reference_variation"
ev_mode = None # so far implemented: "dumb", "flexible"
hp_mode = "dumb" # so far implemented: "dumb", "flexible"


if __name__ == "__main__":
    solver = "gurobi"
    time_increment = pd.to_timedelta('1h')
    ts_reference = pd.read_csv("data/vres_reference_ego100.csv", index_col=0,
                               parse_dates=True)
    sum_energy = ts_reference.sum().sum()
    scaled_ts_reference = ts_reference.divide(ts_reference.sum())
    #ts_reference.loc[:, "solar"] = 0
    shifted_energy_df = pd.DataFrame(columns=["grid_id", "share_pv", "storage_type",
                                              "energy_stored"])
    shifted_energy_rel_df = pd.DataFrame(columns=["grid_id", "share_pv", "storage_type",
                                                  "energy_stored"])
    if hp_mode != "":
        heat_demand = pd.read_csv(os.path.join(grid_dir, "hp_heat_2011.csv"),
                                  header=None)
        cop = pd.read_csv(os.path.join(grid_dir, "COP_2011.csv"))
    for grid_id in grid_ids:
        ts_loads = pd.read_csv(os.path.join(grid_dir, str(grid_id), "load.csv"),
                               index_col=0, parse_dates=True)
        if ev_mode == "dumb":
            ts_loads["EV"] = pd.read_csv(os.path.join(grid_dir, str(grid_id),
                                                      "dumb_charging.csv"),
                                   index_col=0, parse_dates=True)
            ev_energy = 0
        elif ev_mode == "flexible":
            flex_bands_home = pd.read_csv(os.path.join(grid_dir, str(grid_id),
                                                       "flex_ev_home.csv"),
                                          index_col=0, parse_dates=True)
            flex_bands_work = pd.read_csv(os.path.join(grid_dir, str(grid_id),
                                                       "flex_ev_work.csv"),
                                          index_col=0, parse_dates=True)
            flex_bands = flex_bands_work + flex_bands_home
            ev_energy = (flex_bands["upper"][-1] + flex_bands["lower"][-1] -
                         flex_bands["upper"][0] - flex_bands["lower"][0])/2
        else:
            ev_energy = 0
        if hp_mode == "dumb":
            grid_heat_demand, _ = scale_heat_demand_to_grid(heat_demand, grid_id)
            ts_loads["HP"] = (grid_heat_demand.T.divide(cop["COP 2011"].T)).T.values
            hp_energy = 0
        elif hp_mode == "flexible":
            grid_heat_demand, grid_nr_hps = \
                scale_heat_demand_to_grid(heat_demand, grid_id)
            hp_energy = (grid_heat_demand.T.divide(cop["COP 2011"].T)).T.sum().sum()
        else:
            hp_energy = 0
        ts_generators = pd.read_csv(os.path.join(grid_dir, str(grid_id),
                                                 "generation.csv"),
                                    index_col=0, parse_dates=True)
        #ts_generators.loc[:, "solar"] = 0
        for share_pv in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            ts_reference["solar"] = scaled_ts_reference["solar"] * sum_energy * share_pv
            ts_reference["wind"] = \
                scaled_ts_reference["wind"] * sum_energy * (1-share_pv)
            new_res_load, new_load = \
                scale_residual_load(ts_loads.sum(axis=1), ts_generators.sum(axis=1),
                                    ts_reference.sum(axis=1), energy_ev=ev_energy,
                                    energy_hp=hp_energy)
            print("Percentage of energy stored in grid {}: {}".format(
                grid_id, new_res_load.abs().sum()/2/new_load.sum()))
            # set up base model
            model = pm.ConcreteModel()
            model.time_set = pm.RangeSet(0, len(new_res_load) - 1)
            model.time_non_zero = model.time_set - [model.time_set.at(1)]
            model.time_increment = time_increment
            model.weighting = [1, 7, 30, 365]
            model = add_storage_equivalents(model, new_res_load)
            model.objective = pm.Objective(rule=minimize_energy,
                                           sense=pm.minimize,
                                           doc='Define objective function')
            # add EVs
            if ev_mode == "flexible":
                model.times_fixed_soc = pm.Set(initialize=[model.time_set.at(1),
                                                           model.time_set.at(-1)])
                model = add_ev_model(model, flex_bands.reset_index())
            # add HPs
            if hp_mode == "flexible":
                if not hasattr(model, "times_fixed_soc"):
                    model.times_fixed_soc = pm.Set(initialize=[model.time_set.at(1),
                                                               model.time_set.at(-1)])
                grid_capacity_tes = grid_nr_hps * 0.05 # MWh
                grid_p_nom_hp = grid_nr_hps * 0.003 # MW
                model = add_heat_pump_model(model, grid_p_nom_hp, grid_capacity_tes,
                                            cop, grid_heat_demand)
            opt = pm.SolverFactory(solver)
            results = opt.solve(model, tee=True)
            charging = pd.Series(model.charging.extract_values()).unstack()
            energy_levels = pd.Series(model.energy_levels.extract_values()).unstack()
            caps = pd.Series(model.caps_pos.extract_values()) + \
                   pd.Series(model.caps_neg.extract_values())
            caps_neg = pd.Series(model.caps_neg.extract_values())
            relative_energy_levels = (energy_levels.T + caps_neg).divide(caps)
            abs_charging = pd.Series(model.abs_charging.extract_values()).unstack()
            total_demand = new_load.sum() + ev_energy
            df_tmp = (abs_charging.sum(axis=1) / 2).reset_index().rename(
                columns={"index": "storage_type", 0: "energy_stored"})
            df_tmp["grid_id"] = grid_id
            df_tmp["share_pv"] = share_pv
            shifted_energy_df = shifted_energy_df.append(df_tmp, ignore_index=True)
            df_tmp["energy_stored"] = df_tmp["energy_stored"] / total_demand * 100
            shifted_energy_rel_df = shifted_energy_rel_df.append(df_tmp,
                                                                 ignore_index=True)
        # plt.show()
    shifted_energy_df.to_csv("results/storage_equivalents_{}.csv".format(scenario))
    shifted_energy_rel_df.to_csv("results/storage_equivalents_{}_relative.csv".format(
        scenario))
    # shifted_energy_rel_df.T.plot.bar(stacked=True)
    # plt.title("{}".format(scenario))
    # plt.savefig("results/Storage_{}.png".format(scenario))
    plt.show()
    print("SUCCESS")