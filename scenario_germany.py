import pandas as pd
import pyomo.environ as pm
import matplotlib.pyplot as plt

from storage_equivalent import add_storage_equivalents, minimize_energy

if __name__ == "__main__":
    scenario = "Germany"
    solver = "gurobi"
    time_increment = pd.to_timedelta('1h')
    vres = pd.read_csv(r"vres_reference_ego100.csv", index_col=0,
                       parse_dates=True).divide(1000)
    demand = pd.read_csv(r"demand_germany_ego100.csv", index_col=0,
                         parse_dates=True)
    sum_energy = demand.sum().sum()
    scaled_ts_reference = vres.divide(vres.sum())
    shifted_energy_df = pd.DataFrame(columns=["share_pv", "storage_type",
                                              "energy_stored"])
    shifted_energy_rel_df = pd.DataFrame(columns=["grid_id", "share_pv", "storage_type",
                                                  "energy_stored"])
    for share_pv in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        vres["solar"] = scaled_ts_reference["solar"] * sum_energy * share_pv
        vres["wind"] = \
            scaled_ts_reference["wind"] * sum_energy * (1 - share_pv)
        new_res_load = demand.sum(axis=1) - vres.sum(axis=1)
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
        caps = pd.Series(model.caps_pos.extract_values()) + pd.Series(
            model.caps_neg.extract_values())
        caps_neg = pd.Series(model.caps_neg.extract_values())
        relative_energy_levels = (energy_levels.T + caps_neg).divide(caps)
        abs_charging = pd.Series(model.abs_charging.extract_values()).unstack()
        df_tmp = (abs_charging.sum(axis=1) / 2).reset_index().rename(
            columns={"index": "storage_type", 0: "energy_stored"})
        df_tmp["share_pv"] = share_pv
        shifted_energy_df = shifted_energy_df.append(df_tmp, ignore_index=True)
        df_tmp["energy_stored"] = df_tmp["energy_stored"] / sum_energy * 100
        shifted_energy_rel_df = shifted_energy_rel_df.append(df_tmp,
                                                             ignore_index=True)
    shifted_energy_df.to_csv("results/storage_equivalents_{}.csv".format(scenario))
    shifted_energy_rel_df.to_csv("results/storage_equivalents_{}_relative.csv".format(
        scenario))
    shifted_energy_rel_df.loc[shifted_energy_rel_df.storage_type == 0].set_index(
        "share_pv").energy_stored.plot_df.bar(figsize=(4, 2))
    plt.title("Relative energy stored short")
    plt.tight_layout()
    # TODO: plot all
    # TODO: subplots
    print("SUCCESS")