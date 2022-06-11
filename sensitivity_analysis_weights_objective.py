import pandas as pd
import pyomo.environ as pm
import matplotlib.pyplot as plt

from storage_equivalent import add_storage_equivalents, minimize_energy

if __name__ == "__main__":
    scenario = "weights"
    solver = "gurobi"
    time_increment = pd.to_timedelta('1h')
    vres = pd.read_csv(r"vres_reference_ego100.csv", index_col=0,
                       parse_dates=True).divide(1000)
    demand = pd.read_csv(r"demand_germany_ego100.csv", index_col=0,
                         parse_dates=True)
    sum_energy = demand.sum().sum()
    vres = vres.divide(vres.sum().sum()).multiply(sum_energy)
    shifted_energy_df = pd.DataFrame(columns=["relative_weight", "storage_type",
                                              "energy_stored"])
    shifted_energy_rel_df = pd.DataFrame(columns=["relative_weight", "storage_type",
                                                  "energy_stored"])
    for relative_weighting in [1e-1, 1, 1e1, 1e2, 1e3]:
        new_res_load = demand.sum(axis=1) - vres.sum(axis=1)
        model = pm.ConcreteModel()
        model.time_set = pm.RangeSet(0, len(new_res_load) - 1)
        model.time_non_zero = model.time_set - [model.time_set.at(1)]
        model.time_increment = time_increment
        model.weighting = [relative_weighting, relative_weighting**2,
                           relative_weighting**3, relative_weighting**4]
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
        df_tmp["relative_weight"] = relative_weighting
        shifted_energy_df = shifted_energy_df.append(df_tmp, ignore_index=True)
        df_tmp["energy_stored"] = df_tmp["energy_stored"] / sum_energy * 100
        shifted_energy_rel_df = shifted_energy_rel_df.append(df_tmp,
                                                             ignore_index=True)
    shifted_energy_df.to_csv(f"results/storage_equivalents_{scenario}.csv")
    shifted_energy_rel_df.to_csv(f"results/storage_equivalents_{scenario}_relative.csv")
    shifted_energy_rel_df.loc[shifted_energy_rel_df.storage_type == 0].set_index(
        "relative_weight").energy_stored.plot.bar(figsize=(4, 2))
    plt.title("Relative energy stored short")
    plt.tight_layout()
    plt.show()
    # TODO: plot all
    # TODO: subplots
    print("SUCCESS")
