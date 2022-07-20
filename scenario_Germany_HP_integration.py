import pandas as pd
import pyomo.environ as pm
import matplotlib.pyplot as plt

from storage_equivalent import add_storage_equivalents, minimize_energy
from heat_pump_model import add_heat_pump_model

if __name__ == "__main__":
    scenario = "Germany_HP"
    solver = "gurobi"
    hp_mode = "flexible"
    time_increment = pd.to_timedelta('1h')
    vres = pd.read_csv(r"data/vres_reference_ego100.csv", index_col=0,
                       parse_dates=True).divide(1000)
    demand = pd.read_csv(r"data/demand_germany_ego100.csv", index_col=0,
                         parse_dates=True)
    sum_energy = demand.sum().sum()
    scaled_ts_reference = vres.divide(vres.sum().sum())
    shifted_energy_df = pd.DataFrame(columns=["storage_type",
                                              "energy_stored"])
    shifted_energy_rel_df = pd.DataFrame(columns=["storage_type",
                                                  "energy_stored"])
    for nr_hp_mio in range(15):
        nr_hp = nr_hp_mio * 1e6
        heat_demand = pd.read_csv(
            r"data/hp_heat_2011.csv",
            header=None)/1e3
        cop = pd.read_csv(
            r'data/COP_2011.csv')
        capacity_tes = nr_hp * 0.05 * 1e-3  # GWh
        p_nom_hp = nr_hp * 0.003 * 1e-3  # GW
        heat_demand_per_hp = 12 * 1e-3  # GWh
        ts_heat_demand_per_hp = heat_demand / heat_demand.sum() * heat_demand_per_hp
        ts_heat_demand = ts_heat_demand_per_hp * nr_hp
        ts_heat_el = ts_heat_demand.T.divide(cop["COP 2011"]).T
        sum_energy_heat = ts_heat_el.sum().sum()
        vres = scaled_ts_reference * (sum_energy + sum_energy_heat)
        if hp_mode == "flexible":
            new_res_load = demand.sum(axis=1) - vres.sum(axis=1)
        else:
            new_res_load = demand.sum(axis=1) + ts_heat_el.set_index(demand.index).sum(axis=1) - vres.sum(axis=1)
        model = pm.ConcreteModel()
        model.time_set = pm.RangeSet(0, len(new_res_load) - 1)
        model.time_non_zero = model.time_set - [model.time_set.at(1)]
        model.time_increment = time_increment
        model.times_fixed_soc = pm.Set(initialize=[model.time_set.at(1),
                                                   model.time_set.at(-1)])
        model.weighting = [1, 10, 100, 1000]
        if hp_mode == "flexible":
            model = add_heat_pump_model(model, p_nom_hp, capacity_tes, cop, ts_heat_demand)
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
        df_tmp["nr_hp"] = nr_hp_mio
        shifted_energy_df = shifted_energy_df.append(df_tmp, ignore_index=True)
        df_tmp["energy_stored"] = df_tmp["energy_stored"] / sum_energy * 100
        shifted_energy_rel_df = shifted_energy_rel_df.append(df_tmp,
                                                             ignore_index=True)
    shifted_energy_df.to_csv("results/storage_equivalents_{}.csv".format(scenario))
    shifted_energy_rel_df.to_csv("results/storage_equivalents_{}_relative.csv".format(
        scenario))
    # shifted_energy_rel_df.loc[shifted_energy_rel_df.storage_type == 0].set_index(
    #     "share_pv").energy_stored.plot.bar(figsize=(4, 2))
    # plt.title("Relative energy stored short")
    # plt.tight_layout()
    # TODO: plot all
    # TODO: subplots
    print("SUCCESS")