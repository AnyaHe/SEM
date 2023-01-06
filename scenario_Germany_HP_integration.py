import pandas as pd
import pyomo.environ as pm
import os
import json

import storage_equivalent as se
from heat_pump_model import add_heat_pump_model, scale_heat_pumps
from scenario_input import base_scenario, scenario_input_hps
from plotting import plot_storage_equivalent_germany_stacked


if __name__ == "__main__":
    scenario = "test"
    solver = "gurobi"
    hp_mode = "flexible" # None, "flexible", "inflexible"
    # load scenario values
    scenario_dict = base_scenario()
    scenario_dict["hp_mode"] = hp_mode
    if hp_mode is not None:
        scenario_dict = scenario_input_hps(scenario_dict=scenario_dict, mode=hp_mode)
    sum_energy = scenario_dict["ts_demand"].sum().sum()
    scaled_ts_reference = scenario_dict["ts_vres"].divide(
        scenario_dict["ts_vres"].sum().sum())
    # initialise result
    shifted_energy_df = pd.DataFrame(columns=["storage_type",
                                              "energy_stored"])
    shifted_energy_rel_df = pd.DataFrame(columns=["storage_type",
                                                  "energy_stored"])
    for i in range(2):
        # initialise base model
        model = pm.ConcreteModel()
        model.time_set = pm.RangeSet(0, len(scenario_dict["ts_demand"]) - 1)
        model.time_non_zero = model.time_set - [model.time_set.at(1)]
        model.time_increment = pd.to_timedelta(scenario_dict["time_increment"])
        model.times_fixed_soc = pm.Set(initialize=[model.time_set.at(1),
                                                   model.time_set.at(-1)])
        model.weighting = scenario_dict["weighting"]
        # add hps if included
        if hp_mode is not None:
            nr_hp_mio = i * 2.5
            (capacity_tes, p_nom_hp,
             ts_heat_demand, ts_heat_el, sum_energy_heat) = \
                scale_heat_pumps(nr_hp_mio=nr_hp_mio,
                                 scenario_dict=scenario_dict)
            # capacity_tes = capacity_tes * 2
            if hp_mode == "flexible":
                model = add_heat_pump_model(model, p_nom_hp, capacity_tes,
                                            scenario_dict["ts_cop"], ts_heat_demand)
        else:
            ts_heat_el = pd.Series(index=scenario_dict["ts_demand"].index, data=0)
            sum_energy_heat = 0
        # determine new residual load
        vres = scaled_ts_reference * (sum_energy + sum_energy_heat)
        new_res_load = scenario_dict["ts_demand"].sum(axis=1) - vres.sum(axis=1)
        if hp_mode != "flexible":
            new_res_load = new_res_load + \
                           ts_heat_el.set_index(scenario_dict["ts_demand"].index).sum(
                               axis=1)
        # add storage equivalents
        model = se.add_storage_equivalent_model(
            model, new_res_load, time_horizons=scenario_dict["time_horizons"])
        # define objective
        model.objective = pm.Objective(rule=getattr(se, scenario_dict["objective"]),
                                       sense=pm.minimize,
                                       doc='Define objective function')
        opt = pm.SolverFactory(solver)
        if solver == "gurobi":
            opt.options["Method"] = 0
        results = opt.solve(model, tee=True)
        # extract results
        charging = pd.Series(model.charging.extract_values()).unstack()
        energy_levels = pd.Series(model.energy_levels.extract_values()).unstack()
        caps = pd.Series(model.caps_pos.extract_values()) + pd.Series(
            model.caps_neg.extract_values())
        caps_neg = pd.Series(model.caps_neg.extract_values())
        relative_energy_levels = (energy_levels.T + caps_neg).divide(caps)
        abs_charging = pd.Series(model.abs_charging.extract_values()).unstack()
        # save flexible hp operation
        if (hp_mode == "flexible") & (nr_hp_mio == 20.0):
            hp_operation = pd.Series(model.charging_hp_el.extract_values())
            hp_operation.index = scenario_dict["ts_demand"].index
            hp_operation.to_csv(f"results/hp_charging_flexible_{scenario}.csv")
        df_tmp = (abs_charging.sum(axis=1) / 2).reset_index().rename(
            columns={"index": "storage_type", 0: "energy_stored"})
        df_tmp["nr_hp"] = nr_hp_mio
        shifted_energy_df = shifted_energy_df.append(df_tmp, ignore_index=True)
        df_tmp["energy_stored"] = df_tmp["energy_stored"] / sum_energy * 100
        shifted_energy_rel_df = shifted_energy_rel_df.append(df_tmp,
                                                             ignore_index=True)
    res_dir = os.path.join(f"results/{scenario}")
    os.makedirs(res_dir, exist_ok=True)
    shifted_energy_df.to_csv(f"{res_dir}/storage_equivalents.csv")
    shifted_energy_rel_df.to_csv(
        f"{res_dir}/storage_equivalents_relative.csv")
    # plot results
    plot_storage_equivalent_germany_stacked(shifted_energy_df,
                                            parameter={"nr_hp": "Number HPs [Mio.]"})
    # remove timeseries as they cannot be saved in json format
    keys = [key for key in scenario_dict.keys()]
    for key in keys:
        if "ts_" in key:
            del scenario_dict[key]
    # save scenario input
    with open(
            os.path.join(res_dir, "scenario_dict.json"),
            'w', encoding='utf-8') as f:
        json.dump(scenario_dict, f, ensure_ascii=False, indent=4)
    print("SUCCESS")
