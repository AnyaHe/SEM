import pandas as pd
import pyomo.environ as pm
import os

import storage_equivalent as se
from ev_model import add_evs_model, scale_electric_vehicles
from scenario_input import base_scenario, scenario_input_evs, save_scenario_dict
from plotting import plot_storage_equivalent_germany_stacked


if __name__ == "__main__":
    scenario = "Germany_EV"
    solver = "gurobi"
    ev_mode = "flexible"
    # load scenario values
    scenario_dict = base_scenario()
    scenario_dict["ev_mode"] = ev_mode
    scenario_dict["solver"] = solver
    if ev_mode is not None:
        scenario_dict = scenario_input_evs(scenario_dict=scenario_dict, mode=ev_mode)

    sum_energy = scenario_dict["ts_demand"].sum().sum()
    scaled_ts_reference = scenario_dict["ts_vres"].divide(
        scenario_dict["ts_vres"].sum().sum())
    # initialise result
    shifted_energy_df = pd.DataFrame(columns=["storage_type",
                                              "energy_stored"])
    shifted_energy_rel_df = pd.DataFrame(columns=["storage_type",
                                                  "energy_stored"])
    for nr_ev_mio in range(0, 45, 5):
        # initialise base model
        model = pm.ConcreteModel()
        model.time_set = pm.RangeSet(0, len(scenario_dict["ts_demand"]) - 1)
        model.time_non_zero = model.time_set - [model.time_set.at(1)]
        model.time_increment = pd.to_timedelta(scenario_dict["time_increment"])
        model.times_fixed_soc = pm.Set(initialize=[model.time_set.at(1),
                                                   model.time_set.at(-1)])
        model.weighting = scenario_dict["weighting"]
        # add evs if included
        if ev_mode is not None:
            (reference_charging, flexibility_bands) = scale_electric_vehicles(
                nr_ev_mio, scenario_dict)
            if ev_mode == "flexible":
                energy_ev = reference_charging["inflexible"].sum() + \
                            (flexibility_bands["upper_energy"].sum(axis=1)[-1]/0.9 +
                             flexibility_bands["lower_energy"].sum(axis=1)[-1]/0.9)/2
                ref_charging = reference_charging["inflexible"]
                add_evs_model(model, flexibility_bands)
            else:
                energy_ev = reference_charging.sum().sum()
                ref_charging = reference_charging.sum(axis=1)
        else:
            energy_ev = 0
            ref_charging = pd.Series(index=scenario_dict["ts_demand"].index, data=0)
        # determine new residual load
        vres = scaled_ts_reference * (sum_energy + energy_ev)
        new_res_load = scenario_dict["ts_demand"].sum(axis=1) + ref_charging - vres.sum(axis=1)

        model = se.add_storage_equivalent_model(
            model, new_res_load, time_horizons=scenario_dict["time_horizons"])
        model.objective = pm.Objective(rule=getattr(se, scenario_dict["objective"]),
                                       sense=pm.minimize,
                                       doc='Define objective function')
        opt = pm.SolverFactory(solver)
        if solver == "gurobi":
            opt.options["Method"] = 1
        results = opt.solve(model, tee=True)
        # extract results
        charging = pd.Series(model.charging.extract_values()).unstack()
        energy_levels = pd.Series(model.energy_levels.extract_values()).unstack()
        caps = pd.Series(model.caps_pos.extract_values()) + pd.Series(
            model.caps_neg.extract_values())
        caps_neg = pd.Series(model.caps_neg.extract_values())
        relative_energy_levels = (energy_levels.T + caps_neg).divide(caps)
        abs_charging = pd.Series(model.abs_charging.extract_values()).unstack()
        df_tmp = (abs_charging.sum(axis=1) / 2).reset_index().rename(
            columns={"index": "storage_type", 0: "energy_stored"})
        df_tmp["nr_ev"] = nr_ev_mio
        shifted_energy_df = shifted_energy_df.append(df_tmp, ignore_index=True)
        df_tmp["energy_stored"] = df_tmp["energy_stored"] / sum_energy * 100
        shifted_energy_rel_df = shifted_energy_rel_df.append(df_tmp,
                                                             ignore_index=True)
        # save flexible hp operation
        if (ev_mode == "flexible") & (nr_ev_mio == 40.0):
            ev_operation = pd.Series(model.charging_ev.extract_values()).unstack().T
            ev_operation.index = scenario_dict["ts_demand"].index
            ev_operation.to_csv(f"results/ev_charging_flexible_{scenario}.csv")
    res_dir = os.path.join(f"results/{scenario}")
    os.makedirs(res_dir, exist_ok=True)
    shifted_energy_df.to_csv(f"{res_dir}/storage_equivalents.csv")
    shifted_energy_rel_df.to_csv(
        f"{res_dir}/storage_equivalents_relative.csv")
    # plot results
    plot_storage_equivalent_germany_stacked(shifted_energy_df,
                                            parameter={"nr_ev": "Number EVs [Mio.]"})
    # remove timeseries as they cannot be saved in json format
    save_scenario_dict(scenario_dict, res_dir)
    print("SUCCESS")
