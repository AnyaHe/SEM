import pandas as pd
import pyomo.environ as pm
import os

import storage_equivalent as se
from heat_pump_model import add_heat_pump_model, model_input_hps
from ev_model import add_evs_model, model_input_evs
from scenario_input import base_scenario, scenario_input_hps, save_scenario_dict, \
    scenario_input_evs, get_new_residual_load, adjust_timeseries_data
from plotting import plot_storage_equivalent_germany_stacked


if __name__ == "__main__":
    storage_type = 0
    scenario = f"EV_flexible_reduction_{storage_type}"
    reference_scenario = "EV_reference"
    extract_storage_duration = True
    plot_results = False
    solver = "gurobi"
    hp_mode = "flexible" # None, "flexible", "inflexible"
    ev_mode = None # None, "flexible", "inflexible"
    tes_relative_size = 1 # in share standard
    ev_extended_flex = False
    flexible_ev_use_cases = ["home", "work", "public"]
    if ev_extended_flex:
        flexible_ev_use_cases = ["home", "work", "public"]
    # relative_weighting = 1000
    # weights = [relative_weighting, relative_weighting**2,
    #                        relative_weighting**3]
    # create results directory
    res_dir = os.path.join(f"results/{scenario}")
    res_dir_ref = os.path.join(f"results/{reference_scenario}")
    os.makedirs(res_dir, exist_ok=True)
    # load scenario values
    scenario_dict = base_scenario()
    scenario_dict["hp_mode"] = hp_mode
    scenario_dict["ev_mode"] = ev_mode
    scenario_dict["solver"] = solver
    # scenario_dict["weighting"] = weights
    if hp_mode is not None:
        scenario_dict = scenario_input_hps(scenario_dict=scenario_dict, mode=hp_mode)
        scenario_dict["capacity_single_tes"] = \
            tes_relative_size * scenario_dict["capacity_single_tes"]
    if ev_mode is not None:
        scenario_dict = scenario_input_evs(scenario_dict=scenario_dict, mode=ev_mode,
                                           use_cases_flexible=flexible_ev_use_cases,
                                           extended_flex=ev_extended_flex)
    # shift timeseries
    scenario_dict = adjust_timeseries_data(scenario_dict)
    # initialise result
    shifted_energy_df = pd.DataFrame(columns=["storage_type",
                                              "energy_stored"])
    shifted_energy_rel_df = pd.DataFrame(columns=["storage_type",
                                                  "energy_stored"])
    storage_durations = pd.DataFrame()
    for i in range(9):
        # add hps if included
        nr_hp_mio, ts_heat_el, sum_energy_heat, capacity_tes, p_nom_hp, ts_heat_demand = \
            model_input_hps(
                scenario_dict=scenario_dict,
                hp_mode=hp_mode,
                i=i
            )
        nr_ev_mio, flexibility_bands, energy_ev, ref_charging = model_input_evs(
            scenario_dict=scenario_dict,
            ev_mode=ev_mode,
            i=i
        )
        # determine new residual load
        new_res_load = get_new_residual_load(
            scenario_dict=scenario_dict,
            sum_energy_heat=sum_energy_heat,
            energy_ev=energy_ev,
            ref_charging=ref_charging,
            ts_heat_el=ts_heat_el)
        # subtract charging of other storage types
        charging_ref = pd.read_csv(f"{res_dir_ref}/charging_{i}.csv", index_col=0,
                                   parse_dates=True)
        new_res_load = new_res_load - charging_ref[
            charging_ref.columns[charging_ref.columns != str(storage_type)]].sum(axis=1)
        # initialise base model
        model = se.set_up_base_model(scenario_dict=scenario_dict,
                                     new_res_load=new_res_load)
        # add heat pump model if flexible
        if hp_mode == "flexible":
            model = add_heat_pump_model(model, p_nom_hp, capacity_tes,
                                        scenario_dict["ts_cop"], ts_heat_demand)
        # add ev model if flexible
        if ev_mode == "flexible":
            add_evs_model(model, flexibility_bands)
        # add storage equivalents
        model = se.add_storage_equivalent_model(
            model, new_res_load,
            time_horizons=[scenario_dict["time_horizons"][storage_type]])
        # define objective
        model.objective = pm.Objective(rule=getattr(se, scenario_dict["objective"]),
                                       sense=pm.minimize,
                                       doc='Define objective function')
        opt = pm.SolverFactory(solver)
        if solver == "gurobi":
            if ev_mode == "flexible":
                opt.options["Method"] = 1
            else:
                opt.options["Method"] = 0
        results = opt.solve(model, tee=True)
        # extract results
        slacks = pd.Series(model.slack_res_load_neg.extract_values()) + \
                 pd.Series(model.slack_res_load_pos.extract_values())
        if slacks.sum() > 1e-9:
            raise ValueError("Slacks are being used. Please check. Consider increasing "
                             "weights.")
        charging = pd.Series(model.charging.extract_values()).unstack().T.set_index(
            new_res_load.index)
        charging.to_csv(f"{res_dir}/charging_{i}.csv")
        if extract_storage_duration:
            storage_durations = pd.concat([storage_durations,
                                           se.determine_storage_durations(charging, i)])
        energy_levels = \
            pd.Series(model.energy_levels.extract_values()).unstack().T.set_index(
                new_res_load.index)
        energy_levels.to_csv(f"{res_dir}/energy_levels_{i}.csv")
        abs_charging = pd.Series(model.abs_charging.extract_values()).unstack()
        # save flexible hp operation
        if (hp_mode == "flexible") & (nr_hp_mio == 20.0):
            hp_operation = pd.Series(model.charging_hp_el.extract_values())
            hp_operation.index = scenario_dict["ts_demand"].index
            hp_operation.to_csv(f"{res_dir}/hp_charging_flexible.csv")
        # save flexible hp operation
        if (ev_mode == "flexible") & (nr_ev_mio == 40.0):
            ev_operation = pd.Series(model.charging_ev.extract_values()).unstack().T
            ev_operation.index = scenario_dict["ts_demand"].index
            ev_operation.to_csv(f"{res_dir}/ev_charging_flexible.csv")
        df_tmp = (abs_charging.sum(axis=1) / 2).reset_index().rename(
            columns={"index": "storage_type", 0: "energy_stored"})
        df_tmp["nr_hp"] = nr_hp_mio
        df_tmp["nr_ev"] = nr_ev_mio
        shifted_energy_df = shifted_energy_df.append(df_tmp, ignore_index=True)
        df_tmp["energy_stored"] = \
            df_tmp["energy_stored"] / \
            (scenario_dict["ts_demand"].sum().sum() + sum_energy_heat + energy_ev) * 100
        shifted_energy_rel_df = shifted_energy_rel_df.append(df_tmp,
                                                             ignore_index=True)
    shifted_energy_df.to_csv(f"{res_dir}/storage_equivalents.csv")
    shifted_energy_rel_df.to_csv(
        f"{res_dir}/storage_equivalents_relative.csv")
    if extract_storage_duration:
        storage_durations.to_csv(f"{res_dir}/storage_durations.csv")
    # plot results
    if plot_results:
        if hp_mode is not None:
            plot_storage_equivalent_germany_stacked(shifted_energy_df,
                                                    parameter={
                                                        "nr_hp": "Number HPs [Mio.]"})
        if ev_mode is not None:
            plot_storage_equivalent_germany_stacked(shifted_energy_df,
                                                    parameter={
                                                        "nr_ev": "Number EVs [Mio.]"})
    # remove timeseries as they cannot be saved in json format
    save_scenario_dict(scenario_dict, res_dir)
    print("SUCCESS")
