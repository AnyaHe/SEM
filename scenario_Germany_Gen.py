import pandas as pd
import pyomo.environ as pm
import os

from scenario_input import base_scenario, scenario_input_hps, save_scenario_dict, \
    scenario_input_evs, get_new_residual_load, adjust_timeseries_data
from storage_equivalent import set_up_base_model, add_storage_equivalent_model, \
    minimize_energy, determine_storage_durations
from heat_pump_model import add_heat_pump_model, model_input_hps
from ev_model import add_evs_model, model_input_evs
from plotting import plot_storage_equivalent_germany_stacked


if __name__ == "__main__":
    scenario = "test_to_delete"
    extract_storage_duration = True
    solver = "gurobi"
    ev_mode = None # None, "flexible", "inflexible"
    hp_mode = "flexible"  # None, "flexible", "inflexible"
    tes_relative_size = 1  # in share standard
    flexible_ev_use_cases = ["home", "work"]
    # create results directory
    res_dir = os.path.join(f"results/{scenario}")
    os.makedirs(res_dir, exist_ok=True)
    # load scenario values
    scenario_dict = base_scenario()
    scenario_dict["hp_mode"] = hp_mode
    scenario_dict["ev_mode"] = ev_mode
    scenario_dict["solver"] = solver
    if hp_mode is not None:
        scenario_dict = scenario_input_hps(scenario_dict=scenario_dict, mode=hp_mode)
    if ev_mode is not None:
        scenario_dict = scenario_input_evs(scenario_dict=scenario_dict, mode=ev_mode,
                                           use_cases_flexible=flexible_ev_use_cases)
    scenario_dict["share_pv"] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # shift timeseries
    scenario_dict = adjust_timeseries_data(scenario_dict)
    # adjust model input to hps and evs
    nr_hp_mio, ts_heat_el, sum_energy_heat, capacity_tes, p_nom_hp, ts_heat_demand = \
        model_input_hps(
            scenario_dict=scenario_dict,
            hp_mode=hp_mode,
            nr_hp_mio=20
        )
    nr_ev_mio, flexibility_bands, energy_ev, ref_charging = model_input_evs(
        scenario_dict=scenario_dict,
        ev_mode=ev_mode,
        nr_ev_mio=40
    )
    shifted_energy_df = pd.DataFrame(columns=["share_pv", "storage_type",
                                              "energy_stored"])
    shifted_energy_rel_df = pd.DataFrame(columns=["share_pv", "storage_type",
                                                  "energy_stored"])
    storage_durations = pd.DataFrame()
    for share_pv in scenario_dict["share_pv"]:
        new_res_load = get_new_residual_load(scenario_dict,
                                             share_pv=share_pv,
                                             sum_energy_heat=sum_energy_heat,
                                             energy_ev=energy_ev,
                                             ref_charging=ref_charging,
                                             ts_heat_el=ts_heat_el)
        model = set_up_base_model(scenario_dict=scenario_dict,
                                  new_res_load=new_res_load)
        if hp_mode == "flexible":
            model = add_heat_pump_model(model, p_nom_hp, capacity_tes,
                                        scenario_dict["ts_cop"], ts_heat_demand)
        if ev_mode == "flexible":
            add_evs_model(model, flexibility_bands)
        model = add_storage_equivalent_model(model, new_res_load,
                                             time_horizons=[24, 7 * 24, 24 * 366])
        model.objective = pm.Objective(rule=minimize_energy,
                                       sense=pm.minimize,
                                       doc='Define objective function')
        opt = pm.SolverFactory(solver)
        if solver == "gurobi":
            if ev_mode == "flexible":
                opt.options["Method"] = 1
            else:
                opt.options["Method"] = 0
        results = opt.solve(model, tee=True)
        charging = pd.Series(model.charging.extract_values()).unstack().T.set_index(
                new_res_load.index)
        charging.to_csv(f"{res_dir}/charging_{share_pv}.csv")
        if extract_storage_duration:
            storage_durations = pd.concat([storage_durations,
                                           determine_storage_durations(charging, share_pv)])
        energy_levels = \
            pd.Series(model.energy_levels.extract_values()).unstack().T.set_index(
                new_res_load.index)
        energy_levels.to_csv(f"{res_dir}/energy_levels_{share_pv}.csv")
        abs_charging = pd.Series(model.abs_charging.extract_values()).unstack()
        df_tmp = (abs_charging.sum(axis=1) / 2).reset_index().rename(
            columns={"index": "storage_type", 0: "energy_stored"})
        df_tmp["share_pv"] = share_pv
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
    plot_storage_equivalent_germany_stacked(shifted_energy_df,
                                                parameter={
                                                    "share_pv": "Share PV [-]"})
    # remove timeseries as they cannot be saved in json format
    save_scenario_dict(scenario_dict, res_dir)
    print("SUCCESS")
