import numpy as np
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import pyomo.environ as pm
import time

from scenario_input import base_scenario, scenario_input_hps, save_scenario_dict, \
    scenario_input_evs, adjust_timeseries_data
from storage_equivalent.storage_equivalent_model import determine_storage_durations, \
    get_balanced_storage_equivalent_model
from storage_equivalent.heat_pump_model import model_input_hps
from storage_equivalent.ev_model import model_input_evs
from plotting import plot_storage_equivalent_germany_stacked


if __name__ == "__main__":
    scenario = "test_to_delete"
    extract_storage_duration = False
    vres_data_source = "ego" # "ego", "rn"
    year = 2011 # solar_min: 1981, solar_max: 2003, wind_min: 2010, wind_max: 1990
    plot_results = True
    nr_iterations = 1
    max_iteration_balance = 5
    solver = "gurobi"
    use_binaries = True
    hp_mode = None # None, "flexible", "inflexible"
    ev_mode = None  # None, "flexible", "inflexible"
    tes_relative_size = 1 # in share standard
    ev_extended_flex = False
    ev_v2g = False
    flexible_ev_use_cases = ["home", "work"]
    if ev_extended_flex:
        flexible_ev_use_cases = ["home", "work", "public"]
    # create results directory
    res_dir = os.path.join(f"results/two_weeks_weight_one/{scenario}")
    os.makedirs(res_dir, exist_ok=True)
    # load scenario values
    scenario_dict = base_scenario(vres_data_source=vres_data_source, year=year)
    scenario_dict["solver"] = solver
    if hp_mode is not None:
        scenario_dict = scenario_input_hps(scenario_dict=scenario_dict, mode=hp_mode,
                                           use_binaries=use_binaries)
    if ev_mode is not None:
        scenario_dict = scenario_input_evs(scenario_dict=scenario_dict, mode=ev_mode,
                                           use_cases_flexible=flexible_ev_use_cases,
                                           extended_flex=ev_extended_flex,
                                           use_binaries=use_binaries)
    scenario_dict["share_pv"] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # shift timeseries
    scenario_dict = adjust_timeseries_data(scenario_dict)
    # adjust model input to hps and evs
    nr_hp_mio, ts_heat_el, sum_energy_heat, capacity_tes, p_nom_hp, ts_heat_demand = \
        model_input_hps(
            scenario_dict=scenario_dict,
            hp_mode=hp_mode,
            nr_hp_mio=19.4
        )
    nr_ev_mio, flexibility_bands, energy_ev, ref_charging = model_input_evs(
        scenario_dict=scenario_dict,
        ev_mode=ev_mode,
        nr_ev_mio=48.8
    )
    shifted_energy_df = pd.DataFrame()
    shifted_energy_rel_df = pd.DataFrame()
    storage_durations = pd.DataFrame()
    for share_pv in scenario_dict["share_pv"]:
        model, new_res_load = get_balanced_storage_equivalent_model(
            scenario_dict=scenario_dict,
            max_iter=max_iteration_balance,
            ref_charging=ref_charging,
            flexibility_bands=flexibility_bands,
            energy_ev=energy_ev,
            ts_heat_el=ts_heat_el,
            sum_energy_heat=sum_energy_heat,
            ts_heat_demand=ts_heat_demand,
            p_nom_hp=p_nom_hp,
            capacity_tes=capacity_tes,
            use_binaries=use_binaries,
            share_pv=share_pv
        )

        for iter in range(nr_iterations):
            model_tmp = model.clone()
            np.random.seed(int(time.time()))
            opt = pm.SolverFactory(solver)
            if solver == "gurobi":
                opt.options["Seed"] = int(time.time())
                opt.options["Method"] = 3
            opt = pm.SolverFactory(solver)

            results = opt.solve(model_tmp, tee=True)
            # extract results
            charging = pd.Series(model_tmp.charging.extract_values()).unstack().T.set_index(
                new_res_load.index)
            if extract_storage_duration:
                storage_durations = pd.concat([storage_durations,
                                               determine_storage_durations(charging, share_pv)])
            energy_levels = \
                pd.Series(model_tmp.energy_levels.extract_values()).unstack().T.set_index(
                    new_res_load.index)
            discharging = pd.Series(model_tmp.discharging.extract_values()).unstack()
            df_tmp = (discharging.sum(axis=1)).reset_index().rename(
                columns={"index": "storage_type", 0: "energy_stored"})
            df_tmp["share_pv"] = share_pv
            if iter == 0:
                charging.to_csv(f"{res_dir}/charging_{share_pv}.csv")
                energy_levels.to_csv(f"{res_dir}/energy_levels_{share_pv}.csv")
                # save flexible hp operation
                if (hp_mode == "flexible") & (nr_hp_mio == 20.0):
                    hp_operation = pd.Series(model_tmp.charging_hp_el.extract_values())
                    hp_operation.index = scenario_dict["ts_demand"].index
                    hp_operation.to_csv(f"{res_dir}/hp_charging_flexible.csv")
                # save flexible hp operation
                if (ev_mode == "flexible") & (nr_ev_mio == 40.0):
                    ev_operation = pd.Series(model_tmp.charging_ev.extract_values()).unstack().T
                    ev_operation.index = scenario_dict["ts_demand"].index
                    ev_operation.to_csv(f"{res_dir}/ev_charging_flexible.csv")
                shifted_energy_df = pd.concat([shifted_energy_df, df_tmp])
                df_tmp["energy_stored"] = \
                    df_tmp["energy_stored"] / \
                    (scenario_dict["ts_demand"].sum().sum() + sum_energy_heat + energy_ev) * 100
                shifted_energy_rel_df = pd.concat([shifted_energy_rel_df, df_tmp])
            else:
                assert_frame_equal(df_tmp.sort_index(axis=1), shifted_energy_df.loc[
                    (shifted_energy_df["share_pv"] == share_pv)].sort_index(axis=1))
    shifted_energy_df.to_csv(f"{res_dir}/storage_equivalents.csv")
    shifted_energy_rel_df.to_csv(
        f"{res_dir}/storage_equivalents_relative.csv")
    if extract_storage_duration:
        storage_durations.to_csv(f"{res_dir}/storage_durations.csv")
    # remove timeseries as they cannot be saved in json format
    save_scenario_dict(scenario_dict, res_dir)
    # plot results
    if plot_results:
        plot_storage_equivalent_germany_stacked(shifted_energy_df,
                                                parameter={"share_pv": "Share PV [-]"})
    print("SUCCESS")
