import os
import pandas as pd
from pandas.testing import assert_frame_equal

import storage_equivalent as se
from heat_pump_model import model_input_hps
from ev_model import model_input_evs
from scenario_input import base_scenario, scenario_input_hps, save_scenario_dict, \
    scenario_input_evs, adjust_timeseries_data
from plotting import plot_storage_equivalent_germany_stacked


if __name__ == "__main__":
    scenario = "test_HP_reference"
    extract_storage_duration = True
    plot_results = False
    vres_data_source = "ego"
    year = 2011 # solar_min: 1981, solar_max: 2003, wind_min: 2010, wind_max: 1990
    nr_iterations = 10
    solver = "gurobi"
    hp_mode = "inflexible" # None, "flexible", "inflexible"
    ev_mode = None # None, "flexible", "inflexible"
    tes_relative_size = 4 # in share standard
    ev_extended_flex = True
    ev_v2g = True
    flexible_ev_use_cases = ["home", "work", "public"]
    if ev_extended_flex:
        flexible_ev_use_cases = ["home", "work", "public"]
    # relative_weighting = 1000
    # weights = [relative_weighting, relative_weighting**2,
    #                        relative_weighting**3]
    # create results directory
    res_dir = os.path.join(f"results/two_weeks_weight_one/{scenario}")
    os.makedirs(res_dir, exist_ok=True)
    # load scenario values
    scenario_dict = base_scenario(vres_data_source=vres_data_source, year=year)
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
                                           extended_flex=ev_extended_flex, v2g=ev_v2g)
    # shift timeseries
    scenario_dict = adjust_timeseries_data(scenario_dict)
    # initialise result
    shifted_energy_df = pd.DataFrame()
    shifted_energy_rel_df = pd.DataFrame()
    storage_durations = pd.DataFrame()
    for i in range(9):
        print(f"Info: Starting iteration {i} of scenario integration of sector coupling {scenario}")
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

        model, new_res_load = se.get_balanced_storage_equivalent_model(
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
        )

        for iter_i in range(nr_iterations):

            print(f"Info: Starting iteration {iter_i} solving final model.")

            if iter_i == 0:
                model_tmp = model
            else:
                model_tmp = model.clone()
                model_tmp = se.solve_model(model_tmp, solver, hp_mode, ev_mode, ev_v2g)

            # extract results
            charging = pd.Series(model_tmp.charging.extract_values()).unstack().T.set_index(
                new_res_load.index)
            if extract_storage_duration:
                storage_durations = pd.concat([storage_durations,
                                               se.determine_storage_durations(charging, i)])
            energy_levels = \
                pd.Series(model_tmp.energy_levels.extract_values()).unstack().T.set_index(
                    new_res_load.index)
            discharging = pd.Series(model_tmp.discharging.extract_values()).unstack()
            df_tmp = (discharging.sum(axis=1)).reset_index().rename(
                columns={"index": "storage_type", 0: "energy_stored"})
            df_tmp["nr_hp"] = nr_hp_mio
            df_tmp["nr_ev"] = nr_ev_mio

            if iter_i == 0:
                charging.to_csv(f"{res_dir}/charging_{i}.csv")
                energy_levels.to_csv(f"{res_dir}/energy_levels_{i}.csv")
                # save flexible hp operation
                if (hp_mode == "flexible") & (nr_hp_mio == 20.0):
                    hp_operation = pd.Series(model_tmp.charging_hp_el.extract_values())
                    hp_operation.index = scenario_dict["ts_demand"].index
                    hp_operation.to_csv(f"{res_dir}/hp_charging_flexible.csv")
                    charging_tes = pd.Series(model_tmp.charging_tes.extract_values())
                    discharging_tes = pd.Series(model_tmp.discharging_tes.extract_values())
                    tes_operation = charging_tes - discharging_tes
                    tes_operation.index = scenario_dict["ts_demand"].index
                    tes_operation.to_csv(f"{res_dir}/tes_operation_flexible.csv")
                    tes_energy = pd.Series(model_tmp.energy_tes.extract_values())
                    tes_energy.index = scenario_dict["ts_demand"].index
                    tes_energy.to_csv(f"{res_dir}/tes_energy_flexible.csv")
                # save flexible hp operation
                if (ev_mode == "flexible") & (nr_ev_mio == 40.0):
                    ev_operation = pd.Series(model_tmp.charging_ev.extract_values()).unstack().T
                    if ev_v2g:
                        ev_operation -= pd.Series(model_tmp.discharging_ev.extract_values()).unstack().T
                    ev_operation.index = scenario_dict["ts_demand"].index
                    ev_operation.to_csv(f"{res_dir}/ev_charging_flexible.csv")
                shifted_energy_df = pd.concat([shifted_energy_df, df_tmp])
                df_tmp["energy_stored"] = \
                    df_tmp["energy_stored"] / \
                    (scenario_dict["ts_demand"].sum().sum() + sum_energy_heat + energy_ev) * 100
                shifted_energy_rel_df = pd.concat([shifted_energy_rel_df, df_tmp])
            else:
                assert_frame_equal(df_tmp, shifted_energy_df.loc[(shifted_energy_df["nr_hp"] == nr_hp_mio)&
                                                                 (shifted_energy_df["nr_ev"] == nr_ev_mio)])
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
