import numpy as np
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import pyomo.environ as pm
import time

import storage_equivalent as se
from heat_pump_model import model_input_hps
from ev_model import model_input_evs
from scenario_input import base_scenario, scenario_input_hps, save_scenario_dict, \
    scenario_input_evs, adjust_timeseries_data, scenario_variation_electric_vehicles, \
    scenario_variation_heat_pumps, scenario_variation_electric_vehicles_and_heat_pumps
from plotting import plot_storage_equivalent_germany_stacked


if __name__ == "__main__":
    mode = "ev"
    extract_storage_duration = True
    plot_results = False
    nr_iterations = 3
    solver = "gurobi"
    use_binaries = True
    if mode == "hp":
        scenarios = scenario_variation_heat_pumps()
    elif mode == "ev":
        scenarios = scenario_variation_electric_vehicles()
    elif mode == "ev_hp":
        scenarios = scenario_variation_electric_vehicles_and_heat_pumps()
    else:
        raise ValueError("Unknown mode.")
    for scenario_orig, scenario_input in scenarios.items():
        if "reference" in scenario_orig:
            continue
        # try:
        for storage_type in [0, 1, 2]:
            scenario = f"{scenario_orig}_reduction_{storage_type}"
            reference_scenario = f"{mode.upper()}_reference"
            # create results directory
            res_dir = os.path.join(f"results/final_run_paper/{scenario}")
            res_dir_ref = os.path.join(f"results/final_run_paper/{reference_scenario}")
            os.makedirs(res_dir, exist_ok=True)
            if os.path.isfile(os.path.join(res_dir, "storage_equivalents.csv")):
                print(f"Scenario {scenario} already solved. Skipping scenario.")
                continue
            # load scenario values
            scenario_dict = base_scenario()
            scenario_dict.update(scenario_input)
            scenario_dict["solver"] = solver
            # let only examined storage horizon "cost" something in objective
            weights_adjusted = [scenario_dict["weighting"][th] if th == storage_type else 0
                                for th in range(len(scenario_dict["time_horizons"]))]
            scenario_dict["weighting"] = weights_adjusted
            if scenario_dict["hp_mode"] is not None:
                scenario_dict = scenario_input_hps(scenario_dict=scenario_dict,
                                                   mode=scenario_dict["hp_mode"],
                                                   use_binaries=use_binaries)
                scenario_dict["capacity_single_tes"] = \
                    scenario_dict["tes_relative_size"] * scenario_dict["capacity_single_tes"]
            if scenario_dict["ev_mode"] is not None:
                scenario_dict = scenario_input_evs(scenario_dict=scenario_dict,
                                                   mode=scenario_dict["ev_mode"],
                                                   use_cases_flexible=scenario_dict["flexible_ev_use_cases"],
                                                   extended_flex=scenario_dict["ev_extended_flex"],
                                                   v2g=scenario_dict["ev_v2g"],
                                                   use_binaries=use_binaries)
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
                        hp_mode=scenario_dict["hp_mode"],
                        i=i
                    )
                nr_ev_mio, flexibility_bands, energy_ev, ref_charging = model_input_evs(
                    scenario_dict=scenario_dict,
                    ev_mode=scenario_dict["ev_mode"],
                    i=i
                )
                # get shifted energy from other storage units
                charging_ref = pd.read_csv(f"{res_dir_ref}/charging_{i}.csv", index_col=0,
                                           parse_dates=True)
                charging_ref.columns = [int(col) for col in charging_ref.columns]
                shifted_enery_ref = charging_ref[
                    charging_ref.columns[charging_ref.columns != storage_type]]
                shifted_enery_ref = shifted_enery_ref[shifted_enery_ref < 0].abs().sum()

                model, new_res_load = se.get_balanced_storage_equivalent_model(
                    scenario_dict=scenario_dict,
                    ref_charging=ref_charging,
                    flexibility_bands=flexibility_bands,
                    energy_ev=energy_ev,
                    ts_heat_el=ts_heat_el,
                    sum_energy_heat=sum_energy_heat,
                    ts_heat_demand=ts_heat_demand,
                    p_nom_hp=p_nom_hp,
                    capacity_tes=capacity_tes,
                    use_binaries=use_binaries,
                    fixed_shifted_energy=shifted_enery_ref
                )

                for iter_i in range(nr_iterations):

                    print(f"Info: Starting iteration {iter_i} solving final model.")

                    if iter_i == 0:
                        model_tmp = model
                    else:
                        model_tmp = model.clone()
                        model_tmp = se.solve_model(model_tmp, solver, scenario_dict["hp_mode"],
                                                   scenario_dict["ev_mode"], scenario_dict["ev_v2g"])

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
                        if (scenario_dict["hp_mode"] == "flexible") & (nr_hp_mio == 20.0):
                            hp_operation = pd.Series(model_tmp.charging_hp_el.extract_values())
                            hp_operation.index = scenario_dict["ts_demand"].index
                            hp_operation.to_csv(f"{res_dir}/hp_charging_flexible.csv")
                        # save flexible hp operation
                        if (scenario_dict["ev_mode"] == "flexible") & (nr_ev_mio == 40.0):
                            ev_operation = pd.Series(model_tmp.charging_ev.extract_values()).unstack().T
                            ev_operation.index = scenario_dict["ts_demand"].index
                            ev_operation.to_csv(f"{res_dir}/ev_charging_flexible.csv")
                        shifted_energy_df = pd.concat([shifted_energy_df, df_tmp])
                        df_tmp["energy_stored"] = \
                            df_tmp["energy_stored"] / \
                            (scenario_dict["ts_demand"].sum().sum() + sum_energy_heat + energy_ev) * 100
                        shifted_energy_rel_df = pd.concat([shifted_energy_rel_df, df_tmp])
                    else:
                        assert np.isclose(df_tmp, shifted_energy_df.loc[
                            (shifted_energy_df["nr_hp"] == nr_hp_mio) &
                            (shifted_energy_df["nr_ev"] == nr_ev_mio)],
                            rtol=1e-04, atol=1e-02).all().all()
            shifted_energy_df.to_csv(f"{res_dir}/storage_equivalents.csv")
            shifted_energy_rel_df.to_csv(
                f"{res_dir}/storage_equivalents_relative.csv")
            if extract_storage_duration:
                storage_durations.to_csv(f"{res_dir}/storage_durations.csv")
            # plot results
            if plot_results:
                if scenario_dict["hp_mode"] is not None:
                    plot_storage_equivalent_germany_stacked(shifted_energy_df,
                                                            parameter={
                                                                "nr_hp": "Number HPs [Mio.]"})
                if scenario_dict["ev_mode"] is not None:
                    plot_storage_equivalent_germany_stacked(shifted_energy_df,
                                                            parameter={
                                                                "nr_ev": "Number EVs [Mio.]"})
            # remove timeseries as they cannot be saved in json format
            save_scenario_dict(scenario_dict, res_dir)
        # except Exception as e:
        #     print(f"Something went wrong in scenario {scenario}. Skipping.")
        #     print(e)
    print("SUCCESS")
