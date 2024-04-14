import os
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
import pyomo.environ as pm

from storage_equivalent import add_storage_equivalent_model, set_up_base_model, \
    solve_model, determine_storage_durations, minimize_discharging
from scenario_input import base_scenario, scenario_input_bess, \
    scenario_variation_battery_storage, adjust_timeseries_data, get_new_residual_load, \
    save_scenario_dict
from bess_model import add_battery_storage_model

if __name__ == "__main__":
    solver = "gurobi"
    use_binaries = True
    extract_storage_duration=False
    scenarios = scenario_variation_battery_storage()
    for scenario, scenario_input in scenarios.items():
        # try:
        print(f"Start solving scenario {scenario}")
        # create results directory
        res_dir = os.path.join(f"results/Diss2024_final/{scenario}")
        os.makedirs(res_dir, exist_ok=True)
        if os.path.isfile(os.path.join(res_dir, "storage_equivalents.csv")):
            print(f"Scenario {scenario} already solved. Skipping scenario.")
            continue
        # load scenario values
        scenario_dict = base_scenario()
        scenario_dict.update(scenario_input)
        scenario_dict["solver"] = solver
        if scenario_dict["bess_mode"] is not None:
            scenario_dict = scenario_input_bess(
                scenario_dict=scenario_dict,
                mode=scenario_dict["bess_mode"],
                use_binaries=use_binaries)
        # shift timeseries
        scenario_dict = adjust_timeseries_data(scenario_dict)
        # initialise result
        shifted_energy_df = pd.DataFrame()
        shifted_energy_rel_df = pd.DataFrame()
        storage_durations = pd.DataFrame()
        energy_consumed = pd.DataFrame()
        for i in range(11):
            nr_bess_mio = i*1.94
            if scenario_dict["bess_mode"] == "inflexible":
                ts_bess_reference = scenario_dict["ts_bess_reference"]*(i/10)
            else:
                ts_bess_reference = scenario_dict["ts_bess_reference"] * 0
            bess_p_nom = scenario_dict["bess_p_nom"]*(i/10)
            bess_capacity = scenario_dict["bess_capacity"]*(i/10)
            new_res_load = get_new_residual_load(
                    scenario_dict=scenario_dict, ts_bess=ts_bess_reference)
            # initialise base model
            model = set_up_base_model(scenario_dict=scenario_dict,
                                      new_res_load=new_res_load)
            if scenario_dict["bess_mode"] == "flexible":
                model = add_battery_storage_model(
                    model=model,
                    p_nom_bess=bess_p_nom,
                    capacity_bess=bess_capacity,
                    use_binaries=scenario_dict["bess_use_binaries"]
                )
            # add storage equivalents
            model = add_storage_equivalent_model(
                model, new_res_load,
                time_horizons=scenario_dict["time_horizons"])
            # define objective
            model.objective = pm.Objective(
                rule=minimize_discharging,
                sense=pm.minimize,
                doc='Define objective function')
            model = solve_model(model=model,
                                solver=scenario_dict["solver"],
                                bess_mode=scenario_dict["bess_mode"],)
            # extract results
            charging = pd.Series(
                model.charging.extract_values()).unstack().T.set_index(
                new_res_load.index)
            if extract_storage_duration:
                storage_durations = pd.concat([storage_durations,
                                               determine_storage_durations(charging,
                                                                           nr_bess_mio)])
            energy_levels = \
                pd.Series(
                    model.energy_levels.extract_values()).unstack().T.set_index(
                    new_res_load.index)
            discharging = pd.Series(
                model.discharging.extract_values()).unstack()
            df_tmp = (discharging.sum(axis=1)).reset_index().rename(
                columns={"index": "storage_type", 0: "energy_stored"})
            df_tmp["nr_bess_mio"] = nr_bess_mio
            iter_i = 0
            if iter_i == 0:
                charging.to_csv(f"{res_dir}/charging_{nr_bess_mio}.csv")
                energy_levels.to_csv(f"{res_dir}/energy_levels_{nr_bess_mio}.csv")
                # save flexible hp operation
                if (scenario_dict["bess_mode"] == "flexible") & (nr_bess_mio == 19.4):
                    bess_operation = pd.Series(
                        model.charging_bess.extract_values())-pd.Series(
                        model.discharging_bess.extract_values())
                    bess_operation.index = scenario_dict["ts_demand"].index
                    bess_operation.to_csv(f"{res_dir}/bess_charging_flexible.csv")

                shifted_energy_df = pd.concat([shifted_energy_df, df_tmp])

            else:
                assert_frame_equal(df_tmp.sort_index(axis=1), shifted_energy_df.loc[
                    (shifted_energy_df["nr_bess_mio"] == nr_bess_mio)].sort_index(axis=1))
        shifted_energy_df.to_csv(f"{res_dir}/storage_equivalents.csv")
        shifted_energy_rel_df.to_csv(
            f"{res_dir}/storage_equivalents_relative.csv")
        if extract_storage_duration:
            storage_durations.to_csv(f"{res_dir}/storage_durations.csv")
        # remove timeseries as they cannot be saved in json format
        save_scenario_dict(scenario_dict, res_dir)
        # except Exception as e:
        #     print(f"Something went wrong in scenario {scenario}. Skipping.")
        #     print(e)
        print("SUCCESS")
