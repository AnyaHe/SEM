import numpy as np
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import pyomo.environ as pm
import time

from scenario_input import base_scenario, save_scenario_dict, get_new_residual_load, adjust_timeseries_data
from storage_equivalent import set_up_base_model, add_storage_equivalent_model, \
    minimize_discharging, determine_storage_durations


if __name__ == "__main__":
    scenario = "Variation_pricing"
    tariffs = ['Cl', 'Csl', 'Ec', 'Ec_Cl', 'Ec_Clf', 'Ec_Clf_Sv', 'Ec_Cl_Sv', 'Ec_Sv',
               'Edn', 'Edn_Cl', 'Edn_Clf', "Ec_Sr_capped", "Ec_Sr_shifted"]
    ts_dir = r"H:\Tariffs\Timeseries"
    solver = "gurobi"
    extract_storage_duration = True
    plot_results = False
    nr_iterations = 1
    if not extract_storage_duration:
        nr_iterations = 1
    # create results directory
    res_dir = os.path.join(f"results/Diss2024_final/{scenario}")
    os.makedirs(res_dir, exist_ok=True)
    # initialize results
    shifted_energy_df = pd.DataFrame()
    shifted_energy_rel_df = pd.DataFrame()
    storage_durations = pd.DataFrame()
    # load scenario and remove household load
    scenario_dict_base = base_scenario()
    scenario_dict_base["solver"] = solver
    scaling_hh = 2*19.2
    ts_customer_group = pd.read_csv(
        os.path.join(ts_dir, "HH", "Ec", "timeseries.csv"),
        index_col=0, parse_dates=True).sum(axis=1).multiply(scaling_hh)
    scenario_dict_base["ts_demand"]["Residential"] = -ts_customer_group
    # calculate storage equivalent for different tariffs
    for tariff in tariffs:
        scenario_dict = scenario_dict_base.copy()
        ts_customer_group = pd.read_csv(
            os.path.join(ts_dir, "EV_HP", tariff, "timeseries.csv"),
            index_col=0, parse_dates=True).sum(axis=1).multiply(scaling_hh)
        scenario_dict["ts_demand"]["Residential_new"] = ts_customer_group
        # shift timeseries
        scenario_dict = adjust_timeseries_data(scenario_dict)
        new_res_load = get_new_residual_load(scenario_dict)
        model = set_up_base_model(scenario_dict=scenario_dict,
                                  new_res_load=new_res_load)
        model = add_storage_equivalent_model(model, new_res_load,
                                             time_horizons=scenario_dict["time_horizons"])
        model.objective = pm.Objective(rule=minimize_discharging,
                                       sense=pm.minimize,
                                       doc='Define objective function')
        for iter in range(nr_iterations):
            model_tmp = model.clone()
            np.random.seed(int(time.time()))
            opt = pm.SolverFactory(solver)

            results = opt.solve(model_tmp, tee=True)
            # extract results
            charging = pd.Series(model_tmp.charging.extract_values()).unstack().T.set_index(
                new_res_load.index)
            if extract_storage_duration:
                storage_durations = pd.concat([storage_durations,
                                               determine_storage_durations(
                                                   charging, f"{tariff}")])
            energy_levels = \
                pd.Series(model_tmp.energy_levels.extract_values()).unstack().T.set_index(
                    new_res_load.index)
            discharging = pd.Series(model_tmp.discharging.extract_values()).unstack()
            df_tmp = (discharging.sum(axis=1)).reset_index().rename(
                columns={"index": "storage_type", 0: "energy_stored"})
            df_tmp["tariff"] = tariff
            if iter == 0:
                charging.to_csv(f"{res_dir}/charging_{tariff}.csv")
                energy_levels.to_csv(f"{res_dir}/energy_levels_{tariff}.csv")
                shifted_energy_df = pd.concat([shifted_energy_df, df_tmp])
                df_tmp["energy_stored"] = \
                    df_tmp["energy_stored"] / \
                    (scenario_dict["ts_demand"].sum().sum()) * 100
                shifted_energy_rel_df = pd.concat([shifted_energy_rel_df, df_tmp])
            else:
                assert_frame_equal(df_tmp.sort_index(axis=1), shifted_energy_df.loc[
                    (shifted_energy_df["tariff"] == tariff)].sort_index(axis=1))
    shifted_energy_df.to_csv(f"{res_dir}/storage_equivalents.csv")
    shifted_energy_rel_df.to_csv(
        f"{res_dir}/storage_equivalents_relative.csv")
    if extract_storage_duration:
        storage_durations.to_csv(f"{res_dir}/storage_durations.csv")
    # remove timeseries as they cannot be saved in json format
    save_scenario_dict(scenario_dict_base, res_dir)
    print("SUCCESS")
