import numpy as np
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import pyomo.environ as pm
import time

from scenario_input import base_scenario, save_scenario_dict, get_new_residual_load, adjust_timeseries_data
from storage_equivalent import set_up_base_model, add_storage_equivalent_model, \
    minimize_discharging, determine_storage_durations


def load_adapted_timeseries_per_grid(gd_dir, gd_id, nt, ts_hh):
    households = \
        pd.read_csv(os.path.join(gd_dir, str(gd_id),
                                 f"households_{penetration}.csv"), index_col=0)
    household_mapping = pd.read_csv(os.path.join(gd_dir, str(gd_id), "mapping.csv"),
                                    index_col=0).set_index("name_old")
    ts_load = pd.DataFrame()
    for (has_ev, has_hp, has_pv, has_bess), household_group in households.groupby(
            ["has_ev", "has_hp", "has_pv", "has_bess"]):
        # get names of consumers in timeseries
        names_new = household_mapping.loc[household_group.index, "name_new"]
        # determine group of time series
        customer_group = [has_ev * "EV", has_hp * "HP", has_pv * "PV",
                          has_bess * "BESS"]
        customer_group = "_".join(list(filter(("").__ne__, customer_group)))
        if customer_group == "":
            customer_group = "HH"
        # load time series and results
        ts_customer_group = pd.read_csv(
            os.path.join(ts_dir, customer_group, nt, "timeseries.csv"),
            index_col=0, parse_dates=True)
        ts_load_tmp = ts_customer_group.loc[:, names_new]
        ts_load_tmp.columns = household_group.index
        ts_load = pd.concat([ts_load, ts_load_tmp], axis=1)
        # save original load
        if customer_group == "HH":
            ts_hh += ts_customer_group.loc[:, household_mapping.name_new].sum(axis=1)
    return ts_load, ts_hh


if __name__ == "__main__":
    penetration = 0.1

    data_dir = r"H:\Tariffs"
    grid_dir = os.path.join(data_dir, "Grids")
    grids = os.listdir(grid_dir)
    scenario = f"Variation_pricing_{penetration}"
    tariffs = [
        # 'Er-mv',
        # 'Er',
        'Cl',
        'Clf',
        'Csl',
        'Ec',
        'Edn',
        'Ec_Clf',
        # 'Ec_Clf_Sv',
        # 'Ec_Cl_Sv',
        'Ec_Sr',
        'Ec_Sv',
        'Clf_Sr',
        # 'Edn_Cl',
        # 'Edn_Clf'
    ]
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

    # calculate storage equivalent for different tariffs
    for tariff in tariffs:
        scenario_dict = scenario_dict_base.copy()
        # load timeseries for all grids
        ts_households_orig = pd.Series(index=scenario_dict["ts_demand"].index, data=0)
        ts_customer_group = pd.DataFrame()
        for grid_id in grids:
            try:
                int(grid_id)
            except:
                continue
            ts_customer_group_grid, ts_households_orig = load_adapted_timeseries_per_grid(
                gd_dir=grid_dir,
                gd_id=grid_id,
                nt=tariff,
                ts_hh=ts_households_orig
            )
            ts_customer_group = pd.concat([ts_customer_group, ts_customer_group_grid], axis=1)
        scaling_hh = 19400/len(ts_customer_group.columns)
        scenario_dict["ts_demand"]["Residential_new"] = ts_customer_group.sum(axis=1).multiply(scaling_hh)
        scenario_dict_base["ts_demand"]["Residential"] = -ts_households_orig.multiply(scaling_hh)
        scenario_dict["ts_demand"]["Residential_new"].to_csv(os.path.join(res_dir, f"ts_hh_{tariff}.csv"))
        ts_households_orig.multiply(scaling_hh).to_csv(os.path.join(res_dir, f"ts_hh_orig.csv"))
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
