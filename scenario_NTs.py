import numpy as np
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import pyomo.environ as pm
import time

from scenario_input import base_scenario, save_scenario_dict, get_new_residual_load, adjust_timeseries_data
from storage_equivalent import set_up_base_model, add_storage_equivalent_model, \
    minimize_discharging, determine_storage_durations, solve_model
from edisgo.edisgo import import_edisgo_from_files
from heat_pump_model import add_heat_pump_model
from ev_model import add_evs_model
from bess_model import add_battery_storage_model


def load_adapted_timeseries_per_grid(gd_dir, gd_id, nt, ts_hh):
    households = \
        pd.read_csv(os.path.join(gd_dir, str(gd_id),
                                 f"households_{penetration}.csv"), index_col=0)
    household_mapping = pd.read_csv(os.path.join(gd_dir, str(gd_id), "mapping.csv"),
                                    index_col=0).set_index("name_old")
    ts_load = pd.DataFrame()
    # save original load
    ts_customer_group = pd.read_csv(
        os.path.join(ts_dir, "HH", "Ec", "timeseries.csv"),
        index_col=0, parse_dates=True)
    ts_hh += ts_customer_group.loc[:, household_mapping.name_new].sum(axis=1)
    ts_load = update_load_ts(household_mapping, households, nt, ts_load)
    return ts_load, ts_hh


def update_load_ts(household_mapping, households, nt, ts_load):
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
    return ts_load


def load_adapted_timeseries_per_grid_clusters(gd_dir, gd_id, nt, ts_hh):
    clustering = pd.read_csv(os.path.join(gd_dir, "clustering.csv"), index_col=0)
    if "Er-mv" in nt:
        cluster = clustering.loc[f"{grid_id}_{penetration}", "Cluster"]
        return load_adapted_timeseries_per_grid(
            gd_dir=gd_dir,
            gd_id=gd_id,
            nt=f"Er-{cluster}",
            ts_hh=ts_hh
        )
    else:
        # save original load
        household_mapping = pd.read_csv(os.path.join(gd_dir, str(gd_id), "mapping.csv"),
                                        index_col=0).set_index("name_old")
        ts_customer_group = pd.read_csv(
            os.path.join(ts_dir, "HH", "Ec", "timeseries.csv"),
            index_col=0, parse_dates=True)
        ts_hh += ts_customer_group.loc[:, household_mapping.name_new].sum(axis=1)
        # load new time series
        households = \
            pd.read_csv(os.path.join(gd_dir, str(gd_id),
                                     f"households_{penetration}.csv"), index_col=0)
        edisgo_obj = import_edisgo_from_files(
            os.path.join(gd_dir, str(gd_id)),
        )
        # get clusters and households in cluster
        col_names = [f"{grid.id}_{penetration}" for grid in edisgo_obj.topology.lv_grids
                     if f"{grid.id}_{penetration}" in clustering.index]
        clustering_tmp = clustering.loc[col_names]
        clustering_tmp.index = clustering_tmp.index.str.split("_", expand=True).levels[0]

        households["cluster"] = clustering_tmp.loc[edisgo_obj.topology.buses_df.loc[
                                                       households.bus, "lv_grid_id"].astype(int).astype(
            str), "Cluster"].values
        ts_load = pd.DataFrame()
        for cluster in households.cluster.unique():
            households_tmp = households.loc[households.cluster == cluster]
            if tariff == "Er":
                tariff_tmp = f"Er-{cluster}"
            elif tariff == "Er_Clf":
                tariff_tmp = f"Er-{cluster}_Clf"
            else:
                raise NotImplementedError
            ts_load = update_load_ts(household_mapping, households_tmp, tariff_tmp, ts_load)

        return ts_load, ts_hh


def load_flexible_assets_per_grid(gd_dir, gd_id, ts_hh, ts_hh_new, scen_dict, nr_cust):
    households = \
        pd.read_csv(os.path.join(gd_dir, str(gd_id),
                                 f"households_{penetration}.csv"), index_col=0)
    household_mapping = pd.read_csv(os.path.join(gd_dir, str(gd_id), "mapping.csv"),
                                    index_col=0).set_index("name_old")
    # save original load
    ts_customer_group = pd.read_csv(
        os.path.join(ts_dir, "HH", "Ec", "timeseries.csv"),
        index_col=0, parse_dates=True)
    ts_hh += ts_customer_group.loc[:, household_mapping.name_new].sum(axis=1)
    ts_hh_new["hh_load_new"] += \
        ts_customer_group.loc[:, household_mapping.name_new].sum(axis=1)
    nr_cust += len(household_mapping)
    # todo: handle assets separately
    data = load_ts_data()
    for comp in ["has_ev", "has_hp", "has_pv", "has_bess"]:
        household_group = households.loc[households[comp] == True]
        # get names of consumers in timeseries
        names_new = household_mapping.loc[household_group.index, "name_new"]
        # determine group of time series
        if "ev" in comp:
            scen_dict["nr_ev_ref"] += len(names_new)
            scen_dict["ts_ref_charging"]["home"] += \
                data["charging_points_active_power"][names_new].sum(axis=1)

            for band in ["upper_power", "lower_energy", "upper_energy"]:
                scen_dict["ts_flex_bands"][band]["home"] += \
                    data[band][names_new].sum(axis=1)
        if "hp" in comp:
            scen_dict["nr_hp_ref"] += len(names_new)
            scen_dict["ts_heat_demand_ref"] += \
                data["heat_demand"][names_new].sum(axis=1)
            scen_dict["hp_p_nom_ref"] += \
                data["heat_pumps"].loc[names_new, "p_set"].sum()
            scen_dict["hp_capacity_tes_ref"] += \
                data["heat_pumps"].loc[names_new, "tes_capacity"].sum()
            scen_dict["hp_cop_ref"] = \
                pd.concat([scen_dict["hp_cop_ref"], data["cop"][names_new]], axis=1)
        if "pv" in comp:
            ts_hh_new["hh_load_new"] -= data["generators_active_power"][names_new].sum(axis=1)
        if "bess" in comp:
            scen_dict["nr_bess_ref"] += len(names_new)
            scen_dict["bess_p_nom_ref"] += \
                data["storage_units"].loc[names_new, "p_nom"].sum()
            scen_dict["bess_capacity_ref"] += \
                data["storage_units"].loc[names_new, "capacity"].sum()
    return ts_hh_new, ts_hh, scen_dict, nr_cust


def load_ts_data(group_dir = r"H:\Tariffs\Consumer_Groups"):
    # import timeseries of components
    print(f"Starting creation of time series data.")
    base_components = {
        "Inflexible": ["loads_active_power"],
        "Heat_Pumps": ["heat_pumps", "heat_demand", "cop"],
        "Electric_Vehicles": ["upper_power", "lower_energy", "upper_energy",
                              "charging_points_active_power"],
        "Photovoltaic": ["generators_active_power"],
        "Battery_Storage": ["storage_units"]
    }

    data_tmp = {}
    for comp_dir, data_names in base_components.items():
        for data_name in data_names:
            data_tmp[data_name] = \
                pd.read_csv(os.path.join(group_dir, comp_dir, f"{data_name}.csv"),
                            index_col=0, parse_dates=True)
    # adapt names to household loads
    names = data_tmp["loads_active_power"].columns
    for comp in ["heat_pumps", "storage_units"]:
        tmp = data_tmp[comp].iloc[:len(names)]
        tmp.index = names
        data_tmp[comp] = tmp
    for ts in ["heat_demand", "cop", "upper_power", "lower_energy", "upper_energy",
               "generators_active_power", "charging_points_active_power"]:
        tmp = data_tmp[ts].iloc[:, :len(names)]
        tmp.columns = names
        data_tmp[ts] = tmp
    return data_tmp


if __name__ == "__main__":
    penetration = 0.9

    data_dir = r"H:\Tariffs"
    grid_dir = os.path.join(data_dir, "Grids")
    grids = os.listdir(grid_dir)
    scenario = f"Variation_pricing_{penetration}_with_opt_and_ref"
    tariffs = [
        'Optimised',
        'Reference',
        'Ec',
        'Edn',
        'Er',
        'Er-mv',
        'Cl',
        'Csl',
        'Clf',
        'Er_Clf',
        'Ec_Sr',
        'Clf_Sr',
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
        ts_households_orig = pd.Series(index=scenario_dict["ts_demand"].index, data=0.0)
        ts_customer_group = pd.DataFrame()
        nr_customers = 0

        if not tariff == "Optimised":
            for grid_id in grids:
                try:
                    int(grid_id)
                except:
                    continue
                if "Er" in tariff:
                    ts_customer_group_grid, ts_households_orig = load_adapted_timeseries_per_grid_clusters(
                        gd_dir=grid_dir,
                        gd_id=grid_id,
                        nt=tariff,
                        ts_hh=ts_households_orig
                    )
                else:
                    ts_customer_group_grid, ts_households_orig = load_adapted_timeseries_per_grid(
                        gd_dir=grid_dir,
                        gd_id=grid_id,
                        nt=tariff,
                        ts_hh=ts_households_orig
                    )
                ts_customer_group = pd.concat([ts_customer_group, ts_customer_group_grid], axis=1)
                nr_customers += len(ts_customer_group_grid)
        else:
            # prepare initial data
            ts_customer_group["hh_load_new"] = ts_households_orig
            # EVs
            ts_evs = pd.DataFrame(
                columns=["home"], index=scenario_dict["ts_timesteps"], data=0.0)
            scenario_dict["ev_mode"] = "flexible"
            scenario_dict["use_cases_flexible"] = ["home"]
            scenario_dict["ev_charging_efficiency"] = 0.9
            scenario_dict["nr_ev_ref"] = 0
            scenario_dict["ts_ref_charging"] = ts_evs
            scenario_dict["ts_flex_bands"] = {
                "upper_power": ts_evs,
                "lower_energy": ts_evs,
                "upper_energy": ts_evs
            }
            # HPs
            scenario_dict["hp_mode"] = "flexible"
            scenario_dict["efficiency_static_tes"] = 1.0
            scenario_dict["efficiency_dynamic_tes"] = 1.0
            scenario_dict["nr_hp_ref"] = 0
            scenario_dict["ts_heat_demand_ref"] = pd.Series(
                index=scenario_dict["ts_timesteps"], data=0)
            scenario_dict["hp_p_nom_ref"] = 0
            scenario_dict["hp_capacity_tes_ref"] = 0
            scenario_dict["hp_cop_ref"] = pd.DataFrame()
            # BESS
            scenario_dict["bess_mode"] = "flexible"
            scenario_dict["bess_efficiency_charging"] = 1.0
            scenario_dict["bess_efficiency_discharging"] = 1.0
            scenario_dict["nr_bess_ref"] = 0
            scenario_dict["bess_p_nom_ref"] = 0
            scenario_dict["bess_capacity_ref"] = 0

            for grid_id in grids:
                try:
                    int(grid_id)
                except:
                    continue
                ts_customer_group_grid, ts_hh, scenario_dict, nr_customers = \
                    load_flexible_assets_per_grid(
                        gd_dir=grid_dir,
                        gd_id=grid_id,
                        ts_hh=ts_households_orig,
                        ts_hh_new=ts_customer_group,
                        scen_dict=scenario_dict,
                        nr_cust=nr_customers
                    )
        scaling_hh = 19400/nr_customers
        scenario_dict["ts_demand"]["Residential_new"] = \
            ts_customer_group.sum(axis=1).multiply(scaling_hh)
        scenario_dict_base["ts_demand"]["Residential"] = \
            -ts_households_orig.multiply(scaling_hh)
        scenario_dict["ts_demand"]["Residential_new"].to_csv(
            os.path.join(res_dir, f"ts_hh_{tariff}.csv"))
        ts_households_orig.multiply(scaling_hh).to_csv(
            os.path.join(res_dir, f"ts_hh_orig.csv"))
        scenario_dict["ts_cop"] = scenario_dict["hp_cop_ref"].mean(axis=1)
        # shift timeseries
        scenario_dict = adjust_timeseries_data(scenario_dict,)

        scaling_hp = scaling_hh*scenario_dict["nr_hp_ref"]/nr_customers
        ts_heat_el = scenario_dict["ts_heat_demand_ref"].divide(
            scenario_dict["ts_cop"]).multiply(scaling_hp)
        ts_heat_demand = scenario_dict["ts_heat_demand_ref"].multiply(scaling_hp)
        scaling_ev = scaling_hh * scenario_dict["nr_ev_ref"] / nr_customers
        scaling_bess = scaling_hh * scenario_dict["nr_bess_ref"] / nr_customers

        new_res_load = get_new_residual_load(
            scenario_dict=scenario_dict,
            sum_energy_heat=ts_heat_el.sum(),
            energy_ev=scenario_dict["ts_ref_charging"].multiply(scaling_ev).sum().sum()
        )
        model = set_up_base_model(scenario_dict=scenario_dict,
                                  new_res_load=new_res_load)
        # add heat pump model if flexible
        if scenario_dict["hp_mode"] == "flexible":
            model = add_heat_pump_model(
                model=model,
                p_nom_hp=scenario_dict["hp_p_nom_ref"]*scaling_hp,
                capacity_tes=scenario_dict["hp_capacity_tes_ref"]*scaling_hp,
                cop=scenario_dict["ts_cop"],
                heat_demand=ts_heat_demand,
                efficiency_static_tes=scenario_dict["efficiency_static_tes"],
                efficiency_dynamic_tes=scenario_dict["efficiency_dynamic_tes"]
            )
        # add ev model if flexible
        if scenario_dict["ev_mode"] == "flexible":
            flexibility_bands = {}
            for band in ["upper_power", "lower_energy", "upper_energy"]:
                flexibility_bands[band] = \
                    scenario_dict["ts_flex_bands"][band].multiply(scaling_ev)
            add_evs_model(
                model=model,
                flex_bands=flexibility_bands,
                v2g=scenario_dict["ev_v2g"],
                efficiency=scenario_dict["ev_charging_efficiency"],
                discharging_efficiency=scenario_dict["ev_discharging_efficiency"]
            )
        if scenario_dict["bess_mode"] == "flexible":
            model = add_battery_storage_model(
                model=model,
                p_nom_bess=scenario_dict["bess_p_nom_ref"]*scaling_bess,
                capacity_bess=scenario_dict["bess_capacity_ref"]*scaling_bess,
            )
        model = add_storage_equivalent_model(model, new_res_load,
                                             time_horizons=scenario_dict["time_horizons"])
        model.objective = pm.Objective(rule=minimize_discharging,
                                       sense=pm.minimize,
                                       doc='Define objective function')
        for iter in range(nr_iterations):
            model_tmp = model.clone()
            np.random.seed(int(time.time()))
            opt = pm.SolverFactory(solver)

            model = solve_model(model=model,
                                solver=scenario_dict["solver"],
                                bess_mode=scenario_dict["bess_mode"],
                                hp_mode=scenario_dict["hp_mode"],
                                ev_mode=scenario_dict["ev_mode"],
                                ev_v2g=scenario_dict.get("ev_v2g",False))
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
