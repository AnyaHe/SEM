import os
import pandas as pd
import pyomo.environ as pm
import numpy as np

from scenario_input import base_scenario, scenario_variation_distribution_grids, \
    scenario_input_hps, scenario_input_evs, adjust_timeseries_data, \
    get_new_residual_load, save_scenario_dict, shift_and_extend_ts_by_one_timestep
from storage_equivalent import storage_equivalent_model as se
from storage_equivalent.dg_model import add_dg_model
from storage_equivalent.heat_pump_model import model_input_hps
from storage_equivalent.ev_model import model_input_evs


def get_connection_matrix_of_dgs(dgs):
    """
    Method to get connection matrix of DGs. The model includes a connection to the
    overlying grid from every DG and no interconnections between the single grids.

    :return: connections, flows
    """
    idx_upper_grid = "0"
    all_cells = dgs + [idx_upper_grid]
    connections = pd.DataFrame(index=all_cells,
                               columns=all_cells,
                               data=0)
    flows = pd.DataFrame(columns=["from_bus", "to_bus"],
                         index=[i for i in range(len(dgs))])
    connections.loc[idx_upper_grid, dgs] = 1
    connections.loc[dgs, idx_upper_grid] = -1
    flows.loc[:, "from_bus"] = idx_upper_grid
    flows.loc[:, "to_bus"] = dgs
    return connections, flows


def build_cells_connected(isolated=False):
    res_dgs = pd.read_csv("data/vres_reference_dgs_ego100.csv", index_col=0,
                          parse_dates=True).divide(1000)
    dgs = list(set([idx[1] for idx in res_dgs.columns.str.split("_")]))
    # Todo: connections_df necessary?
    connections_df, flows_df = get_connection_matrix_of_dgs(dgs)
    vres = pd.read_csv(r"data/vres_reference_ego100.csv", index_col=0,
                       parse_dates=True).divide(1000)
    wind_dgs = [col for col in res_dgs.columns if "wind" in col]
    solar_dgs = [col for col in res_dgs.columns if "solar" in col]
    res_dgs["wind_0"] = vres["wind"] - res_dgs[wind_dgs].sum(axis=1)
    res_dgs["solar_0"] = vres["solar"] - res_dgs[solar_dgs].sum(axis=1)
    demand_germany = pd.read_csv("data/demand_germany_ego100.csv", index_col=0,
                                 parse_dates=True)
    demand_dgs = pd.DataFrame(columns=connections_df.index)
    demand_dgs["0"] = demand_germany.sum(axis=1)
    demand_dgs[dgs] = 0
    if isolated:
        connections_df.loc[:, :] = 0
    sum_energy = demand_dgs.sum().sum()
    sum_res = res_dgs.sum().sum()
    scaled_ts_reference = pd.DataFrame(columns=demand_dgs.columns)
    for state in scaled_ts_reference.columns:
        if (state == "0") or ("wind_" + state in wind_dgs):
            wind_ts = res_dgs["wind_" + state]
        else:
            wind_ts = 0
        if (state == "0") or ("solar" + state in solar_dgs):
            solar_ts = res_dgs["solar_" + state]
        else:
            solar_ts = 0
        scaled_ts_reference[state] = \
            (wind_ts + solar_ts) / sum_res * sum_energy
    residual_load = demand_dgs - scaled_ts_reference
    return residual_load


def rename_ev_data(flex_bands, dg_name):
    """
    Temporary method to get data in right format

    Parameters
    ----------
    flex_bands

    Returns
    -------

    """
    for name, band in flex_bands.items():
        cols = band.columns
        flex_bands[name].columns = [f"{dg_name}_{col}" for col in cols]
    return flex_bands


def refactor_hp_data(p_nom, c_tes, demand_th, dg_name):
    """
    Temporary method to get data in right format

    Parameters
    ----------
    p_nom
    c_tes
    demand_th
    dg_name

    Returns
    -------

    """
    p_nom = pd.Series(index=[dg_name], data=p_nom)
    c_tes = pd.Series(index=[dg_name], data=c_tes)
    demand_th_dg = pd.DataFrame()
    demand_th_dg[dg_name] = demand_th
    return p_nom, c_tes, demand_th_dg


def load_hp_data_dg(grid_dir, dg_name):
    """
    Method to load data from distribution grid

    Parameters
    ----------
    grid_dir
    dg_name

    Returns
    -------

    """
    # get cop
    cop = pd.read_csv(os.path.join(grid_dir, "heat_pump", "cop.csv"),
                      index_col=0, parse_dates=True)
    cop_dg = pd.DataFrame()
    cop_dg[dg_name] = cop.mean(axis=1)
    # cop_dg = shift_and_extend_ts_by_one_timestep(
    #     cop_dg, value=cop_dg.iloc[-1])
    # extract nominal power heat pumps
    loads = pd.read_csv(os.path.join(grid_dir, "topology", "loads.csv"),
                        index_col=0)
    heat_pumps = loads.loc[loads.type == "heat_pump"]
    # convert electrical to thermal power
    p_nom_thermal = heat_pumps.p_set.multiply(cop[heat_pumps.index].min())
    p_nom = pd.Series(index=[dg_name], data=p_nom_thermal.sum())
    nr_hps = len(heat_pumps)
    # get capacity of tes
    thermal_energy_storage = pd.read_csv(os.path.join(
        grid_dir, "heat_pump", "thermal_storage_units.csv"
    ), index_col=0)
    c_tes = pd.Series(index=[dg_name], data=thermal_energy_storage.capacity.sum())
    # get thermal demand
    heat_demand = pd.read_csv(
        os.path.join(grid_dir, "heat_pump", "heat_demand.csv"), index_col=0,
        parse_dates=True
    )
    demand_th_dg = pd.DataFrame()
    demand_th_dg[dg_name] = heat_demand.sum(axis=1)
    # demand_th_dg = shift_and_extend_ts_by_one_timestep(
    #     demand_th_dg, value=demand_th_dg.iloc[-1])
    return p_nom, c_tes, demand_th_dg, cop_dg, nr_hps


def load_ev_data_dg(grid_dir, dg_name):
    """
    Method to load data from distribution grid

    Parameters
    ----------
    grid_dir
    dg_name

    Returns
    -------

    """
    # extract
    loads = pd.read_csv(os.path.join(grid_dir, "topology", "loads.csv"),
                        index_col=0)
    charging_points = loads.loc[loads.type == "charging_point"]
    nr_evs = len(charging_points)
    # load time series
    timeseries_cps = pd.read_csv(os.path.join(
        grid_dir, "timeseries", "loads_active_power.csv"
    ), index_col=0, parse_dates=True)[charging_points.index]
    # get flexibility bands
    flexibility_bands = {}
    for band in ["upper_power", "upper_energy", "lower_energy"]:
        flexibility_bands[band] = pd.read_csv(os.path.join(
            grid_dir, "electromobility", f"flexibility_band_{band}.csv"
        ), index_col=0, parse_dates=True)
        # flexibility_bands[band] = shift_and_extend_ts_by_one_timestep(
        #     flexibility_bands[band], value=flexibility_bands[band].iloc[-1])
        # aggregate bands per charging use case
        band_tmp = pd.DataFrame()
        flexible_charging_points = charging_points.loc[flexibility_bands[band].columns]
        flexible_use_cases = list(flexible_charging_points["sector"].unique())
        for use_case in flexible_use_cases:
            band_tmp[f"{dg_name}_{use_case}"] = \
                flexibility_bands[band][flexible_charging_points.loc[
                    flexible_charging_points.sector == use_case].index].sum(axis=1)
        flexibility_bands[band] = band_tmp
    # get reference charging
    reference_charging = pd.DataFrame()
    reference_charging[dg_name] = timeseries_cps[timeseries_cps.columns[
        ~timeseries_cps.columns.isin(flexible_charging_points.index)]].sum(axis=1)
    # get energy
    energy_ev = \
        reference_charging.sum().sum() + \
        (flexibility_bands["upper_energy"].sum(axis=1)[-1] -
         flexibility_bands["upper_energy"].sum(axis=1)[0] +
         flexibility_bands["lower_energy"].sum(axis=1)[-1] -
         flexibility_bands["lower_energy"].sum(axis=1)[0]) / 0.9 / 2
    return flexibility_bands, energy_ev, reference_charging, nr_evs, flexible_use_cases


def load_bess_data_df(grid_dir, dg_name):
    bess = pd.read_csv(os.path.join(grid_dir, "topology", "storage_units.csv"),
                       index_col=0)
    bess_dg = pd.DataFrame(columns=["capacity", "p_nom"], index=[dg_name])
    bess_dg.loc[dg_name] = bess[["capacity", "p_nom"]].sum()
    return bess_dg


def load_inflexible_load_and_vres_dgs(grid_dir, scenario_dict, vres_mode="local"):
    """
    Method to update scenario dict with dg load and vres

    Parameters
    ----------
    grid_dir
    scenario_dict
    vres_mode: "local" or "global"
        determines, which timeseries is used for vres feed-in, the time series of the dg
        ("local" - default) or the national time series ("global")

    Returns
    -------

    """
    timeindex = scenario_dict["ts_demand"].index
    # get time series of inflexible loads
    loads = pd.read_csv(os.path.join(grid_dir, "topology", "loads.csv"),
                        index_col=0)
    conventional_loads = loads.loc[loads.type == "conventional_load"]
    ts_conventional = pd.read_csv(os.path.join(
        grid_dir, "timeseries", "loads_active_power.csv"
    ), index_col=0, parse_dates=True)[conventional_loads.index]
    # get time series of renewables
    generators = pd.read_csv(os.path.join(grid_dir, "topology", "generators.csv"),
                             index_col=0)
    ts_generators = pd.read_csv(os.path.join(
        grid_dir, "timeseries", "generators_active_power.csv"
    ), index_col=0, parse_dates=True)
    ts_vres = pd.DataFrame()
    ts_vres["wind"] = \
        ts_generators[generators.loc[generators.type == "wind"].index].sum(axis=1)
    ts_vres["solar"] = \
        ts_generators[generators.loc[generators.type == "solar"].index].sum(axis=1)
    scenario_dict["ts_demand"] = ts_conventional.loc[timeindex]
    if vres_mode == "local":
        scenario_dict["ts_vres"] = ts_vres.loc[timeindex]
    elif vres_mode == "global":
        pass
    else:
        raise ValueError("Unexpected value for vres_mode. Implemented values are global"
                         " and local.")
    return scenario_dict


if __name__ == "__main__":
    solver = "gurobi"
    varied_parameter = "dg_reinforcement"
    v_res_mode = "global"
    include_bess = True
    grid_id = 1056
    feeder_id = 8
    if include_bess:
        path = r"H:\Grids_SE_storage\{}\feeder\{}".format(grid_id, feeder_id)
        orig_dir = r"H:\Grids_SE"  # r"H:\Grids_SE_storage"
    else:
        path = r"H:\Grids_SE\{}\feeder\{}".format(grid_id, feeder_id)
        orig_dir = r"H:\Grids_SE"
    # information for data on grid constraints
    tail = "grid_power_bands.csv"
    variations = ["full_flex", "minimum_reinforcement"] # [1.0, 0.5, 0.2, 0.1, 0.05, 0.0]
    extract_storage_duration = False
    nr_iterations = 1
    dg_names = ["0"]
    # load scenarios
    scenarios = scenario_variation_distribution_grids()
    for scenario, scenario_input in scenarios.items():
        # try:
        print(f"Start solving scenario {scenario}")
        # create results directory
        if include_bess:
            res_dir = os.path.join(
                f"results/se2_single_dgs_storage_{v_res_mode}_vres/{scenario}")
        else:
            res_dir = os.path.join(
                f"results/se2_single_dgs_{v_res_mode}_vres/{scenario}")
        os.makedirs(res_dir, exist_ok=True)
        # if os.path.isfile(os.path.join(res_dir, "storage_equivalents.csv")):
        #     print(f"Scenario {scenario} already solved. Skipping scenario.")
        #     continue
        # load scenario values
        scenario_dict = base_scenario()
        scenario_dict = load_inflexible_load_and_vres_dgs(
            grid_dir=f"{path}/minimum_reinforcement",
            scenario_dict=scenario_dict,
            vres_mode=v_res_mode
        )
        scenario_dict.update(scenario_input)
        scenario_dict["solver"] = solver
        scenario_dict["varied_parameter"] = varied_parameter
        if not (isinstance(variations[0], pd.Series) or
                isinstance(variations[0], pd.DataFrame)):
            scenario_dict[varied_parameter] = variations

        # Load heat pump and EV data from dgs
        scenario_dict["ts_flex_bands"], energy_ev, scenario_dict["ts_ref_charging"], \
            nr_evs, scenario_dict["use_cases_flexible"] = load_ev_data_dg(
            grid_dir=f"{path}/minimum_reinforcement",
            dg_name=dg_names[0]
        )
        scenario_dict["ev_charging_efficiency"] = 0.9
        scenario_dict["ev_discharging_efficiency"] = 0.9
        p_nom_hp, capacity_tes, scenario_dict["ts_heat_demand"], scenario_dict["ts_cop"], \
            nr_hps = load_hp_data_dg(
            grid_dir=f"{path}/minimum_reinforcement",
            dg_name=dg_names[0]
        )
        scenario_dict["efficiency_static_tes"] = 0.99
        scenario_dict["efficiency_dynamic_tes"] = 0.95
        # Load bess data
        if include_bess:
            bess_data = load_bess_data_df(
                grid_dir=f"{path}/minimum_reinforcement",
                dg_name=dg_names[0]
            )
        else:
            bess_data = pd.DataFrame()
        # shift timeseries
        scenario_dict = adjust_timeseries_data(scenario_dict)
        # initialise result
        shifted_energy_df = pd.DataFrame()
        shifted_energy_rel_df = pd.DataFrame()
        storage_durations = pd.DataFrame()
        energy_consumed = pd.DataFrame()
        shedding_dg = pd.DataFrame(columns=["shed_ev", "shed_hp"])

        new_res_load = get_new_residual_load(
            scenario_dict=scenario_dict,
            sum_energy_heat=scenario_dict["ts_heat_demand"].divide(
                scenario_dict["ts_cop"]).sum().sum(),
            energy_ev=energy_ev,
            ref_charging=scenario_dict["ts_ref_charging"],
            ts_heat_el=scenario_dict["ts_heat_demand"].divide(scenario_dict["ts_cop"]),
        )
        # iterate through reinforcement scenarios
        for val in variations:
            if isinstance(val, pd.Series):
                val_id = val.name
            else:
                val_id = val
            # try:
                # read bands Julian
                res_dir_tmp = f"{path}/{val}/{tail}"
                grid_constraints = pd.read_csv(res_dir_tmp, index_col=0,
                                               parse_dates=True)
                grid_constraints = shift_and_extend_ts_by_one_timestep(
                    grid_constraints, value=grid_constraints.iloc[-1])
                grid_constraints.name = val
                dg_constraints = {
                    "upper_power": pd.DataFrame(
                        columns=dg_names,
                        data=grid_constraints["maximize_grid_power"].values,
                        index=new_res_load.index),
                    "lower_power": pd.DataFrame(
                        columns=dg_names,
                        data=grid_constraints["minimize_grid_power"].values,
                        index=new_res_load.index),
                }
                # initialise base model
                model = se.set_up_base_model(scenario_dict=scenario_dict,
                                             new_res_load=new_res_load)
                flexible_evs = scenario_dict["ev_mode"] == "flexible"
                flexible_hps = scenario_dict["hp_mode"] == "flexible"
                # add model of DGs
                model = add_dg_model(
                    model=model,
                    dg_names=dg_names,
                    grid_powers=dg_constraints,
                    flexible_evs=flexible_evs,
                    flexible_hps=flexible_hps,
                    flexible_bess=include_bess,
                    flex_use_cases=scenario_dict["use_cases_flexible"],
                    flex_bands=scenario_dict["ts_flex_bands"],
                    v2g=scenario_dict["ev_v2g"],
                    efficiency=scenario_dict["ev_charging_efficiency"],
                    discharging_efficiency=scenario_dict["ev_discharging_efficiency"],
                    use_binaries=scenario_dict["use_binaries"],
                    p_nom_hp=p_nom_hp,
                    capacity_tes=capacity_tes,
                    cop=scenario_dict["ts_cop"],
                    heat_demand=scenario_dict["ts_heat_demand"],
                    efficiency_static_tes=scenario_dict["efficiency_static_tes"],
                    efficiency_dynamic_tes=scenario_dict["efficiency_dynamic_tes"],
                    storage_units=bess_data,
                    use_linear_penalty=scenario_dict["use_linear_penalty"],
                    weight_ev=scenario_dict["weights_linear_penalty"],
                    weight_hp=scenario_dict["weights_linear_penalty"]
                )
                model = se.add_storage_equivalent_model(
                    model, new_res_load, time_horizons=scenario_dict["time_horizons"])
                # define objective

                model.objective = pm.Objective(rule=getattr(se, scenario_dict["objective"]),
                                               sense=pm.minimize,
                                               doc='Define objective function')
                for iter_i in range(nr_iterations):

                    print(f"Info: Starting iteration {iter_i} solving final model.")

                    if iter_i == 0:
                        model_tmp = model
                    else:
                        model_tmp = model.clone()
                        model_tmp = se.solve_model(model_tmp, solver,
                                                   scenario_dict["hp_mode"],
                                                   scenario_dict["ev_mode"],
                                                   scenario_dict["ev_v2g"])
                    model_tmp = se.solve_model(
                        model=model_tmp,
                        solver=scenario_dict["solver"],
                        hp_mode=scenario_dict["hp_mode"],
                        ev_mode=scenario_dict["ev_mode"],
                    )
                    # extract results
                    charging = pd.Series(
                        model_tmp.charging.extract_values()).unstack().T.set_index(
                        new_res_load.index)
                    if extract_storage_duration:
                        storage_durations = pd.concat([storage_durations,
                                                       se.determine_storage_durations(
                                                           charging, val_id)])
                    energy_levels = \
                        pd.Series(
                            model_tmp.energy_levels.extract_values()).unstack().T.set_index(
                            new_res_load.index)
                    discharging = pd.Series(model_tmp.discharging.extract_values()).unstack()
                    df_tmp = (discharging.sum(axis=1)).reset_index().rename(
                        columns={"index": "storage_type", 0: "energy_stored"})
                    df_tmp[varied_parameter] = val_id
                    # extract energy values hps
                    if scenario_dict["hp_mode"] == "flexible":
                        sum_energy_heat_tmp = pd.Series(
                            model.charging_hp_el.extract_values()).sum()
                    else:
                        sum_energy_heat_tmp = scenario_dict["ts_heat_demand"].divide(
                            scenario_dict["ts_cop"]).sum().sum()
                    # extract energy values evs
                    if scenario_dict["ev_mode"] == "flexible":
                        ev_operation = \
                            pd.Series(model.charging_ev.extract_values()).unstack().T
                        if scenario_dict["ev_v2g"]:
                            ev_operation -= pd.Series(
                                model.discharging_ev.extract_values()).unstack().T
                        energy_ev_tmp = ev_operation.sum().sum() + \
                                        scenario_dict["ts_ref_charging"].sum().sum()
                    else:
                        energy_ev_tmp = energy_ev
                    # write into data frame to save
                    df_energy_tmp = pd.DataFrame(columns=[val_id],
                                                 index=["energy_ev", "energy_hp",
                                                        varied_parameter],
                                                 data=[energy_ev_tmp, sum_energy_heat_tmp,
                                                       val_id]).T
                    if iter_i == 0:
                        charging.to_csv(f"{res_dir}/charging_{val_id}.csv")
                        energy_levels.to_csv(f"{res_dir}/energy_levels_{val_id}.csv")
                        # save flexible hp operation
                        if scenario_dict["hp_mode"] == "flexible":
                            hp_operation = pd.Series(model_tmp.charging_hp_el.extract_values())
                            hp_operation.index = scenario_dict["ts_demand"].index
                            hp_operation.to_csv(f"{res_dir}/hp_charging_flexible.csv")
                            charging_tes = pd.Series(model_tmp.charging_tes.extract_values())
                            discharging_tes = pd.Series(
                                model_tmp.discharging_tes.extract_values())
                            if scenario_dict["use_binaries"]:
                                charging_tes *= pd.Series(
                                    model_tmp.y_charge_tes.extract_values())
                                discharging_tes *= pd.Series(
                                    model_tmp.y_discharge_tes.extract_values())
                            tes_operation = charging_tes - discharging_tes
                            tes_operation.index = scenario_dict["ts_demand"].index
                            tes_operation.to_csv(f"{res_dir}/tes_operation_flexible.csv")
                            tes_energy = pd.Series(model_tmp.energy_tes.extract_values())
                            tes_energy.index = scenario_dict["ts_demand"].index
                            tes_energy.to_csv(f"{res_dir}/tes_energy_flexible.csv")
                        # save flexible ev operation
                        if scenario_dict["ev_mode"] == "flexible":
                            ev_operation = pd.Series(
                                model_tmp.charging_ev.extract_values()).unstack().T
                            if scenario_dict["ev_v2g"]:
                                if scenario_dict["use_binaries"]:
                                    ev_operation *= pd.Series(
                                        model_tmp.y_charge_ev.extract_values()).unstack().T
                                    ev_operation -= pd.Series(
                                        model_tmp.discharging_ev.extract_values()
                                    ).unstack().T.multiply(
                                        pd.Series(
                                            model_tmp.y_discharge_ev.extract_values()
                                        ).unstack().T
                                    )
                                else:
                                    ev_operation -= pd.Series(
                                        model_tmp.discharging_ev.extract_values()
                                    ).unstack().T
                            ev_operation.index = scenario_dict["ts_demand"].index
                            ev_operation.to_csv(f"{res_dir}/ev_charging_flexible.csv")
                        shifted_energy_df = pd.concat([shifted_energy_df, df_tmp])
                        energy_consumed = pd.concat([energy_consumed, df_energy_tmp])
                        if hasattr(model, "shedding_ev"):
                            shedding_dg.loc[val_id, "shed_ev"] = pd.Series(
                                model_tmp.shedding_ev.extract_values()
                            ).unstack().sum().sum().sum()
                        if hasattr(model, "shedding_hp_el"):
                            shedding_dg.loc[val_id, "shed_hp"] = pd.Series(
                                model_tmp.shedding_hp_el.extract_values()
                            ).unstack().sum().sum()
                    else:
                        assert np.isclose(df_tmp, shifted_energy_df.loc[
                            (shifted_energy_df["nr_hp"] == nr_hps) &
                            (shifted_energy_df["nr_ev"] == nr_evs)],
                                          rtol=1e-04, atol=1e-02).all().all()
            # except Exception as e:
            #     print(f"Something went wrong in scenario {scenario}, iteration {val_id}. "
            #           f"Skipping.")
            #     print(e)

        shifted_energy_df.to_csv(f"{res_dir}/storage_equivalents.csv")
        shifted_energy_rel_df.to_csv(
            f"{res_dir}/storage_equivalents_relative.csv")
        shedding_dg.to_csv(
            f"{res_dir}/shedding_dg.csv")
        if extract_storage_duration:
            storage_durations.to_csv(f"{res_dir}/storage_durations.csv")
        energy_consumed.to_csv(f"{res_dir}/energy_consumed.csv")
        # remove timeseries as they cannot be saved in json format
        save_scenario_dict(scenario_dict, res_dir)
    print("SUCCESS")
