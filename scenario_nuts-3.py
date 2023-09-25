import os
import pandas as pd
import pyomo.environ as pm
import numpy as np

from scenario_input import base_scenario, scenario_variation_distribution_grids, \
    scenario_input_hps, scenario_input_evs, adjust_timeseries_data, \
    get_new_residual_load, save_scenario_dict
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


if __name__ == "__main__":
    solver = "gurobi"
    use_binaries = True
    varied_parameter = "dg_reinforcement"
    variations = [1.0, 0.5, 0.2, 0.1, 0.05, 0.0]
    extract_storage_duration = False
    nr_iterations = 1
    dg_names = ["0"]
    # load scenarios
    scenarios = scenario_variation_distribution_grids()
    for scenario, scenario_input in scenarios.items():
        # try:
        print(f"Start solving scenario {scenario}")
        # create results directory
        res_dir = os.path.join(f"results/se2_distribution_grids/{scenario}")
        os.makedirs(res_dir, exist_ok=True)
        # if os.path.isfile(os.path.join(res_dir, "storage_equivalents.csv")):
        #     print(f"Scenario {scenario} already solved. Skipping scenario.")
        #     continue
        # load scenario values
        scenario_dict = base_scenario()
        scenario_dict.update(scenario_input)
        scenario_dict["solver"] = solver
        scenario_dict["varied_parameter"] = varied_parameter
        scenario_dict[varied_parameter] = variations
        # load hp data if required
        if scenario_dict["hp_mode"] is not None:
            scenario_dict = scenario_input_hps(scenario_dict=scenario_dict,
                                               mode=scenario_dict["hp_mode"],
                                               use_binaries=use_binaries)
            if scenario_dict["hp_mode"] == "flexible":
                scenario_dict["capacity_single_tes"] = \
                    scenario_dict["tes_relative_size"] * scenario_dict[
                        "capacity_single_tes"]
        # load ev data if required
        if scenario_dict["ev_mode"] is not None:
            scenario_dict = scenario_input_evs(scenario_dict=scenario_dict,
                                               mode=scenario_dict["ev_mode"],
                                               use_cases_flexible=scenario_dict[
                                                   "flexible_ev_use_cases"],
                                               extended_flex=scenario_dict[
                                                   "ev_extended_flex"],
                                               v2g=scenario_dict["ev_v2g"],
                                               use_binaries=use_binaries)
        # shift timeseries
        scenario_dict = adjust_timeseries_data(scenario_dict)
        # adjust model input to hps and evs
        nr_hp_mio, ts_heat_el, sum_energy_heat, capacity_tes, p_nom_hp, ts_heat_demand = \
            model_input_hps(
                scenario_dict=scenario_dict,
                hp_mode=scenario_dict["hp_mode"],
                nr_hp_mio=19.4
            )
        nr_ev_mio, flexibility_bands, energy_ev, ref_charging = model_input_evs(
            scenario_dict=scenario_dict,
            ev_mode=scenario_dict["ev_mode"],
            nr_ev_mio=48.8
        )
        # Fixme: remove when data is read correctly
        flexibility_bands = \
            rename_ev_data(flex_bands=flexibility_bands, dg_name=dg_names[0])
        p_nom_hp, capacity_tes, heat_demand = refactor_hp_data(
            p_nom=p_nom_hp, c_tes=capacity_tes, demand_th=ts_heat_demand,
            dg_name=dg_names[0]
        )
        dg_constraints = {
            "upper_power": pd.DataFrame(columns=dg_names, data=None),
            "lower_power": pd.DataFrame(columns=dg_names, data=None),
        }
        # initialise result
        shifted_energy_df = pd.DataFrame()
        shifted_energy_rel_df = pd.DataFrame()
        storage_durations = pd.DataFrame()
        energy_consumed = pd.DataFrame()
        shedding_dg = pd.DataFrame(columns=["shed_ev", "shed_hp"])
        # iterate through reinforcement scenarios
        for val in scenario_dict[scenario_dict["varied_parameter"]]:
            try:
                new_res_load = get_new_residual_load(
                    scenario_dict=scenario_dict,
                    sum_energy_heat=sum_energy_heat,
                    energy_ev=energy_ev,
                    ref_charging=ref_charging,
                    ts_heat_el=ts_heat_el,
                )

                # initialise base model
                model = se.set_up_base_model(scenario_dict=scenario_dict,
                                             new_res_load=new_res_load)
                flexible_evs = scenario_dict["ev_mode"] == "flexible"
                flexible_hps = scenario_dict["hp_mode"] == "flexible"
                # Fixme: Should work differently in the end
                if isinstance(val, float):
                    if "upper_power" not in flexibility_bands:
                        p_max_ev = 0
                    else:
                        p_max_ev = flexibility_bands["upper_power"].sum(axis=1)
                    dg_constraints["upper_power"][dg_names[0]] = \
                        val * (p_nom_hp[dg_names[0]] + p_max_ev)
                    dg_constraints["lower_power"][dg_names[0]] = \
                        - val * p_max_ev
                # get flexible charging points
                flexible_cps = scenario_dict["use_cases_flexible"]
                if scenario_dict["ev_extended_flex"]:
                    flexible_cps = ["extended_bevs"]
                # add model of DGs
                model = add_dg_model(
                    model=model,
                    dg_names=dg_names,
                    grid_powers=dg_constraints,
                    flexible_evs=flexible_evs,
                    flexible_hps=flexible_hps,
                    flex_use_cases=flexible_cps,
                    flex_bands=flexibility_bands,
                    v2g=scenario_dict["ev_v2g"],
                    efficiency=scenario_dict["ev_charging_efficiency"],
                    discharging_efficiency=scenario_dict["ev_discharging_efficiency"],
                    use_binaries=use_binaries,
                    p_nom_hp=p_nom_hp,
                    capacity_tes=capacity_tes,
                    cop=scenario_dict["ts_cop"],
                    heat_demand=heat_demand,
                    efficiency_static_tes=scenario_dict["efficiency_static_tes"],
                    efficiency_dynamic_tes=scenario_dict["efficiency_dynamic_tes"],
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
                                                           charging, val)])
                    energy_levels = \
                        pd.Series(
                            model_tmp.energy_levels.extract_values()).unstack().T.set_index(
                            new_res_load.index)
                    discharging = pd.Series(model_tmp.discharging.extract_values()).unstack()
                    df_tmp = (discharging.sum(axis=1)).reset_index().rename(
                        columns={"index": "storage_type", 0: "energy_stored"})
                    df_tmp[varied_parameter] = val
                    # extract energy values hps
                    if scenario_dict["hp_mode"] == "flexible":
                        sum_energy_heat_tmp = pd.Series(
                            model.charging_hp_el.extract_values()).sum()
                    else:
                        sum_energy_heat_tmp = sum_energy_heat
                    # extract energy values evs
                    if scenario_dict["ev_mode"] == "flexible":
                        ev_operation = pd.Series(model.charging_ev.extract_values()).unstack().T
                        if scenario_dict["ev_v2g"]:
                            ev_operation -= pd.Series(
                                model.discharging_ev.extract_values()).unstack().T
                        energy_ev_tmp = ev_operation.sum().sum() + ref_charging.sum()
                    else:
                        energy_ev_tmp = energy_ev
                    # write into data frame to save
                    df_energy_tmp = pd.DataFrame(columns=[val],
                                                 index=["energy_ev", "energy_hp",
                                                        varied_parameter],
                                                 data=[energy_ev_tmp, sum_energy_heat_tmp,
                                                       val]).T
                    if iter_i == 0:
                        charging.to_csv(f"{res_dir}/charging_{val}.csv")
                        energy_levels.to_csv(f"{res_dir}/energy_levels_{val}.csv")
                        # save flexible hp operation
                        if (scenario_dict["hp_mode"] == "flexible") & (nr_hp_mio == 20.0):
                            hp_operation = pd.Series(model_tmp.charging_hp_el.extract_values())
                            hp_operation.index = scenario_dict["ts_demand"].index
                            hp_operation.to_csv(f"{res_dir}/hp_charging_flexible.csv")
                            charging_tes = pd.Series(model_tmp.charging_tes.extract_values())
                            discharging_tes = pd.Series(
                                model_tmp.discharging_tes.extract_values())
                            if use_binaries:
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
                        if (scenario_dict["ev_mode"] == "flexible") & (nr_ev_mio == 40.0):
                            ev_operation = pd.Series(
                                model_tmp.charging_ev.extract_values()).unstack().T
                            if scenario_dict["ev_v2g"]:
                                if use_binaries:
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
                        df_tmp["energy_stored"] = \
                            df_tmp["energy_stored"] / \
                            (scenario_dict["ts_demand"].sum().sum() +
                             sum_energy_heat + energy_ev) * 100
                        shifted_energy_rel_df = pd.concat([shifted_energy_rel_df, df_tmp])
                        energy_consumed = pd.concat([energy_consumed, df_energy_tmp])
                        if hasattr(model, "shedding_ev"):
                            shedding_dg.loc[val, "shed_ev"] = pd.Series(
                                model_tmp.shedding_ev.extract_values()
                            ).unstack().sum().sum().sum()
                        if hasattr(model, "shedding_hp_el"):
                            shedding_dg.loc[val, "shed_hp"] = pd.Series(
                                model_tmp.shedding_hp_el.extract_values()
                            ).unstack().sum().sum()
                    else:
                        assert np.isclose(df_tmp, shifted_energy_df.loc[
                            (shifted_energy_df["nr_hp"] == nr_hp_mio) &
                            (shifted_energy_df["nr_ev"] == nr_ev_mio)],
                                          rtol=1e-04, atol=1e-02).all().all()
            except Exception as e:
                print(f"Something went wrong in scenario {scenario}, iteration {val}. "
                      f"Skipping.")
                print(e)

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
