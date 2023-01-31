import pandas as pd
import pyomo.environ as pm
import os

from scenario_input import base_scenario, scenario_input_hps, save_scenario_dict, \
    scenario_input_evs
from storage_equivalent import add_storage_equivalent_model, minimize_energy, \
    determine_storage_durations
from heat_pump_model import add_heat_pump_model, scale_heat_pumps
from ev_model import add_evs_model, scale_electric_vehicles
from plotting import plot_storage_equivalent_germany_stacked


def get_new_residual_load(scenario_dict, share_pv=None, sum_energy_heat=0, energy_ev=0,
                          ref_charging=None, ts_heat_el=None) :
    """
    Method to calculate new residual load for input into storage equivalent model.

    :param scenario_dict:
    :param sum_energy_heat:
    :param energy_ev:
    :return:
    """
    sum_energy = scenario_dict["ts_demand"].sum().sum()
    if ref_charging is None:
        ref_charging = pd.Series(index=scenario_dict["ts_demand"].index, data=0)
    if ts_heat_el is None:
        ts_heat_el = pd.Series(index=scenario_dict["ts_demand"].index, data=0)
    if share_pv is None:
        scaled_ts_reference = scenario_dict["ts_vres"].divide(
            scenario_dict["ts_vres"].sum().sum())
    else:
        scaled_ts_reference = \
            scenario_dict["ts_vres"].divide(scenario_dict["ts_vres"].sum())
        scaled_ts_reference["solar"] = scaled_ts_reference["solar"] * share_pv
        scaled_ts_reference["wind"] = scaled_ts_reference["wind"] * (1 - share_pv)
    vres = scaled_ts_reference * (sum_energy + sum_energy_heat + energy_ev)
    new_res_load = \
        scenario_dict["ts_demand"].sum(axis=1) + ref_charging - vres.sum(axis=1)
    if scenario_dict["hp_mode"] != "flexible":
        new_res_load = new_res_load + \
                       ts_heat_el
    # shift residual load by one timestep to make sure storage units end at 0
    new_res_load = pd.concat([pd.Series(
        index=[new_res_load.index[0]-pd.to_timedelta(scenario_dict["time_increment"])],
        data=0), new_res_load])
    new_res_load.index = \
        new_res_load.index + pd.to_timedelta(scenario_dict["time_increment"])
    return new_res_load


if __name__ == "__main__":
    scenario = "test_to_delete"
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
        nr_hp_mio = 20
        (capacity_tes, p_nom_hp,
         ts_heat_demand, ts_heat_el, sum_energy_heat) = \
            scale_heat_pumps(nr_hp_mio=nr_hp_mio,
                             scenario_dict=scenario_dict)
        scenario_dict["capacity_single_tes"] = \
            tes_relative_size * scenario_dict["capacity_single_tes"]
    else:
        sum_energy_heat = 0
        ts_heat_el = pd.Series(index=scenario_dict["ts_demand"].index, data=0)
    if ev_mode is not None:
        scenario_dict = scenario_input_evs(scenario_dict=scenario_dict, mode=ev_mode,
                                           use_cases_flexible=flexible_ev_use_cases)
        nr_ev_mio = 40
        (reference_charging, flexibility_bands) = scale_electric_vehicles(
            nr_ev_mio, scenario_dict)
        if ev_mode == "flexible":
            use_cases_inflexible = reference_charging.columns[
                ~reference_charging.columns.isin(scenario_dict["use_cases_flexible"])]
            energy_ev = reference_charging[use_cases_inflexible].sum().sum() + \
                        (flexibility_bands["upper_energy"].sum(axis=1)[-1] / 0.9 +
                         flexibility_bands["lower_energy"].sum(axis=1)[
                             -1] / 0.9) / 2
            ref_charging = reference_charging[use_cases_inflexible].sum(axis=1)
        else:
            energy_ev = reference_charging.sum().sum()
            ref_charging = reference_charging.sum(axis=1)
    else:
        energy_ev = 0
        ref_charging = pd.Series(index=scenario_dict["ts_demand"].index, data=0)
    scenario_dict["share_pv"] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
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
        model = pm.ConcreteModel()
        model.timeindex = scenario_dict["ts_timesteps"]
        model.time_set = pm.RangeSet(0, len(new_res_load) - 1)
        model.time_non_zero = model.time_set - [model.time_set.at(1)]
        model.time_increment = pd.to_timedelta(scenario_dict["time_increment"])
        model.times_fixed_soc = pm.Set(initialize=[model.time_set.at(1),
                                                   model.time_set.at(-1)])
        model.weighting = scenario_dict["weighting"]
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
    storage_durations.to_csv(f"{res_dir}/storage_durations.csv")
    # plot results
    plot_storage_equivalent_germany_stacked(shifted_energy_df,
                                                parameter={
                                                    "share_pv": "Share PV [-]"})
    # remove timeseries as they cannot be saved in json format
    save_scenario_dict(scenario_dict, res_dir)
    print("SUCCESS")
