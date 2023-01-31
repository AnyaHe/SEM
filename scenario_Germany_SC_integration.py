import pandas as pd
import pyomo.environ as pm
import os

import storage_equivalent as se
from heat_pump_model import add_heat_pump_model, scale_heat_pumps
from ev_model import add_evs_model, scale_electric_vehicles
from scenario_input import base_scenario, scenario_input_hps, save_scenario_dict, \
    scenario_input_evs
from plotting import plot_storage_equivalent_germany_stacked


def get_new_residual_load(scenario_dict, sum_energy_heat=0, energy_ev=0,
                          ref_charging=None, ts_heat_el=None) :
    """
    Method to calculate new residual load for input into storage equivalent model.

    :param scenario_dict:
    :param sum_energy_heat:
    :param energy_ev:
    :return:
    """
    def shift_and_extend_ts_by_one_timestep(ts, time_increment="1h", value=0):
        ts = pd.concat([pd.Series(
            index=[ts.index[0] - pd.to_timedelta(
                time_increment)],
            data=value), ts])
        ts.index = \
            ts.index + pd.to_timedelta(time_increment)
        return ts
    sum_energy = scenario_dict["ts_demand"].sum().sum()
    if ref_charging is None:
        ref_charging = pd.Series(index=scenario_dict["ts_demand"].index, data=0)
    if ts_heat_el is None:
        ts_heat_el = pd.Series(index=scenario_dict["ts_demand"].index, data=0)
    scaled_ts_reference = scenario_dict["ts_vres"].divide(
        scenario_dict["ts_vres"].sum().sum())
    vres = scaled_ts_reference * (sum_energy + sum_energy_heat + energy_ev)
    new_res_load = \
        scenario_dict["ts_demand"].sum(axis=1) + ref_charging - vres.sum(axis=1)
    if scenario_dict["hp_mode"] != "flexible":
        new_res_load = new_res_load + \
                       ts_heat_el
    # shift residual load by one timestep to make sure storage units end at 0
    new_res_load = shift_and_extend_ts_by_one_timestep(new_res_load,
                                                       scenario_dict["time_increment"])
    for key in scenario_dict.keys():
        if key.startswith("ts_"):
            if key == "ts_timesteps":
                scenario_dict[key] = new_res_load.index
            elif key == "ts_cop":
                scenario_dict[key] = \
                    shift_and_extend_ts_by_one_timestep(
                        scenario_dict[key], scenario_dict["time_increment"], value=1)
            else:
                scenario_dict[key] = \
                    shift_and_extend_ts_by_one_timestep(
                        scenario_dict[key], scenario_dict["time_increment"], value=0)
    return new_res_load


if __name__ == "__main__":
    scenario = "test_HP_reference"
    extract_storage_duration = True
    solver = "gurobi"
    hp_mode = "inflexible" # None, "flexible", "inflexible"
    ev_mode = None # None, "flexible", "inflexible"
    tes_relative_size = 1 # in share standard
    ev_extended_flex = False
    flexible_ev_use_cases = ["home", "work"]
    if ev_extended_flex:
        flexible_ev_use_cases = ["home", "work", "public"]
    # relative_weighting = 1000
    # weights = [relative_weighting, relative_weighting**2,
    #                        relative_weighting**3]
    # create results directory
    res_dir = os.path.join(f"results/{scenario}")
    os.makedirs(res_dir, exist_ok=True)
    # load scenario values
    scenario_dict = base_scenario()
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
                                           extended_flex=ev_extended_flex)
    # initialise result
    shifted_energy_df = pd.DataFrame(columns=["storage_type",
                                              "energy_stored"])
    shifted_energy_rel_df = pd.DataFrame(columns=["storage_type",
                                                  "energy_stored"])
    storage_durations = pd.DataFrame()
    for i in range(9):
        # add hps if included
        if hp_mode is not None:
            nr_hp_mio = i * 2.5
            (capacity_tes, p_nom_hp,
             ts_heat_demand, ts_heat_el, sum_energy_heat) = \
                scale_heat_pumps(nr_hp_mio=nr_hp_mio,
                                 scenario_dict=scenario_dict)
        else:
            ts_heat_el = pd.Series(index=scenario_dict["ts_demand"].index, data=0)
            sum_energy_heat = 0
            nr_hp_mio = 0
        if ev_mode is not None:
            nr_ev_mio = i * 5
            (reference_charging, flexibility_bands) = scale_electric_vehicles(
                nr_ev_mio, scenario_dict)
            if ev_mode == "flexible":
                use_cases_inflexible = reference_charging.columns[
                    ~reference_charging.columns.isin(scenario_dict["use_cases_flexible"])]
                energy_ev = \
                    reference_charging[use_cases_inflexible].sum().sum() + \
                    (flexibility_bands["upper_energy"].sum(axis=1)[
                         -1] +
                     flexibility_bands["lower_energy"].sum(axis=1)[
                                 -1]) / 0.9 / 2
                ref_charging = reference_charging[use_cases_inflexible].sum(axis=1)
            else:
                energy_ev = reference_charging.sum().sum()
                ref_charging = reference_charging.sum(axis=1)
        else:
            nr_ev_mio = 0
            energy_ev = 0
            ref_charging = pd.Series(index=scenario_dict["ts_demand"].index, data=0)
        # determine new residual load
        new_res_load = get_new_residual_load(
            scenario_dict=scenario_dict,
            sum_energy_heat=sum_energy_heat,
            energy_ev=energy_ev,
            ref_charging=ref_charging,
            ts_heat_el=ts_heat_el)
        # initialise base model
        model = pm.ConcreteModel()
        model.timeindex = scenario_dict["ts_timesteps"]
        model.time_set = pm.RangeSet(0, len(scenario_dict["ts_demand"]) - 1)
        model.time_non_zero = model.time_set - [model.time_set.at(1)]
        model.time_increment = pd.to_timedelta(scenario_dict["time_increment"])
        model.times_fixed_soc = pm.Set(initialize=[model.time_set.at(1),
                                                   model.time_set.at(-1)])
        model.weighting = scenario_dict["weighting"]
        # add heat pump model if flexible
        if hp_mode == "flexible":
            model = add_heat_pump_model(model, p_nom_hp, capacity_tes,
                                        scenario_dict["ts_cop"], ts_heat_demand)
        # add ev model if flexible
        if ev_mode == "flexible":
            add_evs_model(model, flexibility_bands)
        # add storage equivalents
        model = se.add_storage_equivalent_model(
            model, new_res_load, time_horizons=scenario_dict["time_horizons"])
        # define objective
        model.objective = pm.Objective(rule=getattr(se, scenario_dict["objective"]),
                                       sense=pm.minimize,
                                       doc='Define objective function')
        opt = pm.SolverFactory(solver)
        if solver == "gurobi":
            if ev_mode == "flexible":
                opt.options["Method"] = 1
            else:
                opt.options["Method"] = 0
        results = opt.solve(model, tee=True)
        # extract results
        charging = pd.Series(model.charging.extract_values()).unstack().T.set_index(
            new_res_load.index)
        charging.to_csv(f"{res_dir}/charging_{i}.csv")
        if extract_storage_duration:
            storage_durations = pd.concat([storage_durations,
                                           se.determine_storage_durations(charging, i)])
        energy_levels = \
            pd.Series(model.energy_levels.extract_values()).unstack().T.set_index(
                new_res_load.index)
        energy_levels.to_csv(f"{res_dir}/energy_levels_{i}.csv")
        abs_charging = pd.Series(model.abs_charging.extract_values()).unstack()
        # save flexible hp operation
        if (hp_mode == "flexible") & (nr_hp_mio == 20.0):
            hp_operation = pd.Series(model.charging_hp_el.extract_values())
            hp_operation.index = scenario_dict["ts_demand"].index
            hp_operation.to_csv(f"{res_dir}/hp_charging_flexible.csv")
        # save flexible hp operation
        if (ev_mode == "flexible") & (nr_ev_mio == 40.0):
            ev_operation = pd.Series(model.charging_ev.extract_values()).unstack().T
            ev_operation.index = scenario_dict["ts_demand"].index
            ev_operation.to_csv(f"{res_dir}/ev_charging_flexible.csv")
        df_tmp = (abs_charging.sum(axis=1) / 2).reset_index().rename(
            columns={"index": "storage_type", 0: "energy_stored"})
        df_tmp["nr_hp"] = nr_hp_mio
        df_tmp["nr_ev"] = nr_ev_mio
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
