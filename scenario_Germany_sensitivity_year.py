import numpy as np
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import pyomo.environ as pm
import time

from scenario_input import base_scenario, scenario_input_hps, save_scenario_dict, \
    scenario_input_evs, get_new_residual_load, adjust_timeseries_data
from storage_equivalent.storage_equivalent_model import set_up_base_model, add_storage_equivalent_model, \
    minimize_energy, determine_storage_durations
from storage_equivalent.heat_pump_model import add_heat_pump_model, model_input_hps
from storage_equivalent.ev_model import add_evs_model, model_input_evs

if __name__ == "__main__":
    scenario = "Variation_Load_years"
    mode = "demand"
    if mode == "generation":
        years_dict = {
            "ego": [2011],
            "rn": [year for year in range(1980, 2020)]
        }
    elif mode == "demand":
        years_dict = {
            "ego": [2011],
            "entso": [year for year in range(2015, 2023)]
        }
    else:
        raise ValueError("mode not correct")
    extract_storage_duration = True
    plot_results = False
    nr_iterations = 10
    solver = "gurobi"
    ev_mode = None # None, "flexible", "inflexible"
    hp_mode = None  # None, "flexible", "inflexible"
    tes_relative_size = 1 # in share standard
    ev_extended_flex = False
    flexible_ev_use_cases = ["home", "work"]
    if ev_extended_flex:
        flexible_ev_use_cases = ["home", "work", "public"]
    # create results directory
    res_dir = os.path.join(f"results/two_weeks_weight_one/{scenario}")
    os.makedirs(res_dir, exist_ok=True)
    # initialize results
    shifted_energy_df = pd.DataFrame()
    shifted_energy_rel_df = pd.DataFrame()
    storage_durations = pd.DataFrame()
    # load scenario values
    for data_source in years_dict.keys():
        for year in years_dict[data_source]:
            if mode == "generation":
                scenario_dict = base_scenario(vres_data_source=data_source, year=year, share_pv=None)
            elif mode == "demand":
                scenario_dict = base_scenario(demand_data_source=data_source, year=year, reference_demand=None)
            else:
                raise ValueError("Mode not defined")
            scenario_dict["hp_mode"] = hp_mode
            scenario_dict["ev_mode"] = ev_mode
            scenario_dict["solver"] = solver
            if hp_mode is not None:
                scenario_dict = scenario_input_hps(scenario_dict=scenario_dict, mode=hp_mode)
            if ev_mode is not None:
                scenario_dict = scenario_input_evs(scenario_dict=scenario_dict, mode=ev_mode,
                                                   use_cases_flexible=flexible_ev_use_cases,
                                                   extended_flex=ev_extended_flex)
            # shift timeseries
            scenario_dict = adjust_timeseries_data(scenario_dict)
            # adjust model input to hps and evs
            nr_hp_mio, ts_heat_el, sum_energy_heat, capacity_tes, p_nom_hp, ts_heat_demand = \
                model_input_hps(
                    scenario_dict=scenario_dict,
                    hp_mode=hp_mode,
                    nr_hp_mio=20
                )
            nr_ev_mio, flexibility_bands, energy_ev, ref_charging = model_input_evs(
                scenario_dict=scenario_dict,
                ev_mode=ev_mode,
                nr_ev_mio=40
            )
            new_res_load = get_new_residual_load(scenario_dict,
                                                 sum_energy_heat=sum_energy_heat,
                                                 energy_ev=energy_ev,
                                                 ref_charging=ref_charging,
                                                 ts_heat_el=ts_heat_el)
            model = set_up_base_model(scenario_dict=scenario_dict,
                                      new_res_load=new_res_load)
            if hp_mode == "flexible":
                model = add_heat_pump_model(model, p_nom_hp, capacity_tes,
                                            scenario_dict["ts_cop"], ts_heat_demand)
            if ev_mode == "flexible":
                add_evs_model(model, flexibility_bands)
            model = add_storage_equivalent_model(model, new_res_load,
                                                 time_horizons=scenario_dict["time_horizons"])
            model.objective = pm.Objective(rule=minimize_energy,
                                           sense=pm.minimize,
                                           doc='Define objective function')
            for iter in range(nr_iterations):
                model_tmp = model.clone()
                np.random.seed(int(time.time()))
                opt = pm.SolverFactory(solver)
                if solver == "gurobi":
                    opt.options["Seed"] = int(time.time())
                    opt.options["Method"] = 3
                opt = pm.SolverFactory(solver)

                results = opt.solve(model_tmp, tee=True)
                # extract results
                charging = pd.Series(model_tmp.charging.extract_values()).unstack().T.set_index(
                    new_res_load.index)
                if extract_storage_duration:
                    storage_durations = pd.concat([storage_durations,
                                                   determine_storage_durations(
                                                       charging, f"{data_source}_{year}")])
                energy_levels = \
                    pd.Series(model_tmp.energy_levels.extract_values()).unstack().T.set_index(
                        new_res_load.index)
                discharging = pd.Series(model_tmp.discharging.extract_values()).unstack()
                df_tmp = (discharging.sum(axis=1)).reset_index().rename(
                    columns={"index": "storage_type", 0: "energy_stored"})
                df_tmp["data_source"] = data_source
                df_tmp["year"] = year
                if iter == 0:
                    charging.to_csv(f"{res_dir}/charging_{data_source}_{year}.csv")
                    energy_levels.to_csv(f"{res_dir}/energy_levels_{data_source}_{year}.csv")
                    # save flexible hp operation
                    if (hp_mode == "flexible") & (nr_hp_mio == 20.0):
                        hp_operation = pd.Series(model_tmp.charging_hp_el.extract_values())
                        hp_operation.index = scenario_dict["ts_demand"].index
                        hp_operation.to_csv(f"{res_dir}/hp_charging_flexible.csv")
                    # save flexible hp operation
                    if (ev_mode == "flexible") & (nr_ev_mio == 40.0):
                        ev_operation = pd.Series(model_tmp.charging_ev.extract_values()).unstack().T
                        ev_operation.index = scenario_dict["ts_demand"].index
                        ev_operation.to_csv(f"{res_dir}/ev_charging_flexible.csv")
                    shifted_energy_df = pd.concat([shifted_energy_df, df_tmp])
                    df_tmp["energy_stored"] = \
                        df_tmp["energy_stored"] / \
                        (scenario_dict["ts_demand"].sum().sum() + sum_energy_heat + energy_ev) * 100
                    shifted_energy_rel_df = pd.concat([shifted_energy_rel_df, df_tmp])
                else:
                    assert_frame_equal(df_tmp.sort_index(axis=1), shifted_energy_df.loc[
                        (shifted_energy_df["data_source"] == data_source) &
                        (shifted_energy_df["year"] == year)].sort_index(axis=1))
    shifted_energy_df.to_csv(f"{res_dir}/storage_equivalents.csv")
    shifted_energy_rel_df.to_csv(
        f"{res_dir}/storage_equivalents_relative.csv")
    if extract_storage_duration:
        storage_durations.to_csv(f"{res_dir}/storage_durations.csv")
    # remove timeseries as they cannot be saved in json format
    save_scenario_dict(scenario_dict, res_dir)
    print("SUCCESS")
