import pandas as pd
import pyomo.environ as pm
import matplotlib.pyplot as plt
import os

from storage_equivalent import add_storage_equivalent_model, minimize_energy, \
    set_up_base_model
from scenario_input import base_scenario, scenario_input_hps, save_scenario_dict, \
    scenario_input_evs, adjust_timeseries_data
from heat_pump_model import add_heat_pump_model, model_input_hps
from ev_model import add_evs_model, model_input_evs
from scenario_Germany_SC_integration import get_new_residual_load
from plotting import plot_storage_equivalent_germany_stacked

if __name__ == "__main__":
    scenario = "test_to_delete"
    plot_results = False
    solver = "gurobi"
    ev_mode = "flexible"  # None, "flexible", "inflexible"
    hp_mode = None  # None, "flexible", "inflexible"
    tes_relative_size = 1  # in share standard
    ev_extended_flex = False
    flexible_ev_use_cases = ["home", "work"]
    if ev_extended_flex:
        flexible_ev_use_cases = ["home", "work", "public"]
    # create results directory
    res_dir = os.path.join(f"results/two_weeks_weight_one/{scenario}")
    os.makedirs(res_dir, exist_ok=True)
    # load scenario values
    scenario_dict = base_scenario()
    scenario_dict["hp_mode"] = hp_mode
    scenario_dict["ev_mode"] = ev_mode
    scenario_dict["solver"] = solver
    if hp_mode is not None:
        scenario_dict = scenario_input_hps(scenario_dict=scenario_dict, mode=hp_mode)
        scenario_dict["capacity_single_tes"] = \
            tes_relative_size * scenario_dict["capacity_single_tes"]
    if ev_mode is not None:
        scenario_input_evs(scenario_dict=scenario_dict, mode=ev_mode,
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
    scenario_dict["relative_weights"] = [1e-1, 1, 1e1, 1e2, 1e3]
    new_res_load = get_new_residual_load(scenario_dict,
                                         sum_energy_heat=sum_energy_heat,
                                         energy_ev=energy_ev,
                                         ref_charging=ref_charging,
                                         ts_heat_el=ts_heat_el)
    shifted_energy_df = pd.DataFrame(columns=["relative_weight", "storage_type",
                                              "energy_stored"])
    shifted_energy_rel_df = pd.DataFrame(columns=["relative_weight", "storage_type",
                                                  "energy_stored"])
    for relative_weighting in scenario_dict["relative_weights"]:
        model = set_up_base_model(scenario_dict=scenario_dict,
                                  new_res_load=new_res_load)
        model.weighting = [relative_weighting + 1e-3,
                           (relative_weighting + 1e-3)**2,
                           (relative_weighting + 1e-3)**3]
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
        opt = pm.SolverFactory(solver)
        if solver == "gurobi":
            opt.options["Method"] = 3
        results = opt.solve(model, tee=True)
        charging = pd.Series(model.charging.extract_values()).unstack()
        energy_levels = pd.Series(model.energy_levels.extract_values()).unstack()
        caps = pd.Series(model.caps_pos.extract_values()) + pd.Series(
            model.caps_neg.extract_values())
        caps_neg = pd.Series(model.caps_neg.extract_values())
        relative_energy_levels = (energy_levels.T + caps_neg).divide(caps)
        discharging = pd.Series(model.discharging.extract_values()).unstack()
        df_tmp = (discharging.sum(axis=1)).reset_index().rename(
            columns={"index": "storage_type", 0: "energy_stored"})
        df_tmp["relative_weight"] = relative_weighting
        shifted_energy_df = shifted_energy_df.append(df_tmp, ignore_index=True)
        df_tmp["energy_stored"] = \
            df_tmp["energy_stored"] / scenario_dict["ts_demand"].sum().sum() * 100
        shifted_energy_rel_df = shifted_energy_rel_df.append(df_tmp,
                                                             ignore_index=True)
    shifted_energy_df.to_csv(f"{res_dir}/storage_equivalents.csv")
    shifted_energy_rel_df.to_csv(
        f"{res_dir}/storage_equivalents_relative.csv")
    if plot_results:
        plot_storage_equivalent_germany_stacked(shifted_energy_df,
                                                parameter={
                                                    "relative_weight": "Relative Weight [-]"})
    # remove timeseries as they cannot be saved in json format
    save_scenario_dict(scenario_dict, res_dir)
    print("SUCCESS")
