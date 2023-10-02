import pandas as pd
import pyomo.environ as pm
import os
import numpy as np

from storage_equivalent.storage_equivalent_model import add_storage_equivalent_model, \
    minimize_discharging, set_up_base_model
from scenario_input import base_scenario, scenario_input_hps, save_scenario_dict, \
    scenario_input_evs, adjust_timeseries_data, get_new_residual_load, \
    scenario_variation_electric_vehicles_and_heat_pumps, \
    scenario_variation_electric_vehicles, scenario_variation_heat_pumps
from storage_equivalent.heat_pump_model import add_heat_pump_model, model_input_hps
from storage_equivalent.ev_model import add_evs_model, model_input_evs
from plotting import plot_storage_equivalent_germany_stacked

if __name__ == "__main__":
    mode = "linear_penalty" # "implementation_losses", "linear_penalty"
    mode_sc = "ev_hp" # "ev", "hp", "ev_hp"
    scenario = f"sensitivity_{mode}_{mode_sc}"
    plot_results = False
    solver = "gurobi"
    if mode == "implementation_losses":
        varied_parameter = "losses_implementation"
        variations = ["binaries", "penalty", "linear_penalty"]
        weights = 0.0001
    elif mode == "linear_penalty":
        varied_parameter = "weights"
        variations = [0]
        variations += [0.1**(3-i) for i in range(5)]
        implementation = "linear_penalty"
    else:
        raise ValueError("Mode is not known. So far implementation_losses and "
                         "linear_penalty are implemented.")
    # create results directory
    res_dir = os.path.join(f"results/{scenario}")
    os.makedirs(res_dir, exist_ok=True)
    # load scenario values
    scenario_dict = base_scenario()
    if mode_sc == "ev_hp":
        scenario_input = \
            scenario_variation_electric_vehicles_and_heat_pumps()["EV_HP_flex+++"]
    elif mode_sc == "ev":
        scenario_input = \
            scenario_variation_electric_vehicles()["EV_flexible_with_v2g"]
    elif mode_sc == "hp":
        scenario_input = \
            scenario_variation_heat_pumps()["HP_flexible_four_TES"]
    else:
        raise ValueError("Mode for sector coupling is not correct. Should be hp, ev or "
                         "ev_hp.")
    scenario_dict.update(scenario_input)
    scenario_dict["solver"] = solver
    scenario_dict["varied_parameter"] = varied_parameter
    scenario_dict[varied_parameter] = variations
    if scenario_dict["hp_mode"] is not None:
        scenario_dict = scenario_input_hps(scenario_dict=scenario_dict,
                                           mode=scenario_dict["hp_mode"],
                                           use_binaries=True)
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
                                           use_binaries=True)
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
    new_res_load = get_new_residual_load(scenario_dict,
                                         sum_energy_heat=sum_energy_heat,
                                         energy_ev=energy_ev,
                                         ref_charging=ref_charging,
                                         ts_heat_el=ts_heat_el)
    shifted_energy_df = pd.DataFrame(columns=["storage_type",
                                              "energy_stored"])
    shifted_energy_rel_df = pd.DataFrame(columns=["storage_type",
                                                  "energy_stored"])
    for val in scenario_dict[scenario_dict["varied_parameter"]]:
        model = set_up_base_model(scenario_dict=scenario_dict,
                                  new_res_load=new_res_load)
        if mode == "linear_penalty":
            weights = val
        if mode == "implementation_losses":
            implementation = val
        use_binaries = implementation == "binaries"
        use_linear_penalty = implementation == "linear_penalty"

        if scenario_dict["hp_mode"] == "flexible":
            model = add_heat_pump_model(model, p_nom_hp, capacity_tes,
                                        scenario_dict["ts_cop"], ts_heat_demand,
                                        efficiency_static_tes=scenario_dict[
                                            "efficiency_static_tes"],
                                        efficiency_dynamic_tes=scenario_dict[
                                            "efficiency_dynamic_tes"],
                                        use_binaries=use_binaries,
                                        use_linear_penalty=use_linear_penalty,
                                        weight_hp=weights)
        if scenario_dict["ev_mode"] == "flexible":
            add_evs_model(model, flexibility_bands,
                          v2g=scenario_dict["ev_v2g"],
                          efficiency=scenario_dict["ev_charging_efficiency"],
                          discharging_efficiency=scenario_dict[
                              "ev_discharging_efficiency"],
                          use_binaries=use_binaries,
                          use_linear_penalty=use_linear_penalty,
                          weight_ev=weights)
        model = add_storage_equivalent_model(model, new_res_load,
                                             time_horizons=scenario_dict["time_horizons"])
        model.objective = pm.Objective(rule=minimize_discharging,
                                       sense=pm.minimize,
                                       doc='Define objective function')
        opt = pm.SolverFactory(solver)
        # if solver == "gurobi":
        #     opt.options["Method"] = 3
        if implementation == "penalty":
            opt.options["NonConvex"] = 2
        results = opt.solve(model, tee=True)
        charging = pd.Series(model.charging.extract_values()).unstack()
        energy_levels = pd.Series(model.energy_levels.extract_values()).unstack()
        discharging = pd.Series(model.discharging.extract_values()).unstack()
        df_tmp = (discharging.sum(axis=1)).reset_index().rename(
            columns={"index": "storage_type", 0: "energy_stored"})
        df_tmp[scenario_dict["varied_parameter"]] = val
        shifted_energy_df = shifted_energy_df.append(df_tmp, ignore_index=True)
        df_tmp["energy_stored"] = \
            df_tmp["energy_stored"] / scenario_dict["ts_demand"].sum().sum() * 100
        shifted_energy_rel_df = shifted_energy_rel_df.append(df_tmp,
                                                             ignore_index=True)
        # extract charging and discharging of EV and TES
        sc_operations = pd.DataFrame()
        # EV charging and discharging
        if scenario_dict["ev_mode"] is not None:
            ev_charging = pd.Series(
                model.charging_ev.extract_values()).unstack().T
            ev_discharging = pd.Series(
                model.discharging_ev.extract_values()
            ).unstack().T
            if use_binaries:
                ev_charging *= pd.Series(
                    model.y_charge_ev.extract_values()).unstack().T
                ev_discharging *= pd.Series(
                            model.y_discharge_ev.extract_values()
                        ).unstack().T
            sc_operations["EV_charging"] = ev_charging.iloc[:, 0]
            sc_operations["EV_discharging"] = ev_discharging.iloc[:, 0]
        # HP charging and discharging
        if scenario_dict["hp_mode"] is not None:
            charging_tes = pd.Series(model.charging_tes.extract_values())
            discharging_tes = pd.Series(
                model.discharging_tes.extract_values())
            if use_binaries:
                charging_tes *= pd.Series(
                    model.y_charge_tes.extract_values())
                discharging_tes *= pd.Series(
                    model.y_discharge_tes.extract_values())
            sc_operations["TES_charging"] = charging_tes
            sc_operations["TES_discharging"] = discharging_tes
        sc_operations.index = scenario_dict["ts_demand"].index
        if isinstance(val, float):
            val = np.round(val, decimals=3)
        sc_operations.to_csv(f"{res_dir}/sc_operations_{val}.csv")
    shifted_energy_df.to_csv(f"{res_dir}/storage_equivalents.csv")
    shifted_energy_rel_df.to_csv(
        f"{res_dir}/storage_equivalents_relative.csv")
    if plot_results:
        plot_storage_equivalent_germany_stacked(
            shifted_energy_df,
            parameter={
                "relative_weight": "Relative Weight [-]"})
    # remove timeseries as they cannot be saved in json format
    save_scenario_dict(scenario_dict, res_dir)
    print("SUCCESS")
