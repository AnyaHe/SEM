import pandas as pd
import os
import json

from data_preparation.data_preparation import get_heat_pump_timeseries_data


def base_scenario():
    """
    Method defining the default scenario input for three different storage equivalents:
    daily, weekly and seasonal
    :return: dict
        Dictionary with scenario input data
    """
    vres = pd.read_csv(r"data/vres_reference_ego100.csv", index_col=0,
                       parse_dates=True).divide(1000)
    demand = pd.read_csv(r"data/demand_germany_ego100.csv", index_col=0,
                         parse_dates=True)
    timeindex = pd.date_range("2011-01-01", freq="1h", periods=8736)
    return {
        "objective": "minimize_energy",
        "weighting": [1, 7, 364],
        "time_horizons": [24, 7*24, 24*366],
        "time_increment": '1h',
        "ts_vres": vres.loc[timeindex],
        "ts_demand": demand.loc[timeindex]
    }


def scenario_input_hps(scenario_dict={}, mode="inflexible", timesteps=None):
    """
    Method to add relevant information on modelled hps
    :param scenario_dict: dict
        Dictionary with base scenario values, see ref:base_scenario
    :param mode: str
        Mode of heat pump operation, possible values: "flexible" and "inflexible"
    :return:
        Updated dictionary with additional information on HPs
    """
    heat_demand_single_hp = 21.5 * 1e-3  # GWh
    if timesteps is None:
        if hasattr(scenario_dict, "ts_demand"):
            timesteps = scenario_dict["ts_demand"].index
        else:
            timesteps = pd.date_range("1/1/2011 00:00", periods=8760, freq="H")
    scenario_dict.update({
        "hp_weight_air": 0.71,
        "hp_weight_ground": 0.29,
        "hp_weight_floor": 0.6,
        "hp_weight_radiator": 0.4,
        "ts_timesteps": timesteps,
        "hp_dir": r"C:\Users\aheider\Documents\Software\Cost-functions\distribution-grid-expansion-cost-functions\data"
    })
    cop, heat_demand = get_heat_pump_timeseries_data(scenario_dict["hp_dir"], scenario_dict)
    scenario_dict.update({
        "hp_mode": mode,
        "capacity_single_tes": 0.0183 * 1e-3,  # GWh
        "p_nom_single_hp": 0.013 * 1e-3,  # GW
        "heat_demand_single_hp": heat_demand_single_hp,  # GWh
        "ts_heat_demand_single_hp":
            heat_demand / heat_demand.sum() * heat_demand_single_hp,  # GWh
        "ts_cop": cop,
    })
    return scenario_dict


def scenario_input_evs(scenario_dict={}, mode="inflexible",
                       use_cases_flexible=None, extended_flex=False):
    """
    Method to add relevant information on modelled evs
    :param scenario_dict: dict
        Dictionary with base scenario values, see ref:base_scenario
    :param mode: str
        Mode of EV operation, possible values: "flexible" and "inflexible"
    :param use_cases_flexible: list of str
        List of names of use cases that can charge flexibly, default None will result in
        ["home", "work"]
    :param extended_flex: bool
        Indicator whether shifting over standing times is allowed, default: False
    :return:
        Updated dictionary with additional information on EVs
    """
    if use_cases_flexible is None:
        use_cases_flexible = ["home", "work"]
    # set time increment if not already included in the scenario dictionary
    time_increment = scenario_dict.get("time_increment", "1h")
    scenario_dict["time_increment"] = time_increment
    ref_charging = (pd.read_csv(
        r"data/ref_charging_use_case.csv", index_col=0, parse_dates=True) / 1e3).resample(
        scenario_dict["time_increment"]).mean() # GW
    if mode == "flexible":
        flex_bands = {}
        for band in ["upper_power", "upper_energy", "lower_energy"]:
            if not extended_flex:
                flex_bands[band] = pd.read_csv(f"data/{band}_flex+.csv", index_col=0,
                                               parse_dates=True) / 1e3
                nr_ev_ref = 26880 # from SEST
            else:
                flex_bands[band] = pd.read_csv(f"data/{band}_flex++.csv", index_col=0,
                                               parse_dates=True) / 1e3
                nr_ev_ref = 26880 # Todo: adapt
            if "power" in band:
                flex_bands[band] = flex_bands[band].resample(time_increment).mean()
            elif "energy" in band:
                flex_bands[band] = flex_bands[band].resample(time_increment).max()
        scenario_dict.update({"ts_flex_bands": flex_bands})
    scenario_dict.update({
        "ev_mode": mode,
        "use_cases_flexible": use_cases_flexible,
        "ts_ref_charging": ref_charging,
        "nr_ev_ref": nr_ev_ref,
        "extended_flex": extended_flex,
    })
    return scenario_dict


def save_scenario_dict(scenario_dict, res_dir):
    """
    Method to save scenario dict as json. Since dataframes cannot be saved, all
    timeseries (which have to be named as "ts_<...>") are removed first.

    :param scenario_dict: dict
        Dictionary with scenario input, see ref:base_scenario, ref:scenario_input_hps  and
        ref:scenario_input_evs
    :param res_dir: str
        Directory to which scenario dict is saved
    :return: None
    """
    # remove timeseries as they cannot be saved in json format
    keys = [key for key in scenario_dict.keys()]
    for key in keys:
        if "ts_" in key:
            del scenario_dict[key]
    # save scenario input
    with open(
            os.path.join(res_dir, "scenario_dict.json"),
            'w', encoding='utf-8') as f:
        json.dump(scenario_dict, f, ensure_ascii=False, indent=4)

