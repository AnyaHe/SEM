import pandas as pd
import os
import json


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
    return {
        "objective": "minimize_energy",
        "weighting": [1, 7, 364],
        "time_horizons": [24, 7*24, 24*366],
        "time_increment": '1h',
        "ts_vres": vres,
        "ts_demand": demand
    }


def scenario_input_hps(scenario_dict={}, mode="inflexible"):
    """
    Method to add relevant information on modelled hps
    :param scenario_dict: dict
        Dictionary with base scenario values, see ref:base_scenario
    :param mode: str
        Mode of heat pump operation, possible values: "flexible" and "inflexible"
    :return:
        Updated dictionary with additional information on HPs
    """
    heat_demand = pd.read_csv(
        r"data/hp_heat_2011.csv",
        header=None) / 1e3  # GWh
    heat_demand_single_hp = 21.5 * 1e-3  # GWh
    cop = pd.read_csv(
        r'data/COP_2011.csv')["COP 2011"]
    scenario_dict.update({
        "hp_mode": mode,
        "capacity_single_tes": 0.0183 * 1e-3,  # GWh
        "p_nom_single_hp": 0.0067 * 1e-3,  # GW
        "heat_demand_single_hp": heat_demand_single_hp,  # GWh
        "ts_heat_demand_single_hp":
            heat_demand / heat_demand.sum() * heat_demand_single_hp,  # GWh
        "ts_cop": cop
    })
    return scenario_dict


def scenario_input_evs(scenario_dict={}, mode="inflexible"):
    """
    Method to add relevant information on modelled evs
    :param scenario_dict: dict
        Dictionary with base scenario values, see ref:base_scenario
    :param mode: str
        Mode of heat pump operation, possible values: "flexible" and "inflexible"
    :return:
        Updated dictionary with additional information on EVs
    """
    time_increment = scenario_dict.get("time_increment", "1h")
    scenario_dict["time_increment"] = time_increment
    ref_charging = (pd.read_csv(
        r"data/ref_charging.csv", index_col=0, parse_dates=True) / 1e3).resample(
        scenario_dict["time_increment"]).mean() # GW
    if mode == "flexible":
        flex_bands = {}
        for band in ["upper_power", "upper_energy", "lower_energy"]:
            flex_bands[band] = pd.read_csv(f"data/{band}.csv", index_col=0,
                                           parse_dates=True) / 1e3
            if "power" in band:
                flex_bands[band] = flex_bands[band].resample(time_increment).mean()
            elif "energy" in band:
                flex_bands[band] = flex_bands[band].resample(time_increment).max()
        scenario_dict.update({"ts_flex_bands": flex_bands})
    scenario_dict.update({
        "ev_mode": mode,
        "ts_ref_charging": ref_charging,
        "nr_ev_ref": 26880, # from SEST
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

