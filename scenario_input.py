import pandas as pd
import os
import json

from data_preparation.data_preparation import get_heat_pump_timeseries_data


def scenario_variation_heat_pumps():
    scenarios = {
        "HP_reference": {
            "hp_mode": "inflexible"
        },
        "HP_flexible": {
            "hp_mode": "flexible", "tes_relative_size": 1
        },
        "HP_flexible_double_TES": {
            "hp_mode": "flexible", "tes_relative_size": 2
        },
        "HP_flexible_four_TES": {
            "hp_mode": "flexible", "tes_relative_size": 4
        },
    }
    return scenarios


def scenario_variation_electric_vehicles():
    scenarios = {
        "EV_reference": {
            "ev_mode": "inflexible", "flexible_ev_use_cases": [],
            "ev_extended_flex": False, "ev_v2g": False
        },
        "EV_flexible": {
            "ev_mode": "flexible", "flexible_ev_use_cases": ["home", "work"],
            "ev_extended_flex": False, "ev_v2g": False
        },
        "EV_flexible_with_public": {
            "ev_mode": "flexible", "flexible_ev_use_cases": ["home", "work", "public"],
            "ev_extended_flex": False, "ev_v2g": False
        },
        "EV_flexible_with_shifting": {
            "ev_mode": "flexible", "flexible_ev_use_cases": ["home", "work", "public"],
            "ev_extended_flex": True, "ev_v2g": False
        },
        "EV_flexible_with_v2g": {
            "ev_mode": "flexible", "flexible_ev_use_cases": ["home", "work", "public"],
            "ev_extended_flex": True, "ev_v2g": True
        },
    }
    return scenarios


def scenario_variation_electric_vehicles_and_heat_pumps():
    scenarios = {
        "EV_HP_reference": {
            "ev_mode": "inflexible", "flexible_ev_use_cases": [],
            "ev_extended_flex": False, "ev_v2g": False,
            "hp_mode": "inflexible"
        },
        "EV_HP_flexible": {
            "ev_mode": "flexible", "flexible_ev_use_cases": ["home", "work"],
            "ev_extended_flex": False, "ev_v2g": False,
            "hp_mode": "flexible", "tes_relative_size": 1
        },
        "EV_HP_flex+": {
            "ev_mode": "flexible", "flexible_ev_use_cases": ["home", "work", "public"],
            "ev_extended_flex": False, "ev_v2g": False,
            "hp_mode": "flexible", "tes_relative_size": 2
        },
        "EV_HP_flex++": {
            "ev_mode": "flexible", "flexible_ev_use_cases": ["home", "work", "public"],
            "ev_extended_flex": True, "ev_v2g": False,
            "hp_mode": "flexible", "tes_relative_size": 4
        },
        "EV_HP_flex+++": {
            "ev_mode": "flexible", "flexible_ev_use_cases": ["home", "work", "public"],
            "ev_extended_flex": True, "ev_v2g": True,
            "hp_mode": "flexible", "tes_relative_size": 4
        },
    }
    return scenarios


def scenario_variation_distribution_grids():
    scenarios = {
        "DG_reference": {
            "ev_mode": "inflexible", "flexible_ev_use_cases": [],
            "ev_extended_flex": False, "ev_v2g": False,
            "hp_mode": "inflexible"
        },
        "DG_flexible_base": {
            "ev_mode": "flexible", "flexible_ev_use_cases": ["home", "work"],
            "ev_extended_flex": False, "ev_v2g": False,
            "hp_mode": "flexible", "tes_relative_size": 1
        },
        "DG_flexible_max": {
            "ev_mode": "flexible", "flexible_ev_use_cases": ["home", "work", "public"],
            "ev_extended_flex": True, "ev_v2g": True,
            "hp_mode": "flexible", "tes_relative_size": 4
        },
    }
    return scenarios


def scenario_variation_overcapacities():
    scenarios = scenario_variation_distribution_grids()
    scenarios["DG_base"] = {
            "ev_mode": None, "flexible_ev_use_cases": [],
            "ev_extended_flex": False, "ev_v2g": False,
            "hp_mode": None
        }
    return scenarios


def base_scenario(vres_data_source="ego", demand_data_source="ego", **kwargs):
    """
    Method defining the default scenario input for three different storage equivalents:
    daily, weekly and seasonal
    :param vres_data_source: str
        data source of renewable feed-in data, implemented so far: "ego", "rn". The respective
        time series of the ego-project and renewables ninja have to be added to the data folder.
    :param demand_data_source: str
        data source of demand data, implemented so far: "ego", "entso". The respective
        time series of the ego-project and entso-e have to be added to the data folder.
    :param kwargs: dict
        Optional input parameters
        year: int
            Only used for vres_data_source="rn", if set, a different year of the input data
            is used.
    :return: dict
        Dictionary with scenario input data
    """
    timeindex = pd.date_range("2011-01-01", freq="1h", periods=8736)
    if demand_data_source == "ego":
        demand = pd.read_csv(r"data/demand_germany_ego100.csv", index_col=0,
                             parse_dates=True)
    elif demand_data_source == "entso":
        year = kwargs.get("year")
        demand = pd.DataFrame()
        for month in ["01", "02", "03", "04", "05", "06",
                      "07", "08", "09", "10", "11", "12"]:
            demand_tmp = pd.read_csv(
                f"data/load_entso/{year}_{month}_ActualTotalLoad_6.1.A.csv", sep='\t',
                index_col=0, parse_dates=True)
            hourly_demand_germany = demand_tmp.loc[
                demand_tmp.AreaName == "DE CTY"][
                ["TotalLoadValue"]].resample("1h").mean().divide(1000)
            demand = pd.concat([demand, hourly_demand_germany])
        demand = demand.iloc[:len(timeindex)]
        demand.index = timeindex
        # adjust timeseries to reference demand to make more comparable
        demand_ref = kwargs.get("reference_demand", 499299.467829801)
        if demand_ref is not None:
            demand = demand.divide(demand.sum().sum()).multiply(demand_ref)
    elif demand_data_source == "dg_clustering":
        demand = pd.read_csv(r"data/demand_dgs_clustering.csv", index_col=0,
                             parse_dates=True).divide(1000)
    else:
        raise ValueError("Data source for demand not valid.")
    if vres_data_source == "ego":
        vres = pd.read_csv(r"data/vres_reference_ego100.csv", index_col=0,
                           parse_dates=True).divide(1000)
    elif vres_data_source == "rn":
        wind_rn = pd.read_csv("data/ninja_wind_country_DE_current-merra-2_corrected.csv",
                              index_col=0, parse_dates=True, header=2)
        solar_rn = pd.read_csv("data/ninja_pv_country_DE_merra-2_corrected.csv",
                               index_col=0, parse_dates=True, header=2)
        vres = pd.DataFrame()
        vres["wind"] = wind_rn["national"]
        vres["solar"] = solar_rn["national"]
        year = kwargs.get("year", None)
        if year is not None:
            vres = vres.loc[vres.index.year == year].iloc[:len(timeindex)]
            vres.index = timeindex
        # adjust timeseries to reference share pv to make more comparaböe
        share_pv = kwargs.get("share_pv", 0.2817756687234966)
        if share_pv is not None:
            sum_energy = vres.sum().sum()
            vres_scaled = vres.divide(vres.sum())
            vres["solar"] = vres_scaled["solar"] * share_pv * sum_energy
            vres["wind"] = vres_scaled["wind"] * (1-share_pv) * sum_energy
    elif vres_data_source == "flat_generation":
        vres = pd.DataFrame(index=timeindex, columns=["gen"], data=1)
    elif vres_data_source == "dg_clustering":
        vres = pd.read_csv(r"data/vres_dgs_clustering.csv", index_col=0,
                           parse_dates=True).divide(1000)
    else:
        raise ValueError("Data source for vres not valid.")
    return {
        "objective": "minimize_discharging",
        "weighting": [1.001, 1.001**2, 1.001**3],
        "time_horizons": [24, 14*24, 24*365],
        "time_increment": '1h',
        "ts_vres": vres.loc[timeindex],
        "ts_demand": demand.loc[timeindex],
        "hp_mode": None, "tes_relative_size": 1,
        "ev_mode": None, "flexible_ev_use_cases": [],
        "ev_extended_flex": False, "ev_v2g": False,
        "use_binaries": False,
        "use_linear_penalty": True,
        "weights_linear_penalty": 0.0001
    }


def scenario_input_hps(scenario_dict={}, mode="inflexible", timesteps=None,
                       use_binaries=False):
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
        if "ts_demand" in scenario_dict.keys():
            timesteps = scenario_dict["ts_demand"].index
        else:
            timesteps = pd.date_range("1/1/2011 00:00", periods=8760, freq="H")
    scenario_dict.update({
        "hp_weight_air": 0.71,
        "hp_weight_ground": 0.29,
        "hp_weight_floor": 0.6,
        "hp_weight_radiator": 0.4,
        "ts_timesteps": timesteps,
        "hp_dir": r"U:\Software\Cost-functions\distribution-grid-expansion-cost-functions\data"
    })
    heat_demand, cop = \
        get_heat_pump_timeseries_data(scenario_dict["hp_dir"], scenario_dict)
    scenario_dict.update({
        "hp_mode": mode,
        "capacity_single_tes": 0.0183 * 1e-3,  # GWh
        "efficiency_static_tes": 0.99,
        "efficiency_dynamic_tes": 0.95,
        "hp_use_binaries": use_binaries,
        "p_nom_single_hp": 0.013 * 1e-3,  # GW
        "heat_demand_single_hp": heat_demand_single_hp,  # GWh
        "ts_heat_demand_single_hp":
            (heat_demand / heat_demand.sum() * heat_demand_single_hp).loc[timesteps],  # GWh
        "ts_cop": cop.loc[timesteps],
    })
    return scenario_dict


def scenario_input_evs(scenario_dict={}, mode="inflexible",
                       use_cases_flexible=None, extended_flex=False, timesteps=None,
                       v2g=False, use_binaries=False):
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
    if timesteps is None:
        if "ts_demand" in scenario_dict.keys():
            timesteps = scenario_dict["ts_demand"].index
        else:
            timesteps = pd.date_range("1/1/2011 00:00", periods=8760, freq="H")
    # set time increment if not already included in the scenario dictionary
    time_increment = scenario_dict.get("time_increment", "1h")
    scenario_dict["time_increment"] = time_increment
    ref_charging = (pd.read_csv(
        r"data/ref_charging_use_case_bevs.csv", index_col=0, parse_dates=True) / 1e3).resample(
        scenario_dict["time_increment"]).mean() # GW
    nr_ev_ref = 16574  # only BEVs from SEST
    nr_ev_extended_flex = 13842
    if mode == "flexible":
        flex_bands = {}
        for band in ["upper_power", "upper_energy", "lower_energy"]:
            if not extended_flex:
                flex_bands[band] = pd.read_csv(f"data/{band}_bevs.csv", index_col=0,
                                               parse_dates=True) / 1e3
            else:
                flex_bands[band] = pd.read_csv(f"data/{band}_extended_bevs.csv", index_col=0,
                                               parse_dates=True) / 1e3
                # nr_ev_ref = 26837
            if "power" in band:
                flex_bands[band] = \
                    flex_bands[band].resample(time_increment).mean().loc[timesteps]
            elif "energy" in band:
                flex_bands[band] = \
                    flex_bands[band].resample(time_increment).max().loc[timesteps]
        scenario_dict.update({
            "ts_flex_bands": flex_bands})
    # only set ev_use_binaries to True if V2G is used
    ev_use_binaries = use_binaries and v2g
    scenario_dict.update({
        "ev_mode": mode,
        "use_cases_flexible": use_cases_flexible,
        "ts_ref_charging": ref_charging.loc[timesteps],
        "nr_ev_ref": nr_ev_ref,
        "ev_extended_flex": extended_flex,
        "nr_ev_extended_flex": nr_ev_extended_flex,
        "ev_v2g": v2g,
        "ev_charging_efficiency": 0.9,
        "ev_discharging_efficiency": 0.9,
        "ev_use_binaries": ev_use_binaries
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


def shift_and_extend_ts_by_one_timestep(ts, time_increment="1h", value=0):
    if isinstance(value, pd.Series):
        ts_first = pd.DataFrame(
            columns=[ts.index[0] - pd.to_timedelta(time_increment)],
            index=value.index,
            data=value.values).T
    else:
        ts_first = pd.Series(
            index=[ts.index[0] - pd.to_timedelta(time_increment)],
            data=value)
    ts = pd.concat([ts_first, ts])
    ts.index = \
        ts.index + pd.to_timedelta(time_increment)
    return ts


def adjust_timeseries_data(scenario_dict):
    """
    Method to shift all timeseries by one timestep to make sure storage units end at 0
    :param scenario_dict:
    :return:
    """

    for key in scenario_dict.keys():
        if key.startswith("ts_"):
            if (key == "ts_initial") or (key == "ts_timesteps"):
                pass
            elif key == "ts_cop":
                scenario_dict[key] = \
                    shift_and_extend_ts_by_one_timestep(
                        scenario_dict[key], scenario_dict["time_increment"],
                        value=1)
            elif key == "ts_flex_bands":
                for band in scenario_dict[key].keys():
                    scenario_dict[key][band] = \
                        shift_and_extend_ts_by_one_timestep(
                            scenario_dict[key][band],
                            scenario_dict["time_increment"],
                            value=scenario_dict[key][band].iloc[0]
                        )
            else:
                scenario_dict[key] = \
                    shift_and_extend_ts_by_one_timestep(
                        scenario_dict[key], scenario_dict["time_increment"],
                        value=0)
    scenario_dict["ts_timesteps"] = scenario_dict["ts_demand"].index
    return scenario_dict


def get_new_residual_load(scenario_dict, share_pv=None, sum_energy_heat=0, energy_ev=0,
                          ref_charging=None, ts_heat_el=None, share_gen_to_load=1) :
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
    vres = scaled_ts_reference * (sum_energy + sum_energy_heat + energy_ev) * \
        share_gen_to_load
    new_res_load = \
        scenario_dict["ts_demand"].sum(axis=1) + ref_charging - vres.sum(axis=1)
    if scenario_dict["hp_mode"] != "flexible":
        new_res_load = new_res_load + \
                       ts_heat_el
    return new_res_load
