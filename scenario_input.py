import pandas as pd


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
        Dictionary with base scenario values, see ref:scenario_three_storage()
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
