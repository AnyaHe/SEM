import pandas as pd
import numpy as np
import os

BANDS = ["upper_power", "upper_energy", "lower_energy"]


def get_heat_pump_timeseries_data(data_dir, scenario_dict):
    """
    Method to calculate a germany-wide COP time series by building the weighted sum of
    input time-series from when2heat

    (https://data.open-power-system-data.org/when2heat/)

    :param data_dir: str
        directory where when2heat data is stored
    :param scenario_dict: dict
        needs entries for cop, see use :func:`merge_cop_heat_pumps` for required keys
    :return: scaled_cop
    """
    # load input data from when2heat
    hp_path = os.path.join(data_dir, "when2heat.csv")
    profiles_hp = pd.read_csv(hp_path, sep=';', decimal=",", index_col=0,
                              parse_dates=True)
    timesteps = scenario_dict["ts_timesteps"]
    profiles_hp = profiles_hp.loc[timesteps.tz_localize("UTC")]
    # calculate heat_demand and set right timeindex
    heat_demand = sum([
        profiles_hp[f"DE_heat_demand_space_{building}"]for building in ["SFH", "MFH"]
    ])
    heat_demand.index = timesteps
    # calculate cop and set right timeindex
    cop = merge_cop_heat_pumps(profiles_hp, scenario_dict)
    cop.index = timesteps
    return heat_demand, cop


def merge_cop_heat_pumps(profiles_hp, scenario_dict):
    """
    Method to calculate a germany-wide COP time series by building the weighted sum of
    input time-series from when2heat

    (https://data.open-power-system-data.org/when2heat/)
    :param profiles_hp: pd.DataFrame
        input time-series from when2heat
        (https://data.open-power-system-data.org/when2heat/)
    :param scenario_dict: dict
        has to have entries "hp_weight_air", "hp_weight_ground", "hp_weight_floor" and
        "hp_weight_radiator" and "ts_timesteps". Note that hp_weight_air + hp_weight_ground =
        hp_weight_floor + hp_weight_radiator = 1 has to be given.
    :return: pd.Series
        COP time series
    """
    source_names = {"air": "ASHP", "ground": "GSHP"}
    scaled_cop = \
        sum([scenario_dict[f"hp_weight_{source}"] * scenario_dict[f"hp_weight_{sink}"]
             * profiles_hp[f'DE_COP_{source_names[source]}_{sink}']
             for source in ["air", "ground"] for sink in ["floor", "radiator"]])
    return scaled_cop


def import_flexibility_bands(dir, use_cases):
    flexibility_bands = {}

    for band in BANDS:
        band_df = pd.DataFrame()
        for use_case in use_cases:
            flexibility_bands_tmp = \
                pd.read_csv(dir+'/{}_{}.csv'.format(band, use_case),
                            index_col=0, parse_dates=True, dtype=np.float32)
            band_df = pd.concat([band_df, flexibility_bands_tmp],
                                axis=1)
        if band_df.columns.duplicated().any():
            raise ValueError("Charging points with the same name in flexibility bands. "
                             "Please check")
        flexibility_bands[band] = band_df
        # remove numeric problems
        if "upper" in band:
            flexibility_bands[band] = flexibility_bands[band] + 1e-5
        elif "lower" in band:
            flexibility_bands[band] = flexibility_bands[band] - 1e-5
    return flexibility_bands


if __name__ == "__main__":
    scenario_dict = {
        "hp_weight_air": 0.71,
        "hp_weight_ground": 0.29,
        "hp_weight_floor": 0.6,
        "hp_weight_radiator": 0.4,
        "ts_timesteps": pd.date_range("1/1/2011 00:00", periods=8760, freq="H")
    }
    hp_dir = r"C:\Users\aheider\Documents\Software\Cost-functions\distribution-grid-expansion-cost-functions\data"
    scaled_heat_demand, scaled_cop = get_heat_pump_timeseries_data(hp_dir, scenario_dict)
    print("Success")
