import pandas as pd
import numpy as np
import os
import shutil


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
        profiles_hp[f"DE_heat_demand_space_{building}"] +
        profiles_hp[f"DE_heat_demand_water_{building}"] for building in ["SFH", "MFH"]
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


def determine_shifting_times_ev(flex_bands):
    """
    Method to determine the shifting times between upper and lower energy bands.
    :param flex_bands:
    :return:
    """
    shifting_time = pd.DataFrame(columns=flex_bands["upper_energy"].columns,
                                 index=flex_bands["upper_energy"].index)
    for idx_max, energy in flex_bands["upper_energy"].iterrows():
        idx_min = flex_bands["lower_energy"][flex_bands["lower_energy"] <= energy][::-1].idxmax()
        shifting_time.loc[idx_max] = idx_min - idx_max
    return shifting_time


def extract_relevant_data_from_entso(data_dir, save_dir, years=None):
    if years is None:
        years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
    generation = pd.DataFrame()
    for year in years:
        for month in ["01", "02", "03", "04", "05", "06",
                      "07", "08", "09", "10", "11", "12"]:
            hourly_generation_vres_germany = pd.DataFrame()
            generation_tmp = pd.read_csv(
                f"{data_dir}/{year}_{month}_AggregatedGenerationPerType_16.1.B_C.csv", sep='\t',
                index_col=0, parse_dates=True)
            hourly_generation_vres_germany["solar"] = generation_tmp.loc[
                (generation_tmp.AreaName == "DE CTY") &
                (generation_tmp.ProductionType == "Solar")][
                "ActualGenerationOutput"].resample("1h").mean()
            hourly_generation_vres_germany["wind_onshore"] = generation_tmp.loc[
                (generation_tmp.AreaName == "DE CTY") &
                (generation_tmp.ProductionType == "Wind Onshore")][
                "ActualGenerationOutput"].resample("1h").mean()
            hourly_generation_vres_germany["wind_offshore"] = generation_tmp.loc[
                (generation_tmp.AreaName == "DE CTY") &
                (generation_tmp.ProductionType == "Wind Offshore")][
                "ActualGenerationOutput"].resample("1h").mean()
            generation = pd.concat([generation, hourly_generation_vres_germany])
    generation.to_csv(f"{save_dir}/vres_entso-e.csv")


def copy_simbev_data_bevs(grid_dir, grid_ids):
    """
    Method to copy simbev data of BEVs into new directory to be able to use tracBEV
    to create new reference charging timeseries.

    :param grid_dir: str
        Directory where grids are stored. Each grid has to have a subfolder "simbev_run",
        where the data from a previous simbev run is stored. The cleaned data will be
        copied to a new folder "simbev_bevs". The folder structure will be the same as
        in the original simbev run.
    :param grid_ids: list of int or str
        List of mv grid ids of which the data should be copied.
    :return:
    """
    for grid_id in grid_ids:
        dirs = os.listdir(os.path.join(grid_dir, str(grid_id), "simbev_run"))
        for dir_tmp in dirs:
            dir_tmp_full = os.path.join(grid_dir, str(grid_id), "simbev_run", dir_tmp)
            if os.path.isdir(dir_tmp_full):
                evs = os.listdir(dir_tmp_full)
                for ev in evs:
                    if "bev" in ev:
                        old_dir = os.path.join(dir_tmp_full, ev)
                        new_dir = os.path.join(grid_dir, str(grid_id), "simbev_bevs", dir_tmp)
                        os.makedirs(new_dir, exist_ok=True)
                        new_dir = os.path.join(new_dir, ev)
                        shutil.copy(old_dir, new_dir)
    print("Copied simBEV data for BEVS.")


def create_reference_charging_and_flexibility_timeseries(
        grid_dir, grid_ids, simbev_folder="simbev_bevs", efficiency=0.9,
        timedelta="15min", save_dir=None):
    """
    Method to create reference charging time series for the use cases "home", "work",
    "public" and "hpc".

    :param grid_dir: str
        Directory where grids are stored. Each grid has to have a subfolder named simbev_folder,
        where the data from a simbev run is stored.
    :param grid_ids: list of int or str
        List of mv grid ids of which the charging events should be taken into consideration.
    :param simbev_folder: str
        Name of folder where simbev data is stored.
    :return: pd.DataFrame
        Columns are ["home", "work", "public", "hpc"] and index a annual time index with 15 min
        resolution.
    """
    timeindex = pd.date_range("2010-12-25", end='2011-12-31 23:45:00', freq=timedelta)
    timesteps_per_hour = pd.to_timedelta("1h")/pd.to_timedelta(timedelta)
    reference_charging_use_cases = pd.DataFrame(columns=["home", "work", "public", "hpc"],
                                               index=timeindex, data=0).reset_index()
    upper_power = pd.DataFrame(columns=["home", "work", "public", "hpc"],
                               index=timeindex, data=0).reset_index()
    lower_energy = pd.DataFrame(columns=["home", "work", "public", "hpc"],
                                index=timeindex, data=0).reset_index()
    for grid_id in grid_ids:
        dirs = os.listdir(os.path.join(grid_dir, str(grid_id), simbev_folder))
        for dir_tmp in dirs:
            dir_tmp_full = os.path.join(grid_dir, str(grid_id), simbev_folder, dir_tmp)
            if os.path.isdir(dir_tmp_full):
                evs = os.listdir(dir_tmp_full)
                for ev in evs:
                    charging_processes = pd.read_csv(os.path.join(dir_tmp_full, ev),
                                                     index_col=0)
                    charging_processes = \
                        charging_processes.loc[charging_processes.chargingdemand > 0]
                    # iterate through charging processes
                    for _, charging_process in charging_processes.T.items():
                        # extract charging use case
                        if charging_process.location == "7_charging_hub":
                            use_case = "hpc"
                        elif (charging_process.location == "6_home") & \
                                (charging_process.use_case == "private"):
                            use_case = "home"
                        elif (charging_process.location == "0_work") & \
                                (charging_process.use_case == "private"):
                            use_case = "work"
                        else:
                            use_case = "public"
                        # determine power at grid connection point
                        brutto_charging_capacity = charging_process.netto_charging_capacity / efficiency
                        # get charging times
                        charging_timesteps = \
                            charging_process.chargingdemand / brutto_charging_capacity * timesteps_per_hour
                        charging_timesteps_full = int(charging_timesteps)
                        start = charging_process.park_start
                        end = charging_process.park_end
                        if start+charging_timesteps_full < len(timeindex):
                            # add charging power to respective use case
                            reference_charging_use_cases.loc[start:start+charging_timesteps_full-1, use_case] += \
                                brutto_charging_capacity
                            # handle timestep that is only partly charging
                            charging_timestep_part = charging_timesteps - charging_timesteps_full
                            reference_charging_use_cases.loc[start + charging_timesteps_full, use_case] += \
                                brutto_charging_capacity * charging_timestep_part
                            # maximum power for full standing period
                            if end < len(timeindex):
                                upper_power.loc[start: end, use_case] += brutto_charging_capacity
                                # lower band
                                lower_energy.loc[end - charging_timesteps_full + 1: end, use_case] += \
                                    brutto_charging_capacity
                                if charging_timestep_part != 0.0:
                                    lower_energy.loc[end - charging_timesteps_full, use_case] += (
                                            charging_timestep_part * brutto_charging_capacity
                                    )
                            else:
                                upper_power.loc[start:, use_case] += brutto_charging_capacity
                                # lower band
                                if end - charging_timesteps_full < len(timeindex):
                                    if charging_timestep_part != 0.0:
                                        lower_energy.loc[end - charging_timesteps_full, use_case] += (
                                                charging_timestep_part * brutto_charging_capacity
                                        )
                                    if end - charging_timesteps_full + 1 < len(timeindex):
                                        lower_energy.loc[end - charging_timesteps_full + 1:, use_case] += \
                                            brutto_charging_capacity
                        # if end of charging event is later than considered period, full charging until end of period
                        else:
                            reference_charging_use_cases.loc[start:, use_case] += \
                                brutto_charging_capacity
                            # maximum power for full standing period
                            upper_power.loc[start:, use_case] += brutto_charging_capacity
    timeindex = pd.date_range("2011-01-01", end='2011-12-31 23:45:00', freq=timedelta)
    reference_charging_use_cases = reference_charging_use_cases.set_index("index").loc[timeindex].divide(1e3)
    upper_energy = reference_charging_use_cases.cumsum()/timesteps_per_hour
    lower_energy = lower_energy.set_index("index").loc[timeindex].divide(1e3).cumsum()/timesteps_per_hour
    upper_power = upper_power.set_index("index").loc[timeindex].divide(1e3)
    if save_dir is not None:
        reference_charging_use_cases.to_csv(os.path.join(save_dir, "ref_charging_use_case_bevs.csv"))
        upper_energy.to_csv(os.path.join(save_dir, "upper_energy_bevs.csv"))
        lower_energy.to_csv(os.path.join(save_dir, "lower_energy_bevs.csv"))
        upper_power.to_csv(os.path.join(save_dir, "upper_power_bevs.csv"))
    return reference_charging_use_cases, lower_energy, upper_energy, upper_power


if __name__ == "__main__":
    create_reference_charging_and_flexibility_timeseries(
        r"H:\Grids", [176, 177, 1056, 1690, 1811, 2534],
        save_dir=r"U:\Software\Flexibility-Quantification\data")
    #
    # copy_simbev_data_bevs(r"H:\Grids", [176, 177, 1056, 1690, 1811, 2534])
    # extract_relevant_data_from_entso(r"H:\generation_entso",
    #                                  r"U:\Software\Flexibility-Quantification\data")
    # scenario_dict = {
    #     "hp_weight_air": 0.71,
    #     "hp_weight_ground": 0.29,
    #     "hp_weight_floor": 0.6,
    #     "hp_weight_radiator": 0.4,
    #     "ts_timesteps": pd.date_range("1/1/2011 00:00", periods=8760, freq="H")
    # }
    # hp_dir = r"C:\Users\aheider\Documents\Software\Cost-functions\distribution-grid-expansion-cost-functions\data"
    # scaled_heat_demand, scaled_cop = get_heat_pump_timeseries_data(hp_dir, scenario_dict)
    print("Success")
