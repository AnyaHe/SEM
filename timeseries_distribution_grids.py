import pandas as pd
import os
from data_preparation import import_flexibility_bands

base_dir = r"C:\Users\aheider\Documents\Grids"
grid_id = 1811
grid_dir = base_dir + r'\{}'.format(grid_id)
timeseries_dir = os.path.join(base_dir, str(grid_id), "dumb", "timeseries")
topology_dir = os.path.join(base_dir, str(grid_id), "dumb", "topology")


def import_and_save_generation_data():
    # Import data
    generation = pd.read_csv(timeseries_dir + "/generators_active_power.csv", index_col=0,
                             parse_dates=True)
    generators = pd.read_csv(topology_dir + "/generators.csv", index_col=0)
    # Adapt dispatchable generation
    generation_ts = {}
    for gen_type in generators.type.unique():
        if gen_type not in ["wind", "solar"]:
            disp_generation_scaled = pd.read_csv("data/scaled_ts_{}.csv".format(gen_type), index_col=0,
                                                 parse_dates=True)
            disp_generators_installed_capacity = generators.loc[generators.type == gen_type].p_nom.sum()
            disp_generation = disp_generation_scaled * disp_generators_installed_capacity
            generation_ts[gen_type] = disp_generation.asfreq("15min").interpolate()[gen_type]
        else:
            generation_ts[gen_type] = generation[generators.loc[generators.type == gen_type].index].sum(axis=1)
    os.makedirs("data/{}/generation".format(grid_id), exist_ok=True)
    for gen_type, timeseries in generation_ts.items():
        timeseries.to_csv("data/{}/generation/ts_{}".format(grid_id, gen_type))
    return generation_ts


def import_and_save_load_data():
    global load_ts
    # Load data
    load = pd.read_csv(timeseries_dir + "/loads_active_power.csv", index_col=0,
                       parse_dates=True)
    loads = pd.read_csv(topology_dir + "/loads.csv", index_col=0)
    load_ts = {}
    os.makedirs("data/{}/load".format(grid_id), exist_ok=True)
    for load_type in loads.sector.unique():
        load_ts[load_type] = load[loads.loc[loads.sector == load_type].index].sum(axis=1)
        load_ts[load_type].to_csv("data/{}/load/ts_{}".format(grid_id, load_type))
    return load_ts


#generation_ts = import_and_save_generation_data()
#load_ts = import_and_save_load_data()


charging = pd.read_csv(timeseries_dir + "/charging_points_active_power.csv", index_col=0,
                       parse_dates=True)
flexibility_bands = import_flexibility_bands(grid_dir, ["home", "work"])

# extract only inflexible charging
inflexible_charging = charging[charging.columns[
    ~charging.columns.isin(flexibility_bands["upper_power"].columns)]]


print("Success")
