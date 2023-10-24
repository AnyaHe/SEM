# has to be run with edisgo environment

import logging
from numpy.random import choice
import numpy as np
import os
import pandas as pd
from scipy.stats import gamma
from shapely import wkt

from edisgo import EDisGo
from edisgo.network.components import Switch
from edisgo.network.results import Results
from edisgo.network.timeseries import TimeSeries

USER_BASEPATH = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = r"data"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def set_up_status_quo_2018(edisgo):
    """
    Returns generator capacities for each generator type in grid for 2018.

    Only capacities of type wind, solar and biomass are expanded using scaling
    factors based on installed capacities per type and state in 2015 and 2018
    from BNetzA. Other generation types are assumed to stay the same.

    Parameters
    -----------
    edisgo : edisgo object

    Returns
    --------
    dict
        Dictionary with installed capacities in MW in 2018.

    """
    # get state the grid is in
    grid_districts_states = pd.read_csv(
        os.path.join(DATA_PATH, "data", "mv_grid_districts_states.csv"),
        index_col=[0]
    )
    state = grid_districts_states.at[edisgo.topology.id, "state"]

    # get scaling factors for state
    scaling_factors = pd.read_csv(
        os.path.join(DATA_PATH, "data",
                     "generator_scaling_factors_per_state_status_quo.csv"),
        index_col=[0]
    )
    scaling_factors = scaling_factors.loc[state, :]

    # calculate target capacities
    target_capacities = edisgo.topology.generators_df.groupby(
        ["type"]).sum().loc[:, "p_nom"]
    target_capacities_scaled = (target_capacities * scaling_factors).dropna()
    # overwrite target capacities for scaled types
    target_capacities.loc[
        target_capacities_scaled.index] = target_capacities_scaled

    return target_capacities


def remove_1m_lines_from_edisgo(edisgo):
    """
    Method to remove 1m lines to reduce size of edisgo object.
    """
    print("Removing 1m lines for grid {}".format(repr(edisgo)))

    # close switches such that lines with connected switches are not removed
    switches = [
        Switch(id=_, topology=edisgo.topology)
        for _ in edisgo.topology.switches_df.index
    ]
    switch_status = {}
    for switch in switches:
        switch_status[switch] = switch.state
        switch.close()
    # get all lines and remove end 1m lines
    lines = edisgo.topology.lines_df.loc[edisgo.topology.lines_df.length == 0.001]
    for name, line in lines.iterrows():
        remove_1m_end_line(edisgo, line)
    # set switches back to original state
    for switch in switches:
        if switch_status[switch] == "open":
            switch.open()
    return edisgo


def remove_1m_end_line(edisgo, line):
    """
    Method that removes end lines and moves components of end bus to neighboring bus.
    If the line is not an end line, the method will skip this line
    """
    # Check for end buses
    if len(edisgo.topology.get_connected_lines_from_bus(line.bus1)) == 1:
        end_bus = "bus1"
        neighbor_bus = "bus0"
    elif len(edisgo.topology.get_connected_lines_from_bus(line.bus0)) == 1:
        end_bus = "bus0"
        neighbor_bus = "bus1"
    else:
        end_bus = None
        neighbor_bus = None
        print("No end bus found. Implement method.")
        return
    # Move connected elements of end bus to the other bus
    connected_elements = edisgo.topology.get_connected_components_from_bus(
        line[end_bus]
    )
    # move elements to neighboring bus
    rename_dict = {line[end_bus]: line[neighbor_bus]}
    for Type, components in connected_elements.items():
        if not components.empty and Type != "lines":
            setattr(
                edisgo.topology,
                Type.lower() + "_df",
                getattr(edisgo.topology, Type.lower() + "_df").replace(rename_dict),
            )
    # remove line
    edisgo.topology.remove_line(line.name)
    print("{} removed.".format(line.name))


def setup_topology_status_quo_ding0(grids_path, grid_id):
    """
    Reinforces the grids to be stable in the status quo.

    Generator capacities (wind, solar, biomass) are updated to better reflect installed
    capacities in 2018 (generator capacities in ding0 grids are from 2015). Afterwards,
    grids are reinforced.

    Parameters
    -----------
    grids_path : str
        Path to directory containing original ding0 grids.
    grid_id : int or str
        ID of ding0 grid to set up.

    Returns
    ---------
    EDisGo object
        EDisGo object with status quo grid topology and worst case time series.

    """

    dingo_grid_path = os.path.join(grids_path, str(grid_id))
    edisgo_obj = EDisGo(
        ding0_grid=dingo_grid_path,
        config_path=os.path.join(DATA_PATH, "data_preparation", "data", "config_data")
    )

    # install 2018 generator capacities
    logger.debug("Installed capacities 2015: {}".format(
        edisgo_obj.topology.generators_df.loc[
            :, ["p_nom", "type"]].groupby(["type"]).sum().loc[
            :, "p_nom"]
    ))
    target_capacities = set_up_status_quo_2018(
        edisgo_obj
    )
    edisgo_obj.import_generators(
        generator_scenario="nep2035",
        p_target=target_capacities,
        remove_decommissioned=False,
        update_existing=False
    )
    logger.debug("Installed capacities 2018: {}".format(
        edisgo_obj.topology.generators_df.loc[
            :, ["p_nom", "type"]].groupby(["type"]).sum().loc[
            :, "p_nom"]
    ))

    # remove 1 meter lines
    edisgo_obj = remove_1m_lines_from_edisgo(edisgo_obj)

    # set up worst case time series
    edisgo_obj.set_time_series_worst_case_analysis()

    # Reinforce ding0 grid to obtain a stable status quo grid
    logger.debug("Conduct grid reinforcement to obtain stable status quo grid.")
    edisgo_obj.reinforce()

    # Clear results
    edisgo_obj.results = Results(edisgo_obj)

    # Add timeseries
    edisgo_obj.timeseries = TimeSeries()
    edisgo_obj.timeseries.timeindex = \
        pd.date_range("1/1/2011 00:00", periods=8760, freq="H")
    edisgo_obj.set_time_series_active_power_predefined(
        fluctuating_generators_ts="oedb",
        dispatchable_generators_ts="full_capacity",
        conventional_loads_ts="demandlib",
    )
    edisgo_obj.set_time_series_reactive_power_control()

    # Check integrity of edisgo object
    edisgo_obj.check_integrity()

    # Reinforce with timeseries
    edisgo_obj.reinforce(timesteps_pfa='reduced_analysis')
    # # Clear results
    edisgo_obj.results = Results(edisgo_obj)

    logger.debug("Status quo grids ready.")

    return edisgo_obj


def choose_potential_charging_points(use_case, nr_cps, potential_cps, rnd_gen):
    charging_points_tmp = potential_cps.loc[
        potential_cps.use_case == use_case]
    # get relative weights of charging points
    if charging_points_tmp.user_centric_weight.sum() == 0:
        # deal with hpc, there weight is 0 --> equal weighting required
        charging_points_tmp.user_centric_weight += 1.0
    weights = charging_points_tmp.user_centric_weight.divide(
        charging_points_tmp.user_centric_weight.sum())
    # randomly choose charging location
    draw = rnd_gen.choice(charging_points_tmp.index, nr_cps, p=weights)
    return charging_points_tmp.loc[draw]


def scenario_input_data():
    """
    Returns dictionary with relevant scenario data.

    Returns
    --------
    dict
        Dictionary with relevant scenario data. Should be changed manually to adapt
        scenario.
    """
    return {
        "PV_installed_capacities": 0.008, # MW
        "PV_gamma": (190.88895608306424, -46.093268849468814, 0.2853110801380861),
        "BESS_relative_capacity": 1.0, # kWh/kW_PV
        "BESS_relative_power": 0.6, # kW/kWh
        "BESS_charging_efficiency": 1.0,
        "BESS_discharging_efficiency": 1.0,
        "HP_installed_capacities": 0.002, # MW
        "HP_share_air_source": 0.8,
        "Annual_heat_demand_house": 140*45*1e-3, # area in m^2 times heat demand in
                                                 # kWh/m^2 p.a. converted into MWh p.a
        "HP_size_TES": 0.016, # MWh
        "HP_gamma": (5.433654509183175, -0.770275842540082, 2.543186523886985),
    }


def distribute_residential_hps(edisgo_obj, buses, hp_profiles, scenario_dict,
                               seed, penetration=1.0, sizing_mode="gamma"):
    """
    Distributes decentralised residential hps in existing grid. HP is randomly added to
    the grid topology, if no HP already exists at the bus.

    Parameters
    -----------
    edisgo_obj: EDisGo object
        Original edisgo object to which distributed energy resources should be added.
    buses: pandas.Series
        List of buses relevant for adding the generators.
    hp_profiles: pandas.DataFrame
        DataFrame containing scaled heat demand, cop_ground and cop_air
    scenario_dict : dict
        Includes information on installed capacity and share ground and air sourced hps.
    seed : int
        Seed for random distribution.
    penetration: float
        Level of penetration of distributed resources. Has to lie between 0 and 1.
    sizing_mode: str
        Determines how heat pump sizes are determined. So far implemented:
        "gamma": extracts gamma distribution from historical values and adds size from
            distribution
        "constant": all heat pumps are equally sized, the value is taken from scenario
            dict

    Returns
    ---------
    list of str
        Names of added distributed resources.
    """
    def _scale_heat_demand_and_storage(capacities_hp, profiles_hp,
                                       blocked_hours=6, hours_storage=2):
        # get annual consumption
        peak_thermal_loads_hp = \
            (24-blocked_hours)/24 * capacities_hp
        # annual consumption of profile is 1 MWh, therefore new annual consumption can
        # be calcalated directly
        annual_consumptions = peak_thermal_loads_hp / max(profiles_hp["demand"])
        # get storage size
        demand_consecutive = pd.DataFrame(index=profiles_hp["demand"].index)
        for i in range(hours_storage):
            demand_consecutive[i] = pd.concat([profiles_hp["demand"][i:],
                                               profiles_hp["demand"][:i]]).values
        max_storage = demand_consecutive.sum(axis=1).max()
        storage_sizes = max_storage * peak_thermal_loads_hp / max(profiles_hp["demand"])
        return annual_consumptions, storage_sizes
    # Check if heat pump already exists at the buses
    buses = buses[~buses.isin(edisgo_obj.topology.loads_df.loc[
                                  edisgo_obj.topology.loads_df.type == "heat_pump"
                              ].bus)]
    # Divide buses into ones for air-source and ground-source heat pumps
    buses_air_source = buses.sample(frac=scenario_dict["HP_share_air_source"],
                                    random_state=seed)
    buses_ground_source = buses[~buses.isin(buses_air_source)]
    # Randomly choose buses
    buses_air_source = \
        buses_air_source.sample(frac=penetration, random_state=seed)
    buses_ground_source = \
        buses_ground_source.sample(frac=penetration, random_state=seed)
    # Add heat_pumps to topology
    hps = []
    storage_sizes = np.array([])
    for source in ["air", "ground"]:
        buses_source = locals()[f"buses_{source}_source"].values
        # Randomly choose size of HP, assumed maximum of 35 kW
        if sizing_mode == "gamma":
            fit_gamma = scenario_dict["HP_gamma"]
            distribution = gamma(fit_gamma[0], fit_gamma[1], fit_gamma[2])
            np.random.seed(seed=seed)
            installed_capacities_source = \
                distribution.rvs(size=len(buses_source))/1e3
            installed_capacities_source[installed_capacities_source > 0.035] = 0.035
            installed_capacities_source[installed_capacities_source < 0.0] = 0.0
            annual_consumptions_source, storage_sizes_source = _scale_heat_demand_and_storage(
                capacities_hp=installed_capacities_source,
                profiles_hp=hp_profiles,
            )
            # convert thermal installed capacity into electric capacity
            installed_capacities_source = \
                installed_capacities_source/min(hp_profiles[f"cop_{source}"])
        elif sizing_mode == "constant":
            installed_capacities_source = \
                np.array([scenario_dict["HP_installed_capacities"]]*len(buses_source))
            annual_consumptions_source = \
                np.array([scenario_dict["Annual_heat_demand_house"]]*len(buses_source))
            storage_sizes_source = np.array([scenario_dict["HP_size_TES"]]*len(buses_source))
        else:
            raise ValueError("The sizing mode you inserted is not defined yet. Please "
                             "adapt to ´gamma´ or ´constant´ or define a new method.")
        hps += edisgo_obj.topology.add_loads(
            buses=buses_source,
            p_sets=installed_capacities_source,
            types="heat_pump",
            sectors=f"{source}",
            annual_consumptions=annual_consumptions_source
        )
        storage_sizes = np.concatenate([storage_sizes, storage_sizes_source])
    storage_df = pd.DataFrame(
        columns=["capacity", "efficiency", "state_of_charge_initial"],
        index=hps
    )
    # set heat pump properties
    storage_df["capacity"] = storage_sizes
    storage_df["efficiency"] = 1.0
    storage_df["state_of_charge_initial"] = storage_df["capacity"] / 2
    edisgo_obj.heat_pump.thermal_storage_units_df = storage_df
    # scale time series by annual consumption
    edisgo_obj.heat_pump.heat_demand_df = edisgo_obj.topology.loads_df.loc[hps].apply(
        lambda x: hp_profiles["demand"] * x.annual_consumption,
        axis=1,
    ).T
    edisgo_obj.heat_pump.cop_df = edisgo_obj.topology.loads_df.loc[hps].apply(
        lambda x: hp_profiles[f"cop_{x.sector}"],
        axis=1,
    ).T
    return hps


def get_hp_profiles(timesteps):
    """
    Extracts heat demand and cops for ground and air sourced heat pumps. In order to use
    this method, the csv-file from when2heat
    (https://data.open-power-system-data.org/when2heat/) has to be stored in the folder
    distribution-grid-expansion-cost-functions/data. The heat demand is scaled to 1TWh
    per year in the original data and converted to 1MWh per year in this method.

    Parameters
    -----------
    timesteps: pandas DateRange
        Relevant timesteps for which the information should be extracted.

    Returns
    ---------
    pandas DataFrame
        DataFrame containing the columns "demand", "cop_ground", "cop_air" and
        index timesteps
    """
    # Get HP profiles
    hp_path = os.path.join(DATA_PATH, "data", "when2heat.csv")
    profiles_hp = pd.read_csv(hp_path, sep=';', decimal=",", index_col=0,
                              parse_dates=True)
    # profiles_hp.index = profiles_hp['utc_timestamp'].str.split("+")[0]
    profiles_hp = profiles_hp.loc[timesteps.tz_localize("UTC")]
    # Todo: utc timestamp or cet_cest_timestamp?
    #.loc[profiles_hp['cet_cest_timestamp'].str.contains('2011')]
    # get profiles for ground and air sourced hps
    profiles_hp_short = pd.DataFrame()
    profiles_hp_short["demand"] = profiles_hp["DE_heat_profile_space_SFH"].divide(1e6)
    profiles_hp_short["cop_ground"] = profiles_hp['DE_COP_GSHP_floor']
    profiles_hp_short["cop_air"] = profiles_hp['DE_COP_ASHP_floor']
    profiles_hp_short.index = timesteps
    return profiles_hp_short


def reference_operation(profiles_hp):
    """
    Determines the reference operation of air and ground sourced heat pumps. This is
    assumed to be directly providing the heat demand, so by dividing the heat demand
    with the respective cop.

    Parameters
    -----------
    profiles_hp: pandas DataFrame
        DataFrame obtained by the function "get_hp_profiles"

    Returns
    ---------
        pandas DataFrame
        DataFrame containing timeseries for the columns "air_source", "ground_source"
    """
    reference_strategy = pd.DataFrame()
    for source in ["air", "ground"]:
        reference_strategy[source] = \
            profiles_hp["demand"]/profiles_hp[f"cop_{source}"]
    return reference_strategy


if __name__ == "__main__":
    save_dir = r"H:\Grids_SE"
    seed = 2023
    rnd = np.random.default_rng(seed=2023)
    # ++ get base grid ++
    grid_id = 1056
    grids_orig_path = \
        r"C:\Users\aheider\Documents\Grids\ding0_elia_grids"
    edisgo_obj = \
        setup_topology_status_quo_ding0(grids_path=grids_orig_path, grid_id=grid_id)
    # ++ add EVs ++
    # get potential charging sites
    ev_dir = r"C:\Users\aheider\Documents\Grids\{}\dumb\electromobility".format(grid_id)
    potential_charging_points = pd.read_csv(
        os.path.join(ev_dir, "grid_connections.csv"), index_col=0)
    # get added cps from previous project (to get p_set)
    cp_dir = r"C:\Users\aheider\Documents\Grids\{}\dumb\topology".format(grid_id)
    integrated_charging_parks = pd.read_csv(
        os.path.join(cp_dir, "charging_points.csv"), index_col=0
    )
    # get charging time series and flexibility bands of previous project
    ts_dir = r"C:\Users\aheider\Documents\Grids\{}\dumb\timeseries".format(grid_id)
    ts_charging_parks = pd.read_csv(
        os.path.join(ts_dir, "charging_points_active_power.csv"), index_col=0,
        parse_dates=True
    )
    # how many EVs are to be integrated?
    scaling_factor = 4.88 # 4.88
    cp_ids = []
    for use_case in integrated_charging_parks.use_case.unique():
        cps_tmp = integrated_charging_parks.loc[
            integrated_charging_parks.use_case == use_case]
        nr_cps = int(np.round(len(cps_tmp)*scaling_factor))
        locations_cps = choose_potential_charging_points(
            use_case=use_case,
            nr_cps=nr_cps,
            potential_cps=potential_charging_points,
            rnd_gen=rnd
        )
        cps_tmp_dupl = pd.concat([cps_tmp]*int(np.ceil(scaling_factor)))
        cps_idx = choice(len(cps_tmp_dupl), nr_cps, replace=False)
        cps_tmp_int = cps_tmp_dupl.iloc[cps_idx]
        cps_tmp_int["geom"] = locations_cps["geometry"].apply(wkt.loads).values
        cp_ids_use_case = [
            edisgo_obj.integrate_component_based_on_geolocation(
                comp_type="charging_point",
                geolocation=cp.geom,
                p_set=max(cp.p_nom, ts_charging_parks[cp_id].max()),
                sector=use_case,
                add_ts=True,
                ts_active_power=ts_charging_parks[cp_id],
                ts_reactive_power=pd.Series(index=ts_charging_parks.index, data=0.0)
            ) for cp_id, cp in cps_tmp_int.iterrows()]
        cp_ids.append(cp_ids_use_case)
    # ++ add HPs ++
    penetration = 1.0
    scenario_dict = scenario_input_data()
    hp_profiles = get_hp_profiles(edisgo_obj.timeseries.timeindex)
    residential_loads = edisgo_obj.topology.loads_df.loc[
        edisgo_obj.topology.loads_df.sector == "residential"]
    heat_pumps = distribute_residential_hps(
        edisgo_obj=edisgo_obj,
        buses=residential_loads.bus,
        scenario_dict=scenario_dict,
        seed=seed,
        penetration=penetration,
        hp_profiles=hp_profiles,
        sizing_mode="gamma"
    )
    logger.info(f"Number of loads after adding residential HPs: "
                f"{len(edisgo_obj.topology.loads_df)}.")
    reference_operation = reference_operation(hp_profiles)
    edisgo_obj.timeseries.predefined_conventional_loads_by_sector(
        edisgo_object=edisgo_obj,
        ts_loads=reference_operation,
        load_names=heat_pumps
    )
    ts_reactive = edisgo_obj.timeseries.loads_active_power[heat_pumps]
    ts_reactive.loc[:, :] = 0.0
    edisgo_obj.timeseries.add_component_time_series("loads_reactive_power",
                                                    ts_reactive)
    # ++ add VRES ++
    # todo: how? with egon scaling factors? Only PV in MV/LV and others in overlying grid?
    edisgo_obj.check_integrity()
    edisgo_obj.save(os.path.join(save_dir, str(grid_id)))
    print("Success")
