from copy import deepcopy
import logging
import os
import pandas as pd
import time

from edisgo.edisgo import import_edisgo_from_files
from edisgo.opf.lopf import import_flexibility_bands, \
    prepare_time_invariant_parameters, BANDS, update_model, setup_model, optimize
from edisgo.tools.tools import convert_impedances_to_mv

from data_preparation.dg_preparation import get_downstream_nodes_matrix_iterative, \
    get_hp_profiles, reference_operation

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger = logging.getLogger("gurobi")
logger.setLevel(logging.ERROR)


def get_exec_time(start_time):
    """Computes execution time in h:m:s format or ms if <2seconds"""
    exec_time = time.perf_counter() - start_time
    if exec_time > 2:
        exec_time = time.gmtime(exec_time)
        exec_time = time.strftime("%Hh:%Mm:%Ss", exec_time)
    else:
        exec_time = str(int(exec_time * 100)) + "ms"
    return exec_time


def optimised_operation(
    edisgo_orig, ev_base_dir, cps=None, cp_mapping=None, hps=None, bess=None,
        scenario="test"
):
    start_time = time.perf_counter()
    objective = "minimize_loading"
    timesteps_per_iteration = 24
    iterations_per_era = 7
    overlap_iterations = 6

    solver = "gurobi"
    kwargs = {}  # {'v_min':0.91, 'v_max':1.09, 'thermal_limit':0.9}
    config_dict = {
        "objective": objective,
        "solver": solver,
        "timesteps_per_iteration": timesteps_per_iteration,
        "iterations_per_era": iterations_per_era,
        'overlap_iterations': overlap_iterations,
    }
    config_dict.update(kwargs)
    config = pd.Series(config_dict)
    grid_id = edisgo_orig.topology.id
    execution_times = pd.DataFrame(
        columns=["preparation time", "time-invariant parameter",
                 "build time", "update time", "solving time", "total"],
        index=[scenario]
    )

    ev_dir = os.path.join(ev_base_dir, str(grid_id))

    edisgo_obj = convert_impedances_to_mv(deepcopy(edisgo_orig))

    print("Converted impedances to mv.")

    downstream_nodes_matrix = \
        get_downstream_nodes_matrix_iterative(edisgo_obj.topology)

    print("Downstream node matrix imported.")

    if cps is not None:
        optimize_ev = True
        flexibility_bands = import_flexibility_bands(ev_dir, ["home", "work"])

        # todo: update bands with mapping
        print("Flexibility bands imported.")

        # extract data for feeder
        for band in BANDS:
            cp_mapping_tmp = cp_mapping.loc[cp_mapping.name_orig.isin(
                flexibility_bands[band].columns)]
            flexibility_bands[band] = \
                flexibility_bands[band][cp_mapping_tmp.name_orig]
            flexibility_bands[band].columns = cp_mapping_tmp.name_new
            if "power" in band:
                flexibility_bands[band] = \
                    flexibility_bands[band].resample("1h").mean().loc[
                        edisgo_obj.timeseries.timeindex]
            elif "energy" in band:
                flexibility_bands[band] = \
                    flexibility_bands[band].resample("1h").max().loc[
                        edisgo_obj.timeseries.timeindex]
            else:
                raise ValueError("Unknown type of band")
    else:
        optimize_ev = False
        flexibility_bands = {}

    if hps is not None:
        optimize_hp = True
    else:
        optimize_hp = False

    if bess is not None:
        optimize_bess = True
    else:
        optimize_bess = False

    execution_times.loc[scenario, "preparation time"] = get_exec_time(start_time)
    t1 = time.perf_counter()

    # Create dict with time invariant parameters
    parameters = prepare_time_invariant_parameters(
        edisgo_obj,
        downstream_nodes_matrix,
        pu=False,
        optimize_storage=optimize_bess,
        optimize_emob=optimize_ev,
        optimize_hp=optimize_hp,
        ev_flex_bands=flexibility_bands,
        voltage_limits=True
    )
    execution_times.loc[scenario, "time-invariant parameter"] = get_exec_time(t1)
    t2 = time.perf_counter()
    print("Time-invariant parameters extracted.")

    charging_ev = pd.DataFrame()
    charging_hp = pd.DataFrame()
    charging_bess = pd.DataFrame()


    # Todo: handle loading of failed runs
    charging_starts = {"ev": None, "hp": None, "tes": None}
    energy_level_start = {"ev": None, "tes": None}
    start_iter = 0

    for iteration in range(
        start_iter,
        int(len(edisgo_obj.timeseries.timeindex) / timesteps_per_iteration),
    ):

        print("Starting optimisation for iteration {}.".format(iteration))
        # set values for "normal" iterations
        if iteration % iterations_per_era != iterations_per_era - 1:
            timesteps = edisgo_obj.timeseries.timeindex[
                        iteration * timesteps_per_iteration:
                        (iteration + 1) * timesteps_per_iteration + overlap_iterations]
            energy_level_end = {"ev": None, "tes": None}
        # set values for final iteration in era
        else:
            timesteps = edisgo_obj.timeseries.timeindex[
                        iteration * timesteps_per_iteration:
                        (iteration + 1) * timesteps_per_iteration]
            energy_level_end = {"ev": True, "tes": True}
        try:
            model = update_model(
                model,
                timesteps,
                parameters,
                charging_starts=charging_starts,
                energy_level_starts=energy_level_start,
                energy_level_ends=energy_level_end,
                **kwargs
            )
            execution_times.loc[scenario, "update time"] = get_exec_time(t2)
            t2 = time.perf_counter()
        except NameError:
            model = setup_model(
                parameters,
                timesteps,
                objective=objective,
                charging_starts=charging_starts,
                energy_level_starts=energy_level_start,
                energy_level_ends=energy_level_end,
                **kwargs
            )
            execution_times.loc[scenario, "build time"] = get_exec_time(t2)
            t2 = time.perf_counter()

        print("Set up model for iteration {}.".format(iteration))

        result_dict = optimize(model, solver)
        execution_times.loc[scenario, "solving time"] = get_exec_time(t2)
        t2 = time.perf_counter()
        # extract initial charging and energy levels for following iteration
        # (if not last iteration)
        if iteration % iterations_per_era != iterations_per_era - 1:
            charging_starts = {
                "ev": result_dict["charging_ev"].iloc[-overlap_iterations],
                "hp": result_dict["charging_hp_el"].iloc[-overlap_iterations],
                "tes": result_dict["charging_tes"].iloc[-overlap_iterations],
            }
            energy_level_start = {
                "ev": result_dict["energy_level_ev"].iloc[-overlap_iterations],
                "tes": result_dict["energy_level_tes"].iloc[-overlap_iterations],
            }
        # if last iteration per era, reset values
        else:
            charging_starts = None
            energy_level_start = None
        # concatenate charging time series
        if optimize_ev:
            charging_ev = pd.concat([charging_ev, result_dict["charging_ev"]])
            charging_ev = charging_ev.loc[~charging_ev.index.duplicated(keep='last')]
        if optimize_hp:
            charging_hp = pd.concat([charging_hp, result_dict["charging_hp_el"]])
            charging_hp = charging_hp.loc[~charging_hp.index.duplicated(keep='last')]
        if optimize_bess:
            charging_bess = pd.concat([charging_bess, result_dict["charging_bess"]])
            charging_bess = \
                charging_bess.loc[~charging_bess.index.duplicated(keep='last')]

        print("Finished optimisation for iteration {}.".format(iteration))
    execution_times.loc[scenario, "total"] = get_exec_time(start_time)
    return charging_ev, charging_hp, charging_bess, config, execution_times


if __name__ == "__main__":
    scenario = "test_daily"
    save_dir = r"H:\Grids_SE"
    grid_id = 1056
    ev_dir = r"H:\Grids\{}\dumb\electromobility".format(grid_id)
    edisgo_obj = import_edisgo_from_files(
        os.path.join(save_dir, str(grid_id)),
        import_timeseries=True,
        import_heat_pump=True,
    )
    simbev_config = pd.read_csv(os.path.join(ev_dir, "simbev_config.csv"), index_col=0)
    edisgo_obj.electromobility.simbev_config_df = \
        simbev_config.rename(columns={"value": 0}, index={"eta_CP": "eta_cp"}).loc[
            ["stepsize", "year", "eta_cp"]].T.astype("float")
    # add timeseries charging points
    ts_dir = r"H:\Grids\{}".format(grid_id)
    ts_charging_parks = pd.read_csv(
        os.path.join(ts_dir, "charging_points_active_power.csv"), index_col=0,
        parse_dates=True
    )
    ts_charging_parks = ts_charging_parks.resample("1h").mean()
    mapping = pd.read_csv(os.path.join(save_dir, str(grid_id), "cp_mapping.csv"),
                          index_col=0)
    ts_active_power = ts_charging_parks[mapping.name_orig]
    ts_active_power.columns = mapping.name_new
    edisgo_obj.set_time_series_manual(loads_p=ts_active_power)
    # add timeseries heat pumps (cop and demand)
    hp_profiles = get_hp_profiles(edisgo_obj.timeseries.timeindex)
    reference_operation = reference_operation(hp_profiles)
    heat_pumps = edisgo_obj.topology.loads_df.loc[
        edisgo_obj.topology.loads_df.type == "heat_pump"]
    edisgo_obj.timeseries.predefined_conventional_loads_by_sector(
        edisgo_object=edisgo_obj,
        ts_loads=reference_operation,
        load_names=heat_pumps.index
    )
    # add reactive power time series
    edisgo_obj.set_time_series_reactive_power_control()
    edisgo_obj.check_integrity()
    # test
    edisgo_obj.timeseries.timeindex = edisgo_obj.timeseries.timeindex[:7*24]
    results = optimised_operation(
        edisgo_orig=edisgo_obj,
        ev_base_dir=r"H:\Grids",
        cps=True,
        cp_mapping=mapping,
        hps=True,
        scenario=scenario
    )
    # save results
    scenario_dir = os.path.join(save_dir, str(grid_id), scenario)
    os.makedirs(scenario_dir, exist_ok=True)
    results[0].to_csv(os.path.join(scenario_dir, "charging_ev.csv"))
    results[1].to_csv(os.path.join(scenario_dir, "charging_hp_el.csv"))
    results[3].to_csv(os.path.join(scenario_dir, "config.csv"))
    results[4].to_csv(os.path.join(scenario_dir, "execution_times.csv"))
    print("Success")
