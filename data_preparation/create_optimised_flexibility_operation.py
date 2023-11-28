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
    edisgo_orig, ev_base_dir, cps=None, hps=None, bess=None,
        scenario="test"
):
    start_time = time.perf_counter()
    objective = "minimize_loading"
    timesteps_per_iteration = 24*7
    iterations_per_era = 2
    overlap_iterations = 24

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

    edisgo_obj = convert_impedances_to_mv(deepcopy(edisgo_orig))

    print("Converted impedances to mv.")

    downstream_nodes_matrix = \
        get_downstream_nodes_matrix_iterative(edisgo_obj.topology)

    print("Downstream node matrix imported.")

    if cps is not None:
        optimize_ev = True
        flexibility_bands = edisgo_obj.electromobility.flexibility_bands
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
    energy_level_start = {"ev": None, "tes": None}
    start_iter = 0
    energy_level_end = {"ev": None, "tes": None}

    for iteration in range(
        start_iter,
        int(len(edisgo_obj.timeseries.timeindex) / timesteps_per_iteration),
    ):
        try:
            print("Starting optimisation for iteration {}.".format(iteration))
            # set values for "normal" iterations
            if iteration % iterations_per_era != iterations_per_era - 1:
                timesteps = edisgo_obj.timeseries.timeindex[
                            iteration * timesteps_per_iteration:
                            (iteration + 1) * timesteps_per_iteration + overlap_iterations]
                # energy_level_end = {"ev": None, "tes": None}
            # set values for final iteration in era
            else:
                timesteps = edisgo_obj.timeseries.timeindex[
                            iteration * timesteps_per_iteration:
                            (iteration + 1) * timesteps_per_iteration]
                # energy_level_end = {"ev": True, "tes": True}
            # check for violations of ev energy bands and correct
            if energy_level_start["ev"] is not None:
                energy_level_start["ev"] = check_for_energy_band_violations(
                    energy_level_start=energy_level_start["ev"],
                    parameters=parameters,
                    start_time=edisgo_obj.timeseries.timeindex[
                            iteration * timesteps_per_iteration-1]
                )
            try:
                model = update_model(
                    model,
                    timesteps,
                    parameters,
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
                energy_level_start = {
                    "ev": result_dict["energy_level_ev"].iloc[-overlap_iterations-1],
                    "tes": result_dict["energy_level_tes"].iloc[-overlap_iterations-1],
                }
            # if last iteration per era, reset values
            else:
                energy_level_start = {"ev": None, "tes": None}
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
        except Exception as e:
            print(f'Something went wrong in iteration {iteration}. Skipping.')
            print(e)
            energy_level_start = {
                "ev": None,
                "tes": None,
            }
    execution_times.loc[scenario, "total"] = get_exec_time(start_time)
    return charging_ev, charging_hp, charging_bess, config, execution_times


def check_for_energy_band_violations(energy_level_start,
                                     parameters, start_time):
    # Check if problem will be feasible
    violation_lower_bound_cp = []
    violation_upper_bound_cp = []
    upper_energy = parameters["ev_flex_bands"]["upper_energy"]
    lower_energy = parameters["ev_flex_bands"]["lower_energy"]
    for cp_tmp in energy_level_start.index:
        if energy_level_start[cp_tmp] > upper_energy.loc[start_time, cp_tmp]:
            if energy_level_start[cp_tmp] - upper_energy.loc[start_time, cp_tmp] > 1e-4:
                raise ValueError(
                    'Optimisation should not return values higher than upper bound. '
                    'Problem for {}. Initial energy level is {}, but upper bound {}.'.format(
                        cp_tmp, energy_level_start[cp_tmp],
                        upper_energy.loc[start_time, cp_tmp]))
            else:
                energy_level_start[cp_tmp] = \
                    upper_energy.loc[start_time, cp_tmp] - 1e-6
                violation_upper_bound_cp.append(cp_tmp)
        if energy_level_start[cp_tmp] < lower_energy.loc[start_time, cp_tmp]:

            if -energy_level_start[cp_tmp] + \
                    lower_energy.loc[start_time, cp_tmp] > 1e-4:
                raise ValueError(
                    'Optimisation should not return values lower than lower bound. '
                    'Problem for {}. Initial energy level is {}, but lower bound {}.'.format(
                        cp_tmp, energy_level_start[cp_tmp],
                        lower_energy.loc[start_time, cp_tmp]))
            else:
                energy_level_start[cp_tmp] = \
                    lower_energy.loc[start_time, cp_tmp] + 1e-6
                violation_lower_bound_cp.append(cp_tmp)
    print('{} Charging points violate lower bound.'.format(len(violation_lower_bound_cp)))
    print('{} Charging points violate upper bound.'.format(len(violation_upper_bound_cp)))
    return energy_level_start


def setup_edisgo_object(save_dir, grid_id, ev_dir, grid_orig, orig_dir,
                        ev_operation="reference", hp_operation="reference", opt_dir=None):
    edisgo_obj = import_edisgo_from_files(
        os.path.join(save_dir, str(grid_id)),
        import_timeseries=True,
        import_heat_pump=True,
    )
    simbev_config = pd.read_csv(os.path.join(ev_dir, "dumb", "electromobility",
                                             "simbev_config.csv"), index_col=0)
    edisgo_obj.electromobility.simbev_config_df = \
        simbev_config.rename(columns={"value": 0}, index={"eta_CP": "eta_cp"}).loc[
            ["stepsize", "year", "eta_cp"]].T.astype("float")
    # add timeseries charging points
    ts_dir = r"H:\Grids\{}".format(grid_orig)
    ts_charging_parks = pd.read_csv(
        os.path.join(ts_dir, "charging_points_active_power.csv"), index_col=0,
        parse_dates=True
    )
    ts_charging_parks = ts_charging_parks.resample("1h").mean()
    mapping = pd.read_csv(os.path.join(orig_dir, str(grid_orig), "cp_mapping.csv"),
                          index_col=0)
    ts_active_power = ts_charging_parks[mapping.name_orig]
    ts_active_power.columns = mapping.name_new
    ts_cps_grid = ts_active_power[edisgo_obj.topology.charging_points_df.index]
    # import flexibility bands
    flexibility_bands = import_flexibility_bands(ev_dir, ["home", "work"])

    # Update bands with mapping
    print("Flexibility bands imported.")

    # extract data for feeder
    cp_mapping = mapping.loc[
        mapping.name_new.isin(edisgo_obj.topology.charging_points_df.index)]
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
    # Add bands to edisgo object and run integrity check
    edisgo_obj.electromobility.flexibility_bands = flexibility_bands
    edisgo_obj.electromobility.check_integrity()
    if ev_operation == "reference":
        pass
    elif ev_operation == "optimised":
        ev_charging_opt = pd.read_csv(os.path.join(opt_dir, "charging_ev.csv"),
                                      index_col=0, parse_dates=True)
        ts_cps_grid.update(ev_charging_opt)
    elif ev_operation == "flex_50":
        ev_charging = edisgo_obj.electromobility.flexibility_bands["upper_power"].multiply(0.5)
        ts_cps_grid.update(ev_charging)
    elif ev_operation == "full_flex":
        ev_charging = edisgo_obj.electromobility.flexibility_bands["upper_power"]
        ts_cps_grid.update(ev_charging)
    else:
        raise ValueError("Undefined ev operation.")
    edisgo_obj.set_time_series_manual(
        loads_p=ts_cps_grid)
    # add timeseries heat pumps (cop and demand)
    heat_pumps = edisgo_obj.topology.loads_df.loc[
        edisgo_obj.topology.loads_df.type == "heat_pump"]
    if hp_operation == "reference":
        hp_profiles = get_hp_profiles(edisgo_obj.timeseries.timeindex)
        hp_reference_operation = reference_operation(hp_profiles)
        edisgo_obj.timeseries.predefined_conventional_loads_by_sector(
            edisgo_object=edisgo_obj,
            ts_loads=hp_reference_operation,
            load_names=heat_pumps.index
        )
    elif hp_operation == "optimised":
        hp_charging_opt = pd.read_csv(os.path.join(opt_dir, "charging_hp_el.csv"),
                                      index_col=0, parse_dates=True)
        edisgo_obj.set_time_series_manual(
            loads_p=hp_charging_opt)
    elif hp_operation == "flex_50":
        hp_charging = pd.DataFrame(columns=heat_pumps.index, index=edisgo_obj.timeseries.timeindex,
                                   data=0.5)
        hp_charging = hp_charging.multiply(heat_pumps.p_set)
        edisgo_obj.set_time_series_manual(
            loads_p=hp_charging)
    elif hp_operation == "full_flex":
        hp_charging = pd.DataFrame(columns=heat_pumps.index, index=edisgo_obj.timeseries.timeindex,
                                   data=1.0)
        hp_charging = hp_charging.multiply(heat_pumps.p_set)
        edisgo_obj.set_time_series_manual(
            loads_p=hp_charging)
    else:
        raise ValueError("Undefined heat pump operation.")
    # add reactive power time series
    edisgo_obj.set_time_series_reactive_power_control()
    edisgo_obj.check_integrity()
    return edisgo_obj


if __name__ == "__main__":
    scenario = "test_weekly"
    save_dir = r"H:\Grids_SE\feeder"
    grid_id = 8
    orig_dir = r"H:\Grids_SE"
    grid_orig = 1056
    ev_dir = r"H:\Grids\{}".format(grid_orig)
    edisgo_obj = setup_edisgo_object(
        save_dir=save_dir,
        grid_id=grid_id,
        ev_dir=ev_dir,
        grid_orig=grid_orig,
        orig_dir=orig_dir
    )
    # test
    edisgo_obj.timeseries.timeindex = edisgo_obj.timeseries.timeindex[:7*24*2]
    results = optimised_operation(
        edisgo_orig=edisgo_obj,
        ev_base_dir=r"H:\Grids",
        cps=True,
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