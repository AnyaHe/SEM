from copy import deepcopy
import logging
import os
import pandas as pd
import time

from edisgo.edisgo import import_edisgo_from_files
from edisgo.opf.lopf import import_flexibility_bands, \
    prepare_time_invariant_parameters, BANDS, update_model, setup_model, optimize, get_exec_time
from edisgo.tools.tools import convert_impedances_to_mv

from dg_preparation import get_downstream_nodes_matrix_iterative

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger = logging.getLogger("gurobi")
logger.setLevel(logging.ERROR)


def optimised_operation(
    edisgo_orig, objective, optimize_ev=False, optimize_hp=False, optimize_bess=False,
        scenario="test",
):
    start_time = time.perf_counter()
    timesteps_per_iteration = 24*7
    iterations_per_era = 1
    overlap_iterations = 0

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

    if optimize_ev:
        flexibility_bands = edisgo_obj.electromobility.flexibility_bands
        edisgo_obj.electromobility.check_integrity()
    else:
        flexibility_bands = {}

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

    grid_power = pd.DataFrame()

    start_iter = 0

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
            # setup and update model
            try:
                model = update_model(
                    model,
                    timesteps,
                    parameters,
                    **kwargs
                )
                execution_times.loc[scenario, "update time"] = get_exec_time(t2)
                t2 = time.perf_counter()
            except NameError:
                model = setup_model(
                    parameters,
                    timesteps,
                    objective=objective,
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
            # concatenate charging time series
            grid_power = pd.concat([grid_power, result_dict["grid_power_flexible"]])

            print("Finished optimisation for iteration {}.".format(iteration))
        except Exception as e:
            print(f'Something went wrong in iteration {iteration}. Skipping.')
            print(e)
    execution_times.loc[scenario, "total"] = get_exec_time(start_time)
    return grid_power, config, execution_times


if __name__ == "__main__":
    scenario_reinforcement = "minimum_reinforcement"
    save_dir = r"H:\Grids_SE\feeder"
    grid_id = 8
    edisgo_obj = import_edisgo_from_files(
        os.path.join(save_dir, str(grid_id), scenario_reinforcement),
        import_heat_pump=True, import_electromobility=True, import_timeseries=True
    )
    power_bands = pd.DataFrame()
    for objective in ["maximize_grid_power", "minimize_grid_power"]:
        grid_power_flexible, config_run, exec_times = optimised_operation(
            edisgo_orig=edisgo_obj,
            objective=objective,
            optimize_ev=True,
            optimize_hp=True
        )
        power_bands[objective] = grid_power_flexible
    power_bands.to_csv(os.path.join(save_dir, str(grid_id), scenario_reinforcement, "grid_power_bands.csv"))
    print("Success")
