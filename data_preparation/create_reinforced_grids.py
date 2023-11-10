import os

from create_optimised_flexibility_operation import setup_edisgo_object


if __name__ == "__main__":
    scenario_opt = "test_weekly"
    scenario_reinforcement = "minimum_reinforcement"
    save_dir = r"H:\Grids_SE\feeder"
    grid_id = 8
    orig_dir = r"H:\Grids_SE"
    grid_orig = 1056
    ev_dir = r"H:\Grids\{}".format(grid_orig)
    if scenario_reinforcement == "reference":
        flex_operation = "reference"
    elif scenario_reinforcement == "minimum_reinforcement":
        flex_operation = "optimised"
    elif scenario_reinforcement == "flex_50":
        flex_operation = "flex_50"
    elif scenario_reinforcement == "full_flex":
        flex_operation = "full_flex"
    else:
        raise ValueError("Undefined reinforcement scenario.")
    edisgo_obj = setup_edisgo_object(
        save_dir, grid_id, ev_dir, grid_orig, orig_dir, ev_operation=flex_operation,
        hp_operation=flex_operation,
        opt_dir=os.path.join(save_dir, str(grid_id), scenario_opt)
    )
    # test
    edisgo_obj.timeseries.timeindex = edisgo_obj.timeseries.timeindex[:7*24*2]
    edisgo_obj.reinforce(timesteps_pfa='reduced_analysis')
    edisgo_obj.save(os.path.join(save_dir, str(grid_id), scenario_reinforcement),
                    save_heatpump=True, save_electromobility=True)
