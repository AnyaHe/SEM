import os
import pandas as pd
import pyomo as pm
from storage_equivalent.storage_equivalent_model import add_storage_equivalents_model, \
    minimize_energy_multi


def get_connection_matrix_of_dgs(dgs):
    """
    Method to get connection matrix of DGs. The model includes a connection to the
    overlying grid from every DG and no interconnections between the single grids.

    :return: connections, flows
    """
    idx_upper_grid = "0"
    all_cells = dgs + [idx_upper_grid]
    connections = pd.DataFrame(index=all_cells,
                               columns=all_cells,
                               data=0)
    flows = pd.DataFrame(columns=["from_bus", "to_bus"],
                         index=[i for i in range(len(dgs))])
    connections.loc[idx_upper_grid, dgs] = 1
    connections.loc[dgs, idx_upper_grid] = -1
    flows.loc[:, "from_bus"] = idx_upper_grid
    flows.loc[:, "to_bus"] = dgs
    return connections, flows


if __name__ == "__main__":
    isolated = False
    save_results = True
    res_dgs = pd.read_csv("data/vres_reference_dgs_ego100.csv", index_col=0,
                          parse_dates=True).divide(1000)
    dgs = list(set([idx[1] for idx in res_dgs.columns.str.split("_")]))
    # Todo: connections_df necessary?
    connections_df, flows_df = get_connection_matrix_of_dgs(dgs)

    scenario = "DGs_connected"
    solver = "gurobi"
    time_increment = pd.to_timedelta('1h')
    vres = pd.read_csv(r"data/vres_reference_ego100.csv", index_col=0,
                       parse_dates=True).divide(1000)
    wind_dgs = [col for col in res_dgs.columns if "wind" in col]
    solar_dgs = [col for col in res_dgs.columns if "solar" in col]
    res_dgs["wind_0"] = vres["wind"] - res_dgs[wind_dgs].sum(axis=1)
    res_dgs["solar_0"] = vres["solar"] - res_dgs[solar_dgs].sum(axis=1)
    demand_germany = pd.read_csv("data/demand_germany_ego100.csv", index_col=0,
                                parse_dates=True)
    demand_dgs = pd.DataFrame(columns=connections_df.index)
    demand_dgs["0"] = demand_germany.sum(axis=1)
    demand_dgs[dgs] = 0
    if isolated:
        connections_df.loc[:, :] = 0
    sum_energy = demand_dgs.sum().sum()
    sum_res = res_dgs.sum().sum()
    scaled_ts_reference = pd.DataFrame(columns=demand_dgs.columns)
    for state in scaled_ts_reference.columns:
        if (state == "0") or ("wind_" + state in wind_dgs):
            wind_ts = res_dgs["wind_" + state]
        else:
            wind_ts = 0
        if (state == "0") or ("solar" + state in solar_dgs):
            solar_ts = res_dgs["solar_" + state]
        else:
            solar_ts = 0
        scaled_ts_reference[state] = \
            (wind_ts + solar_ts) / sum_res * sum_energy
    shifted_energy_df = pd.DataFrame(columns=["state", "storage_type",
                                              "energy_stored"])
    shifted_energy_rel_df = pd.DataFrame(columns=["state", "storage_type",
                                                  "energy_stored"])
    residual_load = demand_dgs - scaled_ts_reference
    model = pm.ConcreteModel()
    model.time_set = pm.RangeSet(0, len(residual_load) - 1)
    model.time_non_zero = model.time_set - [model.time_set.at(1)]
    model.time_increment = time_increment
    model.weighting = [10, 100, 1000, 10000]
    model = add_storage_equivalents_model(model, residual_load, connections_df, flows_df)
    model.objective = pm.Objective(rule=minimize_energy_multi,
                                   sense=pm.minimize,
                                   doc='Define objective function')

    opt = pm.SolverFactory(solver)
    results = opt.solve(model, tee=True)
    charging = pd.Series(model.charging.extract_values()).unstack()
    energy_levels = pd.Series(model.energy_levels.extract_values()).unstack()
    caps = pd.Series(model.caps_pos.extract_values()).unstack() + pd.Series(
        model.caps_neg.extract_values()).unstack()
    flows = pd.Series(model.flows.extract_values()).unstack()
    if save_results:
        os.makedirs(f"results/{scenario}", exist_ok=True)
        charging.to_csv(f"results/{scenario}/charging.csv")
        energy_levels.to_csv(f"results/{scenario}/energy_levels.csv")
        caps.to_csv(f"results/{scenario}/caps.csv")
        flows.to_csv(f"results/{scenario}/flows.csv")
    abs_charging = pd.Series(model.abs_charging.extract_values()).unstack()
    df_tmp = (abs_charging.sum(axis=1) / 2).reset_index().rename(
        columns={"level_0": "state", "level_1": "storage_type", 0: "energy_stored"})
    df_tmp.to_csv("results/storage_equivalents_{}.csv".format(scenario))
    print("SUCCESS")
