import pandas as pd
import pyomo.environ as pm
from storage_equivalent import add_storage_equivalents_model, minimize_energy_multi
from oedb import import_geolocations_states


def get_connection_matrix_of_states():
    """
    Method to get connection matrix between states.
    :return:
    """
    states_geom, states_data = import_geolocations_states()
    states = states_geom.index.tolist()
    states.append("offshore")
    connections = pd.DataFrame(index=states,
                               columns=states,
                               data=0)
    flows = pd.DataFrame(columns=["from_bus", "to_bus"])
    for index, row in states_geom.iterrows():
        neighbors = states_geom[states_geom.geometry.touches(row['geometry'])].index.tolist()
        if index in neighbors:
            neighbors = neighbors.remove(index)
        for neighbor in neighbors:
            if connections.loc[neighbor, index] == 1:
                connections.loc[index, neighbor] = -1
            else:
                connections.loc[index, neighbor] = 1
                flows = flows.append(pd.Series({"from_bus": index,
                                                "to_bus":neighbor}, name=(index, neighbor)))
    # set offshore wind neighbors
    shore_states = ["DE8", "DE9", "DEF"]
    for state in shore_states:
        connections.loc[state, "offshore"] = -1
        connections.loc["offshore", state] = 1
        flows = flows.append(pd.Series({"from_bus": "offshore",
                                        "to_bus": state}, name=("offshore", state)))
    return connections, flows


if __name__ == "__main__":
    scenario = "States_isolated_test"
    solver = "gurobi"
    time_increment = pd.to_timedelta('1h')
    vres = pd.read_csv(r"vres_reference_ego100.csv", index_col=0,
                       parse_dates=True).divide(1000)
    vres_states = pd.read_csv(r"vres_reference_states_ego100.csv", index_col=0,
                       parse_dates=True).divide(1000)
    vres_states["wind_offshore"] = vres.sum(axis=1) - vres_states.sum(axis=1)
    vres_states["solar_offshore"] = 0
    demand_states = pd.read_csv("demand_states_ego100.csv", index_col=0,
                                parse_dates=True)
    demand_states["offshore"] = 0
    connections, flows = get_connection_matrix_of_states()
    sum_energy = demand_states.sum()
    scaled_ts_reference = pd.DataFrame(columns=demand_states.columns)
    for state in scaled_ts_reference.columns:
        scaled_ts_reference[state] = vres.divide(vres.sum()).sum(axis=1).multiply(sum_energy[state])
    shifted_energy_df = pd.DataFrame(columns=["state", "storage_type",
                                              "energy_stored"])
    shifted_energy_rel_df = pd.DataFrame(columns=["state", "storage_type",
                                                  "energy_stored"])
    residual_load = demand_states - scaled_ts_reference
    model = pm.ConcreteModel()
    model.time_set = pm.RangeSet(0, len(residual_load) - 1)
    model.time_non_zero = model.time_set - [model.time_set.at(1)]
    model.time_increment = time_increment
    model.weighting = [10, 100, 1000, 10000]
    model = add_storage_equivalents_model(model, residual_load, connections, flows)
    model.objective = pm.Objective(rule=minimize_energy_multi,
                                   sense=pm.minimize,
                                   doc='Define objective function')

    opt = pm.SolverFactory(solver)
    results = opt.solve(model, tee=True)
    charging = pd.Series(model.charging.extract_values()).unstack()
    energy_levels = pd.Series(model.energy_levels.extract_values()).unstack()
    caps = pd.Series(model.caps_pos.extract_values()).unstack() + pd.Series(
        model.caps_neg.extract_values()).unstack()
    abs_charging = pd.Series(model.abs_charging.extract_values()).unstack()
    df_tmp = (abs_charging.sum(axis=1) / 2).reset_index().rename(
        columns={"level_0": "state", "level_1": "storage_type", 0: "energy_stored"})
    # df_tmp.to_csv("results/storage_equivalents_{}.csv".format(scenario))
    print("SUCCESS")
