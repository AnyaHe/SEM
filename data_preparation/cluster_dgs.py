import os
import pandas as pd
import pyomo.environ as pm

file_path = r"C:\Users\aheider\Documents\Software\SE2\Flexibility-Quantification\data"
solver = "gurobi"
# read load and generation data
load_dgs = pd.read_csv(os.path.join(file_path, "demand_6dgs.csv"),
                       index_col=0, parse_dates=True)/1000
generation_dgs = pd.read_csv(os.path.join(file_path, "generation_6dgs_full.csv"),
                             index_col=0, parse_dates=True)/1000
generation_dgs_solar_wind = pd.read_csv(os.path.join(file_path, "generation_6dgs.csv"),
                                        index_col=0, parse_dates=True)/1000

# load overall demand and generation
load_germany = pd.read_csv(os.path.join(file_path, "demand_dgs_ego100.csv"),
                           index_col=0, parse_dates=True).sum(axis=1)
generation_germany_dgs = pd.read_csv(
    os.path.join(file_path, "vres_reference_dgs_ego100.csv"), index_col=0,
    parse_dates=True)
wind_cols = [col for col in generation_germany_dgs.columns if "wind" in col]
solar_cols = [col for col in generation_germany_dgs.columns if "solar" in col]

# build combined scenario
# scale all ts to 1
ts_germany_combined = pd.DataFrame()
ts_germany_combined["wind"] = generation_germany_dgs[wind_cols].sum(axis=1)/1000
ts_germany_combined["solar"] = generation_germany_dgs[solar_cols].sum(axis=1)/1000
ts_germany_combined["load"] = load_germany
ts_dgs_combined = {
    "load": load_dgs,
    "wind": generation_dgs_solar_wind[[f"wind_{dg}" for dg in load_dgs.columns]],
    "solar": generation_dgs_solar_wind[[f"solar_{dg}" for dg in load_dgs.columns]]}
ts_dgs_combined["wind"].columns = load_dgs.columns
ts_dgs_combined["solar"].columns = load_dgs.columns


def build_model_ts_approximation(ts_germany, ts_dgs):
    def minimize_squared_error(model):
        """
        objective to minimize squared error of germany-wide and time series of
        distribution grids.
        """
        return sum(sum((model.ts_germany.loc[time, ts] -
                        sum(model.weights[dg] * model.ts_dgs[ts].loc[time, dg]
                            for dg in model.dg_set)
                        ) ** 2 for time in model.time_set) for ts in model.ts_set)
    # build model that minimises RMSE of load
    model = pm.ConcreteModel()
    # define sets (time and distribution grids)
    model.time_set = pm.Set(initialize=ts_germany.index)
    model.ts_set = pm.Set(initialize=ts_germany.columns)
    model.dg_set = pm.Set(initialize=ts_dgs[model.ts_set.at(1)].columns)
    # set fixed parameters
    model.ts_germany = ts_germany
    model.ts_dgs = ts_dgs
    # set variables
    model.weights = pm.Var(model.dg_set, bounds=(0, None))
    # set objective
    model.objective = pm.Objective(rule=minimize_squared_error,
                                   sense=pm.minimize,
                                   doc='Define objective function')
    return model


scenarios = {
    "load": {"ts_germany": pd.DataFrame(load_germany, columns=["load"]),
             "ts_dgs": {"load": load_dgs}},
    "generation": {"ts_germany": pd.DataFrame(generation_germany_dgs.sum(axis=1),
                                              columns=["feed-in"]),
                   "ts_dgs": {"feed-in": generation_dgs}},
    "combined": {"ts_germany": ts_germany_combined,
                 "ts_dgs": ts_dgs_combined}
}
weights = pd.DataFrame()
for scenario, ts in scenarios.items():
    m = build_model_ts_approximation(ts["ts_germany"], ts["ts_dgs"])
    opt = pm.SolverFactory(solver)
    results = opt.solve(m, tee=True)
    weights[scenario] = pd.Series(m.weights.extract_values())
weights.to_csv(os.path.join(file_path, "weights_dgs.csv"))
print("SUCCESS")
