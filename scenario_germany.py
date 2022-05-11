import pandas as pd
import pyomo.environ as pm
import matplotlib.pyplot as plt

from storage_equivalent import add_storage_equivalents, minimize_energy

if __name__ == "__main__":
    solver = "glpk"
    time_increment = pd.to_timedelta('1h')
    vres = pd.read_csv(r"vres_reference_ego100.csv", index_col=0,
                       parse_dates=True).divide(1000)
    demand = pd.read_csv(r"demand_germany_ego100.csv", index_col=0,
                         parse_dates=True)
    scaling = demand.sum().sum()/vres.sum().sum()
    vres_scaled = vres.multiply(scaling)
    new_res_load = demand.sum(axis=1)-vres_scaled.sum(axis=1)
    shifted_energy_df = pd.DataFrame()
    shifted_energy_rel_df = pd.DataFrame()
    model = pm.ConcreteModel()
    model.time_set = pm.RangeSet(0, len(new_res_load) - 1)
    model.time_non_zero = model.time_set - [model.time_set[1]]
    model.time_increment = time_increment
    model.weighting = [1, 7, 30, 365]
    model = add_storage_equivalents(model, new_res_load)
    model.objective = pm.Objective(rule=minimize_energy,
                                   sense=pm.minimize,
                                   doc='Define objective function')
    opt = pm.SolverFactory(solver)
    results = opt.solve(model, tee=True)
    charging = pd.Series(model.charging.extract_values()).unstack()
    energy_levels = pd.Series(model.energy_levels.extract_values()).unstack()
    caps = pd.Series(model.caps_pos.extract_values()) + pd.Series(
        model.caps_neg.extract_values())
    caps_neg = pd.Series(model.caps_neg.extract_values())
    relative_energy_levels = (energy_levels.T + caps_neg).divide(caps)
    abs_charging = pd.Series(model.abs_charging.extract_values()).unstack()
    shifted_energy_df["Germany"] = (abs_charging.sum(axis=1) / 2)
    total_demand = demand.sum().sum()
    shifted_energy_rel_df["Germany"] = (abs_charging.sum(
        axis=1) / 2) / total_demand * 100
    shifted_energy_df.T.plot.bar(stacked=True)
    shifted_energy_rel_df.T.plot.bar(stacked=True)
    plt.show()
    print("SUCCESS")