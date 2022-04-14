import pyomo.environ as pm
import pandas as pd


def add_heat_pump_model(model, p_nom_hp, capacity_tes, cop, heat_demand):
    def energy_balance_hp_tes(model, time):
        return model.charging_hp_el[time] * cop.loc[time, "COP 2011"] == \
               model.heat_demand.loc[time, 0] + model.charging_tes[time]

    def fixed_energy_level_tes(model, time):
        return model.energy_tes[time] == model.capacity_tes/2

    def charging_tes(model, time):
        return model.energy_tes[time] == \
               model.energy_tes[time-1] + model.charging_tes[time]
    # save fix parameters
    model.capacity_tes = capacity_tes
    model.p_nom_hp = p_nom_hp
    model.cop = cop
    model.heat_demand = heat_demand
    # set up variables
    model.energy_tes = pm.Var(model.time_set, bounds=(0, capacity_tes))
    model.charging_tes = pm.Var(model.time_set)
    model.charging_hp_el = pm.Var(model.time_set, bounds=(0, p_nom_hp))
    # add constraints
    model.EnergyBalanceHPTES = pm.Constraint(model.time_set, rule=energy_balance_hp_tes)
    model.FixedEnergyTES = pm.Constraint(model.times_fixed_soc, rule=fixed_energy_level_tes)
    model.ChargingTES = pm.Constraint(model.time_non_zero, rule=charging_tes)
    return model


def reduced_operation(model):
    return sum(model.charging_hp_el[time]**2 for time in model.time_set)


if __name__ == "__main__":
    solver = "gurobi"
    nr_hp = 2983
    heat_demand = pd.read_csv(r"C:\Users\aheider\Documents\Software\Semester Project Scripts\Scripts and Data\grids\176\hp_heat_2011.csv",
                              header=None)
    cop = pd.read_csv(r'C:\Users\aheider\Documents\Software\Semester Project Scripts\Scripts and Data\grids\176\COP_2011.csv')
    capacity_tes = nr_hp * 0.05 # MWh
    p_nom_hp = nr_hp * 0.003 # MW
    model = pm.ConcreteModel()
    model.time_set = pm.RangeSet(0, len(heat_demand) - 1)
    model.time_non_zero = model.time_set - [model.time_set.at(1)]
    model.times_fixed_soc = pm.Set(initialize=[model.time_set.at(1),
                                               model.time_set.at(-1)])
    model = add_heat_pump_model(model, p_nom_hp, capacity_tes, cop, heat_demand)
    model.objective = pm.Objective(rule=reduced_operation,
                                   sense=pm.minimize,
                                   doc='Define objective function')
    opt = pm.SolverFactory(solver)
    results = opt.solve(model, tee=True)
    results_df = pd.DataFrame()
    results_df["charging_hp"] = pd.Series(model.charging_hp_el.extract_values())
    results_df["charging_tes"] = pd.Series(model.charging_tes.extract_values())
    results_df["energy_tes"] = pd.Series(model.energy_tes.extract_values())
    print("SUCCESS")