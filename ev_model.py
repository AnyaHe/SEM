import pyomo.environ as pm
import pandas as pd


def add_ev_model(model, flex_bands, efficiency=0.9):
    def charging_ev(model, time):
        """
        Constraint for charging of EV that has to ly between the lower and upper
        energy band.

        :param model:
        :param charging_point:
        :param time:
        :return:
        """
        return model.energy_level_ev[time] == \
               model.energy_level_ev[time - 1] + \
               model.charging_efficiency * \
               model.charging_ev[time] * \
               (pd.to_timedelta(model.time_increment) / pd.to_timedelta('1h'))

    def fixed_energy_level(model, time):
        '''
        Constraint for fixed value of energy
        :param model:
        :param charging_point:
        :param time:
        :return:
        '''
        return model.energy_level_ev[time] == \
               (model.flex_bands.loc[time, "lower"] + model.flex_bands.loc[time, "upper"]) / 2
    # save fix parameters
    model.charging_efficiency = efficiency
    model.flex_bands = flex_bands
    # set up variables
    model.charging_ev = \
        pm.Var(model.time_set,
               bounds=lambda m, t:
               (0, m.flex_bands.loc[t, "power"]))
    model.energy_level_ev = \
        pm.Var(model.time_set,
               bounds=lambda m, t:
               (m.flex_bands.loc[t, "lower"],
                m.flex_bands.loc[t, "upper"]))
    # add constraints
    model.EVCharging = pm.Constraint(model.time_non_zero, rule=charging_ev)
    model.FixedEVEnergyLevel = pm.Constraint(model.times_fixed_soc, rule=fixed_energy_level)
    return model


def reduced_operation(model):
    return sum(model.charging_ev[time]**2 for time in model.time_set)


if __name__ == "__main__":
    solver = "gurobi"
    grid_dir = r"H:\Grids_IYCE\176"
    time_increment = pd.to_timedelta('1h')
    flex_bands = pd.read_csv(grid_dir + "/flex_ev.csv", index_col=0, parse_dates=True)
    flex_bands = flex_bands.resample(time_increment).mean().reset_index()
    model = pm.ConcreteModel()
    model.time_set = pm.RangeSet(0, len(flex_bands) - 1)
    model.time_non_zero = model.time_set - [model.time_set.at(1)]
    model.times_fixed_soc = pm.Set(initialize=[model.time_set.at(1),
                                               model.time_set.at(-1)])
    model.time_increment = time_increment
    model = add_ev_model(model, flex_bands)
    model.objective = pm.Objective(rule=reduced_operation,
                                   sense=pm.minimize,
                                   doc='Define objective function')
    opt = pm.SolverFactory(solver)
    results = opt.solve(model, tee=True)
    results_df = pd.DataFrame()
    results_df["charging_ev"] = pd.Series(model.charging_ev.extract_values())
    results_df["energy_level_ev"] = pd.Series(model.energy_level_ev.extract_values())
    print("SUCCESS")