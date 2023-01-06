import pyomo.environ as pm
import pandas as pd
import matplotlib.pyplot as plt


def add_storage_equivalent_model(model, residual_load, **kwargs):
    def fix_energy_levels(model, time_horizon, time):
        return model.energy_levels[time_horizon, time] == 0

    def charge_storages(model, time_horizon, time):
        return model.energy_levels[time_horizon, time] == model.energy_levels[time_horizon, time-1] + \
               model.charging[time_horizon, time] * \
               (pd.to_timedelta(model.time_increment) / pd.to_timedelta('1h'))

    def meet_residual_load(model, time):
        if hasattr(model, "charging_hp_el"):
            hp_el = model.charging_hp_el[time]
        else:
            hp_el = 0
        if hasattr(model, "charging_ev"):
            ev = sum([model.charging_ev[cp, time] for cp in model.charging_points_set])
        else:
            ev = 0
        return sum(model.charging[time_horizon, time] for time_horizon in
                   model.time_horizons_set) == \
            model.residual_load.iloc[time] + hp_el + ev

    def maximum_charging(model, time_horizon, time):
        return model.charging_max[time_horizon] >= model.charging[time_horizon, time]

    def charging_cap_ratio_upper(model, time_horizon):
        return model.charging_max[time_horizon] <= \
               (model.caps_pos[time_horizon] + model.caps_neg[time_horizon]) / \
               model.coeff_min[time_horizon]

    def charging_cap_ratio_lower(model, time_horizon):
        return model.charging_max[time_horizon] >= \
               (model.caps_pos[time_horizon] + model.caps_neg[time_horizon]) / \
               model.coeff_max[time_horizon]

    def maximum_capacity(model, time_horizon, time):
        return model.caps_pos[time_horizon] >= model.energy_levels[time_horizon, time]

    def minimum_capacity(model, time_horizon, time):
        return model.caps_neg[time_horizon] >= -model.energy_levels[time_horizon, time]

    def abs_charging_up(model, time_horizon, time):
        return model.abs_charging[time_horizon, time] >= model.charging[time_horizon, time]

    def abs_charging_down(model, time_horizon, time):
        return model.abs_charging[time_horizon, time] >= -model.charging[time_horizon, time]

    # save fix parameters
    model.residual_load = residual_load
    model.time_horizons = kwargs.get("time_horizons", [24, 7*24, 28*24, 24*366])
    model.coeff_min = kwargs.get("coeff_min", [0.25, 0.5, 1, 2])
    model.coeff_max = kwargs.get("coeff_max", [8, 12, 48, 96])
    # add time horizon set
    model.time_horizons_set = pm.RangeSet(0, len(model.time_horizons)-1)
    # set up variables
    model.caps_pos = pm.Var(model.time_horizons_set)
    model.caps_neg = pm.Var(model.time_horizons_set)
    model.energy_levels = pm.Var(model.time_horizons_set, model.time_set)
    model.charging = pm.Var(model.time_horizons_set, model.time_set)
    model.charging_max = pm.Var(model.time_horizons_set)
    model.abs_charging = pm.Var(model.time_horizons_set, model.time_set)
    # add constraints
    for time_horizon in model.time_horizons_set:
        times = []
        for time in model.time_set:
            if time % model.time_horizons[time_horizon] == 0:
                times.append(time)
        setattr(model, "FixEnergyLevels{}".format(time_horizon), pm.Constraint([time_horizon], times,
                                              rule=fix_energy_levels))
    model.ChargingStorages = pm.Constraint(model.time_horizons_set, model.time_non_zero,
                                          rule=charge_storages)
    model.ResidualLoad = pm.Constraint(model.time_set, rule=meet_residual_load)
    model.MaximumCharging = pm.Constraint(model.time_horizons_set, model.time_set,
                                          rule=maximum_charging)
    model.MaximumCapacity = pm.Constraint(model.time_horizons_set, model.time_set,
                                          rule=maximum_capacity)
    model.MinimumCapacity = pm.Constraint(model.time_horizons_set, model.time_set,
                                          rule=minimum_capacity)
    model.UpperChargingCapRatio = pm.Constraint(model.time_horizons_set,
                                                rule=charging_cap_ratio_upper)
    model.LowerChargingCapRatio = pm.Constraint(model.time_horizons_set,
                                                rule=charging_cap_ratio_lower)
    model.UpperAbsCharging = pm.Constraint(model.time_horizons_set, model.time_set,
                                           rule=abs_charging_up)
    model.LowerAbsCharging = pm.Constraint(model.time_horizons_set, model.time_set,
                                           rule=abs_charging_down)
    return model


def add_storage_equivalents_model(model, residual_load, connections, flows, **kwargs):
    def fix_energy_levels(model, cell, time_horizon, time):
        return model.energy_levels[cell, time_horizon, time] == 0

    def charge_storages(model, cell, time_horizon, time):
        return model.energy_levels[cell, time_horizon, time] == \
               model.energy_levels[cell, time_horizon, time-1] + \
               model.charging[cell, time_horizon, time] * \
               (pd.to_timedelta(model.time_increment) / pd.to_timedelta('1h'))

    def meet_residual_load (model, cell, time):
        # add flows between cells
        cell_connections = model.connections.loc[cell]
        pos_flows = cell_connections.loc[cell_connections == 1].index
        neg_flows = cell_connections.loc[cell_connections == -1].index
        return sum(model.charging[cell, time_horizon, time] for time_horizon in
                   model.time_horizons_set) + \
               sum(model.flows[(cell, neighbor), time] for neighbor in pos_flows) - \
               sum(model.flows[(neighbor, cell), time] for neighbor in neg_flows) == \
               model.residual_load[cell].iloc[time]

    def maximum_charging(model, cell, time_horizon, time):
        return model.charging_max[cell, time_horizon] >= \
               model.charging[cell, time_horizon, time]

    def charging_cap_ratio_upper(model, cell, time_horizon):
        return model.charging_max[cell, time_horizon] <= \
               (model.caps_pos[cell, time_horizon] +
                model.caps_neg[cell, time_horizon]) / \
               model.coeff_min[time_horizon]

    def charging_cap_ratio_lower(model, cell, time_horizon):
        return model.charging_max[cell, time_horizon] >= \
               (model.caps_pos[cell, time_horizon] +
                model.caps_neg[cell, time_horizon]) / \
               model.coeff_max[time_horizon]

    def maximum_capacity(model, cell, time_horizon, time):
        return model.caps_pos[cell, time_horizon] >= \
               model.energy_levels[cell, time_horizon, time]

    def minimum_capacity(model, cell, time_horizon, time):
        return model.caps_neg[cell, time_horizon] >= \
               -model.energy_levels[cell, time_horizon, time]

    def abs_charging_up(model, cell, time_horizon, time):
        return model.abs_charging[cell, time_horizon, time] >= \
               model.charging[cell, time_horizon, time]

    def abs_charging_down(model, cell, time_horizon, time):
        return model.abs_charging[cell, time_horizon, time] >= \
               -model.charging[cell, time_horizon, time]

    # save fix parameters
    model.residual_load = residual_load
    model.time_horizons = kwargs.get("time_horizons", [24, 7*24, 28*24, 24*366])
    model.coeff_min = kwargs.get("coeff_min", [0.25, 0.5, 1, 2])
    model.coeff_max = kwargs.get("coeff_max", [8, 12, 48, 96])
    model.connections = connections
    # add sets
    model.time_horizons_set = pm.RangeSet(0, len(model.time_horizons)-1)
    model.cells_set = pm.Set(initialize=residual_load.columns)
    model.flows_set = pm.Set(initialize=flows.index)
    # set up variables
    model.caps_pos = pm.Var(model.cells_set, model.time_horizons_set)
    model.caps_neg = pm.Var(model.cells_set, model.time_horizons_set)
    model.energy_levels = pm.Var(model.cells_set, model.time_horizons_set, model.time_set)
    model.charging = pm.Var(model.cells_set, model.time_horizons_set, model.time_set)
    model.charging_max = pm.Var(model.cells_set, model.time_horizons_set)
    model.abs_charging = pm.Var(model.cells_set, model.time_horizons_set, model.time_set)
    model.flows = pm.Var(model.flows_set, model.time_set)
    # add constraints
    for time_horizon in model.time_horizons_set:
        times = []
        for time in model.time_set:
            if time % model.time_horizons[time_horizon] == 0:
                times.append(time)
        setattr(model, f"FixEnergyLevels{time_horizon}",
                pm.Constraint(model.cells_set, [time_horizon], times,
                              rule=fix_energy_levels))
    model.ChargingStorages = pm.Constraint(model.cells_set, model.time_horizons_set, model.time_non_zero,
                                           rule=charge_storages)
    model.ResidualLoad = pm.Constraint(model.cells_set, model.time_set, rule=meet_residual_load)
    model.MaximumCharging = pm.Constraint(model.cells_set, model.time_horizons_set, model.time_set,
                                          rule=maximum_charging)
    model.MaximumCapacity = pm.Constraint(model.cells_set, model.time_horizons_set, model.time_set,
                                          rule=maximum_capacity)
    model.MinimumCapacity = pm.Constraint(model.cells_set, model.time_horizons_set, model.time_set,
                                          rule=minimum_capacity)
    model.UpperChargingCapRatio = pm.Constraint(model.cells_set, model.time_horizons_set,
                                                rule=charging_cap_ratio_upper)
    model.LowerChargingCapRatio = pm.Constraint(model.cells_set, model.time_horizons_set,
                                                rule=charging_cap_ratio_lower)
    model.UpperAbsCharging = pm.Constraint(model.cells_set, model.time_horizons_set, model.time_set,
                                           rule=abs_charging_up)
    model.LowerAbsCharging = pm.Constraint(model.cells_set, model.time_horizons_set, model.time_set,
                                           rule=abs_charging_down)
    return model


def minimize_cap(model):
    return sum(model.weighting[time_horizon] * (model.caps_pos[time_horizon] +
                                                model.caps_neg[time_horizon])
               for time_horizon in model.time_horizons_set)


def minimize_energy(model):
    return sum(model.weighting[time_horizon] * sum(model.abs_charging[time_horizon, time]
                                                   for time in model.time_set)
               for time_horizon in model.time_horizons_set)


def minimize_energy_and_power(model):
    if hasattr(model, "charging_hp_el") and (model.p_nom_hp > 0):
        hp_el = sum([(model.charging_hp_el[time]/model.p_nom_hp)**2
                     for time in model.time_set])
    else:
        hp_el = 0
    if hasattr(model, "charging_ev") and (model.flex_bands["upper_power"].max().max()>0):
        ev = sum([
            (model.charging_ev[cp, time]/model.flex_bands["upper_power"][cp].max())**2
            for cp in model.charging_points_set for time in model.time_set])
    else:
        ev = 0
    return sum(
        model.weighting[time_horizon] * sum(model.abs_charging[time_horizon, time]
                                            for time in model.time_set)
        for time_horizon in model.time_horizons_set)+1e-9*(hp_el+ev)


def minimize_energy_multi(model):
    return sum(sum(model.weighting[time_horizon] * sum(model.abs_charging[cell, time_horizon, time]
                                                   for time in model.time_set)
               for time_horizon in model.time_horizons_set) for cell in model.cells_set)


if __name__ == "__main__":
    solver = "gurobi"
    grid_dir = r"C:\Users\aheider\Documents\Software\Semester Project Scripts\Scripts and Data\grids\1690"
    time_increment = pd.to_timedelta('1h')
    load = pd.read_csv(grid_dir + "/load.csv", header=None)[0]
    feedin = pd.read_csv(grid_dir + r"\generation_alone.csv", header=None)[0]
    residual_load = (load - feedin)#.resample(time_increment).mean().reset_index()
    model = pm.ConcreteModel()
    model.time_set = pm.RangeSet(0, len(residual_load) - 1)
    model.time_non_zero = model.time_set - [model.time_set.at(1)]
    model.time_increment = time_increment
    model.weighting = [1, 7, 30, 365]
    model = add_storage_equivalent_model(model, residual_load)
    model.objective = pm.Objective(rule=minimize_energy,
                                   sense=pm.minimize,
                                   doc='Define objective function')
    opt = pm.SolverFactory(solver)
    results = opt.solve(model, tee=True)
    charging = pd.Series(model.charging.extract_values()).unstack()
    energy_levels = pd.Series(model.energy_levels.extract_values()).unstack()
    caps = pd.Series(model.caps_pos.extract_values()) + pd.Series(model.caps_neg.extract_values())
    caps_neg = pd.Series(model.caps_neg.extract_values())
    relative_energy_levels = (energy_levels.T + caps_neg).divide(caps)
    abs_charging = pd.Series(model.abs_charging.extract_values()).unstack()
    total_demand = load.sum()
    shifted_energy_df = pd.DataFrame(columns=["energy_stored", "energy_charge", "energy_discharge"],
                                     index=model.time_horizons_set)
    shifted_energy_df["energy_stored"] = abs_charging.sum(axis=1)/2
    shifted_energy_df["energy_charge"] = charging[charging > 0].sum(axis=1)
    shifted_energy_df["energy_discharge"] = charging[charging < 0].sum(axis=1).abs()
    shifted_energy_rel_df = shifted_energy_df.divide(total_demand).multiply(100)
    shifted_energy_rel_df.T.plot.bar(stacked=True)
    plt.show()
    print("SUCCESS")
