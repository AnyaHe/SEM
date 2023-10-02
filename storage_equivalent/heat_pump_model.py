import pyomo.environ as pm
import pandas as pd
from scenario_input import scenario_input_hps


def scale_heat_demand_to_grid(ts_heat_demand, grid_id):
    def population_per_grid():
        """From https://openenergy-platform.org/dataedit/view/grid/ego_dp_mv_griddistrict v0.4.3"""
        return {176: 35373, 177: 32084, 1056: 13007, 1690: 14520, 1811: 19580, 2534: 24562}
    total_nr_hps = 7 # Mio
    total_population = 83.1 # Mio
    heat_demand_per_hp = scenario_input_hps()["heat_demand_single_hp"] # MWh
    ts_heat_demand_per_hp = ts_heat_demand/ts_heat_demand.sum() * heat_demand_per_hp
    grid_nr_hp = int(round(total_nr_hps/total_population*population_per_grid()[grid_id], 0))
    return grid_nr_hp * ts_heat_demand_per_hp, grid_nr_hp


def scale_heat_pumps(nr_hp_mio, scenario_dict):
    """
    Method to scale relevant HP parameters and timeseries to the input nr of HPs

    :param nr_hp_mio: int or float
        Number of HPs in Mio.
    :param scenario_dict: scenario dict, has to contain parameters given in
        ref:add_hps()
    :return:
        Values for total capacity of TES, total electrical capacity of HPs, timeseries
        of total heat demand and total heat consumption
    """
    nr_hp = nr_hp_mio * 1e6
    capacity_tes = nr_hp * scenario_dict["capacity_single_tes"]  # GWh
    p_nom_hp = nr_hp * scenario_dict["p_nom_single_hp"]  # GW
    ts_heat_demand = scenario_dict["ts_heat_demand_single_hp"] * nr_hp
    ts_heat_el = ts_heat_demand.T.divide(scenario_dict["ts_cop"]).T
    sum_energy_heat = ts_heat_el.sum().sum()
    return capacity_tes, p_nom_hp, ts_heat_demand, ts_heat_el, sum_energy_heat


def model_input_hps(scenario_dict, hp_mode, i=None, nr_hp_mio=None):
    if hp_mode is not None:
        # determine number of hps
        if i is not None:
            if nr_hp_mio is not None:
                print("Both i and nr_hp_mio are defined, nr_hp_mio will be used.")
            else:
                nr_hp_mio = i * 2.5
        else:
            if nr_hp_mio is None:
                raise ValueError("Either i or nr_hp_mio have to be provided.")
        (capacity_tes, p_nom_hp,
         ts_heat_demand, ts_heat_el, sum_energy_heat) = \
            scale_heat_pumps(nr_hp_mio=nr_hp_mio,
                             scenario_dict=scenario_dict)
    else:
        ts_heat_el = pd.Series(index=scenario_dict["ts_demand"].index, data=0)
        sum_energy_heat, nr_hp_mio, capacity_tes, p_nom_hp, ts_heat_demand = \
            (0, 0, 0, 0, 0)
    return \
        nr_hp_mio, ts_heat_el, sum_energy_heat, capacity_tes, p_nom_hp, ts_heat_demand


def add_heat_pump_model(model, p_nom_hp, capacity_tes, cop, heat_demand,
                        efficiency_static_tes=0.99, efficiency_dynamic_tes=0.95,
                        use_binaries=False, use_linear_penalty=False, **kwargs):
    def energy_conversion_hp(model, time):
        return model.charging_hp_el[time] * cop.loc[model.timeindex[time]] == \
            model.charging_hp_th[time]

    def energy_balance_hp_tes(model, time):
        if model.use_binaries_hp:
            charging = model.y_charge_tes[time] * model.charging_tes[time]
            discharging = model.y_discharge_tes[time] * model.discharging_tes[time]
        else:
            charging = model.charging_tes[time]
            discharging = model.discharging_tes[time]
        return model.charging_hp_th[time] == \
               model.heat_demand.loc[model.timeindex[time]] + \
               charging - discharging

    def fixed_energy_level_tes(model, time):
        return model.energy_tes[time] == model.capacity_tes/2

    def charging_tes(model, time):
        if time == 0:
            energy_pre = model.capacity_tes/2
        else:
            energy_pre = model.energy_tes[time-1]
        if model.use_binaries_hp:
            charging = model.y_charge_tes[time] * model.charging_tes[time]
            discharging = model.y_discharge_tes[time] * model.discharging_tes[time]
        else:
            charging = model.charging_tes[time]
            discharging = model.discharging_tes[time]
        return model.energy_tes[time] == \
            model.efficiency_static_tes * energy_pre + \
            model.efficiency_dynamic_tes * charging * \
            (pd.to_timedelta(model.time_increment) / pd.to_timedelta('1h')) - \
            discharging * \
            (pd.to_timedelta(model.time_increment) / pd.to_timedelta('1h'))

    def charge_discharge_tes_binaries(model, time):
        return model.y_charge_tes[time] + model.y_discharge_tes[time] <= 1
    # save fix parameters
    model.capacity_tes = capacity_tes
    model.efficiency_static_tes = efficiency_static_tes
    model.efficiency_dynamic_tes = efficiency_dynamic_tes
    model.p_nom_hp = p_nom_hp
    model.cop = cop
    model.heat_demand = heat_demand
    model.use_binaries_hp = use_binaries
    model.use_linear_penalty_tes = use_linear_penalty
    if use_binaries and use_linear_penalty:
        print("Both binaries and linear penalty have been set to True for hp model. "
              "Binaries are used in this case.")
    if use_linear_penalty:
        model.weight_hp = kwargs.get("weight_hp")
    # set up variables
    model.charging_hp_th = pm.Var(model.time_set, bounds=(0, p_nom_hp))
    model.charging_hp_el = pm.Var(model.time_set)
    model.energy_tes = pm.Var(model.time_set, bounds=(0, capacity_tes))
    model.charging_tes = pm.Var(model.time_set, bounds=(0, None))
    model.discharging_tes = pm.Var(model.time_set, bounds=(0, None))
    if use_binaries is True:
        model.y_charge_tes = pm.Var(
            model.time_set,
            within=pm.Binary,
            doc='Binary defining for each timestep t if TES is charging'
        )
        model.y_discharge_tes = pm.Var(
            model.time_set,
            within=pm.Binary,
            doc='Binary defining for each timestep t if TES is discharging'
        )
    # add constraints
    model.EnergyConversionHP = pm.Constraint(model.time_set, rule=energy_conversion_hp)
    model.EnergyBalanceHPTES = pm.Constraint(model.time_set, rule=energy_balance_hp_tes)
    model.FixedEnergyTES = pm.Constraint(model.times_fixed_soc, rule=fixed_energy_level_tes)
    model.ChargingTES = pm.Constraint(model.time_set, rule=charging_tes)
    if use_binaries:
        model.NoSimultaneousChargingAndDischargingTES = pm.Constraint(
            model.time_set, rule=charge_discharge_tes_binaries)
    return model


def add_heat_pump_model_cells(
        model, p_nom_hp, capacity_tes, cop, heat_demand, efficiency_static_tes=0.99,
        efficiency_dynamic_tes=0.95, use_binaries=False):

    def energy_conversion_hp(model, cell, time):
        return model.charging_hp_el[cell, time] * cop.loc[model.timeindex[time]] == \
            model.charging_hp_th[cell, time]

    def energy_balance_hp_tes(model, cell, time):
        if model.use_binaries_hp:
            charging = model.y_charge_tes[cell, time] * model.charging_tes[cell, time]
            discharging = model.y_discharge_tes[cell, time] * \
                          model.discharging_tes[cell, time]
        else:
            charging = model.charging_tes[cell, time]
            discharging = model.discharging_tes[cell, time]
        return model.charging_hp_th[cell, time] == \
               model.heat_demand.loc[model.timeindex[time], cell] + \
               charging - discharging

    def fixed_energy_level_tes(model, cell, time):
        return model.energy_tes[cell, time] == model.capacity_tes[cell]/2

    def charging_tes(model, cell, time):
        if time == 0:
            energy_pre = model.capacity_tes[cell]/2
        else:
            energy_pre = model.energy_tes[cell, time-1]
        if model.use_binaries_hp:
            charging = model.y_charge_tes[cell, time] * model.charging_tes[cell, time]
            discharging = model.y_discharge_tes[cell, time] * \
                          model.discharging_tes[cell, time]
        else:
            charging = model.charging_tes[cell, time]
            discharging = model.discharging_tes[cell, time]
        return model.energy_tes[cell, time] == \
            model.efficiency_static_tes * energy_pre + \
            model.efficiency_dynamic_tes * charging * \
            (pd.to_timedelta(model.time_increment) / pd.to_timedelta('1h')) - \
            discharging * \
            (pd.to_timedelta(model.time_increment) / pd.to_timedelta('1h'))

    def charge_discharge_tes_binaries(model, cell, time):
        return model.y_charge_tes[cell, time] + model.y_discharge_tes[cell, time] <= 1
    # save fix parameters
    model.capacity_tes = capacity_tes
    model.efficiency_static_tes = efficiency_static_tes
    model.efficiency_dynamic_tes = efficiency_dynamic_tes
    model.p_nom_hp = p_nom_hp
    model.cop = cop
    model.heat_demand = heat_demand
    model.use_binaries_hp = use_binaries
    # set up variables
    model.charging_hp_th = pm.Var(model.cells_set, model.time_set,
                                  bounds=lambda m, c, t: (0, p_nom_hp[c]))
    model.charging_hp_el = pm.Var(model.cells_set, model.time_set)
    model.shedding_hp_el = pm.Var(model.cells_set, model.time_set, bounds=(0, None))
    model.energy_tes = pm.Var(model.cells_set, model.time_set,
                              bounds=lambda m, c, t: (0, capacity_tes[c]))
    model.charging_tes = pm.Var(model.cells_set, model.time_set, bounds=(0, None))
    model.discharging_tes = pm.Var(model.cells_set, model.time_set, bounds=(0, None))
    if use_binaries is True:
        model.y_charge_tes = pm.Var(
            model.cells_set,
            model.time_set,
            within=pm.Binary,
            doc='Binary defining for each timestep t if TES is charging'
        )
        model.y_discharge_tes = pm.Var(
            model.cells_set,
            model.time_set,
            within=pm.Binary,
            doc='Binary defining for each timestep t if TES is discharging'
        )
    # add constraints
    model.EnergyConversionHP = \
        pm.Constraint(model.cells_set, model.time_set, rule=energy_conversion_hp)
    model.EnergyBalanceHPTES = \
        pm.Constraint(model.cells_set, model.time_set, rule=energy_balance_hp_tes)
    model.FixedEnergyTES = \
        pm.Constraint(model.cells_set, model.times_fixed_soc,
                      rule=fixed_energy_level_tes)
    model.ChargingTES = \
        pm.Constraint(model.cells_set, model.time_set, rule=charging_tes)
    if use_binaries:
        model.NoSimultaneousChargingAndDischargingTES = pm.Constraint(
            model.cells_set, model.time_set, rule=charge_discharge_tes_binaries)
    return model


def add_hp_energy_level(model, mode="minimize", energy_consumption=None):
    """
    Method to add overall energy level. Used to determine maximum and minimum energy
    band. The energy is the cumulated consumed electrical energy.

    :param model: Model with already added heat pump model
    :param mode: str
        Can be "minimize" or "maximize", determines which band to calculate, the lower
        energy band ("minimize") or the upper energy band ("maximize")
    :return: updated model
    """
    def initial_cumulated_energy_level(model, time):
        """
        Set initial energy level to 0
        """
        return model.energy_level[time] == 0

    def cumulated_energy_level(model, time):
        """
        Define charging of energy level
        """
        return model.energy_level[time] == model.energy_level[time-1] + \
               model.charging_hp_el[time] * \
               (pd.to_timedelta(model.time_increment) / pd.to_timedelta('1h'))

    def energy_level_end(model, time):
        """
        Fixing overall energy consumption
        """
        return model.energy_level[time] == energy_consumption

    def energy_level_hp(model):
        """
        Objective
        """
        return sum(model.energy_level[time] for time in model.time_set)
    # add new variable
    model.energy_level = pm.Var(model.time_set, bounds=(0, None))
    # add new constraints
    model.InitialEnergyLevel = pm.Constraint([model.time_set.at(1)],
                                             rule=initial_cumulated_energy_level)
    model.EnergyLevel = pm.Constraint(model.time_non_zero, rule=cumulated_energy_level)
    if energy_consumption is not None:
        model.EnergyLevelEnd = pm.Constraint([model.time_set.at(-1)],
                                             rule=energy_level_end)
    # add objective
    model.objective = pm.Objective(rule=energy_level_hp,
                                   sense=getattr(pm, mode),
                                   doc='Define objective function')
    return model


def reduced_operation(model):
    return sum(model.charging_hp_el[time]**2 for time in model.time_set)


# def abs_objective(model):
#     return sum(model.charging_tes[time]*model.charging_hp_el[time] for time in model.time_set)


if __name__ == "__main__":
    solver = "gurobi"
    nr_hp = 20*1e6
    objective = "maximize_energy_level"
    heat_demand = pd.read_csv(r"C:\Users\aheider\Documents\Software\Semester Project Scripts\Scripts and Data\grids\176\hp_heat_2011.csv",
                              header=None)
    # heat_demand = pd.concat([heat_demand_orig[5500:], heat_demand_orig[:5500]])
    # heat_demand.index = heat_demand_orig.index
    cop = pd.read_csv(r'C:\Users\aheider\Documents\Software\Semester Project Scripts\Scripts and Data\grids\176\COP_2011.csv')["COP 2011"]
    # cop = pd.concat([cop_orig[5500:], cop_orig[:5500]])
    # cop.index = cop_orig.index
    hp_data = scenario_input_hps()
    capacity_tes = nr_hp * hp_data["capacity_single_tes"] # MWh
    p_nom_hp = nr_hp * hp_data["p_nom_single_hp"] # MW
    model = pm.ConcreteModel()
    model.time_set = pm.RangeSet(0, len(heat_demand) - 1)
    model.time_non_zero = model.time_set - [model.time_set.at(1)]
    model.times_fixed_soc = pm.Set(initialize=[model.time_set.at(1),
                                               model.time_set.at(-1)])
    model.time_increment = pd.to_timedelta('1h')
    model = add_heat_pump_model(model, p_nom_hp, capacity_tes, cop, heat_demand)
    if objective == "reduced_operation":
        model.objective = pm.Objective(rule=reduced_operation,
                                       sense=pm.minimize,
                                       doc='Define objective function')
    elif (objective == "minimize_energy_level") or \
        (objective == "maximize_energy_level"):
        if "minimize" in objective:
            model = add_hp_energy_level(model, "minimize")#, energy_consumption=10639.094336
        elif "maximize" in objective:
            model = add_hp_energy_level(model, "maximize")#, energy_consumption=10639.094336
        else:
            raise ValueError("Define whether energy level should be minimized or "
                             "maximised.")
    opt = pm.SolverFactory(solver)
    results = opt.solve(model, tee=True)
    results_df = pd.DataFrame()
    results_df["charging_hp"] = pd.Series(model.charging_hp_el.extract_values())
    results_df["charging_tes"] = pd.Series(model.charging_tes.extract_values())
    results_df["energy_tes"] = pd.Series(model.energy_tes.extract_values())
    if (objective == "minimize_energy_level") or \
         (objective == "maximize_energy_level"):
        results_df["energy_level"] = pd.Series(model.energy_level.extract_values())
        results_df["energy_level"].to_csv(f"data/{objective}.csv")
    print("SUCCESS")