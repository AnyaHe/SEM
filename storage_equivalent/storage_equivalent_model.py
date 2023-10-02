import pyomo.environ as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from quantification.flexibility_quantification import shifting_time
from scenario_input import get_new_residual_load
from storage_equivalent.heat_pump_model import add_heat_pump_model
from storage_equivalent.ev_model import add_evs_model


def set_up_base_model(scenario_dict, new_res_load):
    """
    Method to set up base optimisation model

    :param scenario_dict:
    :param new_res_load:
    :return:
    """
    model = pm.ConcreteModel()
    model.timeindex = scenario_dict["ts_timesteps"]
    model.time_set = pm.RangeSet(0, len(new_res_load) - 1)
    model.time_increment = pd.to_timedelta(scenario_dict["time_increment"])
    model.times_fixed_soc = pm.Set(initialize=[model.time_set.at(-1)])
    model.weighting = scenario_dict["weighting"]
    return model


def get_power_flexible_technologies(model, time, cells=None):
    """
    Method to get charging powers of flexible sector coupling technologies at specific
    time step t

    Parameters
    ----------
    model
    time
    cells

    Returns
    -------

    """
    if hasattr(model, "charging_hp_el"):
        if cells is not None:
            hp_el = \
                sum(model.charging_hp_el[cell, time] for cell in cells)
        else:
            hp_el = model.charging_hp_el[time]
        if hasattr(model, "shedding_hp_el"):
            hp_el -= sum(model.shedding_hp_el[cell, time] for cell in cells)
    else:
        hp_el = 0
    if hasattr(model, "charging_ev"):
        if hasattr(model, "discharging_ev"):
            if model.use_binaries_ev:
                if cells is not None:
                    discharging_ev = sum(model.y_discharge_ev[cp, cell, time] *
                                         model.discharging_ev[cp, cell, time]
                                         for cp in model.charging_points_set
                                         for cell in cells)
                else:
                    discharging_ev = sum(model.y_discharge_ev[cp, time] *
                                         model.discharging_ev[cp, time]
                                         for cp in model.charging_points_set)
            else:
                if cells is not None:
                    discharging_ev = sum(model.discharging_ev[cp, cell, time]
                                         for cp in model.charging_points_set
                                         for cell in cells)
                else:
                    discharging_ev = sum(model.discharging_ev[cp, time]
                                         for cp in model.charging_points_set)
        else:
            discharging_ev = 0
        if model.use_binaries_ev:
            if cells is not None:
                charging_ev = sum(
                    [model.y_charge_ev[cp, cell, time] *
                     model.charging_ev[cp, cell, time]
                     for cp in model.charging_points_set
                     for cell in cells])
            else:
                charging_ev = sum([model.y_charge_ev[cp, time] *
                                   model.charging_ev[cp, time]
                                   for cp in model.charging_points_set])
        else:
            if cells is not None:
                charging_ev = sum([model.charging_ev[cp, cell, time]
                                   for cp in model.charging_points_set
                                   for cell in cells])
            else:
                charging_ev = sum([model.charging_ev[cp, time]
                                   for cp in model.charging_points_set])

        if hasattr(model, "shedding_ev"):
            shed_ev = sum(
                [model.shedding_ev[cp, cell, time]
                 for cp in model.charging_points_set
                 for cell in cells])
        else:
            shed_ev = 0
        ev = charging_ev - shed_ev - discharging_ev
    else:
        ev = 0
    return ev, hp_el


def add_storage_equivalent_model(model, residual_load, **kwargs):
    def fix_energy_levels(model, time_horizon, time):
        return model.energy_levels[time_horizon, time] == 0

    def charge_storages(model, time_horizon, time):
        if time == 0:
            energy_levels_pre = 0
        else:
            energy_levels_pre = model.energy_levels[time_horizon, time - 1]
        return model.energy_levels[time_horizon, time] == energy_levels_pre + \
               model.charging[time_horizon, time] * \
               (pd.to_timedelta(model.time_increment) / pd.to_timedelta('1h'))

    def meet_residual_load(model, time):
        if hasattr(model, "cells_set"):
            cells = model.cells_set
        else:
            cells = None
        ev, hp_el = get_power_flexible_technologies(model, time, cells)
        return sum(model.charging[time_horizon, time] for time_horizon in
                   model.time_horizons_set) + \
               model.residual_load.iloc[time] + hp_el + ev - model.shedding[time] + \
               model.spilling[time] == 0

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
        return model.abs_charging[time_horizon, time] >= \
               model.charging[time_horizon, time]

    def abs_charging_down(model, time_horizon, time):
        return model.abs_charging[time_horizon, time] >= \
               -model.charging[time_horizon, time]

    def fixed_shifted_energy(model, time_horizon):
        return sum(model.discharging[time_horizon, time]
                   for time in model.time_set) <= \
            model.fixed_shifted_energy[time_horizon]

    def discharging_up(model, time_horizon, time):
        return model.discharging[time_horizon, time] >= 0

    def discharging_down(model, time_horizon, time):
        return model.discharging[time_horizon, time] >= \
               -model.charging[time_horizon, time]

    # save fix parameters
    model.residual_load = residual_load
    model.time_horizons = kwargs.get("time_horizons", [24, 7*24, 28*24, 24*366])
    model.coeff_min = kwargs.get("coeff_min", [0.25, 0.5, 1, 2])
    model.coeff_max = kwargs.get("coeff_max", [8, 12, 48, 96])
    model.weight_slacks = kwargs.get("weights_slacks", 1e3)
    # add time horizon set
    model.time_horizons_set = pm.RangeSet(0, len(model.time_horizons)-1)
    # set up variables
    # model.caps_pos = pm.Var(model.time_horizons_set)
    # model.caps_neg = pm.Var(model.time_horizons_set)
    model.energy_levels = pm.Var(model.time_horizons_set, model.time_set)
    model.charging = pm.Var(model.time_horizons_set, model.time_set)
    # model.charging_max = pm.Var(model.time_horizons_set)
    # model.abs_charging = pm.Var(model.time_horizons_set, model.time_set)
    model.discharging = pm.Var(model.time_horizons_set, model.time_set)
    model.shedding = pm.Var(model.time_set, bounds=(0, None))
    model.spilling = pm.Var(model.time_set, bounds=(0, None))
    # add constraints
    for time_horizon in model.time_horizons_set:
        times = []
        for time in model.time_set:
            if time % model.time_horizons[time_horizon] == 0:
                times.append(time)
        setattr(model, "FixEnergyLevels{}".format(time_horizon), pm.Constraint([time_horizon], times,
                                              rule=fix_energy_levels))
    model.ChargingStorages = pm.Constraint(model.time_horizons_set, model.time_set,
                                          rule=charge_storages)
    model.ResidualLoad = pm.Constraint(model.time_set, rule=meet_residual_load)
    # model.MaximumCharging = pm.Constraint(model.time_horizons_set, model.time_set,
    #                                       rule=maximum_charging)
    # model.MaximumCapacity = pm.Constraint(model.time_horizons_set, model.time_set,
    #                                       rule=maximum_capacity)
    # model.MinimumCapacity = pm.Constraint(model.time_horizons_set, model.time_set,
    #                                       rule=minimum_capacity)
    # model.UpperChargingCapRatio = pm.Constraint(model.time_horizons_set,
    #                                             rule=charging_cap_ratio_upper)
    # model.LowerChargingCapRatio = pm.Constraint(model.time_horizons_set,
    #                                             rule=charging_cap_ratio_lower)
    model.UpperDischarging = pm.Constraint(model.time_horizons_set, model.time_set,
                                           rule=discharging_up)
    model.LowerDischarging = pm.Constraint(model.time_horizons_set, model.time_set,
                                           rule=discharging_down)
    # optional: add constraint of shifted energy for single storage types
    model.fixed_shifted_energy = kwargs.get("fixed_shifted_energy", None)
    if model.fixed_shifted_energy is not None:
        model.FixedShiftedEnergy = pm.Constraint(
            model.fixed_shifted_energy.index, rule=fixed_shifted_energy)
    return model


def add_storage_equivalents_model(model, residual_load, connections, flows, **kwargs):
    def fix_energy_levels(model, cell, time_horizon, time):
        return model.energy_levels[cell, time_horizon, time] == 0

    def charge_storages(model, cell, time_horizon, time):
        if time == 0:
            energy_levels_pre = 0
        else:
            energy_levels_pre = model.energy_levels[cell, time_horizon, time-1]
        return model.energy_levels[cell, time_horizon, time] == \
               energy_levels_pre + \
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
               sum(model.flows[(neighbor, cell), time] for neighbor in neg_flows) - \
               model.shedding[time] + \
               model.spilling[time] + \
               model.residual_load[cell].iloc[time] == 0

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

    def discharging_up(model, cell, time_horizon, time):
        return model.discharging[cell, time_horizon, time] >= 0

    def discharging_down(model, cell, time_horizon, time):
        return model.discharging[cell, time_horizon, time] >= \
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
    # model.caps_pos = pm.Var(model.cells_set, model.time_horizons_set)
    # model.caps_neg = pm.Var(model.cells_set, model.time_horizons_set)
    model.energy_levels = pm.Var(model.cells_set, model.time_horizons_set, model.time_set)
    model.charging = pm.Var(model.cells_set, model.time_horizons_set, model.time_set)
    # model.charging_max = pm.Var(model.cells_set, model.time_horizons_set)
    # model.abs_charging = pm.Var(model.cells_set, model.time_horizons_set, model.time_set)
    model.flows = pm.Var(model.flows_set, model.time_set)
    model.shedding = pm.Var(model.time_set, bounds=(0, None))
    model.spilling = pm.Var(model.time_set, bounds=(0, None))
    # add constraints
    for time_horizon in model.time_horizons_set:
        times = []
        for time in model.time_set:
            if time % model.time_horizons[time_horizon] == 0:
                times.append(time)
        setattr(model, f"FixEnergyLevels{time_horizon}",
                pm.Constraint(model.cells_set, [time_horizon], times,
                              rule=fix_energy_levels))
    model.ChargingStorages = pm.Constraint(model.cells_set, model.time_horizons_set, model.time_set,
                                           rule=charge_storages)
    model.ResidualLoad = pm.Constraint(model.cells_set, model.time_set, rule=meet_residual_load)
    # model.MaximumCharging = pm.Constraint(model.cells_set, model.time_horizons_set, model.time_set,
    #                                       rule=maximum_charging)
    # model.MaximumCapacity = pm.Constraint(model.cells_set, model.time_horizons_set, model.time_set,
    #                                       rule=maximum_capacity)
    # model.MinimumCapacity = pm.Constraint(model.cells_set, model.time_horizons_set, model.time_set,
    #                                       rule=minimum_capacity)
    # model.UpperChargingCapRatio = pm.Constraint(model.cells_set, model.time_horizons_set,
    #                                             rule=charging_cap_ratio_upper)
    # model.LowerChargingCapRatio = pm.Constraint(model.cells_set, model.time_horizons_set,
    #                                             rule=charging_cap_ratio_lower)
    model.UpperAbsCharging = pm.Constraint(model.cells_set, model.time_horizons_set, model.time_set,
                                           rule=abs_charging_up)
    model.LowerAbsCharging = pm.Constraint(model.cells_set, model.time_horizons_set, model.time_set,
                                           rule=abs_charging_down)
    return model


def get_slacks(model):
    # extract slack for simultaneous charging and discharging evs
    if hasattr(model, "discharging_ev") and not model.use_binaries_ev:
        if model.use_linear_penalty_ev:
            slack_ev = sum(
                model.charging_ev[cp, time] + model.discharging_ev[cp, time]
                for cp in model.charging_points_set for time in model.time_set
            ) / model.weight_slacks * model.weight_ev
        else:
            slack_ev = sum(
                model.charging_ev[cp, time]*model.discharging_ev[cp, time]
                for cp in model.charging_points_set for time in model.time_set
            )
    else:
        slack_ev = 0
    # extract shedding ev if dg constraints are present
    if hasattr(model, "shedding_ev"):
        shed_ev = sum(
            [model.shedding_ev[cp, cell, time]
             for cp in model.charging_points_set
             for cell in model.cells_set for time in model.time_set])
    else:
        shed_ev = 0
    # extract slack for simultaneous charging and discharging tes
    if hasattr(model, "discharging_tes") and not model.use_binaries_hp:
        if model.use_linear_penalty_tes:
            slack_tes = sum(
                model.charging_tes[time] + model.discharging_tes[time]
                for time in model.time_set
            ) / model.weight_slacks * model.weight_hp
        else:
            slack_tes = sum(
                model.charging_tes[time]*model.discharging_tes[time]
                for time in model.time_set
            )
    else:
        slack_tes = 0
        # extract shedding ev if dg constraints are present
    if hasattr(model, "shedding_hp_el"):
        shed_hp = sum(
            [model.shedding_hp_el[cell, time]
             for cell in model.cells_set for time in model.time_set])
    else:
        shed_hp = 0
    return sum(model.spilling[time] + model.shedding[time]
               for time in model.time_set) + slack_ev + slack_tes + shed_ev + shed_hp


def minimize_cap(model):
    # todo: determine good weighting factor
    slacks = get_slacks(model) * model.weight_slacks
    return sum(model.weighting[time_horizon] * (model.caps_pos[time_horizon] +
                                                model.caps_neg[time_horizon])
               for time_horizon in model.time_horizons_set) + slacks


def minimize_energy(model):
    # todo: determine good weighting factor
    slacks = get_slacks(model) * model.weight_slacks
    return sum(model.weighting[time_horizon] * sum(model.abs_charging[time_horizon, time]
                                                   for time in model.time_set)
               for time_horizon in model.time_horizons_set) + slacks


def minimize_discharging(model):
    # todo: determine good weighting factor
    slacks = get_slacks(model) * model.weight_slacks
    return sum(model.weighting[time_horizon] * sum(model.discharging[time_horizon, time]
                                                   for time in model.time_set)
               for time_horizon in model.time_horizons_set) + slacks


def minimize_energy_and_power(model):
    # todo: determine good weighting factor
    slacks = get_slacks(model) * model.weight_slacks
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
        for time_horizon in model.time_horizons_set)+1e-9*(hp_el+ev)+slacks


def minimize_energy_multi(model):
    # todo: determine good weighting factor
    slacks = get_slacks(model) * 1e6
    return sum(sum(model.weighting[time_horizon] *
                   sum(model.abs_charging[cell, time_horizon, time]
                       for time in model.time_set)
               for time_horizon in model.time_horizons_set) for
               cell in model.cells_set) + slacks


def determine_storage_durations(charging, index="duration"):
    """
    Method to determine medium storage duration of stored energy.

    :param charging: pd.DataFrame
        Charging timeseries of the different storage types. Index should be
        datetimeindex and columns storage types. The charging time series of a storage
        type should amount to 0 in total
    :return:
    """

    def get_mean_shifting_time(sdi):
        """
        Method to extract mean storage duration
        :param sdi: see output shifting time
        :return:
        """
        sdi["storage_duration_numerical"] = sdi.storage_duration.divide(
            pd.to_timedelta("1h"))
        if sdi.energy_shifted.abs().sum() > 0:
            mean_time_shift = \
                (sdi.storage_duration_numerical * sdi.energy_shifted.abs()).sum() / \
                sdi.energy_shifted.abs().sum()
        else:
            mean_time_shift = 0
        return mean_time_shift * pd.to_timedelta("1h")
    if (charging.sum() > 1e-5).any():
        print("Warning: charging time series do not amount to 0.")
    if type(index) in [str, float, int]:
        index = [index]
    storage_durations = pd.DataFrame(index=index, columns=charging.columns)
    for storage_type in charging.columns:
        sdi = shifting_time(charging[storage_type], reference_curve=0)
        storage_durations.loc[index, storage_type] = get_mean_shifting_time(sdi)
    return storage_durations


def solve_model(model, solver, hp_mode=None, ev_mode=None, ev_v2g=False,
                allow_spill_and_shed=False):
    np.random.seed(int(time.time()))
    opt = pm.SolverFactory(solver)
    if solver == "gurobi":
        opt.options["Seed"] = int(time.time())
        opt.options["Method"] = 3
        if (hp_mode == "flexible") or ev_v2g:
            opt.options["NonConvex"] = 2
    opt.solve(model, tee=True)
    # check that no simultaneous charging and discharging occurs for v2g
    if (ev_mode == "flexible") & ev_v2g:
        charging_ev = pd.Series(model.charging_ev.extract_values()).unstack().T
        discharging_ev = pd.Series(model.discharging_ev.extract_values()).unstack().T
        if model.use_binaries_ev:
            y_charge_ev = pd.Series(model.y_charge_ev.extract_values()).unstack().T
            y_discharge_ev = pd.Series(model.y_discharge_ev.extract_values()).unstack().T
            prefactor = y_charge_ev.multiply(y_discharge_ev)
        else:
            prefactor = 1
        if charging_ev.multiply(discharging_ev).multiply(prefactor).sum().sum() > 1e-3:
            raise ValueError("Simultaneous charging and discharging of EVs. "
                             "Please check.")
    # check that no simultaneous charging and discharging of TES occurs
    if hp_mode == "flexible":
        charging_tes = pd.Series(model.charging_tes.extract_values())
        discharging_tes = pd.Series(model.discharging_tes.extract_values())
        if model.use_binaries_hp:
            charging_tes *= pd.Series(model.y_charge_tes.extract_values())
            discharging_tes *= pd.Series(model.y_discharge_tes.extract_values())
        if charging_tes.multiply(discharging_tes).sum() > 1e-3:
            raise ValueError("Simultaneous charging and discharging of TES. "
                             "Please check.")
    # extract results
    slacks = pd.Series(model.spilling.extract_values()) + \
             pd.Series(model.shedding.extract_values())
    if slacks.sum() > 1e-9:
        if allow_spill_and_shed:
            print(f"Info: {pd.Series(model.spilling.extract_values())} of spilling and "
                  f"{pd.Series(model.shedding.extract_values())} of shedding are being "
                  f"used.")
        else:
            raise ValueError("Slacks are being used. Please check. Consider increasing "
                             "weights.")
    return model


def get_balanced_storage_equivalent_model(scenario_dict, max_iter=3, tol=1e-2,
                                          **kwargs):
    """
    Method to set up base model which is balanced in terms of energy consumption
    and supply. Necessary because of losses of TES and V2G.

    :param scenario_dict:
    :param max_iter:
    :param tol:
    :param kwargs:
    :return:
    """
    # initialise values
    energy_balanced = False
    iter_a = 0

    ref_charging = kwargs.get("ref_charging", None)
    sum_energy_heat = kwargs.get("sum_energy_heat", 0)
    energy_ev = kwargs.get("energy_ev", 0)

    while (not energy_balanced) & (iter_a < max_iter):
        print(f"Info: Starting iteration {iter_a} for energy balance.")
        # determine new residual load
        new_res_load = get_new_residual_load(
            scenario_dict=scenario_dict,
            sum_energy_heat=sum_energy_heat,
            energy_ev=energy_ev,
            ref_charging=ref_charging,
            ts_heat_el=kwargs.get("ts_heat_el", None),
            share_pv=kwargs.get("share_pv", None),
            share_gen_to_load=kwargs.get("share_gen_to_load", 1.0)
        )

        # initialise base model
        model = set_up_base_model(scenario_dict=scenario_dict,
                                  new_res_load=new_res_load)
        # add heat pump model if flexible
        if scenario_dict["hp_mode"] == "flexible":
            model = add_heat_pump_model(
                model=model,
                p_nom_hp=kwargs["p_nom_hp"],
                capacity_tes=kwargs["capacity_tes"],
                cop=scenario_dict["ts_cop"],
                heat_demand=kwargs["ts_heat_demand"],
                efficiency_static_tes=scenario_dict["efficiency_static_tes"],
                efficiency_dynamic_tes=scenario_dict["efficiency_dynamic_tes"],
                use_binaries=scenario_dict["hp_use_binaries"]
            )
        # add ev model if flexible
        if scenario_dict["ev_mode"] == "flexible":
            add_evs_model(
                model=model,
                flex_bands=kwargs["flexibility_bands"],
                v2g=scenario_dict["ev_v2g"],
                efficiency=scenario_dict["ev_charging_efficiency"],
                discharging_efficiency=scenario_dict["ev_discharging_efficiency"],
                use_binaries=scenario_dict["ev_use_binaries"]
            )
        # add storage equivalents
        model = add_storage_equivalent_model(
            model, new_res_load,
            time_horizons=scenario_dict["time_horizons"],
            fixed_shifted_energy=kwargs.get("fixed_shifted_energy"))
        # define objective
        model.objective = pm.Objective(rule=globals()[scenario_dict["objective"]],
                                       sense=pm.minimize,
                                       doc='Define objective function')
        model = solve_model(
            model=model,
            solver=scenario_dict["solver"],
            hp_mode=scenario_dict["hp_mode"],
            ev_mode=scenario_dict["ev_mode"],
            ev_v2g=scenario_dict.get("ev_v2g", False),
            allow_spill_and_shed=kwargs.get("allow_spill_and_shed", False)
        )
        # check that energy is balanced
        energy_hp_balanced = True
        if scenario_dict["hp_mode"] == "flexible":
            sum_energy_heat_opt = pd.Series(model.charging_hp_el.extract_values()).sum()
            if abs(sum_energy_heat_opt - sum_energy_heat) / sum_energy_heat > tol:
                energy_hp_balanced = False
                sum_energy_heat = sum_energy_heat_opt
        energy_ev_balanced = True
        if scenario_dict["ev_mode"] == "flexible":
            ev_operation = pd.Series(model.charging_ev.extract_values()).unstack().T
            if scenario_dict["ev_v2g"]:
                ev_operation -= pd.Series(model.discharging_ev.extract_values()).unstack().T
            energy_ev_opt = ev_operation.sum().sum() + ref_charging.sum()
            if abs(energy_ev_opt - energy_ev) / energy_ev > tol:
                energy_ev_balanced = False
                energy_ev = energy_ev_opt
        if energy_hp_balanced & energy_ev_balanced:
            energy_balanced = True
            print(f"Info: Energy balanced in iteration {iter_a}.")
        iter_a += 1
    if not energy_balanced:
        print(f"Warning: Energy not balanced after maximum of {max_iter} iterations.")
    return model, new_res_load


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
