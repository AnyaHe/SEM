import pyomo.environ as pm
import pandas as pd

from storage_equivalent.storage_equivalent_model import get_power_flexible_technologies, \
    get_slacks
from scenario_input import shift_and_extend_ts_by_one_timestep


def add_price_model(model, residual_load, **kwargs):
    """
    Simulates buying from the market with time varying prices.

    Parameters
    ----------
    model
    residual_load
    kwargs

    Returns
    -------

    """
    def meet_residual_load(model, time):
        if hasattr(model, "cells_set"):
            cells = model.cells_set
        else:
            cells = None
        ev, hp_el, bess = get_power_flexible_technologies(model, time, cells)
        return model.residual_load.iloc[time] + hp_el + ev + bess == \
               model.grid_exchange[time]

    model.residual_load = residual_load
    model.weight_slacks = kwargs.get("weights_slacks", 1e3)
    prices = pd.read_csv(kwargs.get("price_path", "data/prices.csv"), index_col=0,
                         parse_dates=True, header=None)[1]
    prices = shift_and_extend_ts_by_one_timestep(
                        prices, model.time_increment,
                        value=prices.mean()).iloc[:len(model.timeindex)]
    prices.index = model.timeindex
    model.prices = prices
    # define grid exchange: positive buying from grid, negative selling to grid
    model.grid_exchange = pm.Var(model.time_set)
    model.ResidualLoad = pm.Constraint(model.time_set, rule=meet_residual_load)
    return model


def minimize_costs(model):
    slacks = get_slacks(model) * model.weight_slacks
    return sum(model.grid_exchange[time] * model.prices[model.timeindex[time]]
               for time in model.time_set) + slacks


def flatten_grid_import(model):
    slacks = get_slacks(model) * model.weight_slacks
    return sum(model.grid_exchange[time] ** 2
               for time in model.time_set) + slacks
