import pandas as pd
import pyomo.environ as pm


from storage_equivalent.ev_model import add_evs_model_cells
from storage_equivalent.heat_pump_model import add_heat_pump_model_cells
from storage_equivalent.storage_equivalent_model import get_power_flexible_technologies


def add_dg_model(model, dg_names, grid_powers,
                 flexible_evs=True, flexible_hps=True, **kwargs):
    """
    Adds distribution grids to existing model. It can be defined whether electric
    vehicles and heat pumps should be included.

    Parameters
    ----------
    model: pyomo optimisation model
    dg_names: list of str
    grid_powers: dict of Series or DataFrame, keys: "upper_power", "lower_power"
        When series scalar value for power, else time-dependent value
    flexible_evs: bool
        Determines whether flexible EVs should be included in the DGs. Default: True
    flexible_hps: bool
        Determines whether flexible HPs should be included in the DGs. Default: True
    **kwargs: dict


    Returns
    -------

    """
    def grid_upper(model, cell, time):
        """
        Constraint for maximum cumulated power of flexible units within a distribution
        grid

        Parameters
        ----------
        model
        cell
        time

        Returns
        -------

        """
        ev, hp_el = get_power_flexible_technologies(model, time, [cell])
        if isinstance(model.grid_powers["upper_power"], pd.Series):
            upper_power_grid_flex = model.grid_powers["upper_power"][cell]
        elif isinstance(model.grid_powers["upper_power"], pd.DataFrame):
            upper_power_grid_flex = model.grid_powers["upper_power"].iloc[time][cell]
        else:
            raise ValueError("Unexpected type of upper power grid.")
        return ev + hp_el <= upper_power_grid_flex

    def grid_lower(model, cell, time):
        """
        Constraint for minimum cumulated power of flexible units within a distribution
        grid

        Parameters
        ----------
        model
        cell
        time

        Returns
        -------

        """
        ev, hp_el = get_power_flexible_technologies(model, time, [cell])
        if isinstance(model.grid_powers["lower_power"], pd.Series):
            lower_power_grid_flex = model.grid_powers["lower_power"][cell]
        elif isinstance(model.grid_powers["lower_power"], pd.DataFrame):
            lower_power_grid_flex = model.grid_powers["lower_power"].iloc[time][cell]
        else:
            raise ValueError("Unexpected type of lower power grid.")
        return ev + hp_el >= lower_power_grid_flex
    # add new set with dgs
    model.cells_set = pm.Set(initialize=dg_names)
    # add ev model if flexible
    if flexible_evs:
        model = add_evs_model_cells(
            model=model,
            flex_use_cases=kwargs.get("flex_use_cases"),
            flex_bands=kwargs.get("flex_bands"),
            efficiency=kwargs.get("efficiency"),
            v2g=kwargs.get("v2g"),
            discharging_efficiency=kwargs.get("discharging_efficiency"),
            use_binaries=kwargs.get("use_binaries") and kwargs.get("v2g"),
            use_linear_penalty=kwargs.get("use_linear_penalty"),
            weight_ev=kwargs.get("weight_ev")
        )
    # add hp model if flexible
    if flexible_hps:
        model = add_heat_pump_model_cells(
            model=model,
            p_nom_hp=kwargs.get("p_nom_hp"),
            capacity_tes=kwargs.get("capacity_tes"),
            cop=kwargs.get("cop"),
            heat_demand=kwargs.get("heat_demand"),
            efficiency_static_tes=kwargs.get("efficiency_static_tes"),
            efficiency_dynamic_tes=kwargs.get("efficiency_dynamic_tes"),
            use_binaries=kwargs.get("use_binaries"),
            use_linear_penalty=kwargs.get("use_linear_penalty"),
            weight_hp=kwargs.get("weight_hp")
        )
    # add grid constraints
    model.grid_powers = grid_powers
    # add constraints
    if flexible_hps or flexible_evs:
        model.UpperPowerFlex = \
            pm.Constraint(model.cells_set, model.time_set, rule=grid_upper)
        model.LowerPowerFlex = \
            pm.Constraint(model.cells_set, model.time_set, rule=grid_lower)
    return model

