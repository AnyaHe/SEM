import pyomo.environ as pm


from storage_equivalent.ev_model import add_evs_model_cells
from storage_equivalent.heat_pump_model import add_heat_pump_model_cells


def add_dg_model(model, dg_names, flexible_evs=True, flexible_hps=True, **kwargs):
    """
    Adds distribution grids to existing model. It can be defined whether electric
    vehicles and heat pumps should be included.

    Parameters
    ----------
    model: pyomo optimisation model
    dg_names: list of str
    flexible_evs: bool
        Determines whether flexible EVs should be included in the DGs. Default: True
    flexible_hps: bool
        Determines whether flexible HPs should be included in the DGs. Default: True
    **kwargs: dict


    Returns
    -------

    """
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
            use_binaries=kwargs.get("use_binaries") and kwargs.get("v2g")
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
            use_binaries=kwargs.get("use_binaries")
        )
    # add grid constraints

    return model

