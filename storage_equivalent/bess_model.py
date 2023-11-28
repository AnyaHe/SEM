import pandas as pd
import pyomo.environ as pm


def add_battery_storage_model_optimal_cells(
    model,
    storage_units
):
    def charging_bess(model, cell, time):
        """
        Constraint for battery charging
        """
        if time == 0:
            energy_level_bess_pre = model.battery_storage.loc[cell, "capacity"]/2
        else:
            energy_level_bess_pre = model.energy_level_bess[cell, time - 1]
        return model.energy_level_bess[cell, time] == energy_level_bess_pre + \
               model.charging_bess[cell, time] * (
                       pd.to_timedelta(model.time_increment) / pd.to_timedelta("1h")
               )

    def fixed_energy_level_bess(model, cell, time):
        """
        Method to fix energy level of bess to half the storage capacity
        """
        return model.energy_level_bess[cell, time] == \
               model.battery_storage.loc[cell, "capacity"]/2

    # check if efficiencies are 1.0, otherwise raise warning, since only these values
    # can be modelled with the implemented approach
    if "efficiency_store" in storage_units.columns:
        if (storage_units[
                ["efficiency_store", "efficiency_dispatch"]] < 1).any().any():
            print("Warning: Optimised storage units contain efficiencies smaller than "
                  "1.0.\nThe implemented approach is only valid for optimal storage "
                  "systems with efficiencies of 1.0.")
    # save fix parameters
    model.battery_storage = storage_units

    # set up variables
    model.energy_level_bess = pm.Var(
        model.cells_set,
        model.time_set,
        bounds=lambda m, cell, t: (0, m.battery_storage.loc[cell, "capacity"]),
    )

    model.charging_bess = pm.Var(
        model.cells_set,
        model.time_set,
        bounds=lambda m, cell, t: (-m.battery_storage.loc[cell, "p_nom"],
                                   m.battery_storage.loc[cell, "p_nom"]),
    )
    model.ChargingBESS = pm.Constraint(
        model.cells_set, model.time_set, rule=charging_bess
    )
    model.FixedEnergyBESS = pm.Constraint(
        model.cells_set, model.times_fixed_soc, rule=fixed_energy_level_bess
    )
    return model
