import pyomo.environ as pm
import pandas as pd


def add_battery_storage_model(
    model,
    p_nom_bess,
    capacity_bess,
    efficiency_charging_bess=0.98,
    efficiency_discharging_bess=0.98,
    use_binaries=True,
):
    def charging_bess(model, time):
        if time == 0:
            energy_pre = model.capacity_bess / 2
        else:
            energy_pre = model.energy_bess[time - 1]
        if model.use_binaries_bess:
            charging = model.y_charge_bess[time] * model.charging_bess[time]
            discharging = model.y_discharge_bess[time] * model.discharging_bess[time]
        else:
            charging = model.charging_bess[time]
            discharging = model.discharging_bess[time]
        return model.energy_bess[time] == \
               energy_pre + \
               model.efficiency_charging_bess * charging * \
               (pd.to_timedelta(model.time_increment) / pd.to_timedelta('1h')) - \
               model.efficiency_discharging_bess * discharging * \
               (pd.to_timedelta(model.time_increment) / pd.to_timedelta('1h'))

    def fixed_energy_level_bess(model, time):
        return model.energy_bess[time] == model.capacity_bess/2

    def charge_discharge_bess_binaries(model, time):
        return model.y_charge_bess[time] + model.y_discharge_bess[time] <= 1

    if not use_binaries:
        raise NotImplementedError("BESS model only implemented using binaries.")
    # save fix parameters
    model.capacity_bess = capacity_bess
    model.p_nom_bess = p_nom_bess
    model.efficiency_charging_bess = efficiency_charging_bess
    model.efficiency_discharging_bess = efficiency_discharging_bess
    model.use_binaries_bess = use_binaries
    # set up variables
    model.charging_bess = pm.Var(model.time_set, bounds=(0, p_nom_bess))
    model.discharging_bess = pm.Var(model.time_set, bounds=(0, p_nom_bess))
    model.energy_bess = pm.Var(model.time_set, bounds=(0, capacity_bess))
    if use_binaries is True:
        model.y_charge_bess = pm.Var(
            model.time_set,
            within=pm.Binary,
            doc='Binary defining for each timestep t if TES is charging'
        )
        model.y_discharge_bess = pm.Var(
            model.time_set,
            within=pm.Binary,
            doc='Binary defining for each timestep t if TES is discharging'
        )
    # add constraints
    model.ChargingBESS = pm.Constraint(
        model.time_set, rule=charging_bess
    )
    model.FixedEnergyBESS = \
        pm.Constraint(model.times_fixed_soc, rule=fixed_energy_level_bess)
    if use_binaries:
        model.NoSimultaneousChargingAndDischargingBESS = pm.Constraint(
            model.time_set, rule=charge_discharge_bess_binaries)
    return model
