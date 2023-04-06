import pyomo.environ as pm
import pandas as pd
import numpy as np
import os

from data_preparation.data_preparation import determine_shifting_times_ev


def add_ev_model(model, flex_bands, charging_efficiency=0.9, v2g=False,
                 discharging_efficiency=0.9):
    def charging_ev(model, time):
        """
        Constraint for charging of EV that has to ly between the lower and upper
        energy band.

        :param model:
        :param charging_point:
        :param time:
        :return:
        """
        # get previous energy level
        if time == 0:
            energy_level_pre = \
                (model.flex_bands.loc[time, "lower"] +
                 model.flex_bands.loc[time, "upper"]) / 2
        else:
            energy_level_pre = model.energy_level_ev[time - 1]
        # get discharging is v2g is allowed
        if hasattr(model, "discharging_ev"):
            discharging = model.discharging_ev[time]
        else:
            discharging = 0
        # get time increment
        delta_t = (pd.to_timedelta(model.time_increment) / pd.to_timedelta('1h'))
        return model.energy_level_ev[time] == \
            energy_level_pre + \
            model.charging_efficiency * model.charging_ev[time] * delta_t - \
            discharging / model.discharging_efficiency * delta_t

    def fixed_energy_level(model, time):
        '''
        Constraint for fixed value of energy
        :param model:
        :param charging_point:
        :param time:
        :return:
        '''
        return model.energy_level_ev[time] == \
            (model.flex_bands.loc[time, "lower"] +
             model.flex_bands.loc[time, "upper"]) / 2
    # save fix parameters
    model.charging_efficiency = charging_efficiency
    model.discharging_efficiency = discharging_efficiency
    model.flex_bands = flex_bands
    # set up variables
    model.charging_ev = \
        pm.Var(model.time_set,
               bounds=lambda m, t:
               (0, m.flex_bands.loc[t, "power"]))
    if v2g:
        model.discharging_ev = \
            pm.Var(model.time_set,
                   bounds=lambda m, t:
                   (0, m.flex_bands.loc[t, "power"]))
    model.energy_level_ev = \
        pm.Var(model.time_set,
               bounds=lambda m, t:
               (m.flex_bands.loc[t, "lower"],
                m.flex_bands.loc[t, "upper"]))
    # add constraints
    model.EVCharging = pm.Constraint(model.time_set,  rule=charging_ev)
    model.FixedEVEnergyLevel = pm.Constraint(model.times_fixed_soc, rule=fixed_energy_level)
    return model


def add_evs_model(model, flex_bands, efficiency=0.9, v2g=False,
                  discharging_efficiency=0.9, use_binaries=False):
    def charging_ev(model, cp, time):
        """
        Constraint for charging of EV that has to ly between the lower and upper
        energy band.

        :param model:
        :param charging_point:
        :param time:
        :return:
        """
        # get previous energy level
        if time == 0:
            energy_level_pre = (model.flex_bands["lower_energy"].iloc[time][cp] +
                                model.flex_bands["upper_energy"].iloc[time][cp]) / 2
        else:
            energy_level_pre = model.energy_level_ev[cp, time - 1]
        # get discharging is v2g is allowed
        if hasattr(model, "discharging_ev"):
            if model.use_binaries_ev:
                discharging = model.y_discharge_ev[cp, time] * model.discharging_ev[cp, time]
            else:
                discharging = model.discharging_ev[cp, time]
        else:
            discharging = 0
        # get charging
        if model.use_binaries_ev:
            charging = model.y_charge_ev[cp, time] * model.charging_ev[cp, time]
        else:
            charging = model.charging_ev[cp, time]
        # get time increment
        delta_t = (pd.to_timedelta(model.time_increment) / pd.to_timedelta('1h'))
        return model.energy_level_ev[cp, time] == \
            energy_level_pre + \
            model.charging_efficiency_ev * charging * delta_t - \
            discharging / model.discharging_efficiency_ev * delta_t

    def fixed_energy_level(model, cp, time):
        '''
        Constraint for fixed value of energy
        :param model:
        :param charging_point:
        :param time:
        :return:
        '''
        return model.energy_level_ev[cp, time] == \
               (model.flex_bands["lower_energy"].iloc[time][cp] +
                model.flex_bands["upper_energy"].iloc[time][cp]) / 2

    def charge_discharge_ev_binaries(model, cp, time):
        return model.y_charge_ev[cp, time] + model.y_discharge_ev[cp, time] <= 1

    # save fix parameters
    model.charging_efficiency_ev = efficiency
    model.discharging_efficiency_ev = discharging_efficiency
    model.flex_bands = flex_bands
    model.use_binaries_ev = use_binaries
    # set up set
    model.charging_points_set = \
        pm.Set(initialize=model.flex_bands["lower_energy"].columns)
    # set up variables
    model.charging_ev = \
        pm.Var(model.charging_points_set, model.time_set,
               bounds=lambda m, cp, t:
               (0, m.flex_bands["upper_power"].iloc[t][cp]))
    if v2g:
        model.discharging_ev = \
            pm.Var(model.charging_points_set, model.time_set,
                   bounds=lambda m, cp, t:
                   (0, m.flex_bands["upper_power"].iloc[t][cp]))
        if use_binaries is True:
            model.y_charge_ev = pm.Var(
                model.charging_points_set,
                model.time_set,
                within=pm.Binary,
                doc='Binary defining for each car c and timestep t if it is charging'
            )
            model.y_discharge_ev = pm.Var(
                model.charging_points_set,
                model.time_set,
                within=pm.Binary,
                doc='Binary defining for each car c and timestep t if it is discharging'
            )
    model.energy_level_ev = \
        pm.Var(model.charging_points_set, model.time_set,
               bounds=lambda m, cp, t:
               (m.flex_bands["lower_energy"].iloc[t][cp],
                m.flex_bands["upper_energy"].iloc[t][cp]))
    # add constraints
    model.EVCharging = pm.Constraint(model.charging_points_set, model.time_set,
                                     rule=charging_ev)
    if use_binaries:
        model.NoSimultaneousChargingAndDischargingEV = pm.Constraint(
            model.charging_points_set, model.time_set,
            rule=charge_discharge_ev_binaries)
    model.FixedEVEnergyLevel = \
        pm.Constraint(model.charging_points_set, model.times_fixed_soc,
                      rule=fixed_energy_level)
    return model


def import_flexibility_bands(dir, efficiency=0.9):
    flexibility_bands = {}

    for band in ["upper_power", "upper_energy", "lower_energy"]:
        band_df = \
            pd.read_csv(dir+'/{}.csv'.format(band),
                        index_col=0, parse_dates=True, dtype=np.float32)
        if band_df.columns.duplicated().any():
            raise ValueError("Charging points with the same name in flexibility bands. "
                             "Please check")
        flexibility_bands[band] = band_df
        # remove numeric problems
        if "upper" in band:
            flexibility_bands[band] = flexibility_bands[band] + 1e-3
        elif "lower" in band:
            flexibility_bands[band] = flexibility_bands[band] - 1e-3
    # sanity check
    if ((flexibility_bands["upper_energy"] - flexibility_bands["lower_energy"]) < 1e-6).any().any():
        raise ValueError("Lower energy is higher than upper energy bound. Please check.")
    if ((flexibility_bands["upper_energy"].diff() - flexibility_bands["upper_power"]*efficiency) > 1e-6).any().any():
        problematic = flexibility_bands["upper_energy"][((flexibility_bands["upper_energy"].diff() -
                                          flexibility_bands["upper_power"]*efficiency) > 1e-6)].dropna(
            how="all").dropna(how="all", axis=1)
        raise ValueError("Upper energy has power values higher than nominal power. Please check.")
    if ((flexibility_bands["lower_energy"].diff() - flexibility_bands["upper_power"]*efficiency) > -1e-6).any().any():
        raise ValueError("Lower energy has power values higher than nominal power. Please check.")
    return flexibility_bands


def import_flexibility_bands_use_case(dir, use_cases):
    flexibility_bands = {}

    for band in ["upper_power", "upper_energy", "lower_energy"]:
        band_df = pd.DataFrame()
        for use_case in use_cases:
            flexibility_bands_tmp = \
                pd.read_csv(dir+'/{}_{}.csv'.format(band, use_case),
                            index_col=0, parse_dates=True, dtype=np.float32)
            band_df = pd.concat([band_df, flexibility_bands_tmp],
                                axis=1)
        if band_df.columns.duplicated().any():
            raise ValueError("Charging points with the same name in flexibility bands. "
                             "Please check")
        flexibility_bands[band] = band_df
        # remove numeric problems
        if "upper" in band:
            flexibility_bands[band] = flexibility_bands[band]
        elif "lower" in band:
            flexibility_bands[band] = flexibility_bands[band]
    return flexibility_bands


def import_and_merge_flexibility_bands_extended(data_dir, grid_ids=[],
                                                append="extended", only_bevs=True):
    """
    Method to import and merge flexibility bands that allow shifting over standing
    times. The number of EVs is extracted to update nr_ev_ref in scenario_input_ev.

    :param data_dir:
    :param grid_ids:
    :param append:
    :return:
    """
    flexibility_bands = {}
    nr_ev = 0
    for band in ["upper_power", "upper_energy", "lower_energy"]:
        band_df = pd.DataFrame(columns=[append])
        for grid_id in grid_ids:
            flexibility_bands_tmp = \
                pd.read_csv(data_dir+f'/{grid_id}/{band}_{append}.csv',
                            index_col=0, parse_dates=True, dtype=np.float32)

            # Extract BEVs if only they should be extracted, otherwise use all columns
            if only_bevs:
                bevs = []
                dirs = os.listdir(os.path.join(data_dir, str(grid_id), "simbev_run"))
                for dir_tmp in dirs:
                    if os.path.isdir(os.path.join(data_dir, str(grid_id), "simbev_run", dir_tmp)):
                        evs = os.listdir(os.path.join(data_dir, str(grid_id), "simbev_run", dir_tmp))
                        for ev in evs:
                            if "bev" in ev:
                                bevs.append(True)
                            else:
                                bevs.append(False)
                cols = [str(i) for i in range(len(bevs))
                        if bevs[i] & (str(i) in flexibility_bands_tmp.columns)]
            else:
                cols = flexibility_bands_tmp.columns
            if band_df.index.empty:
                band_df = \
                    pd.DataFrame(columns=[append], index=flexibility_bands_tmp.index,
                                 data=0)
            band_df[append] += flexibility_bands_tmp[cols].sum(axis=1)
            if band == "upper_power":
                nr_ev += len(flexibility_bands_tmp[cols].columns)
            print(f"Finished grid {grid_id} {band}")
        # remove numeric problems
        if "upper" in band:
            flexibility_bands[band] = band_df + 1e-6
        elif "lower" in band:
            flexibility_bands[band] = band_df - 1e-6
        # remove first week
        timeindex_full = pd.date_range("2010-12-25", end='2011-12-31 23:45:00', freq="15min")
        timeindex = pd.date_range("2011-01-01", end='2011-12-31 23:45:00', freq="15min")
        flexibility_bands[band].index = timeindex_full
        flexibility_bands[band] = flexibility_bands[band].loc[timeindex]
    print(f"Total of {nr_ev} EVs imported.")
    return flexibility_bands


def determine_shifting_times(data_dir, grid_ids, use_cases):
    flexibility_bands = {}
    for grid_id in grid_ids:
        for use_case in use_cases:
            for band in ["upper_energy", "lower_energy"]:
                flexibility_bands[band] = \
                    pd.read_csv(data_dir + f'/{grid_id}/{band}_{use_case}.csv',
                                index_col=0, parse_dates=True, dtype=np.float32)

            print(f"Extracting shifting times for grid {grid_id}.")
            shifting_times = determine_shifting_times_ev(flexibility_bands)
            shifting_times.to_csv(data_dir+f'/{grid_id}/shifting_times_{use_case}.csv')


def scale_electric_vehicles(nr_ev_mio, scenario_dict):
    nr_ev = nr_ev_mio * 1e6
    # scale bands and demand to new nr EV, resample to one hour
    ref_charging = scenario_dict["ts_ref_charging"].divide(
        scenario_dict["nr_ev_ref"]).multiply(
        nr_ev)
    flex_bands = {}
    if scenario_dict["ev_mode"] == "flexible":
        for band in ["upper_power", "upper_energy", "lower_energy"]:
            if not scenario_dict["ev_extended_flex"]:
                flex_bands[band] = scenario_dict["ts_flex_bands"][band].divide(
                    scenario_dict["nr_ev_ref"]).multiply(nr_ev)[
                    scenario_dict["use_cases_flexible"]]
            else:
                flex_bands[band] = scenario_dict["ts_flex_bands"][band].divide(
                    scenario_dict["nr_ev_ref"]).multiply(nr_ev)
    return ref_charging, flex_bands


def model_input_evs(scenario_dict, ev_mode, i=None, nr_ev_mio=None):
    if ev_mode is not None:
        # determine number of evs
        if i is not None:
            if nr_ev_mio is not None:
                print("Both i and nr_ev_mio are defined, nr_ev_mio will be used.")
            else:
                nr_ev_mio = i * 5
        else:
            if nr_ev_mio is None:
                raise ValueError("Either i or nr_ev_mio have to be provided.")
        # scale input accordingly
        (reference_charging, flexibility_bands) = scale_electric_vehicles(
            nr_ev_mio, scenario_dict)
        if ev_mode == "flexible":
            use_cases_inflexible = reference_charging.columns[
                ~reference_charging.columns.isin(scenario_dict["use_cases_flexible"])]
            energy_ev = \
                reference_charging[use_cases_inflexible].sum().sum() + \
                (flexibility_bands["upper_energy"].sum(axis=1)[
                     -1] +
                 flexibility_bands["lower_energy"].sum(axis=1)[
                     -1]) / 0.9 / 2
            ref_charging = reference_charging[use_cases_inflexible].sum(axis=1)
        else:
            energy_ev = reference_charging.sum().sum()
            ref_charging = reference_charging.sum(axis=1)
            flexibility_bands = {}
    else:
        nr_ev_mio = 0
        energy_ev = 0
        ref_charging = pd.Series(index=scenario_dict["ts_demand"].index, data=0)
        flexibility_bands = {}
    return nr_ev_mio, flexibility_bands, energy_ev, ref_charging


def reduced_operation(model):
    return sum(model.charging_ev[time]**2 for time in model.time_set)


def reduced_operation_multi(model):
    return sum(model.charging_ev[cp, time]**2 for cp in model.charging_points_set
               for time in model.time_set)


if __name__ == "__main__":
    mode = "extended"
    save_files = True
    if mode == "extended":
        merge_bands = True
        extract_shifting_time = False
    solver = "gurobi"
    time_increment = pd.to_timedelta('1h')
    if mode == "single":
        grid_dir = r"H:\Grids_IYCE\177"
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
    elif mode == "multi":
        grid_dir = r"C:\Users\aheider\Documents\Grids\Julian\emob_debugging\1056\feeder\01\electromobility"
        flex_bands = import_flexibility_bands(grid_dir, efficiency=1.0)
        for name, band in flex_bands.items():
            flex_bands[name] = band.resample(time_increment).mean().reset_index().drop(columns=["index"])
        model = pm.ConcreteModel()
        model.time_set = pm.RangeSet(0, len(flex_bands["upper_energy"]) - 1)
        model.time_non_zero = model.time_set - [model.time_set.at(1)]
        model.times_fixed_soc = pm.Set(initialize=[model.time_set.at(1),
                                                   model.time_set.at(-1)])
        model.time_increment = time_increment
        model = add_evs_model(model, flex_bands)
        model.objective = pm.Objective(rule=reduced_operation_multi,
                                       sense=pm.minimize,
                                       doc='Define objective function')
        opt = pm.SolverFactory(solver)
        results = opt.solve(model, tee=True)
        results_df = pd.DataFrame()
        results_df["charging_ev"] = pd.Series(model.charging_ev.extract_values())
        results_df["energy_level_ev"] = pd.Series(model.energy_level_ev.extract_values())
    elif mode == "use_case":
        if save_files:
            grid_dir = r"C:\Users\aheider\Documents\Grids"
            grid_ids = [176, 177, 1056, 1690, 1811, 2534]
            use_cases = ["home", "work", "public", "hpc"]
            flex_use_cases = ["home", "work", "public"]
            flex_bands = {case: {"upper_power": pd.DataFrame(),
                           "upper_energy": pd.DataFrame(),
                           "lower_energy": pd.DataFrame()} for case in flex_use_cases}
            ref_charging = pd.DataFrame(columns=use_cases)
            for grid_id in grid_ids:
                print(f"Start loading grid {grid_id}")
                cps = pd.read_csv(os.path.join(grid_dir, str(grid_id), "dumb",
                                               "topology", "charging_points.csv"),
                                  index_col=0)
                charging = pd.read_csv(os.path.join(grid_dir, str(grid_id), "dumb",
                                                    "timeseries",
                                                    "charging_points_active_power.csv"),
                                       index_col=0, parse_dates=True)
                if ref_charging.empty:
                    for case in cps.use_case.unique():
                        ref_charging[case] = charging[cps.loc[cps.use_case.isin(
                            [case])].index].sum(axis=1)
                else:
                    for case in use_cases:
                        ref_charging[case] = \
                            ref_charging[case] + charging[cps.loc[cps.use_case.isin(
                                [case])].index].sum(axis=1)
                print("Reference charging loaded")
                flex_bands_tmp = {}
                for use_case in flex_use_cases:
                    flex_bands_tmp[use_case] = import_flexibility_bands_use_case(
                        os.path.join(grid_dir, str(grid_id)), use_cases=[use_case])
                for band in flex_bands_tmp["home"].keys():
                    for use_case in flex_use_cases:
                        flex_bands[use_case][band] = \
                            pd.concat([flex_bands[use_case][band],
                                       flex_bands_tmp[use_case][band]],
                                      axis=1)
                print("Bands imported")
            ref_charging.to_csv("data/ref_charging_use_case.csv")
            flex_bands_final = {"upper_power": pd.DataFrame(),
                          "upper_energy": pd.DataFrame(),
                          "lower_energy": pd.DataFrame()}
            for band in flex_bands_final.keys():
                for use_case in flex_use_cases:
                    flex_bands_final[band][use_case] = \
                        flex_bands[use_case][band].sum(axis=1)
                flex_bands_final[band].to_csv(f"data/{band}_flex+.csv")
    elif mode == "extended":
        grid_dir = r"H:\Grids"
        grid_ids = [176, 177, 1056, 1690, 1811, 2534] # 176, 177, 1056, 1690, 1811, 2534
        use_case = "extended"
        if merge_bands:
            bands = import_and_merge_flexibility_bands_extended(grid_dir, grid_ids=grid_ids,
                                                                append=use_case)
            if save_files:
                for band in bands.keys():
                    bands[band].to_csv(f"data/{band}_extended_bevs.csv")
        if extract_shifting_time:
            determine_shifting_times(grid_dir, grid_ids=grid_ids, use_cases=[use_case])
    print("SUCCESS")
