import pyomo.environ as pm
import pandas as pd
import numpy as np
import os


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


def add_evs_model(model, flex_bands, efficiency=0.9):
    def charging_ev(model, cp, time):
        """
        Constraint for charging of EV that has to ly between the lower and upper
        energy band.

        :param model:
        :param charging_point:
        :param time:
        :return:
        """
        return model.energy_level_ev[cp, time] == \
               model.energy_level_ev[cp, time - 1] + \
               model.charging_efficiency * \
               model.charging_ev[cp, time] * \
               (pd.to_timedelta(model.time_increment) / pd.to_timedelta('1h'))

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

    # save fix parameters
    model.charging_efficiency = efficiency
    model.flex_bands = flex_bands
    # set up set
    model.charging_points_set = \
        pm.Set(initialize=model.flex_bands["lower_energy"].columns)
    # set up variables
    model.charging_ev = \
        pm.Var(model.charging_points_set, model.time_set,
               bounds=lambda m, cp, t:
               (0, m.flex_bands["upper_power"].iloc[t][cp]))
    model.energy_level_ev = \
        pm.Var(model.charging_points_set, model.time_set,
               bounds=lambda m, cp, t:
               (m.flex_bands["lower_energy"].iloc[t][cp],
                m.flex_bands["upper_energy"].iloc[t][cp]))
    # add constraints
    model.EVCharging = pm.Constraint(model.charging_points_set, model.time_non_zero,
                                     rule=charging_ev)
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
                                                append="extended"):
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
            if band_df.index.empty:
                band_df = \
                    pd.DataFrame(columns=[append], index=flexibility_bands_tmp.index,
                                 data=0)
            band_df[append] += flexibility_bands_tmp.sum(axis=1)
            if band == "upper_power":
                nr_ev += len(flexibility_bands_tmp.columns)
        # remove numeric problems
        if "upper" in band:
            flexibility_bands[band] = band_df + 1e-6
        elif "lower" in band:
            flexibility_bands[band] = band_df - 1e-6
    print(f"Total of {nr_ev} EVs imported.")
    return flexibility_bands


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


def reduced_operation(model):
    return sum(model.charging_ev[time]**2 for time in model.time_set)


def reduced_operation_multi(model):
    return sum(model.charging_ev[cp, time]**2 for cp in model.charging_points_set
               for time in model.time_set)


if __name__ == "__main__":
    mode = "extended"
    save_files = True
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
        grid_ids = [176, 177, 1056, 1690, 1811, 2534]
        use_case = "extended"
        bands = import_and_merge_flexibility_bands_extended(grid_dir, grid_ids=grid_ids,
                                                            append=use_case)
        if save_files:
            for band in bands.keys():
                bands[band].to_csv(f"data/{band}_flex++.csv")
    print("SUCCESS")