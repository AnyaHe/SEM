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
               (model.flex_bands["lower_energy"].loc[time, cp] +
                model.flex_bands["upper_energy"].loc[time, cp]) / 2

    # save fix parameters
    model.charging_efficiency = efficiency
    model.flex_bands = flex_bands
    # set up set
    model.charging_points_set = pm.Set(initialize=model.flex_bands["lower_energy"].columns)
    # set up variables
    model.charging_ev = \
        pm.Var(model.charging_points_set, model.time_set,
               bounds=lambda m, cp, t:
               (0, m.flex_bands["upper_power"].loc[t, cp]))
    model.energy_level_ev = \
        pm.Var(model.charging_points_set, model.time_set,
               bounds=lambda m, cp, t:
               (m.flex_bands["lower_energy"].loc[t, cp],
                m.flex_bands["upper_energy"].loc[t, cp]))
    # add constraints
    model.EVCharging = pm.Constraint(model.charging_points_set, model.time_non_zero, rule=charging_ev)
    model.FixedEVEnergyLevel = pm.Constraint(model.charging_points_set, model.times_fixed_soc, rule=fixed_energy_level)
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


def reduced_operation(model):
    return sum(model.charging_ev[time]**2 for time in model.time_set)


def reduced_operation_multi(model):
    return sum(model.charging_ev[cp, time]**2 for cp in model.charging_points_set
               for time in model.time_set)


if __name__ == "__main__":
    mode = "use_case"
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
        grid_dir = r"H:\Grids Ladina\177"
        flex_bands = import_flexibility_bands(grid_dir)
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
            flex_bands_home = {"upper_power": pd.DataFrame(),
                               "upper_energy": pd.DataFrame(),
                               "lower_energy": pd.DataFrame()}
            flex_bands_work = {"upper_power": pd.DataFrame(),
                               "upper_energy": pd.DataFrame(),
                               "lower_energy": pd.DataFrame()}
            ref_charging = pd.DataFrame(columns=["flex", "inflexible"])
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
                    ref_charging["flex"] = charging[cps.loc[cps.use_case.isin(
                        ["home", "work"])].index].sum(axis=1)
                    ref_charging["inflexible"] = charging[cps.loc[~cps.use_case.isin(
                        ["home", "work"])].index].sum(axis=1)
                else:
                    ref_charging["flex"] = \
                        ref_charging["flex"] + charging[cps.loc[cps.use_case.isin(
                            ["home", "work"])].index].sum(axis=1)
                    ref_charging["inflexible"] = \
                        ref_charging["inflexible"] + charging[cps.loc[~cps.use_case.isin(
                            ["home", "work"])].index].sum(axis=1)
                print("Reference charging loaded")
                flex_bands_home_tmp = \
                    import_flexibility_bands_use_case(
                        os.path.join(grid_dir, str(grid_id)), use_cases=["home"])
                flex_bands_work_tmp = \
                    import_flexibility_bands_use_case(
                        os.path.join(grid_dir, str(grid_id)), use_cases=["work"])
                for band in flex_bands_home.keys():
                    flex_bands_home[band] = \
                        pd.concat([flex_bands_home[band], flex_bands_home_tmp[band]],
                                  axis=1)
                    flex_bands_work[band] = \
                        pd.concat([flex_bands_work[band], flex_bands_work_tmp[band]],
                                  axis=1)
                print("Bands imported")
            ref_charging.to_csv("data/ref_charging.csv")
            flex_bands = {"upper_power": pd.DataFrame(),
                          "upper_energy": pd.DataFrame(),
                          "lower_energy": pd.DataFrame()}
            for band in flex_bands.keys():
                flex_bands[band]["home"] = flex_bands_home[band].sum(axis=1)
                flex_bands[band]["work"] = flex_bands_work[band].sum(axis=1)
                flex_bands[band].to_csv(f"data/{band}.csv")
    print("SUCCESS")