import pandas as pd
import pyomo.environ as pm
import matplotlib.pyplot as plt

from storage_equivalent import add_storage_equivalents_model, minimize_energy
from ev_model import add_evs_model


def import_electric_vehicles(nr_ev_mio):
    nr_ev = nr_ev_mio * 1e6
    ref_charging = pd.read_csv(
        r"data/ref_charging.csv",
        index_col=0, parse_dates=True)
    nr_ev_ref = 26880 # from SEST
    flex_bands = {}
    for band in ["upper_power", "upper_energy", "lower_energy"]:
        flex_bands[band] = pd.read_csv(f"data/{band}.csv", index_col=0, parse_dates=0)
    # scale bands and demand to new nr EV, resample to one hour
    ref_charging = ref_charging.divide(nr_ev_ref).multiply(nr_ev).resample("1h").mean()
    for band in flex_bands.keys():
        flex_bands[band] = flex_bands[band].divide(nr_ev_ref).multiply(nr_ev)
        if "power" in band:
            flex_bands[band] = flex_bands[band].resample("1h").mean()
        elif "energy" in band:
            flex_bands[band] = flex_bands[band].resample("1h").max()
    return ref_charging, flex_bands


if __name__ == "__main__":
    scenario = "Germany_EV"
    solver = "gurobi"
    ev_mode = "inflexible"
    time_increment = pd.to_timedelta('1h')
    vres = pd.read_csv(r"data/vres_reference_ego100.csv", index_col=0,
                       parse_dates=True).divide(1000)
    demand = pd.read_csv(r"data/demand_germany_ego100.csv", index_col=0,
                         parse_dates=True)
    sum_energy = demand.sum().sum()
    scaled_ts_reference = vres.divide(vres.sum().sum())
    shifted_energy_df = pd.DataFrame(columns=["storage_type",
                                              "energy_stored"])
    shifted_energy_rel_df = pd.DataFrame(columns=["storage_type",
                                                  "energy_stored"])
    for nr_ev_mio in range(40):
        (reference_charging, flexibility_bands) = import_electric_vehicles(nr_ev_mio)
        vres = scaled_ts_reference * (sum_energy + reference_charging.sum().sum())
        if ev_mode == "flexible":
            new_res_load = \
                demand.sum(axis=1) + reference_charging["inflexible"] - vres.sum(axis=1)
        else:
            new_res_load = \
                demand.sum(axis=1) + reference_charging.sum(axis=1) - vres.sum(axis=1)
        model = pm.ConcreteModel()
        model.time_set = pm.RangeSet(0, len(new_res_load) - 1)
        model.time_non_zero = model.time_set - [model.time_set.at(1)]
        model.time_increment = time_increment
        model.times_fixed_soc = pm.Set(initialize=[model.time_set.at(1),
                                                   model.time_set.at(-1)])
        model.weighting = [1, 10, 100, 1000]
        if ev_mode == "flexible":
            model = add_evs_model(model, flexibility_bands)
        model = add_storage_equivalents_model(model, new_res_load)
        model.objective = pm.Objective(rule=minimize_energy,
                                       sense=pm.minimize,
                                       doc='Define objective function')
        opt = pm.SolverFactory(solver)
        results = opt.solve(model, tee=True)
        charging = pd.Series(model.charging.extract_values()).unstack()
        energy_levels = pd.Series(model.energy_levels.extract_values()).unstack()
        caps = pd.Series(model.caps_pos.extract_values()) + pd.Series(
            model.caps_neg.extract_values())
        caps_neg = pd.Series(model.caps_neg.extract_values())
        relative_energy_levels = (energy_levels.T + caps_neg).divide(caps)
        abs_charging = pd.Series(model.abs_charging.extract_values()).unstack()
        df_tmp = (abs_charging.sum(axis=1) / 2).reset_index().rename(
            columns={"index": "storage_type", 0: "energy_stored"})
        df_tmp["nr_ev"] = nr_ev_mio
        shifted_energy_df = shifted_energy_df.append(df_tmp, ignore_index=True)
        df_tmp["energy_stored"] = df_tmp["energy_stored"] / sum_energy * 100
        shifted_energy_rel_df = shifted_energy_rel_df.append(df_tmp,
                                                             ignore_index=True)
    shifted_energy_df.to_csv(f"results/storage_equivalents_{scenario}_{ev_mode}.csv")
    shifted_energy_rel_df.to_csv(
        f"results/storage_equivalents_{scenario}_{ev_mode}_relative.csv")
    # shifted_energy_rel_df.loc[shifted_energy_rel_df.storage_type == 0].set_index(
    #     "share_pv").energy_stored.plot.bar(figsize=(4, 2))
    # plt.title("Relative energy stored short")
    # plt.tight_layout()
    # TODO: plot all
    # TODO: subplots
    print("SUCCESS")