import pandas as pd
import pyomo.environ as pm
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from storage_equivalent import add_storage_equivalent_model, minimize_energy
from scenario_Germany_EV_integration import import_electric_vehicles
from scenario_Germany_HP_integration import import_heat_pumps


def plot_storage_timeseries(df_storage):
    plot_df = pd.DataFrame()
    plot_df["PV"] = df_storage["pv_feed-in"]
    plot_df["Residuallast ohne Speicher"] = df_storage["residual_load_pre"]
    plot_df["Residuallast mit Speicher"] = df_storage["residual_load"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = sns.color_palette("Paired")
    plot_df.plot(ax=ax, color=["gold", colors[3], colors[3]],
                 style=["-", "--", "-"])
    plt.fill_between(x=plot_df.index, y1=plot_df["Residuallast mit Speicher"],
                     color=colors[2])
    plt.fill_between(x=plot_df.index, y1=plot_df["PV"], color="gold")
    ts_charge_storage = df_storage.loc[
        df_storage["residual_load"] > df_storage["residual_load_pre"]].index
    ts_discharge_storage = df_storage.loc[
        df_storage["residual_load"] < df_storage["residual_load_pre"]].index
    plot_df["Einspeichern"] = plot_df["Residuallast mit Speicher"]
    plot_df.loc[ts_charge_storage, "Einspeichern"] = \
        plot_df.loc[ts_charge_storage, "Residuallast ohne Speicher"]
    plot_df["Ausspeichern"] = plot_df["Residuallast mit Speicher"]
    plot_df.loc[ts_discharge_storage, "Ausspeichern"] = \
        plot_df.loc[ts_discharge_storage, "Residuallast ohne Speicher"]
    plt.fill_between(x=plot_df.index, y1=plot_df["Residuallast mit Speicher"],
                     y2=plot_df["Einspeichern"],
                     color=colors[1], label="Laden der Speicher")
    plt.fill_between(x=plot_df.index, y1=plot_df["Residuallast mit Speicher"],
                     y2=plot_df["Ausspeichern"],
                     color=colors[0], label="Entladen der Speicher")
    plt.axhline(y=plot_df["Residuallast ohne Speicher"].max(), color='grey', linestyle=':')
    plt.axhline(y=plot_df["Residuallast mit Speicher"].max(),
                color='grey', linestyle='-')
    plt.ylabel("Leistung in MW")
    plt.legend(loc="upper left")
    plt.ylim((0, 12.0))
    plt.tight_layout()
    plt.savefig("results/storage_operation_v3.png", dpi=300)
    plt.show()


def plot_storage_equivalent_germany_stacked(
        storage_equivalent,
        parameter={"scenario": "Deutschland"},
        ylabel="Gespeicherte Energie [GWh]",
        var="storage equivalent",
        figsize=(5, 3),
        loc="upper left",
        append="three_storage",
        ylim=None,
        language="english"
):
    param = list(parameter.items())[0][0]
    ylabel = ylabel
    xlabel = list(parameter.items())[0][1]
    if (language == "german") or (language == "deutsch"):
        type_dict = {0: "Tag", 1: "Woche", 2: "Saisonal"}
    else:
        type_dict = {0: "Day", 1: "Week", 2: "Seasonal"}
    fig, ax = plt.subplots(figsize=figsize)
    colors = matplotlib.cm.get_cmap("Blues")
    for storage_type in range(storage_equivalent.storage_type.max()+1):

        plot_df = pd.DataFrame(index=storage_equivalent[param].unique())
        plot_df[ylabel] = storage_equivalent.loc[
            storage_equivalent.storage_type >= storage_type].groupby(
            param).sum().energy_stored.divide(1e3)
        plot_df[xlabel] = storage_equivalent[param].unique()

        sns.barplot(x=xlabel, y=ylabel, color=colors(1.0-0.3*storage_type),
                    data=plot_df, ax=ax, label=type_dict[storage_type])
    if loc is not None:
        plt.legend(loc=loc)#loc="lower left"
    #plt.title(f"{var.capitalize()} Germany")
    if ylim is not None:
        # ax.get_ylim()
        ax.set_ylim(ylim)
    if param in ["nr_ev"]:
        xticklabels = ax.get_xticklabels()
        ax.set_xticklabels([int(float(tick.get_text())) for tick in xticklabels])
    plt.tight_layout()
    save_name = var.replace(" ", "_")
    plt.savefig(f"results/{save_name}_Germany_{param}_{append}.png", dpi=300)
    plt.show()


def run_and_plot_single_scenario(new_res_load, scenario,
                                 append="three_storage", ylim=None,
                                 ylabel="Gespeicherte Energie [GWh]",):
    shifted_energy_df = pd.DataFrame(columns=["storage_type",
                                              "energy_stored"])
    shifted_energy_rel_df = pd.DataFrame(columns=["storage_type",
                                                  "energy_stored"])

    model = pm.ConcreteModel()
    model.time_set = pm.RangeSet(0, len(new_res_load) - 1)
    model.time_non_zero = model.time_set - [model.time_set.at(1)]
    model.time_increment = time_increment
    model.times_fixed_soc = pm.Set(initialize=[model.time_set.at(1),
                                               model.time_set.at(-1)])
    model.weighting = [1, 10, 100, 1000]
    model = add_storage_equivalent_model(model, new_res_load,
                                         time_horizons=[24, 7 * 24, 24 * 366])
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
    df_tmp["scenario"] = scenario
    shifted_energy_df = shifted_energy_df.append(df_tmp, ignore_index=True)
    df_tmp["energy_stored"] = df_tmp["energy_stored"] / sum_energy * 100
    shifted_energy_rel_df = shifted_energy_rel_df.append(df_tmp,
                                                         ignore_index=True)
    if (language == "german") or (language == "deutsch"):
        parameter = {"scenario": "Deutschland"}
    else:
        parameter = {"scenario": "Germany"}
    plot_storage_equivalent_germany_stacked(
        shifted_energy_df,
        figsize=(1.5, 2.5),
        parameter=parameter,
        ylabel=ylabel,
        language=language,
        loc=None,
        append=append,
        ylim=ylim
    )


if __name__ == "__main__":
    plot_ts = False
    plot_operation = False
    optimise_storage = False
    plot_scenarios = True
    language = "english"
    solver = "gurobi"
    ev_mode = "flexible"
    time_increment = pd.to_timedelta('1h')
    vres = pd.read_csv(r"data/vres_reference_ego100.csv", index_col=0,
                       parse_dates=True).divide(1000)
    demand = pd.read_csv(r"data/demand_germany_ego100.csv", index_col=0,
                         parse_dates=True)
    sum_energy = demand.sum().sum()
    scaled_ts_reference = vres.divide(vres.sum().sum())
    vres = scaled_ts_reference * sum_energy
    new_res_load = demand.sum(axis=1) - vres.sum(axis=1)
    if (language == "german") or (language == "deutsch"):
        ylabel = "Gespeicherte Energie [TWh]"
        solar_name = "Photovoltaik"
        title_name = "Erzeugung"
        ylabel = "Leistung [GW]"
        demand_name = "Verbrauch"
    else:
        ylabel = "Stored Energy [TWh]"
        solar_name = "Photovoltaic"
        title_name = "Generation"
        ylabel = "Power [GW]"
        demand_name = "Demand"
    colors = matplotlib.cm.get_cmap("Blues")
    if plot_ts:

        df_plot = pd.DataFrame()
        df_plot[solar_name] = vres["solar"]
        df_plot["Wind"] = vres.sum(axis=1)
        plt.figure()
        df_plot[[solar_name, "Wind"]].iloc[48:24 * 9].plot(
            figsize=(5, 2.5), color=["orange", colors(1.0)],
            title=title_name)
        plt.fill_between(x=df_plot.index, y1=df_plot[solar_name], color="orange",
                         alpha=0.5)
        plt.fill_between(x=df_plot.index, y2=df_plot["Wind"], y1=df_plot[solar_name],
                         color=colors(1.0), alpha=0.5)
        plt.ylabel(ylabel)
        ylim = plt.ylim()
        plt.tight_layout()
        plt.savefig("generation.png", dpi=300)
        plt.show()
        plt.figure()
        demand.iloc[48:24 * 9].sum(axis=1).plot(figsize=(5, 2.5), color=["green"],
                                                title=demand_name, label=demand_name)
        plt.fill_between(x=demand.index, y1=demand.sum(axis=1),
                         color="green", alpha=0.5)
        plt.legend()
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.ylim(ylim)
        plt.savefig("demand.png", dpi=300)
        plt.show()
        df_plot[demand_name] = demand.sum(axis=1)
        plt.figure()
        ax=df_plot.iloc[24*5:24*6].plot(figsize=(2, 2.5), color=["orange", colors(1.0),
                                                                 "green"])
        df_plot["Einspeichern"] = df_plot["Wind"]
        df_plot.loc[df_plot.Wind > df_plot[demand_name], "Einspeichern"] = \
            df_plot.loc[df_plot.Wind > df_plot[demand_name], demand_name]
        df_plot["Ausspeichern"] = df_plot[demand_name]
        df_plot.loc[df_plot.Wind < df_plot[demand_name], "Ausspeichern"] = \
            df_plot.loc[df_plot.Wind < df_plot[demand_name], "Wind"]
        plt.fill_between(x=df_plot.index, y1=df_plot[solar_name], color="orange",
                         alpha=0.5)
        plt.fill_between(x=df_plot.index, y2=df_plot["Wind"], y1=df_plot[solar_name],
                         color=colors(1.0), alpha=0.5)
        plt.fill_between(x=df_plot.index, y1=df_plot[demand_name],
                         color="green", alpha=0.5)
        ax.get_legend().remove()
        plt.ylim((ylim[0], 109.29))
        ax.set_xticks([])
        ax.set_xticklabels("")
        ax.get_xaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig("detail_clean.png", dpi=300)
        plt.fill_between(x=df_plot.index, y2=df_plot["Wind"], y1=df_plot["Einspeichern"],
                         color=colors(1.0))
        plt.savefig("detail_charge.png", dpi=300)
        plt.fill_between(x=df_plot.index, y2=df_plot[demand_name], y1=df_plot["Ausspeichern"],
                         color="green")
        plt.savefig("detail_discharge.png", dpi=300)
        plt.show()
    if plot_operation:
        scenario = "Germany_EV_three_storage_operation"
        optimised_charging_ev = \
            pd.read_csv(f"results/ev_charging_flexible_{scenario}.csv", index_col=0,
                        parse_dates=True)
        (reference_charging, flexibility_bands) = import_electric_vehicles(40)

        scaled_ts_reference = vres.divide(vres.sum().sum())
        vres = scaled_ts_reference * (sum_energy + reference_charging.sum().sum())
        df_plot = pd.DataFrame()
        df_plot[solar_name] = vres["solar"]
        df_plot["Wind"] = vres.sum(axis=1)
        for season in ["winter", "summer"]:
            start_days = {"winter":2, "summer": 190}
            start_iter = start_days[season] * 24
            plt.figure()
            df_plot[[solar_name, "Wind"]].iloc[start_iter:start_iter+24*7].plot(
                figsize=(5, 2.5), color=["orange", colors(1.0)],
                title=title_name)
            plt.fill_between(x=df_plot.index, y1=df_plot[solar_name], color="orange",
                             alpha=0.5)
            plt.fill_between(x=df_plot.index, y2=df_plot["Wind"], y1=df_plot[solar_name],
                             color=colors(1.0), alpha=0.5)
            plt.ylabel(ylabel)
            ylim = plt.ylim()
            plt.tight_layout()
            plt.savefig(f"generation_ev_{season}.png", dpi=300)

            plt.figure()
            demand.iloc[start_iter:start_iter+24*7].sum(axis=1).plot(
                figsize=(5, 2.5), color=["green"], title="Reference Operation", label=demand_name)
            plt.fill_between(x=demand.index, y1=demand.sum(axis=1),
                             color="green", alpha=0.5)
            tmp = demand.sum(axis=1)+reference_charging.sum(axis=1)
            tmp[start_iter:start_iter+24*7].plot(color=["darkolivegreen"], label="EV")
            df_plot["Wind"][start_iter:start_iter+24*7].plot(color="k", label="Generation")
            plt.fill_between(x=demand.index, y1=tmp, y2=demand.sum(axis=1),
                             color="darkolivegreen", alpha=0.5)
            plt.legend(loc="upper right")
            plt.ylabel(ylabel)
            plt.ylim(ylim)
            plt.tight_layout()
            plt.savefig(f"ev_charging_reference_{season}.png", dpi=300)
            plt.show()
            plt.figure()
            demand.iloc[start_iter:start_iter+24*7].sum(axis=1).plot(figsize=(5, 2.5), color=["green"],
                                                    title="Optimised Operation", label=demand_name)
            plt.fill_between(x=demand.index, y1=demand.sum(axis=1),
                             color="green", alpha=0.5)
            tmp = demand.sum(axis=1) + reference_charging["inflexible"] + optimised_charging_ev.sum(axis=1)
            tmp[start_iter:start_iter+24*7].plot(color=["darkolivegreen"], label="EV")
            df_plot["Wind"][start_iter:start_iter+24*7].plot(color="k", label="Generation")
            plt.fill_between(x=demand.index, y1=tmp, y2=demand.sum(axis=1),
                             color="darkolivegreen", alpha=0.5)
            # plt.legend()
            plt.ylabel(ylabel)
            plt.ylim(ylim)
            plt.tight_layout()
            plt.savefig(f"ev_charging_optimised_{season}.png", dpi=300)
            plt.show()
        # HPs
        scenario = "Germany_HP_three_storage_operation"
        hp_operation = pd.read_csv(f"results/hp_charging_flexible_{scenario}.csv",
                                   index_col=0, parse_dates=True)
        (heat_demand, cop, capacity_tes, p_nom_hp,
         ts_heat_demand, ts_heat_el, sum_energy_heat) = import_heat_pumps(20)
        ts_heat_el.index = demand.index
        vres = scaled_ts_reference * (sum_energy + sum_energy_heat)
        df_plot = pd.DataFrame()
        df_plot[solar_name] = vres["solar"]
        df_plot["Wind"] = vres.sum(axis=1)
        for season in ["winter", "summer"]:
            start_days = {"winter":2, "summer": 190}
            start_iter = start_days[season] * 24
            plt.figure()
            df_plot[[solar_name, "Wind"]].iloc[start_iter:start_iter+24*7].plot(
                figsize=(5, 2.5), color=["orange", colors(1.0)],
                title=title_name)
            plt.fill_between(x=df_plot.index, y1=df_plot[solar_name], color="orange",
                             alpha=0.5)
            plt.fill_between(x=df_plot.index, y2=df_plot["Wind"], y1=df_plot[solar_name],
                             color=colors(1.0), alpha=0.5)
            plt.ylabel(ylabel)
            ylim = plt.ylim()
            plt.tight_layout()
            plt.savefig(f"generation_hp_{season}.png", dpi=300)
            plt.figure()
            demand.iloc[start_iter:start_iter+24*7].sum(axis=1).plot(
                figsize=(5, 2.5), color=["green"], title="Reference Operation", label=demand_name)
            plt.fill_between(x=demand.index, y1=demand.sum(axis=1),
                             color="green", alpha=0.5)
            tmp = demand.sum(axis=1) + ts_heat_el[0]
            tmp[start_iter:start_iter+24*7].plot(color=["darkcyan"], label="HP")
            plt.fill_between(x=demand.index, y1=tmp, y2=demand.sum(axis=1),
                             color="darkcyan", alpha=0.5)
            df_plot["Wind"][start_iter:start_iter+24*7].plot(color="k", label="Generation")

            plt.legend(loc="upper right")
            plt.ylabel(ylabel)
            plt.ylim(ylim)
            plt.tight_layout()
            plt.savefig(f"hp_charging_reference_{season}.png", dpi=300)
            plt.show()
            plt.figure()
            demand.iloc[start_iter:start_iter+24*7].sum(axis=1).plot(
                figsize=(5, 2.5), color=["green"], title="Optimised Operation", label=demand_name)
            plt.fill_between(x=demand.index, y1=demand.sum(axis=1),
                             color="green", alpha=0.5)
            tmp = demand.sum(axis=1) + hp_operation["0"]
            tmp[start_iter:start_iter+24*7].plot(color=["darkcyan"], label="HP")
            df_plot["Wind"][start_iter:start_iter+24*7].plot(color="k", label="Generation")
            plt.fill_between(x=demand.index, y1=tmp, y2=demand.sum(axis=1),
                             color="darkcyan", alpha=0.5)
            # plt.legend()
            plt.ylabel(ylabel)
            plt.ylim(ylim)
            plt.tight_layout()
            plt.savefig(f"hp_charging_optimised_{season}.png", dpi=300)
            plt.show()
    if optimise_storage:
        if (language == "german") or (language == "deutsch"):
            ylabel = "Gespeicherte Energie [TWh]"
        else:
            ylabel = "Stored Energy [TWh]"
        run_and_plot_single_scenario(new_res_load=new_res_load,
                                     scenario="100% RES",
                                     ylabel=ylabel)
        new_res_load_flat = demand.sum(axis=1) - demand.sum(axis=1).mean()
        run_and_plot_single_scenario(new_res_load=new_res_load_flat,
                                     scenario="Flat Gen.",
                                     ylabel=ylabel,
                                     append="three_storage_flat",
                                     ylim=(0.0, 138.4322296470711))
    if plot_scenarios:
        if (language == "german") or (language == "deutsch"):
            ylabel = "Gespeicherte Energie [TWh]"
        else:
            ylabel = "Stored Energy [TWh]"
        scenario = "weights_hp_three_storage"
        storage_equivalent = pd.read_csv(
            "results/storage_equivalents_{}.csv".format(scenario),
            index_col=0)
        if (language == "german") or (language == "deutsch"):
            parameter = {"relative_weight": "Relative Gewichtung [-]"}
        else:
            parameter = {"relative_weight": "Relative Weight [-]"}

        plot_storage_equivalent_germany_stacked(
            storage_equivalent,
            parameter=parameter,
            ylabel=ylabel,
            language=language,
            append="three_storage_sensitivity",
            loc="lower left"
        )
        scenario = "Germany_three_storage"
        storage_equivalent = pd.read_csv(
            "results/storage_equivalents_{}.csv".format(scenario),
            index_col=0)
        if (language == "german") or (language == "deutsch"):
            parameter = {"share_pv": "Anteil PV [-]"}
        else:
            parameter = {"share_pv": "Share PV [-]"}
        plot_storage_equivalent_germany_stacked(
            storage_equivalent,
            parameter=parameter,
            ylabel=ylabel,
            language=language
        )
        scenario = "Germany_EV_three_storage_flexible"
        storage_equivalent = pd.read_csv(
            "results/storage_equivalents_{}.csv".format(scenario),
            index_col=0)
        if (language == "german") or (language == "deutsch"):
            parameter = {"nr_ev": "Anzahl EVs [Mio.]"}
        else:
            parameter = {"nr_ev": "Number EVs [Mio.]"}
        plot_storage_equivalent_germany_stacked(
            storage_equivalent,
            parameter=parameter,
            ylabel=ylabel,
            loc=None,
            append="three_storage_flexible",
            ylim=(0, 162.89322),
            language=language
        )
        scenario_2 = "Germany_EV_three_storage_inflexible"
        storage_equivalent_2 = pd.read_csv(
            f"results/storage_equivalents_{scenario_2}.csv",
            index_col=0)
        plot_storage_equivalent_germany_stacked(
            storage_equivalent_2,
            parameter=parameter,
            ylabel=ylabel,
            loc=None
        )
        diff = storage_equivalent.copy()
        diff["energy_stored"] = storage_equivalent["energy_stored"] - \
                                storage_equivalent_2["energy_stored"]
        plot_storage_equivalent_germany_stacked(
            diff,
            parameter=parameter,
            ylabel=ylabel,
            append="three_storage_diff",
            loc=None,
            ylim=(-24.461, 0.0),
            language=language
        )
        growth = storage_equivalent.copy()
        ref = storage_equivalent_2.loc[storage_equivalent_2.nr_ev == 0]
        for nr_ev in storage_equivalent_2.nr_ev.unique():
            tmp = storage_equivalent_2.loc[storage_equivalent_2.nr_ev == nr_ev]
            growth.loc[tmp.index, "energy_stored"] = \
                tmp["energy_stored"].values-ref["energy_stored"].values
        growth_percentage_ev = \
            growth.groupby("nr_ev").sum().energy_stored.divide(ref.energy_stored.sum())
        growth_percentage_ev_flex = \
            (growth.groupby("nr_ev").sum() + diff.groupby("nr_ev").sum())\
                .energy_stored.divide(ref.energy_stored.sum())
        plot_storage_equivalent_germany_stacked(
            growth,
            parameter=parameter,
            ylabel=ylabel,
            append="three_storage_growth",
            language=language
        )
        scenario = "Germany_hp_integration_flexible"
        storage_equivalent = pd.read_csv(
            "results/storage_equivalents_{}.csv".format(scenario),
            index_col=0)
        if (language == "german") or (language == "deutsch"):
            parameter = {"nr_hp": "Anzahl WP [Mio.]"}
        else:
            parameter = {"nr_hp": "Number HPs [Mio.]"}
        plot_storage_equivalent_germany_stacked(
            storage_equivalent,
            parameter=parameter,
            ylabel=ylabel,
            loc=None,
            append="three_storage_flexible",
            ylim=(0, 187.29552),
            language=language
        )
        scenario_2 = "Germany_HP_three_storage_inflexible"
        storage_equivalent_2 = pd.read_csv(
            f"results/storage_equivalents_{scenario_2}.csv",
            index_col=0)
        plot_storage_equivalent_germany_stacked(
            storage_equivalent_2,
            parameter=parameter,
            ylabel=ylabel,
            loc=None,
            language=language
        )
        diff = storage_equivalent.copy()
        diff["energy_stored"] = storage_equivalent["energy_stored"] - \
                                storage_equivalent_2["energy_stored"]
        plot_storage_equivalent_germany_stacked(
            diff,
            parameter=parameter,
            ylabel=ylabel,
            append="three_storage_diff",
            loc=None,
            ylim=(-39.863, 0.0),
            language=language
        )
        growth = storage_equivalent.copy()
        ref = storage_equivalent_2.loc[storage_equivalent_2.nr_hp == 0]
        for nr_hp in storage_equivalent_2.nr_hp.unique():
            tmp = storage_equivalent_2.loc[storage_equivalent_2.nr_hp == nr_hp]
            growth.loc[tmp.index, "energy_stored"] = \
                tmp["energy_stored"].values - ref["energy_stored"].values
        plot_storage_equivalent_germany_stacked(
            growth,
            parameter=parameter,
            ylabel=ylabel,
            append="three_storage_growth",
            language=language
        )
        growth_percentage_hp = growth.groupby("nr_hp").sum().energy_stored.divide(
            ref.energy_stored.sum())
        growth_percentage_hp_flex = \
            (growth.groupby("nr_hp").sum() + diff.groupby("nr_hp").sum()) \
                .energy_stored.divide(ref.energy_stored.sum())
        # scenario variation double tes
        scenario = "Germany_HP_three_storage_double_tes_flexible"
        storage_equivalent = pd.read_csv(
            "results/storage_equivalents_{}.csv".format(scenario),
            index_col=0)
        if (language == "german") or (language == "deutsch"):
            parameter = {"nr_hp": "Anzahl WP [Mio.]"}
        else:
            parameter = {"nr_hp": "Number HPs [Mio.]"}
        plot_storage_equivalent_germany_stacked(
            storage_equivalent,
            parameter=parameter,
            ylabel=ylabel,
            loc=None,
            append="three_storage_flexible_double_tes",
            ylim=(0, 187.29552),
            language=language
        )
        diff = storage_equivalent.copy()
        diff["energy_stored"] = storage_equivalent["energy_stored"] - \
                                storage_equivalent_2["energy_stored"]
        plot_storage_equivalent_germany_stacked(
            diff,
            parameter=parameter,
            ylabel=ylabel,
            append="three_storage_diff_double_tes",
            loc=None,
            ylim=(-39.863, 0.0),
            language=language
        )
        growth_percentage_hp_flex_2 = \
            (growth.groupby("nr_hp").sum() + diff.groupby("nr_hp").sum()) \
                .energy_stored.divide(ref.energy_stored.sum())
    print("Success")
