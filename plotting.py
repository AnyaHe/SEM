import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd


def plot_storage_equivalent_germany_by_storage_type(storage_equivalent):
    plot_df = pd.DataFrame()
    ylabel = "Energy stored [GWh]"
    xlabel = "Share PV [-]"
    huelabel = "Storage type"
    type_dict = {0: "Day", 1: "Week", 2: "Month", 3: "Year"}
    plot_df[ylabel] = storage_equivalent.energy_stored.divide(1e3)
    plot_df[xlabel] = storage_equivalent.share_pv
    plot_df[huelabel] = "Unknown"
    for storage_type in storage_equivalent.storage_type.unique():
        plot_df.loc[storage_equivalent.storage_type == storage_type, huelabel] = \
            type_dict[storage_type]
    plt.figure(figsize=(5, 3))
    sns.barplot(x=xlabel, y=ylabel, hue=huelabel,
                data=plot_df)
    plt.title("Storage equivalent Germany")
    plt.tight_layout()
    plt.show()


def plot_storage_equivalent_germany_stacked(
        storage_equivalent,
        parameter={"share_pv": "Share PV [-]"},
        ylabel="Energy stored [GWh]",
        var="storage equivalent"
):
    param = list(parameter.items())[0][0]
    ylabel = ylabel
    xlabel = list(parameter.items())[0][1]
    type_dict = {0: "Day", 1: "Week", 2: "Month", 3: "Year"}
    fig, ax = plt.subplots(figsize=(5, 3))
    colors = matplotlib.cm.get_cmap("Blues")
    for storage_type in range(storage_equivalent.storage_type.max()+1):

        plot_df = pd.DataFrame(index=storage_equivalent[param].unique())
        plot_df[ylabel] = storage_equivalent.loc[
            storage_equivalent.storage_type >= storage_type].groupby(
            param).sum().energy_stored.divide(1e3)
        plot_df[xlabel] = storage_equivalent[param].unique()

        sns.barplot(x=xlabel, y=ylabel, color=colors(1-0.2*storage_type),
                    data=plot_df, ax=ax, label=type_dict[storage_type])
    plt.legend(loc="lower left")#loc="lower left"
    plt.title(f"{var.capitalize()} Germany")
    plt.tight_layout()
    save_name = var.replace(" ", "_")
    plt.savefig(f"results/{save_name}_Germany_{param}.pdf")
    plt.show()


if __name__ == "__main__":
    scenario = "weights_hp"
    storage_equivalent = pd.read_csv(
        "results/storage_equivalents_{}.csv".format(scenario),
        index_col=0)
    plot_storage_equivalent_germany_stacked(storage_equivalent,
                                            {"relative_weight": "Relative weight"})#{"relative_weight": "Relative weight"}

    scenario_2 = "Germany_HP_dumb"
    storage_equivalent_2 = pd.read_csv(
        f"results/storage_equivalents_{scenario_2}.csv",
        index_col=0)
    diff = storage_equivalent.copy()
    diff["energy_stored"] = storage_equivalent["energy_stored"] - \
                            storage_equivalent_2["energy_stored"]
    plot_storage_equivalent_germany_stacked(diff,
                                            {"nr_hp": "Number of HPs [Mio.]"})
    growth = storage_equivalent.copy()
    ref = storage_equivalent_2.loc[storage_equivalent_2.nr_hp==0]
    for nr_hp in storage_equivalent_2.nr_hp.unique():
        tmp = storage_equivalent_2.loc[storage_equivalent_2.nr_hp==nr_hp]
        growth.loc[tmp.index, "energy_stored"] = tmp["energy_stored"].values-ref["energy_stored"].values
    storage_equivalent_Germany = storage_equivalent.groupby("storage_type").sum().reset_index()
    storage_equivalent_Germany["state"] = "Germany"
    storage_equivalent_Germany_2 = storage_equivalent_2.groupby(
        "storage_type").sum().reset_index()
    storage_equivalent_Germany_2["state"] = "Germany_isolated"
    plot_storage_equivalent_germany_stacked(storage_equivalent_2,
                                            {"state": "State"})
    plot_storage_equivalent_germany_stacked(pd.concat([storage_equivalent_Germany,
                                                       storage_equivalent_Germany_2]),
                                            {"state": "State"})

    cap_1 = pd.read_csv(f"results/{scenario}/caps.csv", index_col=0)
    cap_2 = pd.read_csv(f"results/{scenario_2}/caps.csv", index_col=0)
    cap_1.plot.bar(stacked=True)
    cap_2.plot.bar(stacked=True)
    el_1 = pd.read_csv(
        f"results/{scenario}/energy_levels.csv", index_col=0).reset_index().set_index(
        ["Unnamed: 1", "index"]).T
    el_2 = pd.read_csv(
        f"results/{scenario_2}/energy_levels.csv", index_col=0).reset_index().set_index(
        ["Unnamed: 1", "index"]).T

    charging = pd.read_csv(
        f"results/{scenario}/charging.csv", index_col=0).reset_index().set_index(
        ["Unnamed: 1", "index"]).T
    charging_2 = pd.read_csv(
        f"results/{scenario_2}/charging.csv", index_col=0).reset_index().set_index(
        ["Unnamed: 1", "index"]).T

    plt.show()
    # plot_storage_equivalent_germany_stacked(
    #     cap_1, parameter={"state": "State"}, ylabel="Storage capacity [GW]",
    #     var="storage capacity")
    # plot_storage_equivalent_germany_stacked(
    #     cap_2, parameter={"state": "State"}, ylabel="Storage capacity [GW]",
    #     var="storage capacity")
    print("SUCCESS")