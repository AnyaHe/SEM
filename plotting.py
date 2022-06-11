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


def plot_storage_equivalent_germany_stacked(storage_equivalent,
                                            parameter={"share_pv": "Share PV [-]"}):
    param = list(parameter.items())[0][0]
    ylabel = "Energy stored [GWh]"
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
    plt.legend()#loc="lower left"
    plt.title("Storage equivalent Germany")
    plt.tight_layout()
    plt.savefig(f"results/storage_equivalent_Germany_{param}.pdf")
    plt.show()


if __name__ == "__main__":
    scenario = "Germany_v1"
    storage_equivalent = pd.read_csv(
        "results/storage_equivalents_{}.csv".format(scenario),
        index_col=0)
    plot_storage_equivalent_germany_stacked(storage_equivalent,
                                            )#{"relative_weight": "Relative weight"}
    print("SUCCESS")