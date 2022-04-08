import pandas as pd
from pypsa import Network
import matplotlib.pyplot as plt

grid_dir = r"C:\Users\aheider\Documents\Grids\diss_ulf_eGo100_base_results_5lopf"

nw = Network(grid_dir)

reference_timeseries = pd.DataFrame()
scaled_timeseries = pd.DataFrame()
for carrier, components in nw.generators.groupby("carrier").groups.items():
    reference_timeseries[carrier] = nw.generators_t.p[components].sum(axis=1)
    scaled_timeseries[carrier] = \
        reference_timeseries[carrier]/nw.generators.loc[components, "p_nom"].sum()

# Save reference timeseries of Germany
reference_timeseries[["solar", "wind_onshore", "wind_offshore"]].sum(axis=1).to_csv(
    "data/vres_reference.csv")
for disp in ["biomass", "gas", "geothermal", "reservoir", "run_of_river"]:
    scaled_timeseries[disp].to_csv("data/scaled_ts_{}.csv".format(disp))

reference_timeseries.plot()
plt.show()
print("SUCCESS")