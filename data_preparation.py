import pandas as pd
import numpy as np

BANDS = ["upper_power", "upper_energy", "lower_energy"]


def import_flexibility_bands(dir, use_cases):
    flexibility_bands = {}

    for band in BANDS:
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
            flexibility_bands[band] = flexibility_bands[band] + 1e-5
        elif "lower" in band:
            flexibility_bands[band] = flexibility_bands[band] - 1e-5
    return flexibility_bands