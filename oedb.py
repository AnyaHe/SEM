import os
import pandas as pd
from sqlalchemy import func
import logging
import sys
import numpy

sys.path.append(r"U:\Software\SEST\eDisGo")

from edisgo.tools import session_scope
from edisgo.tools.config import Config
from edisgo.network.timeseries import import_load_timeseries

logger = logging.getLogger("edisgo")

if "READTHEDOCS" not in os.environ:
    from egoio.db_tables import model_draft, supply, demand
    from shapely.ops import transform
    from shapely.wkt import loads as wkt_loads


def oedb(generator_scenario):
    """
    Gets generator park for specified scenario from oedb and integrates them
    into the grid.

    The importer uses SQLAlchemy ORM objects.
    These are defined in
    `ego.io <https://github.com/openego/ego.io/tree/dev/egoio/db_tables/>`_.

    For further information see also :attr:`~.EDisGo.import_generators`.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    generator_scenario : str
        Scenario for which to retrieve generator data. Possible options
        are 'nep2035' and 'ego100'.

    Other Parameters
    ----------------
    remove_decommissioned : bool
        If True, removes generators from network that are not included in
        the imported dataset (=decommissioned). Default: True.
    update_existing : bool
        If True, updates capacity of already existing generators to
        capacity specified in the imported dataset. Default: True.
    p_target : dict or None
        Per default, no target capacity is specified and generators are
        expanded as specified in the respective scenario. However, you may
        want to use one of the scenarios but have slightly more or less
        generation capacity than given in the respective scenario. In that case
        you can specify the desired target capacity per technology type using
        this input parameter. The target capacity dictionary must have
        technology types (e.g. 'wind' or 'solar') as keys and corresponding
        target capacities in MW as values.
        If a target capacity is given that is smaller than the total capacity
        of all generators of that type in the future scenario, only some of
        the generators in the future scenario generator park are installed,
        until the target capacity is reached.
        If the given target capacity is greater than that of all generators
        of that type in the future scenario, then each generator capacity is
        scaled up to reach the target capacity. Be careful to not have much
        greater target capacities as this will lead to unplausible generation
        capacities being connected to the different voltage levels.
        Also be aware that only technologies specified in the dictionary are
        expanded. Other technologies are kept the same.
        Default: None.
    allowed_number_of_comp_per_lv_bus : int
        Specifies, how many generators are at most allowed to be placed at
        the same LV bus. Default: 2.

    """

    def _import_res_generators(session):
        """
        Import data for renewable generators from oedb.

        Returns
        -------
        (:pandas:`pandas.DataFrame<DataFrame>`,
         :pandas:`pandas.DataFrame<DataFrame>`)
            Dataframe containing data on all renewable MV and LV generators.
            You can find a full list of columns in
            :func:`edisgo.io.import_data.update_grids`.

        Notes
        -----
        If subtype is not specified it is set to 'unknown'.

        """

        # build basic query
        generators_sqla = (
            session.query(
                orm_re_generators.columns.id,
                orm_re_generators.columns.id.label("generator_id"),
                orm_re_generators.columns.subst_id,
                orm_re_generators.columns.la_id,
                orm_re_generators.columns.mvlv_subst_id,
                orm_re_generators.columns.electrical_capacity.label("p_nom"),
                orm_re_generators.columns.generation_type.label(
                    "generator_type"),
                orm_re_generators.columns.generation_subtype.label(
                    "subtype"),
                orm_re_generators.columns.voltage_level,
                orm_re_generators.columns.w_id.label("weather_cell_id"),
                func.ST_AsText(
                    func.ST_Transform(
                        orm_re_generators.columns.rea_geom_new, srid
                    )
                ).label("geom"),
                func.ST_AsText(
                    func.ST_Transform(orm_re_generators.columns.geom, srid)
                ).label("geom_em"),
            ).filter(
                orm_re_generators_version)
        )

        # extend basic query for MV generators and read data from db

        gens = pd.read_sql_query(
            generators_sqla.statement,
            session.bind,
            index_col="id"
        )

        # define generators with unknown subtype as 'unknown'
        gens.loc[
            gens["subtype"].isnull(), "subtype"
        ] = "unknown"

        # convert capacity from kW to MW
        gens.p_nom = pd.to_numeric(gens.p_nom) / 1e3


        return gens

    oedb_data_source = "versioned"
    srid = 4326

    # load ORM names

    orm_re_generators_name = (
            't_ego_dp_res_powerplant_'
            + generator_scenario
            + '_mview'
    )

    data_version = 'v0.4.5'

    # import ORMs
    orm_re_generators = supply.__getattribute__(orm_re_generators_name)

    # set version condition
    orm_re_generators_version = (
            orm_re_generators.columns.version == data_version
    )

    # get conventional and renewable generators
    with session_scope() as session:
        generators = _import_res_generators(session)

    return generators

    # update time series if they were already set
    # if not edisgo_object.timeseries.generators_active_power.empty:
    #     add_generators_timeseries(
    #         edisgo_obj=edisgo_object,
    #         generator_names=edisgo_object.topology.generators_df.index)


def get_wind_power_classes(df):
    df["power_class"] = 7
    df.loc[df.p_nom < 3.1, "power_class"] = 6
    df.loc[df.p_nom < 2.4, "power_class"] = 5
    df.loc[df.p_nom < 2.1, "power_class"] = 4
    df.loc[df.p_nom < 1.6, "power_class"] = 3
    df.loc[df.p_nom < 1.1, "power_class"] = 2
    df.loc[df.p_nom < 0.7, "power_class"] = 1
    df.loc[df.subtype == "wind_offshore", "power_class"] = 0
    return df


def import_feedin_timeseries(weather_cell_ids, timeindex):
    """
    Import RES feed-in time series data and process

    ToDo: Update docstring.

    Parameters
    ----------
    config_data : dict
        Dictionary containing config data from config files.
    weather_cell_ids : :obj:`list`
        List of weather cell id's (integers) to obtain feed-in data for.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with time series for active power feed-in, normalized to
        a capacity of 1 MW.

    """

    def _retrieve_timeseries_from_oedb(session, timeindex):
        """Retrieve time series from oedb

        """
        # ToDo: add option to retrieve subset of time series
        # ToDo: find the reference power class for mvgrid/w_id and insert
        #  instead of 4
        feedin_sqla = (
            session.query(
                orm_feedin.w_id, orm_feedin.source, orm_feedin.feedin, orm_feedin.power_class
            )
            .filter(orm_feedin.w_id.in_(weather_cell_ids))
            .filter(orm_feedin_version)
            .filter(
                orm_feedin.weather_year.in_(timeindex.year.unique().values)
            )
        )

        feedin = pd.read_sql_query(
            feedin_sqla.statement, session.bind,
            index_col=["source", "w_id", "power_class"]
        )
        return feedin

    orm_feedin_name = 'EgoRenewableFeedin'
    orm_feedin = supply.__getattribute__(orm_feedin_name)
    orm_feedin_version = (
        orm_feedin.version == 'v0.4.5'
    )

    if timeindex is None:
        timeindex = pd.date_range("1/1/2011", periods=8760, freq="H")

    with session_scope() as session:
        feedin = _retrieve_timeseries_from_oedb(session, timeindex)

    if feedin.empty:
        raise ValueError(
            "The year you inserted could not be imported from "
            "the oedb. So far only 2011 is provided. Please "
            "check website for updates."
        )

    feedin.sort_index(axis=0, inplace=True)

    recasted_feedin_dict = {}
    for type_w_id in feedin.index:
        recasted_feedin_dict[type_w_id] = feedin.loc[type_w_id, :].values[0]

    # Todo: change when possibility for other years is given
    conversion_timeindex = pd.date_range("1/1/2011", periods=8760, freq="H")
    feedin = pd.DataFrame(recasted_feedin_dict, index=conversion_timeindex)

    # rename 'wind_onshore' and 'wind_offshore' to 'wind'
    # new_level = [
    #     _ if _ not in ["wind_onshore", "wind_offshore"] else "wind"
    #     for _ in feedin.columns.levels[0]
    # ]
    # feedin.columns.set_levels(new_level, level=0, inplace=True, verify_integrity=False)

    feedin.columns.rename("type", level=0, inplace=True)
    feedin.columns.rename("weather_cell_id", level=1, inplace=True)

    return feedin.loc[timeindex]


def combine_solar_timeseries(gens_solar, generation):
    return gens_solar.apply(
        lambda x: generation["solar"][
                      x.name
                  ][0].T
                  * x.p_nom,
        axis=1,
    ).T


def combine_wind_timeseries(gens_wind, generation):
    return gens_wind.apply(
        lambda x: generation[x.name[0]][
                    (int(x.name[1]), x.name[2])
                  ].T
                  * x.p_nom,
        axis=1,
    ).T


def oedb_import_demand():
    def _retrieve_load_fs_from_oedb(session):
        """Retrieve time series from oedb

        """
        load_fs_sqla = (
            session.query(
                orm_load_fs.federal_states,
                orm_load_fs.elec_consumption_households,
                orm_load_fs.elec_consumption_industry,
                orm_load_fs.elec_consumption_tertiary_sector
            )
        )

        load = pd.read_sql_query(
            load_fs_sqla .statement, session.bind,
            index_col=["federal_states"]
        )
        return load

    def _retrieve_load_la_from_oedb(session):
        """Retrieve time series from oedb

        """
        # ToDo: add option to retrieve subset of time series
        # ToDo: find the reference power class for mvgrid/w_id and insert
        #  instead of 4
        load_la_sqla = (
            session.query(
                orm_load_la.id, orm_load_la.sector_consumption_residential,
                orm_load_la.sector_consumption_retail,
                orm_load_la.sector_consumption_industrial,
                orm_load_la.sector_consumption_agricultural
            )
            .filter(orm_load_la_version)
        )

        load_la = pd.read_sql_query(
            load_la_sqla.statement, session.bind,
            index_col=["id"]
        )
        return load_la
    orm_load_fs_name = 'EgoDemandFederalstate'
    orm_load_fs = demand.__getattribute__(orm_load_fs_name)

    with session_scope() as session:
        load_fs = _retrieve_load_fs_from_oedb(session)

    orm_load_la_name = 'EgoDpLoadarea'
    orm_load_la = demand.__getattribute__(orm_load_la_name)
    orm_load_la_version = (
            orm_load_la.version == 'v0.4.5'
    )
    with session_scope() as session:
        load_la = _retrieve_load_la_from_oedb(session)

    return load_fs, load_la


if __name__ == "__main__":
    load_fs, load_la = oedb_import_demand()
    load_fs.loc["Deutschland", "retail"] = \
        load_la["sector_consumption_retail"].sum()
    load_fs.loc["Deutschland", "agricultural"] = \
        load_la["sector_consumption_agricultural"].sum()
    load_fs.rename(columns={"elec_consumption_households": "residential",
                            "elec_consumption_industry": "industrial"},
                   inplace=True)
    annual_consumptions = load_fs.loc["Deutschland",
                                      ["agricultural", "industrial",
                                       "residential", "retail"]]
    config = Config()
    timeseries = import_load_timeseries(config, "demandlib", 2011)
    scaled_ts = timeseries.multiply(annual_consumptions)
    scaled_ts.to_csv("demand_germany_ego100.csv")
    generators = oedb("ego100")
    # generators["weather_cell_id"] = generators["weather_cell_id"].astype(int)
    # Extract wind generators and group them into power classes
    wind_generators = generators.loc[generators.generator_type == "wind"]
    wind_generators = get_wind_power_classes(wind_generators)
    grouped_wind_generators = wind_generators[["subtype", "p_nom", "weather_cell_id", "power_class"]]\
        .groupby(["subtype", "weather_cell_id", "power_class"]).sum()
    # Extract solar generators and group them into weather_cells
    solar_generators = generators.loc[generators.generator_type == "solar"]
    grouped_solar_generators = solar_generators[["p_nom", "weather_cell_id"]]\
        .groupby("weather_cell_id").sum()
    # get timeseries data
    timeindex = pd.date_range("1/1/2011", periods=8760, freq="H")
    weather_cell_ids = [int(id) for id in wind_generators.append(solar_generators).weather_cell_id.unique()
                        if not numpy.isnan(id)]
    feedin_ts = import_feedin_timeseries(weather_cell_ids, timeindex)
    #included_weather_cells = feedin_ts.columns.get_level_values(1)
    #wind_weather_cells = grouped_wind_generators.index.get_level_values(0)
    # combine both
    solar_ts = combine_solar_timeseries(grouped_solar_generators, feedin_ts)
    wind_ts = combine_wind_timeseries(grouped_wind_generators, feedin_ts)
    reference_ts = pd.DataFrame()
    reference_ts["wind"] = wind_ts.sum(axis=1)
    reference_ts["solar"] = solar_ts.sum(axis=1)
    reference_ts.to_csv("data/vres_reference_ego100.csv")
    print("SUCCESS")