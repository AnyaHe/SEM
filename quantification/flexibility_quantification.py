import pandas as pd
import numpy as np
#hello

def shifting_time(netload, reference_curve=None):
    """
    Method to determine the required shifting times in order to obtain the
    course of the inserted reference curve. The annual consumption of the
    reference curve should be the same as for netload. If no reference curve
    is passed the shifting times required to obtain a flat netload curve are
    determined.

    :param netload: timeseries of residual load at interest
    :param reference_curve: reference curve which should be met
    :return: pd.DataFrame
    """
    storage_equivalent = \
        pd.DataFrame(columns=['surplus_index', 'deficit_index',
                              'storage_duration', 'energy_shifted'])
    energy_imbalance = pd.DataFrame()
    # get difference between actual netload and aimed reference
    if reference_curve is None:
        annual_consumption = netload.sum()
        mean_consumption = annual_consumption/len(netload)
        energy_imbalance['difference'] = netload - mean_consumption
    else:
        energy_imbalance['difference'] = netload - reference_curve
    # initialise used energy and counter for shifting events
    energy_imbalance['used_amount'] = 0
    # determine shifting events
    for timestep in energy_imbalance.index:
        # if timestep == pd.to_datetime('2011-01-02T16:15:00'):
        #     print('check this.')
        amount_to_shift = energy_imbalance.loc[timestep, 'difference'] - \
                          energy_imbalance.loc[timestep, 'used_amount']
        if amount_to_shift == 0:
            continue
        available_shifting = \
            energy_imbalance['difference']-energy_imbalance['used_amount']
        storage_equivalent, energy_imbalance = \
            determine_all_shifting_events_for_single_timeindex(
                amount_to_shift, energy_imbalance,
                storage_equivalent, timestep, available_shifting)
    return storage_equivalent


def determine_all_shifting_events_for_single_timeindex(amount_to_shift,
                                                       energy_imbalance,
                                                       storage_equivalent,
                                                       timestep,
                                                       available_shifting):
    """
    Method to determine required shifting by checking the next available
    points in time with opposite sign.

    :param amount_to_shift:
    :param energy_imbalance:
    :param storage_equivalent:
    :param timestep:
    :param available_shifting:
    :return:
    """
    # Handling of last timestep
    if timestep == energy_imbalance.iloc[-1].name:
        if np.isclose(amount_to_shift, 0):
            print('Energy balanced. Shifting time finished.')
        else:
            print('Energy is not balanced. Rest of {} is not shifted.'.format(
                amount_to_shift))
        return storage_equivalent, energy_imbalance
    # Intermediate timesteps
    if amount_to_shift > 0:
        usable_cumulative_shifting = \
            available_shifting.loc[available_shifting < 0].cumsum()
        if len(usable_cumulative_shifting) == 1:
            number_of_shifts = 1
        else:
            number_of_shifts = (np.abs(usable_cumulative_shifting) <
                                np.abs(amount_to_shift)).sum() + 1
        cumulative_shifting = \
            usable_cumulative_shifting.iloc[:number_of_shifts]
        deficit_indices = cumulative_shifting.index
        surplus_indices = [timestep] * len(deficit_indices)
    else:
        usable_cumulative_shifting = \
            available_shifting.loc[available_shifting > 0].cumsum()
        if len(usable_cumulative_shifting) == 1:
            number_of_shifts = 1
        else:
            number_of_shifts = (np.abs(usable_cumulative_shifting) <
                                np.abs(amount_to_shift)).sum() + 1
        cumulative_shifting = \
            usable_cumulative_shifting.iloc[:number_of_shifts]
        surplus_indices = cumulative_shifting.index
        deficit_indices = [timestep] * len(surplus_indices)

    storage_duration = cumulative_shifting.index - timestep
    if (storage_duration < pd.to_timedelta(0)).any():
        print('DEBUG: check this.')
    # update energy imbalance
    energy_imbalance.loc[timestep, 'used_amount'] = \
        energy_imbalance.loc[timestep, 'used_amount'] + amount_to_shift
    if number_of_shifts > 1:
        energy_imbalance.loc[cumulative_shifting.index[:-1], 'used_amount'] = \
            energy_imbalance.loc[cumulative_shifting.index[:-1], 'difference']
        energy_imbalance.loc[cumulative_shifting.index[-1], 'used_amount'] = \
            -(amount_to_shift + cumulative_shifting.iloc[-2])
    else:
        energy_imbalance.loc[cumulative_shifting.index, 'used_amount'] = \
            energy_imbalance.loc[cumulative_shifting.index, 'used_amount'] -\
            amount_to_shift

    energy_shifted = energy_imbalance.loc[cumulative_shifting.index,
                                          'used_amount'].values
    storage_equivalent = storage_equivalent.append(
        pd.DataFrame({'storage_duration': storage_duration,
                      'surplus_index': surplus_indices,
                      'deficit_index': deficit_indices,
                      'energy_shifted': energy_shifted}), ignore_index=True)
    return storage_equivalent, energy_imbalance


def determine_shifting(amount_to_shift, counter_shifts, energy_imbalance,
                       storage_equivalent, surplus_index,  deficit_index,
                       stored_energy):
    """

    :param amount_to_shift:
    :param counter_shifts:
    :param energy_imbalance:
    :param storage_equivalent:
    :param surplus_index:
    :param deficit_index:
    :param stored_energy:
    :return:
    """
    energy_imbalance.loc[surplus_index, 'used_amount'] = stored_energy
    amount_to_shift += stored_energy
    if surplus_index < deficit_index:
        storage_duration = deficit_index - surplus_index  # ggf. timediff?
    else:
        storage_duration = surplus_index - deficit_index
        amount_to_shift -= stored_energy
    storage_equivalent.loc[counter_shifts, 'storage_duration'] = storage_duration
    storage_equivalent.loc[counter_shifts, 'surplus_index'] = surplus_index
    storage_equivalent.loc[counter_shifts, 'deficit_index'] = deficit_index
    storage_equivalent.loc[counter_shifts, 'energy_shifted'] = stored_energy
    counter_shifts += 1
    return amount_to_shift, counter_shifts, storage_equivalent, energy_imbalance


def set_storage_equivalent(storage_equivalent, event_count,
                           surplus_index, deficit_index, stored_energy):
    """
    Method to set parameters of storage equivalent.

    :param storage_equivalent:
    :param event_count:
    :param surplus_index:
    :param deficit_index:
    :param stored_energy:
    :return:
    """
    if surplus_index < deficit_index:
        storage_duration = deficit_index - surplus_index # ggf. timediff?
    else:
        storage_duration = surplus_index - deficit_index
    storage_equivalent.at[event_count, 'storage_duration'] = storage_duration
    storage_equivalent.at[event_count, 'surplus_index'] = surplus_index
    storage_equivalent.at[event_count, 'deficit_index'] = deficit_index
    storage_equivalent.at[event_count, 'energy_shifted'] = stored_energy
    return storage_equivalent
