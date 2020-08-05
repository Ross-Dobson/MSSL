import urllib.request
import datetime
import pathlib

import pandas as pd
import numpy as np


def import_omni_month(year, month, resolution='1min', cols='All'):
    """
    Finds local OMNI Data files, if not available attempts to download.

    Downloads files from
    https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/monthly_1min/

    Rules of the road at https://omniweb.sci.gsfc.nasa.gov/html/citing.html

    Args:
        year: the year of data to import
        month: the month of data to import
        resolution: 1min or 5min (only 1min implemented)

    Returns:
        data: a pandas DataFrame object.

    Author: Andy Smith
    Updated: Ross Dobson, August 2020

    """
    # get string version of the year and month to use in the filenames
    year_str = str(year)
    if month < 10:
        month_str = '0'+str(month)
    else:
        month_str = str(month)
    
    leap_year = leapcheck(year)

    # List end day for each month so we can cycle through
    month_end_dates = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if leap_year:
        month_end_dates[1] = 29

    start_datetime = datetime.datetime(year, month, 1)
    end_datetime = datetime.datetime(year, month, month_end_dates[(month-1)],
                                     23, 59, 59)

    # See https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/hroformat.txt
    omni_header = ['Year', 'Day', 'Hour', 'Minute', 'B_IMF_ScID',
                   'Plasma_ScID', 'IMFAvPoints', 'PlasmaAvPoints',
                   'PercentInterp', 'Timeshift', 'RMSTimeshift',
                   'RMSPhaseFrontNormal', 'TimeBetweenObs', 'B', 'B_X_GSM',
                   'B_Y_GSE', 'B_Z_GSE', 'B_Y_GSM', 'B_Z_GSM', 'RMSBScalar',
                   'RMSFieldVector', 'V', 'V_X_GSE', 'V_Y_GSE', 'V_Z_GSE',
                   'n_p', 'T', 'P', 'E', 'Beta', 'AlfvenMachff', 'X_SC_GSE',
                   'Y_SC_GSE', 'Z_SC_GSE', 'X_BSN_GSE', 'Y_BSN_GSE',
                   'Z_BSN_GSE', 'AE', 'AL', 'AU', 'SymD', 'SymH', 'AsyD',
                   'AsyH', 'PCN', 'MagnetosonicMach', '10MeVProton',
                   '30MeVProton', '60MeVProton']

    # Check if already downloaded as these files are big bois
    asc_dir = pathlib.Path('Data/OMNI/')
    asc_fname = 'OMNI_1min_' + year_str + month_str + '.asc'
    asc_path = asc_dir / asc_fname

    try:
        # headers are NOT stored in the data, so header=None
        # instead, passed in via 'names'
        data = pd.read_csv(asc_path, sep='\s+', names=omni_header, header=None)
        print('Local data found at', asc_path)

        data['DateTime'] = data.apply(
            lambda row:
            datetime.datetime(int(row.Year), 1, 1)
            + datetime.timedelta(
                days=int(row.Day) - 1)
            + datetime.timedelta(seconds=row.Hour*60*60 + row.Minute*60),
            axis=1)

    # FileNotFoundError from pd.read_csv means we need to download the data.
    except FileNotFoundError:

        print('Local data not found -> '
              + 'downloading from https://spdf.sci.gsfc.nasa.gov')
        asc_url = ('https://spdf.sci.gsfc.nasa.gov/pub/data/omni/'
                   + 'high_res_omni/monthly_1min/omni_min'
                   + year_str + month_str + '.asc')

        print('Creating local directory at ', asc_dir,
              ' (if it doesn\'t already exist)')
        pathlib.Path(asc_dir).mkdir(exist_ok=True)

        print('Done. Downloading data to: ', asc_path)
        urllib.request.urlretrieve(asc_url, asc_path)  # Saves to asc_path
        print('Data downloaded.')

        # headers NOT in data, passed in via 'names' parameter instead
        data = pd.read_csv(asc_path, sep='\s+', names=omni_header, header=None)

        data['DateTime'] = data.apply(
            lambda row:
            datetime.datetime(int(row.Year), 1, 1)
            + datetime.timedelta(days=int(row.Day) - 1)
            + datetime.timedelta(seconds=row.Hour*60*60+row.Minute*60),
            axis=1)

    # Select the data within our range
    data = data[(data.DateTime >= start_datetime)
                & (data.DateTime <= end_datetime)]

    # Bodge any borked data with NaN, as pandas knows not to plot this
    data = data.replace(99.99, np.nan)
    data = data.replace(999.9, np.nan)
    data = data.replace(999.99, np.nan)
    data = data.replace(9999.99, np.nan)
    data = data.replace(99999.9, np.nan)
    data = data.replace(9999999., np.nan)

    # Make DateTime the index of the dataframe - ie the row labels
    data.index = data['DateTime']

    # We defined this up the top, remember? Passed in.
    if cols != 'All':
        data = data[cols]

    return data


def leapcheck(year):
    '''Calculate whether the year is a leap year, return a True/False bool.

    Args:
      year: Integer of the year to check

    Returns:
      A boolean True/False on whether the year is a leap year.
    '''
    leap_year = False
    if(year % 4 != 0):
        leap_year = False
    elif(year % 100 == 0):
        if(year % 400 == 0):
            leap_year = True
        else:
            leap_year = False
    else:
        leap_year = True

    return leap_year


def import_omni_year(year):
    """docstring lolol"""

    pkl_dir = pathlib.Path('Data/OMNI/')
    pkl_fname = str(year) + '.pkl'
    pkl_path = pkl_dir / pkl_fname

    try:
        year_df = pd.read_pickle(pkl_path)
        print("Pickle found for", year, ".")

    except FileNotFoundError:
        print("Pickle not found. Looking for local .asc files to create it.")

        # load in each month with import_omni_month, store in array
        df_array = []
        for i in range(0, 12):
            this_month_df = import_omni_month(year, (i+1))
            df_array.append(this_month_df)

        # concat all the month's into one df
        year_df = pd.concat(df_array)

        # store as pickle - MUCH faster loading, saves us doing this again
        year_df.to_pickle(pkl_path)

    return year_df


def main():
    print("Don't execute me. Import me and use my functions.")


if __name__ == "__main__":
    # execute only if run as a script
    main()
