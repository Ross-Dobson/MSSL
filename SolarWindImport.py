import urllib.request
import datetime
import pathlib  # for compatibility with non UNIX/POSIX systems (ie Windows)

import pandas as pd
import numpy as np

def import_omni_month(year, month, resolution='1min', cols='All'):
    """
    Finds local OMNI Data files, if not available attempts to download.

    Downloads files from
    https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/monthly_1min/

    Rules of the road at https://omniweb.sci.gsfc.nasa.gov/html/citing.html

    Args:
        year: the year of the data to import
        month: the month of the data to import
        resolution: 1min or 5min (only 1min implemented)

    Returns:
        data: a pandas DataFrame object.

    Author: Andy Smith
    Updated: Ross Dobson, August 2020

    """

    # get string version of the year and month to use in the filenames
    year_str = str(year)
    if month < 10:
        month_str = '0'+str(month)  # needs to be e.g. "05" not "5"
    else:
        month_str = str(month)

    leap_year = leapcheck(year)  # is it a leap year (is feb 28 or 29 days)

    # List end day for each month so we can cycle through
    month_end_dates = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if leap_year:
        month_end_dates[1] = 29  # feb now has 29 as leap year

    # make datetime objects. End needs to be 23:59:59 to get all of last day
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

    # Use pickles, MUCH faster than the .asc files
    pkl_dir = pathlib.Path('Data/OMNI/pickles')

    mon_str = str(month)
    if (month < 10):
        mon_str = '0' + str(month)

    pkl_fname = str(year) + mon_str + '.pkl'
    pkl_path = pkl_dir / pkl_fname

    # Look for the pickle
    try:
        data = pd.read_pickle(pkl_path)
        print("Pickle found:", pkl_fname)

    except FileNotFoundError:
        print("Pickle not found. Looking for local .asc files to create it.")

        # Check if already downloaded because these files are big
        asc_dir = pathlib.Path('Data/OMNI/asc/')
        asc_fname = 'OMNI_1min_' + year_str + month_str + '.asc'
        asc_path = asc_dir / asc_fname
        try:
            # headers are NOT stored in the NASA asc data files, so header=None
            # instead, manually passed in via 'names'. Not ideal, but it works
            data = pd.read_csv(asc_path, sep='\s+',
                               names=omni_header, header=None)
            print('Local data found at', asc_path)

            # not entirely sure what this is doing - just generating the
            # column of datetimes to use as the index?
            data['DateTime'] = data.apply(
                lambda row:
                datetime.datetime(int(row.Year), 1, 1)
                + datetime.timedelta(
                    days=int(row.Day) - 1)
                + datetime.timedelta(seconds=row.Hour*60*60 + row.Minute*60),
                axis=1)

        # FileNotFoundError from pd.read_csv means we need to download the data
        except FileNotFoundError:

            print('Local .asc not found -> '
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
        data = pd.read_csv(asc_path, sep='\s+',
                           names=omni_header, header=None)

        # same as above, think it just generates datetimes
        data['DateTime'] = data.apply(
            lambda row:
            datetime.datetime(int(row.Year), 1, 1)
            + datetime.timedelta(days=int(row.Day) - 1)
            + datetime.timedelta(seconds=row.Hour*60*60+row.Minute*60),
            axis=1)

        # Select the data within our time range
        data = data[(data.DateTime >= start_datetime)
                    & (data.DateTime <= end_datetime)]

        # Bodge broken data with NaN, easier to interpolate, pd is happier
        data = data.replace(99.99, np.nan)
        data = data.replace(999.9, np.nan)
        data = data.replace(999.99, np.nan)
        data = data.replace(9999.99, np.nan)
        data = data.replace(99999.9, np.nan)
        data = data.replace(9999999., np.nan)

        # Make DateTime the index of the dataframe - ie the row labels
        data.index = data['DateTime']

        # In case we only wanted specific columns
        if cols != 'All':
            data = data[cols]

        # store as pickle - MUCH faster loading, saves us doing this again
        data.to_pickle(pkl_path)

    return data


def leapcheck(year):
    '''Calculate whether the year is a leap year, return a True/False.
    TBH I based this off psuedo-code from Wikipedia.

    Args:
      year: Integer of the year to check

    Returns:
      A boolean True/False on whether the year is a leap year.
    '''
    leap_year = False
    if(year % 4 != 0):  # has to be divisible by 4
        leap_year = False
    elif(year % 100 == 0):
        if(year % 400 == 0):  # year XX00 isn't leap unless multiple of 400
            leap_year = True
        else:
            leap_year = False
    else:
        leap_year = True

    return leap_year


def import_omni_year(year):
    """Uses import_omni_month but concatenates it into an entire year.

    See the dosctring for import_omni_month above.

    Args:
      year: the year of data to get

    Returns:
      data: a pandas DataFrame object containing the year of data.
    """

    # Use pickles, MUCH faster than the .asc files
    pkl_dir = pathlib.Path('Data/OMNI/pickles')
    pkl_fname = str(year) + '.pkl'
    pkl_path = pkl_dir / pkl_fname

    # Look for the pickle
    try:
        year_df = pd.read_pickle(pkl_path)
        print("Pickle found:", pkl_fname)

    except FileNotFoundError:
        print("Pickle not found. Looking for local .asc files to create it.")

        # load in each month with import_omni_month, store the df in array
        df_array = []
        for i in range(0, 12):  # as 0->11, we need to use i+1 for months
            this_month_df = import_omni_month(year, (i+1))
            df_array.append(this_month_df)

        # concat all the month's into one df
        year_df = pd.concat(df_array)

        # store as pickle - MUCH faster loading, saves us doing this again
        year_df.to_pickle(pkl_path)

    return year_df


def import_storm_week(year, month, day):
    """
    Imports the week of data surrounding a storm.
    Can deal with storms spanning month boundary as imports multiple months.

    Args:
        year: the year of the data to import
        month: the month of the data to import

    Returns:
        data: a pandas DataFrame object.

    Ross Dobson, September 2020
    """

    df_array = []
    for i in range(month-1, month+2):
        if (i == 13):
            this_month_df = import_omni_month(year+1, 1)
        elif (i == 0):
            this_month_df = import_omni_month(year-1, 12)
        else:
            this_month_df = import_omni_month(year, i)

        df_array.append(this_month_df)

    # concat the three months together into one df_2003
    storm_df = pd.concat(df_array)

    # storm datetime - START of storm
    storm_dt = datetime.datetime(year, month, day)
    # start datetime -3 days
    start_dt = storm_dt - datetime.timedelta(days=3)
    # end datetime +4 days
    end_dt = storm_dt + datetime.timedelta(days=4)
    # this way we get 3 days either side
    storm_df = storm_df[(storm_df.DateTime >= start_dt)
                        & (storm_df.DateTime <= end_dt)]

    return storm_df


def storm_interpolator(my_df):
    """Straight-line interpolates gaps (NaNs) less than 15 minutes.
    After that, it just removes any remaining NaNs inplace."""
    # df_resampled = my_df.resample(
    #     '1T', loffset=datetime.timedelta(seconds=30.)).mean()
    df_resampled = my_df.interpolate(method='linear', limit=15)
    df_resampled.dropna(axis='index', how='any', inplace=True)

    return df_resampled


def storm_chunker(y_true, y_pred, y_pers, resolution='1h'):
    """
    Splits the storm into chunks of timesteps (1h and 6h implemented)
    If there's any gaps due to removed NaNs, it just means that hour will
    have slightly less datapoints in it. E.g. the borders of the chunk will
    still be 60 minutes apart, if not 60 datapoints apart.

    Args:
      y_true: the AL of true (discretized) AL to compare against
      y_pred: the predicted values of AL from the model
      y_pers: the persistence/AL history values

    Returns:
      chunks: array of 3-col dfs, each col is true, pred, pers

    Author:
      Ross Dobson 2020-10-06
    """

    first_dt = y_true.index[0]  # start_dt cannot precede this 
    limit_dt = y_true.index[-1]  # end_dt cannot exceed this

    year = int(first_dt.strftime("%Y"))
    month = int(first_dt.strftime("%m"))
    day = int(first_dt.strftime("%d"))
    hour = int(first_dt.strftime("%H"))

    # set it to the first (hopefully) full hour
    if (resolution == '1h'):
        start_dt = datetime.datetime(year, month, day, hour, 0)
        end_dt = start_dt + datetime.timedelta(minutes=59)
        interval = 1

    elif (resolution == '6h'):
        start_dt = datetime.datetime(year, month, day, hour, 0)
        end_dt = start_dt + datetime.timedelta(hours=5, minutes=59)
        interval = 6

    else:
        raise ValueError('This resolution', resolution, 'is not implemented.')

    chunks = []
    while (end_dt <= limit_dt):

        # get the chunk
        true_chunk = y_true.loc[start_dt:end_dt]
        pers_chunk = y_pers.loc[start_dt:end_dt]
        pred_chunk = y_pred.loc[start_dt:end_dt]

        # make one df for all three, append
        chunks.append(pd.concat(
            [true_chunk, pred_chunk, pers_chunk], axis=1, sort=False))

        # by iterating hours=1, we keep the XX:00 -> XX:59 spacing intact
        start_dt += datetime.timedelta(hours=interval)
        end_dt += datetime.timedelta(hours=interval)

    # now that every chunk for this storm is generated, return
    return chunks



def main():
    print("Don't execute this file. Import it and use its functions.")


if __name__ == "__main__":
    # execute only if run as a script
    main()
