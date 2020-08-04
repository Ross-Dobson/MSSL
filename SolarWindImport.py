import urllib.request
import datetime
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # ignore the 'unused' error, pandas needs it


def import_omni_month(StartDateTime, EndDateTime, Resolution='1min',
                      Columns='All'):

    """
    Finds local OMNI Data files, if not available attempts to download from
    https://spdf.sci.gsfc.nasa.gov.

    Rules of the Road: https://omniweb.sci.gsfc.nasa.gov/html/citing.html

    Arguments:

    StartDateTime - datetime for the start of the requested data interval
    EndDateTime -   datetime for the end of the requested data interval
    Resolution -    1min or 5min -> only 1min implemented

    Returns:

    Data -  DataFrame

    Author: Andy Smith
    Updated: Ross Dobson, July 2020

    """

    Year = datetime.datetime.strftime(StartDateTime, '%Y')
    Month = datetime.datetime.strftime(StartDateTime, '%m')

    # See https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/hroformat.txt
    Header = ['Year', 'Day', 'Hour', 'Minute', 'B_IMF_ScID', 'Plasma_ScID',
              'IMFAvPoints', 'PlasmaAvPoints', 'PercentInterp', 'Timeshift',
              'RMSTimeshift', 'RMSPhaseFrontNormal', 'TimeBetweenObs', 'B',
              'B_X_GSM', 'B_Y_GSE', 'B_Z_GSE', 'B_Y_GSM', 'B_Z_GSM',
              'RMSBScalar', 'RMSFieldVector', 'V', 'V_X_GSE', 'V_Y_GSE',
              'V_Z_GSE', 'n_p', 'T', 'P', 'E', 'Beta', 'AlfvenMachff',
              'X_SC_GSE', 'Y_SC_GSE', 'Z_SC_GSE', 'X_BSN_GSE', 'Y_BSN_GSE',
              'Z_BSN_GSE', 'AE', 'AL', 'AU', 'SymD', 'SymH', 'AsyD', 'AsyH',
              'PCN', 'MagnetosonicMach', '10MeVProton', '30MeVProton',
              '60MeVProton']

    # Check if already downloaded as these files are big bois
    FDirLocal = pathlib.Path('Data/OMNI/')
    FNameLocal = FDirLocal / ('OMNI_1min_'+Year+Month+'.asc')

    try:
        # headers NOT in data, passed in via 'names' parameter instead
        Data = pd.read_csv(FNameLocal, sep='\s+', names=Header, header=None)
        print('Local data found at', FNameLocal)

        Data['DateTime'] = Data.apply(
            lambda row:
            datetime.datetime(int(row.Year), 1, 1)
            + datetime.timedelta(
                days=int(row.Day) - 1)
            + datetime.timedelta(seconds=row.Hour*60*60 + row.Minute*60),
            axis=1)

    # FileNotFoundError from pd.read_csv means we need to download.
    except FileNotFoundError:

        print('Local data not found -> '
              + 'downloading from https://spdf.sci.gsfc.nasa.gov')
        FNameWeb = ('https://spdf.sci.gsfc.nasa.gov/pub/data/omni/' +
                    'high_res_omni/monthly_1min/omni_min'+Year+Month+'.asc')

        print('Creating local directory (skips if already exists)')
        pathlib.Path(FDirLocal).mkdir(exist_ok=True)

        print('Done. Downloading data to: ', FNameLocal)
        urllib.request.urlretrieve(FNameWeb, FNameLocal)  # Saves to FNameLocal
        print('Data downloaded.')

        # headers NOT in data, passed in via 'names' parameter instead
        Data = pd.read_csv(FNameLocal, sep='\s+', names=Header, header=None)

        Data['DateTime'] = Data.apply(
            lambda row:
            datetime.datetime(int(row.Year), 1, 1)
            + datetime.timedelta(days=int(row.Day) - 1)
            + datetime.timedelta(seconds=row.Hour*60*60+row.Minute*60),
            axis=1)

    # Select the data within our range
    Data = Data[(Data.DateTime >= StartDateTime)
                & (Data.DateTime <= EndDateTime)]

    # Bodge any borked data with NaN, as pandas knows not to plot this
    Data = Data.replace(99.99, np.nan)
    Data = Data.replace(999.9, np.nan)
    Data = Data.replace(999.99, np.nan)
    Data = Data.replace(9999.99, np.nan)
    Data = Data.replace(99999.9, np.nan)
    Data = Data.replace(9999999., np.nan)

    # Make DateTime the index of the dataframe - ie the row labels
    Data.index = Data['DateTime']

    # We defined this up the top, remember? Passed in.
    if Columns != 'All':
        Data = Data[Columns]

    return Data


def main():

    # Let's start with an empty dataframe, but with our headers
    omni_headers = ['Year', 'Day', 'Hour', 'Minute', 'B_IMF_ScID',
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

    # make two blank dataframes - one for the year, one for our data
    data = pd.DataFrame(columns=omni_headers)
    october_data = pd.DataFrame(columns=omni_headers)

    year = 2003

    # Check if it's a leap year or not
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

    # List end day for each month so we can cycle through
    month_end_dates = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if(leap_year):
        month_end_dates[1] = 29

    # Blank array to store our DateTime objects in
    startDT = []
    endDT = []

    for i in range(0, 12):
        startDT.append(datetime.datetime(year, (i+1), 1))
        endDT.append(datetime.datetime(year, (i+1), month_end_dates[i],
                                       23, 59, 59))
        this_month_data = import_omni_month(
            startDT[i], endDT[i], Resolution='1min', Columns='All')
        if (i == 9):
            october_data = this_month_data
        if (i == 10):
            november_data = this_month_data
        # print(this_month_data)
        data = pd.concat([data, this_month_data])  # add it into the main DF

    interest_data = pd.concat([october_data,november_data])
    print(data)
    plot_vals = ['B_X_GSM', 'B_Y_GSM', 'B_Z_GSM', 'n_p', 'AL', 'P', 'V']
    for val in plot_vals:
        
        # year_title = str(val) + ' in ' + str(year)
        # data.plot(x='DateTime', y=val, title=year_title)
       
        # interest_title = str(val) + ' in October and November 2003'
        # interest_data.plot(x='DateTime', y=val, title=interest_title)
        plotvals_data = data[plot_vals]
        plotvals_interest = interest_data[plot_vals]

        print(plotvals_data.corr())
        print(plotvals_interest.corr())

        
main()
