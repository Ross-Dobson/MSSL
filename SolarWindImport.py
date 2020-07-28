import urllib.request
import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import sys


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

    """

    Year = datetime.datetime.strftime(StartDateTime, '%Y')
    # DOY = datetime.strftime(StartDateTime, '%j')
    Month = datetime.datetime.strftime(StartDateTime, '%m')
    # DayOfMonth = datetime.strftime(StartDateTime, '%d')

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
    try:
        FDirLocal = Path('Data/OMNI/')
        FNameLocal = FDirLocal / ('OMNI_1min_'+Year+Month+'.asc')
        Data = pd.read_csv(FNameLocal, sep='\s+', names=Header, header=None)
        Data['DateTime'] = Data.apply(lambda row:
                                      datetime(int(row.Year), 1, 1)
                                      + timedelta(days=int(row.Day) - 1)
                                      + timedelta(seconds=row.Hour*60*60
                                                  + row.Minute*60),
                                      axis=1)
        print('Found local data at '+FNameLocal)

    # IOError from pd.read_csv if the file doesn't exist. Need to download.
    except IOError:

        print('Local data not found -> '
              + 'downloading from https://spdf.sci.gsfc.nasa.gov')
        FNameWeb = ('https://spdf.sci.gsfc.nasa.gov/pub/data/omni/' +
                    'high_res_omni/monthly_1min/omni_min'+Year+Month+'.asc')
        urllib.request.urlretrieve(FNameWeb, FNameLocal)  # Saves to FNameLocal
        print('Data downloaded.')

        Data = pd.read_csv(FNameLocal, sep='\s+', names=Header, header=None)
        Data['DateTime'] = Data.apply(lambda row:
                                      datetime(int(row.Year), 1, 1)
                                      + timedelta(days=int(row.Day) - 1)
                                      + timedelta(seconds=row.Hour*60*60
                                                  + row.Minute*60),
                                      axis=1)

        Data = Data[(Data.DateTime >= StartDateTime)
                    & (Data.DateTime <= EndDateTime)]

        # Bodge any borked data with NaN.
        Data = Data.replace(99.99, np.nan)
        Data = Data.replace(999.9, np.nan)
        Data = Data.replace(999.99, np.nan)
        Data = Data.replace(9999.99, np.nan)
        Data = Data.replace(99999.9, np.nan)
        Data = Data.replace(9999999., np.nan)

    Data.index = Data['DateTime']

    if Columns != 'All':

        Data = Data[Columns]

    return Data


def main():
    # I'll probably have to change this for looping over a whole year, but
    # thats future me's problem
    startDT = datetime.datetime(2003, 10, 1)
    endDT = datetime.datetime(2003, 10, 31)
    data = import_omni_month(startDT, endDT)
    data


main()
