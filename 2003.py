import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
# remember to run 'matplotlib tk' in the interpreter

from SolarWindImport import import_omni_month, import_omni_year


def main():

    # 1: IMPORTING THE DATA

    year = 2003
    df_2003 = import_omni_year(year)

    pkl_dir = pathlib.Path('Data/OMNI/pickles')
    oct_nov_pkl_path = pkl_dir / 'oct_nov_2003.pkl'

    try:
        df_oct_nov_2003 = pd.read_pickle((oct_nov_pkl_path))
        print("Found the pickle for October and November 2003.")

    except FileNotFoundError:
        print("Couldn't find the pickle for October and November 2003."
              + "-> Generating one from the .asc files.")

        print('Creating local directory at ', pkl_dir,
              ' (if it doesn\'t already exist):')
        pathlib.Path(pkl_dir).mkdir(exist_ok=True)
        print("Done.")

        df_oct_2003 = import_omni_month(2003, 10)
        df_nov_2003 = import_omni_month(2003, 11)
        df_oct_nov_2003 = pd.concat([df_oct_2003, df_nov_2003])
        df_oct_nov_2003.to_pickle(oct_nov_pkl_path)

    print(df_2003)

    # plotting

    plot_vals = ['B_X_GSM', 'B_Y_GSM', 'B_Z_GSM', 'n_p', 'AL', 'P', 'V']

    # for val in plot_vals:
    #     year_title = str(val) + ' in ' + str(year)
    #     df_2003.plot(x='DateTime', y=val, title=year_title)

    #     oct_nov_title = str(val) + ' in October and November 2003'
    #     df_oct_nov_2003.plot(x='DateTime', y=val, title=oct_nov_title)

    print("\nCorrelation matrix for 2003\n")
    corr_2003 = df_2003[plot_vals].corr()
    print(corr_2003)

    print("\nCorrelation matrix for October & November 2003\n")
    corr_oct_nov_2003 = df_oct_nov_2003[plot_vals].corr()
    print(corr_oct_nov_2003)

    csv_dir = pathlib.Path('Data/OMNI/corr')
    pathlib.Path(csv_dir).mkdir(exist_ok=True)
    csv_path = csv_dir / 'corr_2003.csv'
    corr_2003.to_csv(csv_path)

    csv_oct_nov_2003_path = csv_dir / 'corr_oct_nov_2003.csv'
    corr_oct_nov_2003.to_csv(csv_oct_nov_2003_path)

    # 2: STANDARDIZE THE DATA

    # strip headers -> raw data only. Seperate AL as that's the output/target
    data1 = df_2003[['B_X_GSM', 'B_Y_GSM', 'B_Z_GSM', 'n_p', 'P', 'V']].to_numpy()
    AL1 = df_2003['AL'].to_numpy()

    # Standardize the data array
    scaler = StandardScaler()
    scaler.fit(data1)
    data2 = scaler.transform(data1)

    # let's check the histograms
    data3 = np.transpose(data2)
    for i, param in enumerate(['B_X_GSM', 'B_Y_GSM', 'B_Z_GSM', 'n_p', 'P', 'V']):
        plt.figure()
        plt.hist(data3[i], bins=30)
        plt.title('Histogram of ' + param)
        plt.show()


main()
