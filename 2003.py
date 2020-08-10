import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# remember to run 'matplotlib tk' in the interpreter

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

from SolarWindImport import import_omni_year


def main():

    # 1: IMPORTING THE DATA

    year = 2003
    df_2003 = import_omni_year(year)

    # pkl_dir = pathlib.Path('Data/OMNI/pickles')
    # oct_nov_pkl_path = pkl_dir / 'oct_nov_2003.pkl'

    # try:
    #     df_oct_nov_2003 = pd.read_pickle((oct_nov_pkl_path))
    #     print("Found the pickle for October and November 2003.")

    # except FileNotFoundError:
    #     print("Couldn't find the pickle for October and November 2003."
    #           + "-> Generating one from the .asc files.")

    #     print('Creating local directory at ', pkl_dir,
    #           ' (if it doesn\'t already exist):')
    #     pathlib.Path(pkl_dir).mkdir(exist_ok=True)
    #     print("Done.")

    #     df_oct_2003 = import_omni_month(2003, 10)
    #     df_nov_2003 = import_omni_month(2003, 11)
    #     df_oct_nov_2003 = pd.concat([df_oct_2003, df_nov_2003])
    #     df_oct_nov_2003.to_pickle(oct_nov_pkl_path)

    print("DF 2003:")
    print(df_2003)

    # PLOTTING

    hist_dir = pathlib.Path('Figures')
    pathlib.Path(hist_dir).mkdir(exist_ok=True)

    plot_vals = ['B_X_GSM', 'B_Y_GSM', 'B_Z_GSM', 'n_p', 'AL', 'P', 'V']
    # for val in plot_vals:
    #     year_title = str(val) + ' in ' + str(year)
    #     df_2003.plot(x='DateTime', y=val, title=year_title)

    #     oct_nov_title = str(val) + ' in October and November 2003'
    #     df_oct_nov_2003.plot(x='DateTime', y=val, title=oct_nov_title)

    print("\nCorrelation matrix for 2003\n")
    corr_2003 = df_2003[plot_vals].corr()
    print(corr_2003)
    # print("\nCorrelation matrix for October & November 2003\n")
    # corr_oct_nov_2003 = df_oct_nov_2003[plot_vals].corr()
    # print(corr_oct_nov_2003)

    # SCALER/TRANSFORM THE DATA

    # the features we care about
    model_vals = ['B_X_GSM', 'B_Y_GSM', 'B_Z_GSM', 'n_p', 'P', 'V']

    # now get just that data. Get AL seperately, as it's the output/target here
    df2 = df_2003[model_vals]
    AL = df_2003['AL']

    print("\nDF2:")
    print(df2)
    df3 = df2.to_numpy()
    AL2 = AL.to_numpy()
    AL3 = AL2.reshape(-1, 1)

    # scale the main features
    scaler = StandardScaler()
    scaler = scaler.fit(df3)
    df4 = scaler.transform(df3)
    df5 = df4.T

    # scale AL
    scaler2 = StandardScaler()
    scaler2 = scaler2.fit(AL3)
    AL4 = scaler2.transform(AL3)

    # fig, axs = plt.subplots(2, 3, tight_layout=True,
    #                        sharex=False, sharey=False)
    # fig.suptitle('Histograms after StandardScaler')

    # plot_vals = np.array(model_vals).reshape(2, 3)

    # for (i, j), param in np.ndenumerate(plot_vals):
    #     axs[i, j].hist(df5[i, j], bins=10)
    #     axs[i, j].set_title(param)

    for i, param in enumerate(model_vals):
        plt.figure()
        plt.hist(df5[i], bins=36)
        plt.title('Histogram of '+param+' after StandardScaler')

    plt.figure()
    plt.hist(AL4, bins=36)
    plt.title('Histogram of AL after StandardScaler')

    # # TRAIN TEST SPLIT

    # # split the data into two equal chunks, for training and testing
    # X_train, X_test, y_train, y_test = train_test_split(
    #     df3, AL2, test_size=0.33, random_state=47)

    # # now, let's do some cross-validation


main()
