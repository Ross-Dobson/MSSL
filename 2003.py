import pathlib  # used for compatibility with non-POSIX systems
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# remember to run 'matplotlib tk' in the interpreter

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

from SolarWindImport import import_omni_year, import_omni_month


def main():

    # ---------------------------------------------------------------
    # IMPORTING THE DATA

    # The method import_omni_year checks for Pickles itself
    # so we do not need to enclose this in a try statement
    year = 2003
    df_2003 = import_omni_year(year)

    # ---------------------------------------------------------------
    # IMPORTING OCTOBER AND NOVEMBER SEPERATELY TO ZOOM IN

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
    #     print("Done. Importing the data for October and November.")

    #     df_oct_2003 = import_omni_month(2003, 10)
    #     df_nov_2003 = import_omni_month(2003, 11)
    #     df_oct_nov_2003 = pd.concat([df_oct_2003, df_nov_2003])
    #     df_oct_nov_2003.to_pickle(oct_nov_pkl_path)  # store for future use

    print("All 2003 data has been loaded.")
    print("\nDF 2003:")
    print(df_2003)

    # ---------------------------------------------------------------
    # PLOTTING THE PARAMETERS

    plot_vals = ['B_X_GSM', 'B_Y_GSM', 'B_Z_GSM', 'n_p', 'AL', 'P', 'V']
    # for val in plot_vals:
    #     year_title = str(val) + ' in ' + str(year)
    #     df_2003.plot(x='DateTime', y=val, title=year_title)

    #     oct_nov_title = str(val) + ' in October and November 2003'
    #     df_oct_nov_2003.plot(x='DateTime', y=val, title=oct_nov_title)

    # ---------------------------------------------------------------
    # CORRELATION MATRIX BEFORE SCALER

    print("\nCorrelation matrix for 2003:")
    corr_2003 = df_2003[plot_vals].corr()
    print(corr_2003)

    # print("\nCorrelation matrix for October & November 2003\n")
    # corr_oct_nov_2003 = df_oct_nov_2003[plot_vals].corr()
    # print(corr_oct_nov_2003)

    # ---------------------------------------------------------------
    # SCALER/TRANSFORMING THE DATA

    # The features we care about
    model_vals = ['B_X_GSM', 'B_Y_GSM', 'B_Z_GSM', 'n_p', 'P', 'V']

    # Now get just that data. Get AL seperately, as it's the output/target here
    # This also makes it more futureproof for changing the target
    df2 = df_2003[model_vals]
    AL = df_2003['AL']

    print("\nDF2:")
    print(df2)

    # strip the labels and column headers
    df3 = df2.to_numpy()  # we don't need to .T, each column is a feature
    AL2 = AL.to_numpy()
    # print(df3.shape)
    AL3 = AL2.reshape(-1, 1)  # however, this needs to be .T'd into a column

    # scale the main features
    scaler = StandardScaler()
    scaler = scaler.fit(df3)
    df4 = scaler.transform(df3)
    df5 = df4.T  # make each ROW a feature, not each column, to plot histogram

    # scale AL
    scaler2 = StandardScaler()
    scaler2 = scaler2.fit(AL3)
    AL4 = scaler2.transform(AL3)
    AL5 = AL4.T

    # ---------------------------------------------------------------
    # CORR AFTER STANDARDISATION

    # think I need to add AL5's [0] back into AL's place (but i'th row instead
    # of i'th column number). Then transpose the whole thing, then we can add
    # the index and column labels back in?

    # ---------------------------------------------------------------
    # PLOT HISTOGRAMS:

    # hist_dir = pathlib.Path('Figures')
    # pathlib.Path(hist_dir).mkdir(exist_ok=True)

    # for i, param in enumerate(model_vals):
    #     plt.figure()
    #     plt.hist(df5[i], bins=36)
    #     plt.title('Histogram of '+param+' after StandardScaler')

    # plt.figure()
    # plt.hist(AL4, bins=36)
    # plt.title('Histogram of AL after StandardScaler')

    # ---------------------------------------------------------------
    # TRAIN TEST SPLIT

    # df4/AL4 not 5 here! Needs shape (n_samples, n_features) -> 525600 samples
    # Split the data into two parts, one for training and testing
    print(df4.shape)
    print(AL4.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        df4, AL4, test_size=0.4, random_state=47)

    # ---------------------------------------------------------------
    # LINEAR REGRESSION
    # create the linear regression object
    # regr = linear_model.LinearRegression()

    # # train it
    # regr.fit(X_train, y_train)

    # cross-validation
    # scores = cross_val_score(regr, X_train, y_train, cv=10)
    # print(scores)


main()
