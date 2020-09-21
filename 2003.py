import pathlib  # used for compatibility with non-POSIX systems
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
# remember to run 'matplotlib tk' in the interpreter

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from SolarWindImport import import_omni_year, import_omni_month, import_storm_week, storm_interpolator


def main():

    # ---------------------------------------------------------------
    # IMPORTING THE DATA

    # The method import_omni_year checks for Pickles itself
    # so we do not need to enclose this in a try statement
    year = 2003
    data_2003 = import_omni_year(year)

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
    # print("\n2003 data:")
    # print(data_2003)

    # ---------------------------------------------------------------
    # PLOTTING THE PARAMETERS

    plot_vals = ['B_X_GSM', 'B_Y_GSM', 'B_Z_GSM', 'n_p', 'AL', 'P', 'V']

    # for val in plot_vals:
        # year_title = str(val) + ' in ' + str(year)
        # data_2003.plot(x='DateTime', y=val, title=year_title)

        # oct_nov_title = str(val) + ' in October and November 2003'
        # df_oct_nov_2003.plot(x='DateTime', y=val, title=oct_nov_title)

    # ---------------------------------------------------------------
    # CORRELATION MATRIX BEFORE SCALER

    # print("\nCorrelation matrix before standardization:\n")
    # corr_2003 = data_2003[plot_vals].corr()
    # print(corr_2003)

    # print("\nCorrelation matrix for October & November 2003\n")
    # corr_oct_nov_2003 = df_oct_nov_2003[plot_vals].corr()
    # print(corr_oct_nov_2003)

    # ---------------------------------------------------------------
    # SCALER/TRANSFORMING THE DATA

    # The features we care about
    model_vals = ['B_X_GSM', 'B_Y_GSM', 'B_Z_GSM', 'n_p', 'P', 'V']

    # get just that data
    df_2003 = data_2003[model_vals]
    df_2003_cols = df_2003.columns  # column labels
    df_2003_index = df_2003.index  # row labels

    df_AL = data_2003['AL']
    df_AL_index = df_AL.index  # get the index (row labels), they are datetimes

    # scale the main features
    scaler = StandardScaler()
    scaler = scaler.fit(df_2003)
    arr_2003_scaled = scaler.transform(df_2003)

    # need to add it back into a DF
    df_2003 = pd.DataFrame(arr_2003_scaled,
                           columns=df_2003_cols, index=df_2003_index)
    print("2003 data has been scaled.")
    # print(df_2003)

    # scale AL
    scaler2 = StandardScaler()
    arr_AL = df_AL.to_numpy()
    arr_AL = arr_AL.reshape(-1, 1)
    scaler2 = scaler2.fit(arr_AL)
    arr_AL_scaled = scaler2.transform(arr_AL)

    # need to add it back into a DF
    df_AL = pd.DataFrame(arr_AL_scaled, columns=['AL'], index=df_AL_index)
    # print("\nAL after scaling\n")
    # print(df_AL)

    # Add AL back in to the df
    df_2003.insert(4, "AL", arr_AL_scaled)

    # ---------------------------------------------------------------
    # CORR AFTER STANDARDISATION

    # print("\nCorr after standardization:\n")
    # print(df_2003.corr())

    # ---------------------------------------------------------------
    # REMOVING USELESS PARAMETERS

    # removing n_p - in the words of Mayur
    # "by far weakest correlation with AL and a strong correlation with P"
    df_2003 = df_2003.drop(["n_p"], axis=1)

    # ---------------------------------------------------------------
    # INTERPOLATING GAPS
    df_2003 = storm_interpolator(df_2003)
    print("2003 data interpolated.")

    # ---------------------------------------------------------------
    # PLOT HISTOGRAMS:

    # hist_dir = pathlib.Path('Figures')
    # pathlib.Path(hist_dir).mkdir(exist_ok=True)

    # for i, param in enumerate(plot_vals):
    #     plt.figure()
    #     plt.hist(df_2003[param], bins=35)
    #     plt.title('Histogram of '+param+' after StandardScaler')

    # ---------------------------------------------------------------
    # DISCRETIZE AL

    # seperate AL from the main df
    df_AL = df_2003['AL'].copy()
    df_2003 = df_2003.drop(["AL"], axis=1)

    # Roll right, 30 minutes.
    # E.g. 12:00-12:30 -> 12:30, 12:01-12:31 -> 12:31
    discrete_AL = df_AL.rolling(30).min()

    # create a copy before shifting - this is for persistence model later
    pers_AL = discrete_AL.copy()
    pers_AL_index = pers_AL.index

    # now, because this isn't a native argument, we roll left by shifting
    discrete_AL = discrete_AL.shift(-30)

    # ---------------------------------------------------------------
    # SOLUTION TO NAN ISSUES FROM TIMESHIFTING

    # drop the nans in the last 30 elements of the discretized AL
    discrete_AL.dropna(axis='index', how='any', inplace=True)

    # now, drop the last 30 minutes of all the other data so shapes stay equal
    df_2003.drop(df_2003.tail(30).index, inplace=True)

    # for the persistence model, we do the same, but they'll drop from head
    pers_AL.dropna(axis='index', how='any', inplace=True)

    # ---------------------------------------------------------------
    # TRAIN TEST SPLIT

    # Needs shape (n_samples, n_features)
    # Split the data into two parts, one for training and testing

    # 60% of data for training, 40% of data held back for testing
    # using 47 as seed for repeatbility
    X_train, X_test, y_train, y_test = train_test_split(
        df_2003, discrete_AL, test_size=0.4, random_state=47)

    # ---------------------------------------------------------------
    # LINEAR REGRESSION
    # create the linear regression object
    regr = LinearRegression()

    # train it on the training data
    regr.fit(X_train, y_train)

    # ---------------------------------------------------------------
    # CROSS VALIDATION
    regr_scores = cross_val_score(regr, df_2003, discrete_AL, cv=10)
    print("The linear regression CV scores are", regr_scores)
    print("Accuracy: %0.2f (+/- %0.2f)" %
          (regr_scores.mean(), regr_scores.std() * 2))

    y_pred = regr.predict(X_test)

    # put y pred back in a dataframe, just makes life easier for future
    y_test_index = y_test.index
    y_pred = pd.DataFrame(y_pred, columns=['AL'], index=y_test_index)

    # ---------------------------------------------------------------
    # MAKING PERSISTENCE MATCH

    # earliest datetime that != NaN (because of rolling left by default)
    earliest_dt = pers_AL.index[0]

    # similarly, latest that != Nan (because this is rolled right, shifted -30)
    latest_dt = discrete_AL.index[-1]

    # so now we can make an index of these happy values to use
    happy_index = y_test_index[(y_test_index >= earliest_dt)
                               & (y_test_index <= latest_dt)]

    test_df = pd.DataFrame(
        y_test[happy_index], columns=['AL'], index=happy_index)

    pred_df = pd.DataFrame(y_pred, columns=['AL'], index=y_test_index)
    pred_df = pred_df.loc[happy_index]

    pers_df = pd.DataFrame(pers_AL, columns=['AL'], index=pers_AL_index)
    pers_df = pers_df.loc[happy_index]

    # ---------------------------------------------------------------
    # EVALUATING OUR MODEL VS PERSISTENCE MODEL

    def storm_metrics(y_true, y_pred, y_pers):
        """
        Runs various regression metrics from sklearn.metrics
        """

        print("\nExplained variance score (higher is better, best 1.0):")
        print("Discretized AL:",
              explained_variance_score(y_true, y_pred))
        print("Persistence AL:",
              explained_variance_score(y_true, y_pers))

        print("\nMean absolute error (lower is better, best 0.0):")
        print("Discretized AL:", mean_absolute_error(y_true, y_pred))
        print("Persistence AL:", mean_absolute_error(y_true, y_pers))

        print("\nMean squared error (lower is better, best 0.0):")
        print("Discretized AL:", mean_squared_error(y_true, y_pred))
        print("Persistence AL:", mean_squared_error(y_true, y_pers))

        # this penalizes underpreciction more than overprediction
        # Mean squared logarithmic error (lower is better, best 0.0)
        # Penalizes underpreciction better. Good for exponential growth.
        # Unsure of usefulness here."
        # print("Discretized AL:", mean_squared_log_error(
        #     y_true, y_pred))
        # print("Persistence AL:", mean_squared_log_error(
        #     y_true, y_pers))

        # not too affected by outliers - good choice of metric?
        print("\nMedian absolute error (lower is better, best 0.0):")
        print("Discretized AL:", median_absolute_error(
            y_true, y_pred))
        print("Persistence AL:", median_absolute_error(
            y_true, y_pers))

        # variance is dependent on dataset, might be a pitfall
        print("\nR2 coefficient of determination (higher=better), best 1)")
        print("Discretized AL:", r2_score(y_true, y_pred))
        print("Persistence AL:", r2_score(y_true, y_pers))

    print("DEBUG 470")
    storm_metrics(test_df, pred_df, pers_df)

    # ***************************************************************
    # ***************************************************************
    # IMPORTING THE TEST STORMS, AND PREPARING THE DATA
    # ***************************************************************
    # ***************************************************************

    # ---------------------------------------------------------------
    # IMPORT TEST STORMS
    # from doi:10.1002/swe.20056

    # DONT USE STORM 1 as its in the 2003 model training data!
    # storm_1 = import_storm_week(2003, 10, 29)
    storm_2 = import_storm_week(2006, 12, 14)
    storm_3 = import_storm_week(2001, 8, 31)
    storm_4 = import_storm_week(2005, 8, 31)
    storm_5 = import_storm_week(2010, 4, 5)
    storm_6 = import_storm_week(2011, 8, 5)

    storm_array = [storm_2, storm_3, storm_4, storm_5, storm_6]

    storm_str_array = ["2006-12-14 to 2006-12-16",
                       "2001-08-31 to 2001-09-01",
                       "2005-08-31 to 2005-09-01",
                       "2010-04-05 to 2010-04-06",
                       "2011-08-05 to 2011-08-06"]

    print("Test storms imported successfully.")

    # ---------------------------------------------------------------
    # STANDARD SCALING
    # we use the same scalers we used earlier. Remember, "scaler" for features
    # and "scaler2" for AL

    # The features we care about
    model_vals = ['B_X_GSM', 'B_Y_GSM', 'B_Z_GSM', 'n_p', 'P', 'V']

    X_array = []
    y_array = []
    for i, storm in enumerate(storm_array):
        storm_index = storm.index  # save this for reconstructing DF later
        X = storm[model_vals]
        y = storm['AL']

        # scale the main features
        X_trans = scaler.transform(X)

        # need to add it back into a DF
        X_array.append(pd.DataFrame(X_trans,
                                    columns=df_2003_cols, index=storm_index))

        # scale AL
        y = y.to_numpy()
        y = y.reshape(-1, 1)
        y_trans = scaler2.transform(y)

        # need to add it back into a DF
        y_array.append(pd.DataFrame(y_trans,
                                    columns=['AL'], index=storm_index))

    # ---------------------------------------------------------------
    # DROPPING USELESS PARAMETERS
    # removing n_p - in the words of Mayur
    # "by far weakest correlation with AL and a strong correlation with P"
    for i, X in enumerate(X_array):
        X_array[i] = X.drop(["n_p"], axis=1)

    # ---------------------------------------------------------------
    # INTERPOLATE AND DROPNA IN THE TEST STORMS
    # have to add AL back in so that the dropped datetimes are consistent!

    interpolated_array = []
    index_array = []  # each storm will drop seperate dt, need to store indexes

    for i, X in enumerate(X_array):
        X.insert(4, "AL", y_array[i])  # add AL back in

        # interpolate the storm - pars AND AL in one go
        interpolated_array.append(storm_interpolator(X))

        # store the new index for this storm, for later
        index_array.append(interpolated_array[i].index)

    # seperate AL back out again!
    for i, storm in enumerate(interpolated_array):
        X_array[i] = storm[["B_X_GSM", "B_Y_GSM", "B_Z_GSM", "P", "V"]]
        y_array[i] = storm["AL"]

    print("Test storms interpolated.")
    # ---------------------------------------------------------------
    # PREDICTING THE DATA
    y_pred_array = []
    for X in X_array:
        y_pred_array.append(regr.predict(X))

    # ---------------------------------------------------------------
    # PLOTTING PREDICTED VS REAL (DISCRETIZED) AL

    # # Plot the predicted data vs our discretized AL
    # for i in range(0, len(y_array)):
    #     plt.figure()
    #     plt.title("Storm from " + storm_str_array[i])
    #     plt.plot(index_array[i], y_pred_array[i], label="Predicted AL")
    #     plt.plot(index_array[i], y_array[i], label="Actual AL")
    #     plt.legend(loc='best')

# 2020-09-01 meeting
# TODO: REMOVING USELESS PARAMETERS
# mutual info can also tell us something about "drivers"

# TODO: CROSS VALIDATION
# If we have sufficient extreme data in each fold, we can reduce each fold
# dependent on amount of data
#
# might be doing a good job predicting when "Nothing" is happening but not good
# for predicitng extremes.
# so test the model with another year's storm - use the NASA link?
# visualisation - two lines - here's what data did, here's what model did
# but not great as ony predicting one timestep ahead
# instead: linegraph of AL over time. Pick out one point in the storm
# "this is the solar wind input I'm providing". Then present entry X, answer Y
# for a particular point in the storm. So feed in a datetime etc. Need to be
# careful with shapes (.reshape()).
# So, to start, plug in a storm from a different year, then plot real AL vs
# predicted AL for that other year.
#
# OTHER METRICS
# method 1: use another metric? R^2, mean square error, etc.
# Dont imagine they give much difference...
# Method 2: engineer your output - AL varies between +- 10~20 nanoTesla
# but then drops to -100, -200. Engi the output to make it more sensitive
# to extreme events. Take absolute, then take Log, which will scale the whole.
# Might compress the metrics to either take more account, or neglect the
# extremes. Eg ground mag field perturbation studies, takes the Log10
# because by default it varies by 5 order of mag. Will ignore small stuff if
# its not logged.
# way to test this: feed in validation storm from another year, then have loads
# of dashed lines from all the different ways of metrics, different models
#
# Summary so far:
#
# Feature selection - look at mutual information between aprameters
# might pick up squared relationships etc
# Metrics and outputs: ways to check they are doign what we think they are
#
# Extension:
# another baseline model we want to compare to
# first one: persistence models - if it does well, nothing reallys happening
# and you model is not adding any value.
# The way to code this is that your previous timestep is the answer to the next
# Persistence models wont pick up any interesting events.
# You can do this by shifting all the answers along by 1? Then compare
# something like sum-of-squares by providing the 1-shift set as your answers
# Ideally, we want to outperform our persistence model - or we're not doing
# anything useful! A way of benchmarking the forecast.
# Mean of previous day, 4hr rolling cyclical forecast various methods and cases
# Solar physics often use ~24 days as sun spins every ~24d. So they wanna
# beat that.
#
# Week centred on the storm would be a good predictor period.
#
# MEETING Notes 2020-09-08
# model is a linear reg. Good benchmark. Simple model, we can compare to easily
# anny more complicated model SHOULD beat this. If we dont beat it, then
# something has gone wrong, or its topo complicated and confusing itself
# whatever metric we compare, keep it stored and compare later on with neural
# nets. So, if we use mean squared or whatever metric.
# So, this week, get metrics out, that'd be nice.
# ALSO, persistence model. "we have complicated model, but it beats linreg AND
# persistence", thats a good justification.
# evaluate during the storms intervals. Thats the time thats really critical.
# else its just boring times. persistence will do really well, as nothing is
# changing. But storm is the time we really should see improvement.
# Also, mutual inforation, which is more important for NN as limiting the
# inputs saves a LOTTTTT of time training, and perhaps even better results.
# sklearn mutual info score.
# persistence model:
# undo the shift, keep the rolling minimum, so its off to the right?
# LONG TERM STRETCH GOAL
# look into artifical neural networks. Recurrent are also good for time series
# forecast. but ANNs are good starting point. LSTMs are an example of recurrent
# neural network, with a "forget" function. Quicker than normal RNN.

# meeting notes 15th
# first array are the unshifted AL values 0 to 30
# second array is the shifted persistence AL
#
# mutual info:
# start with, feed in density and pressure (np and p) - they should be related
# mutual_info_regression: do the feature array vs AL
# and each feature vs feature
#
# checking parameters that are more easily predicted eg solar wind speed
# rather than just using the L1 data that we are using currently
# buuuuut this is more something for the discussion
#
# svr - random forest regressor
# good idea - random forest, do a graph: x is no features, y is any old metric
# then pick the highest score. GOod way to investigate features
#
# stretch goal: shorter time period - (12h either side)? desirable in future


main()
