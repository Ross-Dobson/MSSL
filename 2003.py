import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # remember to run 'matplotlib tk'
import matplotlib.dates as dt
import pathlib  # used for compatibility with non-POSIX systems
import datetime
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from SolarWindImport import import_omni_year, import_omni_month, import_storm_week, storm_interpolator, storm_chunker


def main():

    # ---------------------------------------------------------------
    # IMPORTING THE DATA

    # The method import_omni_year checks for Pickles itself
    # so we do not need to enclose this in a try statement
    data_2003 = import_omni_year(2003)

    # ---------------------------------------------------------------
    # IMPORTING OCTOBER AND NOVEMBER SEPERATELY TO ZOOM IN

    # df_oct_2003 = import_omni_month(2003, 10)
    # df_nov_2003 = import_omni_month(2003, 11)
    # df_oct_nov_2003 = pd.concat([df_oct_2003, df_nov_2003])
    # df_oct_nov_2003.to_pickle(oct_nov_pkl_path)  # store for future use

    print("All 2003 data has been loaded.")
    # print("\n2003 data:")
    # print(data_2003)

    # ---------------------------------------------------------------
    # PLOTTING THE PARAMETERS

    # plot_vals = ['B_X_GSM', 'B_Y_GSM', 'B_Z_GSM', 'n_p', 'P', 'V', 'AL']

    # for val in plot_vals:
    #    # year_title = str(val) + ' in ' + str(year)
    #    # data_2003.plot(x='DateTime', y=val, title=year_title)

    #    # oct_nov_title = str(val) + ' in October and November 2003'
    #    # df_oct_nov_2003.plot(x='DateTime', y=val, title=oct_nov_title)

    # ---------------------------------------------------------------
    # SCALER/TRANSFORMING THE DATA

    # The features we care about
    model_vals = ['B_X_GSM', 'B_Y_GSM', 'B_Z_GSM', 'n_p', 'P', 'V']

    # get just that data in a new dataframe
    df_2003 = data_2003[model_vals].copy()

    # get the AL values separately, again in a new df/series
    raw_AL = data_2003['AL'].copy()

    # ---------------------------------------------------------------
    # PERSISTENCE TIME HISTORY OF AL

    # Roll right, 30 minutes.
    # E.g. 12:00-12:30 -> 12:30, 12:01-12:31 -> 12:31
    pers_AL = raw_AL.copy()
    pers_AL = pers_AL.rolling(30).min()

    # create a copy before shifting - this is for persistence model later
    disc_AL = pers_AL.copy()

    # now, for our y/target data, we do a roll left by shifting
    # e.g. 12:00 <- 12:00-12:30, 12:01 <- 12:01-12:31
    disc_AL = disc_AL.shift(-30)

    # ---------------------------------------------------------------
    # SOLUTION TO NAN ISSUES FROM TIMESHIFTING

    # drop the nans in the last 30 elements of the discretized AL
    disc_AL.dropna(axis='index', how='any', inplace=True)

    # now, drop the last 30 minutes of all the other data so shapes stay equal
    df_2003.drop(df_2003.tail(30).index, inplace=True)
    pers_AL.drop(pers_AL.tail(30).index, inplace=True)
    raw_AL.drop(raw_AL.tail(30).index, inplace=True)

    # for the persistence model, we do the same, but they'll drop from head
    pers_AL.dropna(axis='index', how='any', inplace=True)

    # drop the beginning 29 minutes of all other data to match (why 29? idk)
    df_2003.drop(df_2003.head(29).index, inplace=True)
    disc_AL.drop(disc_AL.head(29).index, inplace=True)
    raw_AL.drop(raw_AL.head(29).index, inplace=True)

    # ---------------------------------------------------------------
    # ADD PERSISTENCE AS A FEATURE TO THE X ARRAY
    df_2003.insert(6, "AL_hist", pers_AL.copy())
    model_vals.append("AL_hist")

    # take cols and index for remaking DF after scaling
    df_index = df_2003.index

    # ---------------------------------------------------------------
    # CORRELATION MATRIX BEFORE SCALER

    # add disc AL
    df_2003.insert(7, "disc_AL", disc_AL)

    # This is obsolete, but commented here as reminder of current state of
    # the dataframes
    # corr_pars = ['B_X_GSM', 'B_Y_GSM', 'B_Z_GSM', 'n_p', 'P', 'V',
                 # 'AL_hist', 'disc_AL']

    print("\nCorrelation matrix before standardization:\n")
    print(df_2003.corr())

    # print("\nCorrelation matrix for October & November 2003\n")
    # corr_oct_nov_2003 = df_oct_nov_2003[plot_vals].corr()
    # print(corr_oct_nov_2003)

    # remove disc AL
    df_2003 = df_2003.drop(["disc_AL"], axis=1)

    # ---------------------------------------------------------------
    # SCALE THE MAIN FEATURES

    # fit the scaler, then transform the features
    X_scaler = StandardScaler()
    X_scaler = X_scaler.fit(df_2003)
    X_scaled = X_scaler.transform(df_2003)

    # need to add it back into a DF
    df_2003 = pd.DataFrame(X_scaled, columns=model_vals, index=df_index)
    print("\n2003 features have been scaled.")

    # scale AL. We need to scale the discrete AL, and the raw AL
    y_scaler = StandardScaler()
    raw_scaler = StandardScaler()

    # reshape because StandardScaler needs (n_samples, n_features)
    disc_AL = disc_AL.to_numpy().reshape(-1, 1)
    raw_AL = raw_AL.to_numpy().reshape(-1, 1)

    # fit the scaler to the data
    y_scaler = y_scaler.fit(disc_AL)
    raw_scaler = raw_scaler.fit(raw_AL)

    # get our newly scaled y values
    y_scaled = y_scaler.transform(disc_AL)
    raw_scaled = raw_scaler.transform(raw_AL)

    # need to add it back into a DF
    disc_AL = pd.DataFrame(y_scaled, columns=['disc_AL'], index=df_index)
    raw_AL = pd.DataFrame(raw_scaled, columns=['raw_AL'], index=df_index)

    print("2003 AL has been scaled.")

    # ---------------------------------------------------------------
    # CORR AFTER STANDARDISATION

    # Add AL back in to the df
    df_2003.insert(7, "disc_AL", disc_AL)

    print("\nCorrelation matrix after standardization:\n")
    print(df_2003.corr())

    # drop AL again, don't want it as a feature anymore
    df_2003 = df_2003.drop(["disc_AL"], axis=1)

    # ---------------------------------------------------------------
    # VIOLIN PLOTS
    # to check the distributions
    fig, ax = plt.subplots()
    ax = sns.violinplot(data=df_2003)
    _ = ax.set_xticklabels(df_2003.keys(), rotation=90)
    ax.set_title('Violin plot of df_2003')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Normalized')

    # ---------------------------------------------------------------
    # # MUTUAL INFORMATION

    print("\nMUTUAL INFORMATION:")

    # 1 year data crashes. Lets use 1 week, centred on 24h storm
    storm_dt = datetime.datetime(2003, 10, 27, 0, 0, 0)  # start of storm
    start_dt = storm_dt - datetime.timedelta(days=3)  # start of week: -3d
    end_dt = storm_dt + datetime.timedelta(days=4)  # end of week: +4d

    # make copy so we cant break anything
    mi_2003 = df_2003.copy()

    # reinsert AL for consistent NaN drop
    mi_2003.insert(7, 'disc_AL', disc_AL)

    # narrow to week
    mi_2003 = mi_2003.loc[start_dt:end_dt]

    # drop NaNs, MI doesn't like them
    mi_2003.dropna(axis='index', how='any', inplace=True)

    mi_AL = mi_2003['disc_AL']
    mi_2003 = mi_2003.drop(['disc_AL'], axis=1)

    print("\nExample scenario: n_p and P should have high MI:")
    print(mutual_info_regression(
        mi_2003['P'].to_numpy().reshape(-1, 1), mi_2003['n_p']))

    print("Discrete AL vs", model_vals, ":")
    print(mutual_info_regression(mi_2003, mi_AL))

    for i, feature in enumerate(model_vals):
        print("\nMutual information for", feature, "vs the others:")
        feature_array = model_vals.copy()
        feature_array = np.delete(feature_array, i)
        big_df = mi_2003.copy()
        big_df = big_df.drop([feature], axis=1)
        big_df = big_df.to_numpy()
        small_df = mi_2003[feature]
        small_df = small_df.to_numpy()
        print(feature_array)
        print(mutual_info_regression(big_df, small_df))

    # ---------------------------------------------------------------
    # REMOVING UNNEEDED PARAMETERS

    # removing n_p - in the words of Mayur
    # "by far weakest correlation with AL and a strong correlation with P"
    df_2003 = df_2003.drop(["n_p"], axis=1)

    # ---------------------------------------------------------------
    # INTERPOLATING GAPS

    # to make the indexes match, we're gonna combine the whole damn thing
    # and chuck it all through interpolator: df_2003 + disc_AL + raw_AL
    df_2003.insert(5, "disc_AL", disc_AL)
    df_2003.insert(6, "raw_AL", raw_AL)

    # run the interpolation
    df_2003 = storm_interpolator(df_2003)

    print("\n2003 features interpolated.")

    # seperate these two back off, drop them
    raw_AL = df_2003['raw_AL'].copy()
    disc_AL = df_2003['disc_AL'].copy()
    df_2003 = df_2003.drop(["raw_AL", "disc_AL"], axis=1)

    # ---------------------------------------------------------------
    # PLOT HISTOGRAMS:

    # hist_dir = pathlib.Path('Figures')
    # pathlib.Path(hist_dir).mkdir(exist_ok=True)

    # for i, param in enumerate(plot_vals):
    #     plt.figure()
    #     plt.hist(df_2003[param], bins=35)
    #     plt.title('Histogram of '+param+' after StandardScaler')

    # ---------------------------------------------------------------
    # KERAS PREP STEP:
    pkl_dir = pathlib.Path('~/Documents/Python/Projects/MSSL/Data/OMNI/pickles')

    pkl_path = pkl_dir / ('2003_' + 'X' + '.pkl')
    df_2003.to_pickle(pkl_path)

    pkl_path = pkl_dir / ('2003_' + 'y' + '.pkl')
    disc_AL.to_pickle(pkl_path)

    # ---------------------------------------------------------------
    # TRAIN TEST SPLIT

    # Needs shape (n_samples, n_features)
    # Split the data into two parts, one for training and testing
    # 5 splits, which I think leads to 6 sets of data, 2 months each
    # first 5 (k) are training, 6th (k+1)th one is testing

    print("\nFolding the data (cross-validation):")
    tscv = TimeSeriesSplit()
    fold_counter = 0

    for train_index, test_index in tscv.split(df_2003):

        fold_counter += 1
        print("Fold", fold_counter, "...")
        # print("TRAIN:", train_index, "TEST:", test_index)  # debug
        X_train, X_test = df_2003.iloc[train_index], df_2003.iloc[test_index]
        y_train, y_test = disc_AL.iloc[train_index], disc_AL.iloc[test_index]

    print("Folding complete.")

    # ---------------------------------------------------------------
    # LINEAR REGRESSION

    # create the linear regression object
    regr = LinearRegression()

    # train it on the training data
    regr.fit(X_train, y_train)

    # ---------------------------------------------------------------
    # LINEAR REGRESSION PREDICTION AND SCORES
    y_pred = regr.predict(X_test)

    print("\nLinear Regression R^2 score (best 1.0)",
          regr.score(X_test, y_test))
    # print(r2_score(y_test, y_pred))

    # put y pred in a dataframe, just makes life easier for future
    y_test_index = y_test.index
    y_pred = pd.DataFrame(y_pred, columns=['pred_AL'], index=y_test_index)

    # ---------------------------------------------------------------
    # GET PERSISTENCE AGAIN

    pers_AL = df_2003['AL_hist'].copy()

    # ---------------------------------------------------------------
    # EVALUATING OUR MODEL VS PERSISTENCE MODEL

    def storm_metrics(y_true, y_pred, y_pers):
        """Runs various regression metrics from sklearn.metrics

        Args:
          y_true: The target values of discrete rolled-left AL
          y_pred: The predicted values of discrete rolled-left AL from model
          y_pers: The persistence (rolled-right) time history of AL

        Returns:
          None
        """

        # Explained variance score (higher is better, best 1.0)
        evs_true = explained_variance_score(y_true, y_pred)
        evs_pers = explained_variance_score(y_true, y_pers)

        # Mean absolute error (lower is better, best 0.0)
        mean_true = mean_absolute_error(y_true, y_pred)
        mean_pers = mean_absolute_error(y_true, y_pers)

        # Mean squared error (lower is better, best 0.0)
        mse_true = mean_squared_error(y_true, y_pred)
        mse_pers = mean_squared_error(y_true, y_pers)

        # not too affected by outliers - good choice of metric?
        # Median absolute error (lower is better, best 0.0)
        medi_true = median_absolute_error(y_true, y_pred)
        medi_pers = median_absolute_error(y_true, y_pers)

        # variance is dependent on dataset, might be a pitfall
        # R2 coefficient of determination (higher=better), best 1.0
        r2_true = r2_score(y_true, y_pred)
        r2_pers = r2_score(y_true, y_pers)

        return evs_true, evs_pers, mean_true, mean_pers, mse_true, mse_pers, medi_true, medi_pers, r2_true, r2_pers

    metrics = ["Explained variance score",
               "Mean absolute error",
               "Mean squared error",
               "Median absolute error",
               "R2 score"]

    metrics_desc = ["higher is better, best 1.0",
                    "lower is better, best 0.0",
                    "lower is better, best 0.0",
                    "lower is better, best 0.0",
                    "higher is better, best 1.0"]

    # 2003 data - don't actually want to run metrics on this though
    # storm_metrics(y_test, pred_df, pers_AL)

    def storm_metrics(y_true, y_pred, y_pers):
        """Runs various regression metrics from sklearn.metrics
        Args:
          y_true: The target values of discrete rolled-left AL
          y_pred: The predicted values of discrete rolled-left AL from model
          y_pers: The persistence (rolled-right) time history of AL
        Returns:
          None
        """

        # Explained variance score (higher is better, best 1.0)
        evs_true = explained_variance_score(y_true, y_pred)
        evs_pers = explained_variance_score(y_true, y_pers)

        # Mean absolute error (lower is better, best 0.0)
        mean_true = mean_absolute_error(y_true, y_pred)
        mean_pers = mean_absolute_error(y_true, y_pers)

        # Mean squared error (lower is better, best 0.0)
        mse_true = mean_squared_error(y_true, y_pred)
        mse_pers = mean_squared_error(y_true, y_pers)

        # not too affected by outliers - good choice of metric?
        # Median absolute error (lower is better, best 0.0)
        medi_true = median_absolute_error(y_true, y_pred)
        medi_pers = median_absolute_error(y_true, y_pers)

        # variance is dependent on dataset, might be a pitfall
        # R2 coefficient of determination (higher=better), best 1.0
        r2_true = r2_score(y_true, y_pred)
        r2_pers = r2_score(y_true, y_pers)

        return evs_true, evs_pers, mean_true, mean_pers, mse_true, mse_pers, medi_true, medi_pers, r2_true, r2_pers

    metrics = ["Explained variance score",
               "Mean absolute error",
               "Mean squared error",
               "Median absolute error",
               "R2 score"]

    metrics_desc = ["higher is better, best 1.0",
                    "lower is better, best 0.0",
                    "lower is better, best 0.0",
                    "lower is better, best 0.0",
                    "higher is better, best 1.0"]

    # 2003 data - don't actually want to run metrics on this though
    # storm_metrics(y_test, pred_df, pers_AL)

    # ***************************************************************
    # ***************************************************************
    # IMPORTING THE VALIDATION STORMS, AND PREPARING THE DATA
    # ***************************************************************
    # ***************************************************************

    # ---------------------------------------------------------------
    # IMPORT VALIDATION STORMS
    # from doi:10.1002/swe.20056

    print("\nImporting the validation storms:")

    # DONT USE STORM 1 as its in the 2003 model training data!
    # storm_1 = import_storm_week(2003, 10, 29)
    storm_2 = import_storm_week(2006, 12, 14)
    storm_3 = import_storm_week(2001, 8, 31)
    storm_4 = import_storm_week(2005, 8, 31)
    storm_5 = import_storm_week(2010, 4, 5)
    storm_6 = import_storm_week(2011, 8, 5)

    storm_array = [storm_2, storm_3, storm_4, storm_5, storm_6]

    storm_str_array = ["2006-12-14 12:00 to 2006-12-16 00:00",
                       "2001-08-31 00:00 to 2001-09-01 00:00",
                       "2005-08-31 10:00 to 2005-09-01 12:00",
                       "2010-04-05 00:00 to 2010-04-06 00:00",
                       "2011-08-05 09:00 to 2011-08-06 00:00"]

    # for pickling any storms later
    storm_fname_array = ["2006-12-14",
                        "2001-08-31",
                        "2005-08-31",
                        "2010-04-05",
                        "2011-08-05"]

    storm_start_array = [datetime.datetime(2006, 12, 14, 12, 0, 0),
                         datetime.datetime(2001, 8, 31, 0, 0, 0),
                         datetime.datetime(2005, 8, 31, 10, 0, 0),
                         datetime.datetime(2010, 4, 5, 0, 0, 0),
                         datetime.datetime(2011, 8, 5, 9, 0, 0)]

    storm_end_array = [datetime.datetime(2006, 12, 16, 0, 0, 0),
                       datetime.datetime(2001, 9, 1, 0, 0, 0),
                       datetime.datetime(2005, 9, 1, 12, 0, 0),
                       datetime.datetime(2010, 4, 6, 0, 0, 0),
                       datetime.datetime(2011, 8, 6, 0, 0, 0)]

    print("Validation storms imported successfully.")

    # ---------------------------------------------------------------
    # GET FEATURES DATA AND AL
    # The features we care about - recall, no persistence yet
    model_vals = ['B_X_GSM', 'B_Y_GSM', 'B_Z_GSM', 'n_p', 'P', 'V']

    X_array = []
    raw_array = []
    for i, storm in enumerate(storm_array):
        storm_index = storm.index  # save this for reconstructing DF later
        X = storm[model_vals].copy()
        raw = storm['AL'].copy()

        # each storm's features to a df, append to X_array of dataframes
        X_array.append(pd.DataFrame(X, columns=model_vals, index=storm_index))
        raw_array.append(pd.DataFrame(raw, columns=['AL'], index=storm_index))

    # ---------------------------------------------------------------
    # PERSISTENCE AND DISCRETIZATION (is that even a word)

    disc_array = []
    pers_array = []
    df_index_array = []

    # i.e. for each storm's raw AL dataframe
    for i, raw in enumerate(raw_array):
        pers_y = raw.copy()
        pers_y = pers_y.rolling(30).min()  # roll right, 30 minute window
        disc_y = pers_y.copy()
        disc_y = disc_y.shift(-30)  # roll left for discrete AL

        # drop the 30 minutes of NaNs from the shifting
        disc_y.dropna(axis='index', how='any', inplace=True)

        # drop last 30 minutes of all others to match disc
        pers_y.drop(pers_y.tail(30).index, inplace=True)
        raw.drop(raw.tail(30).index, inplace=True)

        # for persistence, the NaNs will dorp from head
        pers_y.dropna(axis='index', how='any', inplace=True)

        # drop first 29 minutes to match
        disc_y.drop(disc_y.head(29).index, inplace=True)
        raw.drop(raw.head(29).index, inplace=True)

        # add to the arrays outside the loop
        pers_array.append(pers_y)
        disc_array.append(disc_y)
        raw_array[i] = raw.copy()

        # store the indexes for later (same for disc, pers, raw)
        df_index_array.append(disc_y.index)

    # go through the five X feature arrays
    for i, X in enumerate(X_array):

        # make them match AL
        X.drop(X.tail(30).index, inplace=True)
        X.drop(X.head(29).index, inplace=True)

        # ADD THEIR PERSISTENCE AS A FEATURE
        X.insert(6, "AL_hist", pers_array[i].to_numpy())
        X_array[i] = X.copy()

    # outside the for loop so we dont add it more than once!
    model_vals.append("AL_hist")

    # ---------------------------------------------------------------
    # STANDARD SCALING
    # we use the same scalers we used earlier. X_scaler, y_scaler, raw_scaler
    # recall X_array, raw_array. disc_array already defined, so we overwrite

    for i, storm_index in enumerate(df_index_array):

        # scale the main features
        X_scaled = X_scaler.transform(X_array[i])

        # each storm's features to a df, append to X_array of dataframes
        X_array[i] = pd.DataFrame(X_scaled,
                                  columns=model_vals, index=storm_index)

        # scale disc AL
        disc = disc_array[i].to_numpy()
        disc = disc.reshape(-1, 1)
        disc_scaled = y_scaler.transform(disc)

        # scale raw AL
        raw = raw_array[i].to_numpy()
        raw = raw.reshape(-1, 1)
        raw_scaled = raw_scaler.transform(raw)

        # store in the main arrays outside the loop
        disc_array[i] = pd.DataFrame(disc_scaled, columns=['disc_AL'],
                                     index=storm_index)
        raw_array[i] = pd.DataFrame(raw_scaled, columns=['raw_AL'],
                                    index=storm_index)

    print("\nValidation storm features and AL have been scaled.")

    # ---------------------------------------------------------------
    # REMOVING N_P
    # "by far weakest correlation with AL, and a strong correlation with P"

    for i, X in enumerate(X_array):
        X = X.drop(["n_p"], axis=1)
        X_array[i] = X

    # ---------------------------------------------------------------
    # INTERPOLATE AND DROPNA IN THE TEST STORMS
    # remember, no longer need to interpolate y because its discretized

    # each storm will drop seperate dt, so store index to remake df
    interped_index_array = []

    for i, X in enumerate(X_array):

        # to make the indexes match, we're gonna combine everything
        # and chuck it all through the interpolator: df_200 + disc_AL + raw_AL
        X.insert(5, "disc_AL", disc_array[i])
        X.insert(6, "raw_AL", raw_array[i])

        # interpolate the storm
        X_array[i] = storm_interpolator(X)

        # seperate these two back off, drop them
        raw_array[i] = X_array[i]['raw_AL'].copy()
        disc_array[i] = X_array[i]['disc_AL'].copy()
        X_array[i] = X_array[i].drop(['raw_AL', 'disc_AL'], axis=1)

        # store the new index for this storm, for later
        interped_index_array.append(X_array[i].index)

    print("\nTest storms interpolated.")

    # ---------------------------------------------------------------
    # PREDICTING THE DATA
    # remember we fitted the LinearRegression object regr on 2003 data

    y_pred_array = []
    for i, X in enumerate(X_array):
        prediction = regr.predict(X)
        pred_y = pd.DataFrame(
            prediction, columns=["pred_AL"], index=interped_index_array[i])
        y_pred_array.append(pred_y)

    # ---------------------------------------------------------------
    # CHUNKING STORMS
    chunk_res = '6h'

    # will store the array of chunks for each of the storms
    chunks_array = []

    print("\nChunking storms:")

    for i in range(0, len(storm_array)):
        print("Chunking the week around storm", storm_str_array[i])

        # call the storm_chunker function to get array of chunks
        chunks_array.append(
            storm_chunker(disc_array[i], y_pred_array[i],
                          X_array[i]['AL_hist'], resolution=chunk_res))

    print("Validation storms have been chunked.")

    # ---------------------------------------------------------------
    # CALCULATING THE METRICS FOR EACH CHUNK

    # iterate through each storm
    for i, storm in enumerate(chunks_array):

        metrics_array = [[], [], [], [], [], [], [], [], [], []]

        # index array to plot the metric against
        index_array = []

        # for each chunk
        for chunk in storm:

            # run the metrics
            if (len(chunk) != 0):
                temp_metrics = storm_metrics(
                    chunk['disc_AL'], chunk['pred_AL'], chunk['AL_hist'])

                # append it into our metrics array
                for j in range(0, len(temp_metrics)):
                    metrics_array[j].append(temp_metrics[j])

                # get just index[0] as we're appending to a whole array
                index_array.append(chunk['disc_AL'].index[0].to_numpy())

            else:
                raise Exception("It's all gone a bit J.G. Ballard out there.")

        # OKAY lets get plotting for each metric!
        for j in range(0, 5):

            # this deals with model->pers->model->pers layout of metrics array
            a = 2*j
            b = a+1

            fig, axs = plt.subplots(2, sharex=False)
            fig.suptitle(metrics[j] + " - " + chunk_res + " - "
                         + storm_str_array[i])

            axs[0].plot(index_array, metrics_array[a], label='Model',
                        marker='.')
            axs[0].plot(index_array, metrics_array[b], label='Persistence',
                        marker='.')
            axs[0].axvline(storm_start_array[i], c='k', ls='--',
                           label='Storm period')
            axs[0].axvline(storm_end_array[i], c='k', ls='--')
            axs[0].axhline(0, c='k', alpha=0.5, ls='dotted')
            axs[0].set_ylabel('Metric score: ' + metrics_desc[j])
            axs[0].set_xlabel('DateTime')

            # get the ylim so we can check if the default is okay
            old_ylim = axs[0].get_ylim()
            new_low = old_ylim[0]
            new_top = old_ylim[1]

            if (new_low <= -10):
                new_low = -10

            if (new_top >= 10):
                new_top = 10

            # if a best 1.0 metric, no need to go above 2
            if j == 0 or j == 4:
                new_top = 2
                new_low = -8
                axs[0].set_yticks(np.arange(-8, 3, step=1))

            # if a best 0.0 metric, no need to go below -0.5
            if j == 1 or j == 2 or j == 3:
                new_low = -0.5

            # set the ylim again
            axs[0].set_ylim((new_low, new_top))

            # generate the legend, auto-location
            axs[0].legend(loc='best')

            # the subplot - disc AL vs raw AL vs persistence AL
            start = index_array[0]
            end = index_array[-1]

            axs[1].plot(disc_array[i].loc[start:end], label='discrete',
                        alpha=0.5)
            axs[1].plot(raw_array[i].loc[start:end], label='true', alpha=0.5)

            axs[1].axvline(storm_start_array[i], c='k', ls='--',
                           label='Storm period')
            axs[1].axvline(storm_end_array[i], c='k', ls='--')

            axs[1].legend(loc='best')


main()
