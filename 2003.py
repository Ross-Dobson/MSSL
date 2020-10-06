import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # remember to run 'matplotlib tk'
import matplotlib.dates as dt
import pathlib  # used for compatibility with non-POSIX systems
import datetime

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
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

    # get just that data
    df_2003 = data_2003[model_vals].copy()

    # get the AL values separately
    df_AL = data_2003['AL'].copy()

    # ---------------------------------------------------------------
    # PERSISTENCE TIME HISTORY OF AL

    # Roll right, 30 minutes.
    # E.g. 12:00-12:30 -> 12:30, 12:01-12:31 -> 12:31
    disc_AL = df_AL.rolling(30).min()

    # create a copy before shifting - this is for persistence model later
    pers_AL = disc_AL.copy()
    pers_AL_index = pers_AL.index

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

    # for the persistence model, we do the same, but they'll drop from head
    pers_AL.dropna(axis='index', how='any', inplace=True)

    # drop the beginning 29 minutes of all other data to match (why 29? idk)
    df_2003.drop(df_2003.head(29).index, inplace=True)
    disc_AL.drop(disc_AL.head(29).index, inplace=True)

    # ---------------------------------------------------------------
    # ADD PERSISTENCE AS A FEATURE
    df_2003.insert(6, "AL_hist", pers_AL)
    model_vals.append("AL_hist")

    # take cols and index for remaking DF after scaling
    df_index = df_2003.index

    # ---------------------------------------------------------------
    # CORRELATION MATRIX BEFORE SCALER

    # # add disc AL
    # df_2003.insert(7, "disc_AL", disc_AL)

    # # # This is obsolete, but commented here as reminder of current state of
    # # # the dataframes
    # # # corr_pars = ['B_X_GSM', 'B_Y_GSM', 'B_Z_GSM', 'n_p', 'P', 'V',
    # #              # 'AL_hist', 'disc_AL']

    # print("\nCorrelation matrix before standardization:\n")
    # print(df_2003.corr())

    # # print("\nCorrelation matrix for October & November 2003\n")
    # # corr_oct_nov_2003 = df_oct_nov_2003[plot_vals].corr()
    # # print(corr_oct_nov_2003)

    # # remove disc AL
    # df_2003 = df_2003.drop(["disc_AL"], axis=1)

    # ---------------------------------------------------------------
    # SCALE THE MAIN FEATURES

    # fit the scaler, then transform the features
    X_scaler = StandardScaler()
    X_scaler = X_scaler.fit(df_2003)
    X_scaled = X_scaler.transform(df_2003)

    # need to add it back into a DF
    df_2003 = pd.DataFrame(X_scaled, columns=model_vals, index=df_index)
    print("\n2003 features have been scaled.")

    # scale AL
    y_scaler = StandardScaler()
    arr_AL = disc_AL.to_numpy()
    arr_AL = arr_AL.reshape(-1, 1)
    y_scaler = y_scaler.fit(arr_AL)
    y_scaled = y_scaler.transform(arr_AL)

    # need to add it back into a DF
    disc_AL = pd.DataFrame(y_scaled, columns=['AL'], index=df_index)
    print("2003 AL has been scaled.")

    # ---------------------------------------------------------------
    # # CORR AFTER STANDARDISATION

    # # Add AL back in to the df
    # df_2003.insert(7, "disc_AL", y_scaled)

    # print("\nCorrelation matrix after standardization:\n")
    # print(df_2003.corr())

    # # drop AL again, don't want it as a feature anymore
    # df_2003 = df_2003.drop(["disc_AL"], axis=1)

    # ---------------------------------------------------------------
    # # MUTUAL INFORMATION

    # print("\nMUTUAL INFORMATION:")

    # # 1 year data crashes. Lets use 1 week, centred on 24h storm
    # storm_dt = datetime.datetime(2003, 10, 27, 0, 0, 0)  # start of storm
    # start_dt = storm_dt - datetime.timedelta(days=3)  # start of week: -3d
    # end_dt = storm_dt + datetime.timedelta(days=4)  # end of week: +4d

    # # make copy so we cant break anything
    # mi_2003 = df_2003.copy()

    # # reinsert AL for consistent NaN drop
    # mi_2003.insert(7, 'disc_AL', disc_AL)

    # # narrow to week
    # mi_2003 = mi_2003.loc[start_dt:end_dt]

    # # drop NaNs, MI doesn't like them
    # mi_2003.dropna(axis='index', how='any', inplace=True)

    # mi_AL = mi_2003['disc_AL']
    # mi_2003 = mi_2003.drop(['disc_AL'], axis=1)

    # print("\nExample scenario: n_p and P should have high MI:")
    # print(mutual_info_regression(
    #     mi_2003['P'].to_numpy().reshape(-1, 1), mi_2003['n_p']))

    # print("")
    # print(model_vals, "vs AL:")
    # print(mutual_info_regression(mi_2003, mi_AL))

    # for i, feature in enumerate(model_vals):
    #     print("\nMutual information for", feature, "vs the others:")
    #     feature_array = model_vals.copy()
    #     feature_array = np.delete(feature_array, i)
    #     big_df = mi_2003.copy()
    #     big_df = big_df.drop([feature], axis=1)
    #     big_df = big_df.to_numpy()
    #     small_df = mi_2003[feature]
    #     small_df = small_df.to_numpy()
    #     print(feature_array)
    #     print(mutual_info_regression(big_df, small_df))

    # ---------------------------------------------------------------
    # REMOVING USELESS PARAMETERS

    # removing n_p - in the words of Mayur
    # "by far weakest correlation with AL and a strong correlation with P"
    df_2003 = df_2003.drop(["n_p"], axis=1)

    # ---------------------------------------------------------------
    # INTERPOLATING GAPS
    df_2003 = storm_interpolator(df_2003)

    print("\n2003 features interpolated.")

    # ---------------------------------------------------------------
    # PLOT HISTOGRAMS:

    # hist_dir = pathlib.Path('Figures')
    # pathlib.Path(hist_dir).mkdir(exist_ok=True)

    # for i, param in enumerate(plot_vals):
    #     plt.figure()
    #     plt.hist(df_2003[param], bins=35)
    #     plt.title('Histogram of '+param+' after StandardScaler')

    # ---------------------------------------------------------------
    # TRAIN TEST SPLIT

    # Needs shape (n_samples, n_features)
    # Split the data into two parts, one for training and testing
    # 5 splits, which I think leads to 6 sets of data, 2 months each
    # first 5 (k) are training, 6th (k+1)th one is testing

    print("\nFolding the data (cross-validation):")
    tscv = TimeSeriesSplit()
    fold_counter = 1
    for train_index, test_index in tscv.split(df_2003):
        # print("TRAIN:", train_index, "TEST:", test_index)  # debug
        print("Fold", fold_counter, "...")
        X_train, X_test = df_2003.iloc[train_index], df_2003.iloc[test_index]
        y_train, y_test = disc_AL.iloc[train_index], disc_AL.iloc[test_index]
        fold_counter += 1

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

    print("\nLinear Regression score (R^2 coeff, best 1.0):")
    print("Score:", regr.score(X_test, y_test))

    # put y pred back in a dataframe, just makes life easier for future
    y_test_index = y_test.index
    y_pred = pd.DataFrame(y_pred, columns=['pred_AL'], index=y_test_index)

    # ---------------------------------------------------------------
    # MAKING PERSISTENCE MATCH

    # earliest datetime that != NaN (because of rolling left by default)
    earliest_dt = pers_AL.index[0]

    # similarly, latest that != Nan (because this is rolled right, shifted -30)
    latest_dt = disc_AL.index[-1]

    # so now we can make an index of these happy values to use
    happy_index = y_test_index[(y_test_index >= earliest_dt)
                               & (y_test_index <= latest_dt)]

    # If running metrics on 2003 data, this would be what you used
    # Leaving here as reference so easier to generate for other storms
    # test_df = pd.DataFrame(
    #     y_test.loc[happy_index], columns=['AL'], index=happy_index)

    # pred_df = y_pred.loc[happy_index]

    pers_df = pd.DataFrame(pers_AL, columns=['AL'], index=pers_AL_index)
    pers_df = pers_df.loc[happy_index]

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
        # print("\nSTORM METRICS:")

        my_array = []
        # print("\nExplained variance score (higher is better, best 1.0):")
        # print("Discretized AL:",
              # explained_variance_score(y_true, y_pred))
        # print("Persistence AL:",
              # explained_variance_score(y_true, y_pers))
        my_array.append(explained_variance_score(y_true, y_pred))
        my_array.append(explained_variance_score(y_true, y_pers))

        # print("\nMean absolute error (lower is better, best 0.0):")
        # print("Discretized AL:", mean_absolute_error(y_true, y_pred))
        # print("Persistence AL:", mean_absolute_error(y_true, y_pers))
        my_array.append(mean_absolute_error(y_true, y_pred))
        my_array.append(mean_absolute_error(y_true, y_pers))
        
        # print("\nMean squared error (lower is better, best 0.0):")
        # print("Discretized AL:", mean_squared_error(y_true, y_pred))
        # print("Persistence AL:", mean_squared_error(y_true, y_pers))
        my_array.append(mean_squared_error(y_true, y_pred))
        my_array.append(mean_squared_error(y_true, y_pers))
        
        # this penalizes underpreciction more than overprediction
        # Mean squared logarithmic error (lower is better, best 0.0)
        # Penalizes underpreciction better. Good for exponential growth.
        # Unsure of usefulness here."
        # print("Discretized AL:", mean_squared_log_error(
        #     y_true, y_pred))
        # print("Persistence AL:", mean_squared_log_error(
        #     y_true, y_pers))

        # not too affected by outliers - good choice of metric?
        # print("\nMedian absolute error (lower is better, best 0.0):")
        # print("Discretized AL:", median_absolute_error(
            # y_true, y_pred))
        # print("Persistence AL:", median_absolute_error(
            # y_true, y_pers))
        my_array.append(median_absolute_error(y_true, y_pred))
        my_array.append(median_absolute_error(y_true, y_pers))

        # variance is dependent on dataset, might be a pitfall
        # print("\nR2 coefficient of determination (higher=better), best 1)")
        # print("Discretized AL:", r2_score(y_true, y_pred))
        # print("Persistence AL:", r2_score(y_true, y_pers))
        my_array.append(r2_score(y_true, y_pred))
        my_array.append(r2_score(y_true, y_pers))
        
        return my_array

    metrics = ["Explained variance score",
               "Mean absolute error",
               "Mean squared error",
               "Median absolute error",
               "R2 score"]

    metrics_desc = ["higher is better, best 1.0",
                    "lower is better, best 0.0",
                    "lower is better, best 0.0",
                    "lower is better, best 0.0",
                    "best value is 1"]
    # 2003 data - don't actually want to run metrics on this though
    # storm_metrics(test_df, pred_df, pers_df)

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

    storm_str_array = ["2006-12-14 to 2006-12-16",
                       "2001-08-31 to 2001-09-01",
                       "2005-08-31 to 2005-09-01",
                       "2010-04-05 to 2010-04-06",
                       "2011-08-05 to 2011-08-06"]

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
    y_array = []
    for i, storm in enumerate(storm_array):
        storm_index = storm.index  # save this for reconstructing DF later
        X = storm[model_vals].copy()
        y = storm['AL'].copy()

        # each storm's features to a df, append to X_array of dataframes
        X_array.append(pd.DataFrame(X, columns=model_vals, index=storm_index))
        y_array.append(pd.DataFrame(y, columns=['AL'], index=storm_index))

    # ---------------------------------------------------------------
    # PERSISTENCE AND DISCRETIZATION (is that even a word)

    disc_array = []
    pers_array = []
    df_index_array = []
    # i.e. for each AL dataframe
    for i, y in enumerate(y_array):
        disc_y = y.rolling(30).min()  # roll right, 30 minute window
        pers_y = disc_y.copy()
        disc_y = disc_y.shift(-30)  # roll left for discrete AL

        # drop the 30 minutes of NaNs from the shifting
        disc_y.dropna(axis='index', how='any', inplace=True)
        pers_y.dropna(axis='index', how='any', inplace=True)

        # make the two match up
        pers_y.drop(pers_y.tail(30).index, inplace=True)
        disc_y.drop(disc_y.head(29).index, inplace=True)

        # add to the master arrays
        pers_array.append(pers_y)
        disc_array.append(disc_y)

        # the indexes to remake the DF after scaling
        df_index_array.append(disc_y.index)

    for i, X in enumerate(X_array):
        X.drop(X.tail(30).index, inplace=True)
        X.drop(X.head(29).index, inplace=True)

        # ADD PERSISTENCE AS A FEATURE
        X.insert(6, "AL_hist", pers_array[i].to_numpy())
        X_array[i] = X

    # outside the for loop so we dont add it more than once!
    model_vals.append("AL_hist")

    # ---------------------------------------------------------------
    # STANDARD SCALING
    # we use the same scalers we used earlier. X_scaler and y_scaler

    y_array = []  # this is really, really bad practice...
    for i, storm_index in enumerate(df_index_array):

        # scale the main features
        X_trans = X_scaler.transform(X_array[i])

        # each storm's features to a df, append to X_array of dataframes
        X_array[i] = pd.DataFrame(X_trans,
                                  columns=model_vals, index=storm_index)

        # scale AL
        y = disc_array[i].to_numpy()
        y = y.reshape(-1, 1)
        y_trans = y_scaler.transform(y)

        # each storm's AL to a df, append to y_array of dataframes
        y_array.append(pd.DataFrame(
            y_trans, columns=['AL'], index=storm_index))

    # ---------------------------------------------------------------
    # REMOVING N_P
    # "by far weakest correlation with AL and a strong correlation with P"

    for i, X in enumerate(X_array):
        X = X.drop(["n_p"], axis=1)
        X_array[i] = X

    # ---------------------------------------------------------------
    # INTERPOLATE AND DROPNA IN THE TEST STORMS
    # remember, no longer need to interpolate y because its discretized

    # each storm will drop seperate dt, so store index to remake df
    week_index_array = []

    for i, X in enumerate(X_array):

        # interpolate the storm
        X_array[i] = storm_interpolator(X)

        # store the new index for this storm, for later
        week_index_array.append(X_array[i].index)

    # make sure y matches the X index (ie match the dropped points)
    for i, index in enumerate(week_index_array):
        y_array[i] = y_array[i].loc[index]

    print("\nTest storms interpolated.")

    # ---------------------------------------------------------------
    # PREDICTING THE DATA
    # remember we fitted the LinearRegression object regr on 2003 data

    y_pred_array = []
    for i, X in enumerate(X_array):
        prediction = regr.predict(X)
        pred_y = pd.DataFrame(
            prediction, columns=["pred_AL"], index=week_index_array[i])
        y_pred_array.append(pred_y)

    # ---------------------------------------------------------------
    # CHUNKING INTO ONE HOUR

    def storm_chunker(y_true, y_pred, y_pers, resolution='1h'):
        """
        Splits the storm into chunks of timesteps (only 1 hour implemented)
        If there's any gaps due to removed NaNs, it just means that hour will 
        have slightly less datapoints in it. E.g. the borders of the chunk will
        still be 60 minutes apart, if not 60 datapoints apart.

        Args:
          y_true: the AL of true (discretized) AL to compare against
          y_pred: the predicted values of AL from the model
          y_pers: the persistence/AL history values

        Returns:
          chunks: array of 3-col dfs, each col is true, pred, pers
        """

        first_dt = y_true.index[0]  # start_dt cannot precede this 
        limit_dt = y_true.index[-1]  # end_dt cannot exceed this

        year = int(first_dt.strftime("%Y"))
        month = int(first_dt.strftime("%m"))
        day = int(first_dt.strftime("%d"))
        hour = int(first_dt.strftime("%H"))

        # set it to the first (hopefully) full hour
        if (resolution=='1h'):
            start_dt = datetime.datetime(year, month, day, int(hour)+1, 0)
            end_dt = start_dt + datetime.timedelta(minutes=59)

        chunks = []
        while (end_dt <= limit_dt):

            # get the chunk
            true_chunk = y_true.loc[start_dt:end_dt]
            pers_chunk = y_pers.loc[start_dt:end_dt]
            pred_chunk = y_pred.loc[start_dt:end_dt]

            chunk_index = true_chunk.index

            # make one df for all three, append
            chunks.append(pd.concat(
                [true_chunk, pred_chunk, pers_chunk], axis=1, sort=False))

            # by iterating hours=1, we keep the XX:00 -> XX:59 spacing intact
            start_dt += datetime.timedelta(hours=1)
            end_dt += datetime.timedelta(hours=1)


        # now that every chunk for this storm is generated, return
        return chunks

    # will store the array of chunks for each of the storms
    chunks_array = []
    print("\nChunking storms:")
    for i in range(0,len(storm_array)):
        print(storm_str_array[i])

        # call the storm_chunker function to get array of chunks
        chunks_array.append(
            storm_chunker(y_array[i], y_pred_array[i], X_array[i]['AL_hist']))

    print("\nValidation storms have been chunked.")

    # ---------------------------------------------------------------
    # CALCULATING THE METRICS FOR EACH CHUNK

    # iterate through each storm
    for i,storm in enumerate(chunks_array):

        # 10 - pred then pers of each 5 metrics (5 lots of 2)
        metrics_array = [[],[],[],[],[],[],[],[],[],[]]

        # index array to plot metric against
        index_array = []
        
        # for each 1 hour chunk
        for chunk in storm:

            # run the metrics
            if (len(chunk) != 0):

                temp_metrics = storm_metrics(
                    chunk['AL'],chunk['pred_AL'],chunk['AL_hist'])
                
                # append it into our array
                for j in range(0, len(temp_metrics)):
                    metrics_array[j].append(temp_metrics[j])

                # also, get the index
                index_array.append(chunk['AL'].index[0].to_numpy())

        # OKAY lets get plotting for each metric!

        for j in range(0,5):
            a = 2*j
            b = a+1

            plt.figure()
            plt.title(metrics[j] + " " + storm_str_array[j])

            plt.plot(index_array, metrics_array[a], label='Model')
            plt.plot(index_array, metrics_array[b], label='Persistence')
            plt.axvline(storm_start_array[i], c='k', ls='--',
                        label='Storm period')
            plt.axvline(storm_end_array[i], c='k', ls='--')
            plt.ylabel('Metric score: ' + metrics_desc[i])
            plt.xlabel('DateTime')

            old_ylim = plt.ylim()
            new_low = old_ylim[0]
            new_top = old_ylim[1]
            
            
            print("\n")
            print("old_top:", new_top)
            print("old_low:", new_low)

            amax = np.amax(metrics_array[a])
            bmax = np.amax(metrics_array[b])
            amin = np.amin(metrics_array[a])
            bmin = np.amin(metrics_array[b])

            if (new_low <= -10):
                if amin <= bmin and amin>= -10:
                    new_low = 1.5*amin
                elif bmin <= amin and bmin >= -10:
                    new_low = 1.5*bmin
                else:
                    new_low = -10

            if (new_top >= 10):
                if amax >= bmax and amax<= 10:
                    new_top = 1.5*amax
                elif bmin <= amin and amin <= 10:
                    new_top = 1.5*bmax
                else:
                    new_top = 10

            print("new_top:", new_top)
            print("new_low:", new_low)

            plt.ylim((new_low, new_top))
            print(j, plt.ylim())
            plt.legend(loc='best')


main()
