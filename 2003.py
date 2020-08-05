import pathlib
import pandas as pd
# remember to run 'matplotlib tk' in the interpreter

from SolarWindImport import import_omni_month, import_omni_year


def main():

    year = 2003
    df_2003 = import_omni_year(year)

    pkl_dir = pathlib.Path('Data/OMNI/')
    oct_nov_pkl_name = pkl_dir / 'oct_nov_2003.pkl'

    try:
        df_oct_nov_2003 = pd.read_pickle((oct_nov_pkl_name))
        print("Found the pickle for October and November 2003.")

    except FileNotFoundError:
        print("Couldn't find the pickle for October and November 2003."
              + "-> Generating one from the .asc files.")

        df_oct_2003 = import_omni_month(2003, 10)
        df_nov_2003 = import_omni_month(2003, 11)
        df_oct_nov_2003 = pd.concat([df_oct_2003, df_nov_2003])
        df_oct_nov_2003.to_pickle(oct_nov_pkl_name)

    print(df_2003)

    # plotting

    plot_vals = ['B_X_GSM', 'B_Y_GSM', 'B_Z_GSM', 'n_p', 'AL', 'P', 'V']

    for val in plot_vals:
        year_title = str(val) + ' in ' + str(year)
        df_2003.plot(x='DateTime', y=val, title=year_title)

        oct_nov_title = str(val) + ' in October and November 2003'
        df_oct_nov_2003.plot(x='DateTime', y=val, title=oct_nov_title)

        print(df_2003[plot_vals].corr())
        print(df_oct_nov_2003[plot_vals].corr())


main()
