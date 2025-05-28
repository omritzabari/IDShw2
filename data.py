import pandas as pd
from datetime import datetime
import numpy as np

def load_data(path):
    return pd.read_csv(path)


def add_new_columns(df):
    """
    new column for season name, new columns for the time stamps, and the column for the holiday weekend
    :param df:
    :return: dataframe with new columns
    """
    print("Part A: ")
    season_map = {0: "spring", 1: "summer", 2: "fall", 3: "winter"} #dictoinary that maps number to season
    df['season_name'] = df['season'].apply(lambda x: season_map.get(x, 'unknown')) #put the maping in the dataframe
    #add 4 column
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M')
    df['Hour'] = df['timestamp'].apply(lambda x: x.hour)
    df['Day'] = df['timestamp'].apply(lambda x: x.day)
    df['Month'] = df['timestamp'].apply(lambda x: x.month)
    df['Year'] = df['timestamp'].apply(lambda x: x.year)
    #add the holiday column based on the function
    df['is_weekend_holiday'] = df.apply(lambda row: holiday_weekend_check(row['is_holiday'], row['is_weekend']), axis=1)
    df['t_diff'] = df.apply(lambda x: x['t2'] - x['t1'], axis=1)
    return df


def holiday_weekend_check(holiday, weekend):
    """
    according to the instruction for the is_weekend_holiday column
    :param holiday:
    :param weekend:
    :return: the correct value for the "is_weekend_holiday" column
    """
    if (holiday == 0 and weekend == 0):
        return 1
    elif (holiday == 0 and weekend == 1):
        return 2
    elif (holiday == 1 and weekend == 0):
        return 3
    elif (holiday == 1 and weekend == 1):
        return 4


def data_analysis(df):
    """
    creating a correlation table and printing the 5 most and least correlated features in the dataframe
    calculating the mean of the t_diff column by each season and the overall mean of the t_diff column and printing
    :param df:
    :return:
    """
    print("describe output:")
    print(df.describe().to_string())
    print()
    print("corr output:")
    corr = df.corr()
    print(corr.to_string())
    print()

    correlation_pairs = {} #creating thr dictionary
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):  # Start from i + 1 to avoid self-correlation and duplicates
            feature1 = corr.columns[i]
            feature2 = corr.columns[j]
            abs_correlation_value = np.abs(corr.loc[feature1, feature2])
            correlation_pairs[(feature1, feature2)] = abs_correlation_value
    # Sort the pairs by their absolute correlation values
    sorted_highest_corr = sorted(correlation_pairs.items(), key=lambda item: item[1], reverse=True)
    sorted_lowest_corr = sorted(correlation_pairs.items(), key=lambda item: item[1])

    # Print the 5 highest correlated pairs
    print("Highest correlated are: ")
    for i in range(5):
        pair, value = sorted_highest_corr[i]
        print(f"{i + 1}. {pair} with {value:.6f}")
    print()

    # Print the 5 lowest correlated pairs
    print("Lowest correlated are: ")
    for i in range(5):
        pair, value = sorted_lowest_corr[i]
        print(f"{i + 1}. {pair} with {value:.6f}")
    print()

    #using groupby to calculate the mean of t_diff based on each season
    avg_tdiff = df.groupby('season_name')['t_diff'].mean()
    for seson in avg_tdiff.index:
        print(f"{seson} average t_diff is {avg_tdiff[seson]:.2f}")
    totalTDiffMean = df['t_diff'].mean()
    print(f"All average t_diff is {totalTDiffMean:.2f}\n")
