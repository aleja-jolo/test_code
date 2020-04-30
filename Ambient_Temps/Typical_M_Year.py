from typing import Union

import pandas as pd
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import calendar
import json
import pickle

from pandas import Series, DataFrame

# Function converts json input into a dataframe
def convert_to_df(time_temp):
    if not isinstance(time_temp, dict):
        raise TypeError('Please provide a dictionary')
    dtemps = []
    for t in time_temp:
        for i in range(0, len(time_temp[t]['time'])):
            if 'time' not in time_temp[t] or 'temperature' not in time_temp[t]:
                raise Exception(f'time or temperature not found on {t}')

            day = datetime.datetime.strptime(time_temp[t]['time'][i],'%d/%m/%Y %H:%M:%S')
            dtemps.append([day, time_temp[t]['temperature'][i]])
    if len(dtemps) > 0:
        df = pd.DataFrame(dtemps, columns=['DateTime', 'Temp_C'])
        df['DateTime'] = df['DateTime']
    else:
        raise Exception(f"No data found in temperature input")
    df.sort_values('DateTime', inplace=True)
    df.drop_duplicates(subset='DateTime', inplace=True)
    print(df)
    return df

# Function uses original dataframe data to interpolate missing dates
# Will only keep interpolated values if the number of consecutive days to interpolate was less than 6
def interpolate_raw(df):
    df['diff'] = df['DateTime'].diff().astype('timedelta64[h]')
    df.loc[:, 'diff'] = df['diff'].fillna(1)
    df.index = df['DateTime']
    del df['DateTime']
    df = df.resample('H').mean().reset_index()
    ndel = df[df['diff'] > 1]['DateTime']
    df['diff'] = df['diff'].fillna(method='bfill')
    for d in ndel:
        df.loc[df['DateTime'] == d, 'diff'] = 1
    if 6 in df['diff'].unique():
        print("Warning: data has at least one instance of 6 or more consecutive hours missing. "
              "Gaps of 6 or more hours are not interplated. t")
    df = df.query("diff<6")
    df['Temp_C'] = df['Temp_C'].interpolate(method='cubic')
    df.drop(columns=['diff'],inplace=True)
    return df

def process_data(df):
    # Process data
    # Strip df of unnecessary dates
    # Remove any datetimes inbetween whole hours (datetime with minutes > 0)
    #### to remove --> df = df.query("DateTime<datetime.datetime(2020,1,1) and DateTime>=datetime.datetime(2004,1,1)")
    #df.assign['Year'] =df['DateTime'].apply(lambda x: int(x.year))
    df.loc[:, 'Year'] = df['DateTime'].apply(lambda x: int(x.year))
    df.loc[:, 'Month'] = df['DateTime'].apply(lambda x: int(x.month))
    df.loc[:, 'Day'] = df['DateTime'].apply(lambda x: int(x.day))
    df.loc[:, 'Hour'] = df['DateTime'].apply(lambda x: int(x.hour))
    df.loc[:, 'Minute'] = df['DateTime'].apply(lambda x: int(x.minute))
    df = df.query("Minute==0")
    df = df.drop_duplicates()
    return df


# Class below is to calculate typical meteorological year (tmy)
# for a set of timeseries temperatures (over multiple years)
# User needs to pass dataframe of times and temperatures
class Sandia:
    # Initialize class with:
    # short term daily CDFs (daily means/min/max/range of each (year,month,day)) ==> day_df
    # long term daily CDFs (means/min/max/range of each (month,day) over all the years available) ==> lt_df
    def __init__(self, tdf):
        # Some days are missing many hours (i.e. only has Temps for 16 out of 24 hours)
        # We prefer not to interpolate over multiple missing hours
        # Therefore remove months with any days that have Temps for less than 22 hours of that day
        dfc = tdf.groupby(['Year', 'Month', 'Day'])['DateTime'].count().reset_index()
        remove_df = dfc.query("DateTime<22")
        for index, row in remove_df.iterrows():
            tdf = tdf[~((tdf['Year'] == int(row['Year'])) & (tdf['Month'] == int(row['Month'])))]

        tdf['date'] = tdf['DateTime'].apply(lambda x: x.date())
        self.df = tdf
        print(datetime.datetime.now())
        self.day_df = tdf.groupby(['date'])['Temp_C']. \
            describe()[['mean', 'min', 'max']].reset_index()
        self.day_df['Year'] = self.day_df['date'].apply(lambda x: x.year)
        self.day_df['Month'] = self.day_df['date'].apply(lambda x: x.month)
        self.day_df['Day'] = self.day_df['date'].apply(lambda x: x.day)
        print(datetime.datetime.now())
        self.day_df['range'] = self.day_df['max'] - self.day_df['min']
        self.lt_df = tdf.groupby(['Month', 'Day'])['Temp_C']. \
            describe()[['mean', 'min', 'max']].reset_index()
        self.lt_df['range'] = self.day_df['max'] - self.day_df['min']

    # Perform Steps 1 and 2 of Sandia method
    # These steps first calculates the FS statistic
    # The FS stat takes the difference between the long-term and short-term CDFs ..
    #   (for each day of the month)
    # FS is calculated for the mean, min, max, and range CDFs
    # The FS stats for the above are then weighted (equally) and summed
    # This produces one weighted FS (WS) stat for each month/year combination
    def get_Fstat(self):
        self.Fstat = pd.DataFrame(columns=['Year', 'Month', 'WS'])
        stats = ['mean', 'min', 'max', 'range']
        # weights are equal for the stats (above)
        weights = [0.25, 0.25, 0.25, 0.25]
        index = 0
        for m, mdf in self.day_df.groupby(['Month']):
            # long term for month of interest
            lt_m = self.lt_df[self.lt_df['Month'] == m]
            for y, ymdf in mdf.groupby('Year'):
                WS = 0
                # need to account discrepancy between 28 and 29 days in February
                if m == 2 and len(ymdf) >= 28:
                    lt_mi = lt_m[lt_m['Day'].isin(ymdf['Day'].values)]
                else:
                    lt_mi = lt_m.copy()
                # Only calculate FS stat if all the days of month are present in the short term df
                if len(lt_mi) == len(ymdf):
                    for j in range(len(stats)):
                        FSj = weights[j] * \
                              (np.sum(abs(np.array(sorted(ymdf[stats[j]].values)) -
                                          np.array(sorted(lt_mi[stats[j]].values)))) / len(ymdf))
                        WS += FSj
                    self.Fstat.at[index, 'Year'] = y
                    self.Fstat.at[index, 'Month'] = m
                    self.Fstat.at[index, 'WS'] = WS
                    index += 1
                else:
                    # If short term df is missing dates, it is skipped
                    # month/year is not counted towards TMY
                    # print(m, y, len(lt_mi), len(ymdf))
                    continue

        self.Fstat['Year'] = self.Fstat['Year'].astype(np.int64)
        self.Fstat['Month'] = self.Fstat['Month'].astype(np.int64)
        self.Fstat['WS'] = self.Fstat['WS'].astype(np.float64)

    # Function narrows down top five candiates for each month based on step 2..
    # and performs steps 3 and 4 of Sandia Method
    # Step 3 reorganizes top five:
    #   it first takes difference between based short and long terms ..
    #   and also short and long term medians ..
    #   for each month/year it picks the max out of the two ..
    #   and sorts top five based on ascending order of these values
    # Step 4 calculates the persitance of mean daily temperatures:
    #   it sorts short term monthly temps from min to max
    #   it then keeps only temperatures that are ..
    #   above 67th percentile of the long term temperatures for that month ..
    #   and below the 33rd percentile of the long term temps for that month
    #   From the remaining temps it calculates:
    #       # of runs (how many times you see consecutive days where temps are low or high)
    #       length of each run (sum of consecutive days)
    def get_TopFive(self):
        self.get_Fstat()
        self.TopFive = pd.DataFrame()
        # get top five from step 2
        for m, mdf in self.Fstat.groupby('Month'):
            self.TopFive = pd.concat([self.TopFive, mdf.nsmallest(5, ['WS'])], axis=0)

        # do step 3
        for m, mdf in self.TopFive.groupby('Month'):
            lt_m = self.lt_df[self.lt_df['Month'] == m]
            for y in mdf['Year'].unique():
                ymdf = self.day_df[(self.day_df['Year'] == y) & (self.day_df['Month'] == m)]
                mean = abs(ymdf['mean'].mean() - lt_m['mean'].mean())
                median = abs(ymdf['mean'].median() - lt_m['mean'].median())
                self.TopFive.loc[(self.TopFive['Year'] == y) &
                                 (self.TopFive['Month'] == m), 'Diff'] = np.max([mean, median])
        # sort top five base on step 3
        self.TopFive.sort_values(['Month', 'Diff'], inplace=True)

        # do step 4 (persistence)
        for m, mdf in self.TopFive.groupby('Month'):
            lt_m = self.lt_df[self.lt_df['Month'] == m]
            # get long term percentiles
            low = lt_m['mean'].quantile(0.33)
            high = lt_m['mean'].quantile(0.67)
            for y in mdf['Year'].unique():
                ymdf = self.day_df[(self.day_df['Year'] == y) & (self.day_df['Month'] == m)]
                ymdf = ymdf.sort_values(['mean'])
                # filter ymdf by values <=33p and >=67p
                tdf = pd.concat([ymdf[(ymdf['mean'] <= low)], ymdf[(ymdf['mean'] >= high)]], axis=0)
                # see how many consecutive days there are
                daydiff = abs(tdf.diff())
                daydiff.loc[daydiff['Day'] != 1, 'Day'] = 0
                # runs column accounts for each individual run
                # value is run lenghth - 1
                # len column adds 1 to run column to get actual run length value
                c = daydiff['Day'].eq(1)
                daydiff['run'] = daydiff[c].groupby(c.cumsum()).cumcount() + 1
                daydiff['len'] = daydiff['run'].apply(lambda x: x + 1 if x >= 1 else x)
                daydiff = daydiff[daydiff['len'].isnull() == False]
                if len(daydiff) > 0:
                    run = len(daydiff)
                    rlen = int(daydiff['len'].max())
                else:
                    run = 0
                    rlen = 0
                self.TopFive.loc[(self.TopFive['Year'] == y) &
                                 (self.TopFive['Month'] == m), 'runs'] = run
                self.TopFive.loc[(self.TopFive['Year'] == y) &
                                 (self.TopFive['Month'] == m), 'rlen'] = rlen

    # Function uses run and run lengths from above function to..
    # Narrow down top five and eventually pick top candidate
    # Rules are based on reference
    def get_TMY_candites(self):
        self.get_TopFive()
        Top_1 = pd.DataFrame()
        # part 1 of step 5 below
        for m, mdf in self.TopFive.groupby(['Month']):
            if mdf['runs'].sum() == 0:
                mdf = mdf.drop(mdf.tail(len(mdf) - 1).index)
            elif abs(mdf['runs'].diff()).sum() == 0:
                if mdf['rlen'].diff().sum() != 0:
                    mdf = mdf[~(mdf['rlen'] == mdf['rlen'].max())]
                else:
                    mdf = mdf.drop(mdf.tail(1).index)
            elif abs(mdf['runs'].diff()).sum() != 0:
                mdf = mdf[~(mdf['runs'] == mdf['runs'].max())]
            Top_1 = pd.concat([Top_1, mdf], axis=0)

        # unlike reference, runs that are == 0 are eliminated before next step
        # in reference it's the last thing it does but this way works better
        # part 3 of step 5
        for m, mdf in Top_1.groupby(['Month']):
            if len(mdf) > 1:
                Top_1 = Top_1[~((Top_1['Month'] == m) & (Top_1['runs'] == 0))]

        # part 2 of step 5
        Top_2 = pd.DataFrame()
        for m, mdf in Top_1.groupby(['Month']):
            if len(mdf) > 1:
                if mdf['rlen'].diff().sum() != 0:
                    mdf = mdf[~(mdf['rlen'] == mdf['rlen'].max())]
                else:
                    if abs(mdf['runs'].diff()).sum() == 0:
                        mdf = mdf.drop(mdf.tail(1).index)
                    else:
                        mdf = mdf[~(mdf['runs'] == mdf['runs'].max())]
            Top_2 = pd.concat([Top_2, mdf], axis=0)

        self.Top = Top_2.drop_duplicates(subset='Month', keep='first')

    # function returns array of 8670 temperatures (24 for each day of the year)
    # temps are based on candiate year/month from TMY Sandia calculations abovce
    def get_tmy_dic(self, dates):
        self.get_TMY_candites()
        # reduce starting temp dataframe to only contain candiate year/months
        self.TMY = pd.merge(self.df, self.Top, on=['Year', 'Month'])
        # create datetimes based on requested year
        yr = dates[1].year
        candyr = self.Top[self.Top['Month'] == 2]['Year'].iloc[0]
        curr_yr_days = (datetime.datetime(yr, 3, 1) - datetime.datetime(yr, 2, 1)).days
        cand_yr_days = (datetime.datetime(int(candyr), 3, 1) - datetime.datetime(int(candyr), 2, 1)).days
        print(cand_yr_days,curr_yr_days)
        # when candiate month is from a leap year but TMY year is not a leap year:
        if cand_yr_days > curr_yr_days:
            self.TMY = self.TMY[~((self.TMY['Year'] == int(candyr)) & (self.TMY['Month'] == 2) & (
                    self.TMY['Day'] == 29))]
        self.TMY['Time'] = self.TMY.apply(lambda x: datetime.datetime(yr, int(x['Month']), \
                                                                      int(x['Day']), int(x['Hour'])), axis=1)
        # when candidate month is not from a leap year but chosen TMY year is a leap year:
            # use the same temp distribution as previous day
        if curr_yr_days > cand_yr_days:
            print('add a day!')
            newday = self.TMY[(self.TMY['Month'] == 2) & (self.TMY['Day'] == 28)]
            newday.loc[:,'Time'] = newday['Time'].apply(lambda x: datetime.datetime(x.year, x.month, 29, x.hour))
            self.TMY = pd.concat([self.TMY, newday], axis=0)
        assert isinstance(self.TMY[['Time', 'Temp_C']].sort_values('Time').drop_duplicates, object)
        self.TMY = self.TMY[['Time', 'Temp_C']].sort_values('Time').drop_duplicates(subset='Time')
        # remove last six hours of last day of month and first six hours of next day
        # will interpolate so that it's smoothed out
        for i in range(1, 12):
            lastday = calendar.monthrange(2021, i)[1]
            self.TMY = self.TMY[~((self.TMY['Time'] > datetime.datetime(yr, i, lastday, 17)) & \
                                  (self.TMY['Time'] < datetime.datetime(yr, i + 1, 1, 6)))]
        # not every day has all 24 hours .. some are missing 1-2 hours in that day
        # need to fill in missing dates:
        if self.TMY['Time'].min() != np.min(dates):
            self.TMY = self.TMY.append({'Time': np.min(dates)}, ignore_index=True)
        dmax = np.max(dates)
        if self.TMY['Time'].max() != datetime.datetime(yr, dmax.month, dmax.day, 23):
            self.TMY = self.TMY.append({'Time': datetime.datetime(yr, dmax.month, dmax.day, 23)}, ignore_index=True)
        self.TMY['Time'] = pd.to_datetime(self.TMY['Time'])
        self.TMY.index = self.TMY['Time']
        del self.TMY['Time']
        self.TMY = self.TMY.resample('H').mean()  # fills missing dates
        # interpolate missing temperatures
        self.TMY['Temp_C'] = self.TMY['Temp_C'].interpolate(method='cubic')
        self.TMY.reset_index(inplace=True)
        # clean up df for final output
        self.TMY['datetime'] = self.TMY['Time'].apply(lambda x: x.date())
        self.TMY = self.TMY.round({'Temp_C': 2})
        # output is a dictionary with key == date and value == array of 24 temperatures
        tmy = {}
        for date, df in self.TMY.groupby('datetime'):
            tmy[str(date)] = list(df['Temp_C'].values)
        return (tmy)


# Function that returns p99 temperatures for each season
def get_p99(df,dates):
    # create assign all dates the same year and sort df by seasons
    yr = dates[1].year
    df = df[~((df['Month'] == 2) & (df['Day'] == 29))]
    df['Time'] = df.apply(lambda x:
                          datetime.datetime(yr, int(x['Month']), int(x['Day'])), axis=1)
    df.Month = df.Month.astype(np.int64)
    df.loc[(df['Time'] >= datetime.datetime(yr, 1, 1)) &
           (df['Time'] < datetime.datetime(yr, 3, 22)), 'Season'] = 'winter'
    df.loc[(df['Time'] >= datetime.datetime(yr, 3, 22)) &
           (df['Time'] < datetime.datetime(yr, 6, 22)), 'Season'] = 'spring'
    df.loc[(df['Time'] >= datetime.datetime(yr, 6, 22)) &
           (df['Time'] < datetime.datetime(yr, 9, 22)), 'Season'] = 'summer'
    df.loc[(df['Time'] >= datetime.datetime(yr, 9, 22)) &
           (df['Time'] < datetime.datetime(yr, 12, 22)), 'Season'] = 'fall'
    df.loc[(df['Time'] >= datetime.datetime(yr, 12, 22)), 'Season'] = 'winter'

    # go through each season and grab p99 of each hour
    season = {}
    for s, sdf in df.groupby(['Season']):
        print(s)
        season[s] = []
        for hr, hdf in sdf.groupby('Hour'):
            season[s].append(round(hdf['Temp_C'].quantile(.99), 2))
    return(season)


if __name__ == '__main__':
    # Eventually this code will be wrapped into a function where user puts in date array
    # For now we generate ourselves:
    dcurr = datetime.datetime(2021, 1, 1)
    dend = datetime.datetime(2022, 1, 1)
    dates = []
    while True:
        dates.append(dcurr)
        dcurr += relativedelta(days=+1)
        if dcurr.year == dend.year and dend.month == dend.month and dend.day == dend.day:  # end date
            break

    # load json file with temp dictionary
    time_temp = json.load(open("Lancaster_temps.json"))

    # convert dictionary to dataframe
    df = convert_to_df(time_temp)

    # interpolate missing values in daframe
    # note: interpolation will only happen when there are less than 6 consecutive values
    df = interpolate_raw(df)

    # Process dataframe
    df_p = process_data(df)

    # Get 99 percentile temperatures for each season
    p99_seasons = get_p99(df_p,dates)

    # Get typical metereological year temperatures
    # call class Sandia
    sand = Sandia(df_p)
    # Call function that returns dictionary
    # tmy[day]=[array of 24 hr temperatures]
    # i.e. tmy["2021-03-14"]=[23,24,19,21,....x24]
    tmy = sand.get_tmy_dic(dates)
    # print(tmy)

    # Below is to save and open files
    # f = open('Dublin_Tprofile.pkl', 'wb')
    # pickle.dump(tmy, f)
    # f.close()

    # f = open('Dublin_Tprofile.pkl','rb')
    # temps = pickle.load(f)
    # f.close()

