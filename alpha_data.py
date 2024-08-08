import pytz
import yfinance
import requests
import threading
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from utils import ReAlpha

def get_sp500_tickers():
    res = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(res.content,'html')
    table = soup.find_all('table')[0] 
    df = pd.read_html(str(table))
    tickers = list(df[0].Symbol)
    return tickers

def get_history(ticker, period_start, period_end, granularity="1d", tries=0):
    try:
        df = yfinance.Ticker(ticker).history(
            start=period_start,
            end=period_end,
            interval=granularity,
            auto_adjust=True
        ).reset_index()
    except Exception as err:
        if tries < 5:
            return get_history(ticker, period_start, period_end, granularity, tries+1)
        return pd.DataFrame()
    
    df = df.rename(columns={
        "Date":"datetime",
        "Open":"open",
        "High":"high",
        "Low":"low",
        "Close":"close",
        "Volume":"volume"
    })
    if df.empty:
        return pd.DataFrame()
    
    #df["datetime"] = df["datetime"].dt.tz_localize(pytz.utc)
    df.datetime=df.datetime.dt.tz_convert(pytz.utc)
    df.datetime=df.datetime.dt.normalize()
    df = df.drop(columns=["Dividends", "Stock Splits"])
    df = df.set_index("datetime",drop=True)
    return df

def get_histories(tickers, period_starts,period_ends, granularity="1d"):
    dfs = [None]*len(tickers)
    def _helper(i):
        print(tickers[i])
        df = get_history(
            tickers[i],
            period_starts[i], 
            period_ends[i], 
            granularity=granularity
        )
        dfs[i] = df
    threads = [threading.Thread(target=_helper,args=(i,)) for i in range(len(tickers))]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    tickers = [tickers[i] for i in range(len(tickers)) if not dfs[i].empty]
    input(f'tickers: {tickers}')
    dfs = [df for df in dfs if not df.empty]
    return tickers, dfs

def get_ticker_dfs(start,end):
    from utils import load_pickle,save_pickle
    try:
        tickers, ticker_dfs = load_pickle("dataset.obj")
    except Exception as err:
        tickers = get_sp500_tickers()
        starts=[start]*len(tickers)
        ends=[end]*len(tickers)
        tickers,dfs = get_histories(tickers,starts,ends,granularity="1d")
        ticker_dfs = {ticker:df for ticker,df in zip(tickers,dfs)}
        save_pickle("dataset.obj", (tickers,ticker_dfs))
    return tickers, ticker_dfs 


class YahooDataConverter:
    def __init__(self):
        pass

    #staticmethod
    def convert_to_data_frames(tickers,dfs,period_start,period_end):
        date_range=pd.date_range(start=period_start,end=period_end)
    
        #assume all tickers have the same columns
        columns=dfs[tickers[0]].columns
        dfs_panel={}
        for column in columns:
            dfs_panel[column]=pd.DataFrame(index=date_range)
            for ticker in tickers:
                dfs_panel[column][ticker]=dfs[ticker][column]
        return dfs_panel

    # Function to convert timezone-aware datetimes to naive datetimes
    #staticmethod
    def make_timezone_naive(df):
        df.index=df.index.tz_localize(None)
        df.index.name='datetime'
        for col in df.columns:
            if pd.api.types.is_datetime64tz_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)
        return df
        #staticmethod
        def convert_to_csv(tickers,dfs,period_start,period_end):
            date_range=pd.date_range(start=period_start,end=period_end)
            df=pd.DataFrame(index=date_range)

from utils import Alpha
from utils import save_pickle, load_pickle
period_start = datetime(2010,1,1, tzinfo=pytz.utc)
period_end = datetime.now(pytz.utc)



import pdb
#pdb.set_trace()
tickers, ticker_dfs = get_ticker_dfs(start=period_start,end=period_end)
#ticker_dfs2=ReAlpha.create_sim_data_df_panel(tickers,period_start,period_end,ticker_dfs)
#input(f'tickers_dfs["close_filled"]: {ticker_dfs2["close_filled"]}')
testfor = 20
tickers = tickers[:testfor]
input(f'tickers: {tickers}')

df_dict=YahooDataConverter.convert_to_data_frames(tickers,ticker_dfs,period_start,period_end)
ticker_dfs2=ReAlpha.create_sim_data_df_panel(tickers,period_start,period_end,df_dict)
portfolio_df=ReAlpha.init_portfolio_settings(pd.date_range(start=period_start,end=period_end))
input(f'portfolio_df: {portfolio_df}')
df_signal_dist=ReAlpha.compute_signal_distribution(ticker_dfs2,tickers)

ReAlpha.compute_daily_pnl(ticker_dfs2,tickers,portfolio_df,portfolio_df)
with pd.ExcelWriter('output.xlsx', engine='xlsxwriter') as writer:
    # Iterate through the dictionary and write each DataFrame to a separate sheet
    for sheet_name, df in df_dict.items():
        #df.index
        df=YahooDataConverter.make_timezone_naive(df)
        df.to_excel(writer, sheet_name=sheet_name, index=True)
#input(f'df: {df}')

"""
from alpha1 import Alpha1
from alpha2 import Alpha2
from alpha3 import Alpha3

alpha1 = Alpha1(insts=tickers,dfs=ticker_dfs,start=period_start,end=period_end)
alpha2 = Alpha2(insts=tickers,dfs=ticker_dfs,start=period_start,end=period_end)
alpha3 = Alpha3(insts=tickers,dfs=ticker_dfs,start=period_start,end=period_end)
"""
