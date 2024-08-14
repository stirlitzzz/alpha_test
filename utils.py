import lzma
import dill as pickle

def load_pickle(path):
    with lzma.open(path,"rb") as fp:
        file = pickle.load(fp)
    return file

def save_pickle(path,obj):
    with lzma.open(path,"wb") as fp:
        pickle.dump(obj,fp)

def get_pnl_stats(date, prev, portfolio_df, insts, idx, dfs):
    day_pnl = 0
    nominal_ret = 0
    for inst in insts:
        units = portfolio_df.loc[idx - 1, "{} units".format(inst)]
        if units != 0:
            delta = dfs[inst].loc[date,"close"] - dfs[inst].loc[prev,"close"]
            inst_pnl = delta * units
            day_pnl += inst_pnl
            nominal_ret += portfolio_df.loc[idx - 1, "{} w".format(inst)] * dfs[inst].loc[date, "ret"]
    capital_ret = nominal_ret * portfolio_df.loc[idx - 1, "leverage"]
    portfolio_df.loc[idx,"capital"] = portfolio_df.loc[idx - 1,"capital"] + day_pnl
    portfolio_df.loc[idx,"day_pnl"] = day_pnl
    portfolio_df.loc[idx,"nominal_ret"] = nominal_ret
    portfolio_df.loc[idx,"capital_ret"] = capital_ret
    return day_pnl, capital_ret

import numpy as np
import pandas as pd
from copy import deepcopy

class AbstractImplementationException(Exception):
    pass

class Alpha():
    
    def __init__(self, insts, dfs, start, end, portfolio_vol=0.20):
        self.insts = insts
        self.dfs = deepcopy(dfs)
        self.start = start 
        self.end = end
        self.portfolio_vol = portfolio_vol

    def init_portfolio_settings(self, trade_range):
        portfolio_df = pd.DataFrame(index=trade_range)\
            .reset_index()\
            .rename(columns={"index":"datetime"})
        portfolio_df.loc[0,"capital"] = 10000
        portfolio_df.loc[0,"day_pnl"] = 0.0
        portfolio_df.loc[0,"capital_ret"] = 0.0
        portfolio_df.loc[0,"nominal_ret"] = 0.0
        return portfolio_df

    def pre_compute(self,trade_range):
        pass

    def post_compute(self,trade_range):
        pass
    
    def compute_signal_distribution(self, eligibles, date):
        raise AbstractImplementationException("no concrete implementation for signal generation")

    def compute_meta_info(self,trade_range):
        self.pre_compute(trade_range=trade_range)
        self.dfs = ReAlpha.create_sim_data_df(insts=self.insts,start=self.start,end=self.end,dfs=self.dfs)
        self.post_compute(trade_range=trade_range)
        return 
    
    def get_strat_scaler(self, target_vol, ewmas, ewstrats):
        ann_realized_vol = np.sqrt(ewmas[-1] * 253)
        return target_vol / ann_realized_vol * ewstrats[-1]
                    
    def run_simulation(self):
        date_range = pd.date_range(start=self.start,end=self.end, freq="D")
        self.compute_meta_info(trade_range=date_range)
        portfolio_df = self.init_portfolio_settings(trade_range=date_range)
        self.ewmas, self.ewstrats = [0.01], [1]
        self.strat_scalars = []
        for i in portfolio_df.index:
            date = portfolio_df.loc[i,"datetime"]
            eligibles = [inst for inst in self.insts if self.dfs[inst].loc[date,"eligible"]]
            non_eligibles = [inst for inst in self.insts if inst not in eligibles]
            strat_scalar = 2

            if i != 0:
                date_prev = portfolio_df.loc[i-1, "datetime"]
                
                strat_scalar = self.get_strat_scaler(
                    target_vol=self.portfolio_vol,
                    ewmas=self.ewmas,
                    ewstrats=self.ewstrats
                )

                day_pnl, capital_ret = get_pnl_stats(
                    date=date,
                    prev=date_prev,
                    portfolio_df=portfolio_df,
                    insts=self.insts,
                    idx=i,
                    dfs=self.dfs
                )
                self.ewmas.append(0.06 * (capital_ret**2) + 0.94 * self.ewmas[-1] if capital_ret != 0 else self.ewmas[-1])
                self.ewstrats.append(0.06 * strat_scalar + 0.94 * self.ewstrats[-1] if capital_ret != 0 else self.ewstrats[-1])

            self.strat_scalars.append(strat_scalar)
            forecasts, forecast_chips = self.compute_signal_distribution(eligibles,date)
            
            for inst in non_eligibles:
                portfolio_df.loc[i, "{} w".format(inst)] = 0
                portfolio_df.loc[i, "{} units".format(inst)] = 0
            
            vol_target = (self.portfolio_vol / np.sqrt(253)) * portfolio_df.loc[i,"capital"]

            nominal_tot = 0
            for inst in eligibles:
                forecast = forecasts[inst]
                scaled_forecast = forecast / forecast_chips if forecast_chips != 0 else 0
                position = \
                    strat_scalar * \
                    scaled_forecast \
                    * vol_target \
                    / (self.dfs[inst].loc[date, "vol"] * self.dfs[inst].loc[date,"close"])

                portfolio_df.loc[i, inst + " units"] = position 
                nominal_tot += abs(position * self.dfs[inst].loc[date,"close"])

            for inst in eligibles:
                units = portfolio_df.loc[i, inst + " units"]
                nominal_inst = units * self.dfs[inst].loc[date,"close"]
                inst_w = nominal_inst / nominal_tot
                portfolio_df.loc[i, inst + " w"] = inst_w
            
            portfolio_df.loc[i, "nominal"] = nominal_tot
            portfolio_df.loc[i, "leverage"] = nominal_tot / portfolio_df.loc[i, "capital"]
            if i%100 == 0: print(portfolio_df.loc[i])
        return portfolio_df

class ReAlpha():
    def __init__(self,insts,start,end,portfolio_vol) -> None:
        pass

    @staticmethod
    def create_sim_data_df(insts,start,end,dfs):
        trade_range=pd.date_range(start=start,end=end,freq='D')
        df=pd.DataFrame(index=trade_range)
        dfs_out=deepcopy(dfs)

        for inst in insts:
            inst_vol = (-1 + dfs[inst]["close"]/dfs[inst]["close"].shift(1)).rolling(30).std()
            dfs_out[inst] = df.join(dfs[inst]).fillna(method="ffill").fillna(method="bfill")
            #dfs_out has weekends and holidays filled in
            dfs_out[inst]["ret"] = -1 + dfs_out[inst]["close"]/dfs_out[inst]["close"].shift(1)
            dfs_out[inst]["vol"] = inst_vol
            dfs_out[inst]["vol"] = dfs_out[inst]["vol"].fillna(method="ffill").fillna(0)       
            dfs_out[inst]["vol"] = np.where(dfs_out[inst]["vol"] < 0.005, 0.005, dfs_out[inst]["vol"])
            sampled = dfs_out[inst]["close"] != dfs_out[inst]["close"].shift(1).fillna(method="bfill")
            eligible = sampled.rolling(5).apply(lambda x: int(np.any(x))).fillna(0)
            dfs_out[inst]["eligible"] = eligible.astype(int) & (dfs_out[inst]["close"] > 0).astype(int)
        return dfs_out
   
    @staticmethod
    def compute_meta_info(start,end,dfs):
        trade_range=pd.date_range(start=start,end=end,freq='D')
        df=pd.DataFrame(index=trade_range)
        dfs_out=deepcopy(dfs)
        dfs_out["close_raw"]=dfs_out["close"]

        dfs_out["close"]=dfs_out["close_raw"].fillna(method="ffill").fillna(method="bfill")
        dfs_out["ret"]=dfs_out["close_raw"].diff()/dfs_out["close_raw"].shift(1)
        dfs_out["vol"]=dfs_out["ret"].rolling(30).std()*np.sqrt(253)*np.sqrt(30/(30-dfs_out["close_raw"].isna().rolling(30).sum()))

        dfs_out["eligible"]=pd.DataFrame(np.where(dfs_out["close_raw"].isna(),0,1)&np.where(dfs_out["close"]>0,1,0),index=dfs_out["close"].index,columns=dfs_out["close"].columns)
        return dfs_out  

    @staticmethod
    def compute_signal_distribution(dfs,insts):
        alpha_scores = {}
        df=dfs["close"]
        random_data = np.random.uniform(-1, 1, size=df.shape)
        forecast_data=np.where(dfs["eligible"],random_data,0)
        df_scores = pd.DataFrame(random_data, index=df.index, columns=df.columns)
        df_forecasts = pd.DataFrame(forecast_data, index=df.index, columns=df.columns)
        df_forecast_chips=df_forecasts.abs().sum(axis=1)
        #input(f'df_forecast_chips: {df_forecast_chips}')

        return df_forecasts, df_forecast_chips 

    @staticmethod
    def compute_daily_pnl_bak(dfs,insts,portfolio_df,positions):
        df_closes=dfs["close"]
        df_rets=dfs["ret"]
        #input(f'df_rets: {df_rets.iloc[:-1,:]}')

        port_weights=deepcopy(positions)
        port_weights.iloc[:,1:]=positions.iloc[:,1:].div(positions.iloc[:,1:].abs().sum(axis=1),axis=0)

        for i in range(1,len(portfolio_df)):
            date=portfolio_df.loc[i,"datetime"]
            prev=portfolio_df.loc[i-1,"datetime"]

            day_pnl=positions.iloc[i-1,1:].to_numpy()@(df_closes.loc[date,:]-df_closes.loc[prev,:])
            #input(f'day_pnl={day_pnl}')
            portfolio_df.at[i,"day_pnl"]=day_pnl
            portfolio_df.at[i,"capital"]=portfolio_df.at[i-1,"capital"]+day_pnl

            #portfolio_df.at[i,"day_pnl"]=day_pnl
            #portfolio_df.at[i,"capital"]=portfolio_df.at[i-1,"capital"]+day_pnl
        portfolio_df["nominal_ret"].iloc[1:]=(port_weights.iloc[:-1,1:].to_numpy()*df_rets.iloc[1:,:].to_numpy()).sum(axis=1)
        #input(f'port_weights: {port_weights.iloc[:,1:]}')
        #input(f'df_rets: {df_rets.iloc[:-1,:]}')

        ret_check=portfolio_df["day_pnl"]-portfolio_df["capital"].shift(1)*portfolio_df["nominal_ret"]
        input(f'ret_check: {ret_check}')

    @staticmethod
    def compute_daily_pnl(mkt_data_dict,tickers,portfolio_df,df_weights=None,diagnostics=False):
        portfolio_df.at[0,"nominal"]=portfolio_df.at[0,"capital"]#will probably need to be removed
        portfolio_df.at[0,"leverage"]=portfolio_df.at[0,"nominal"]/portfolio_df.at[0,"capital"]#will probably need to be removed
        df_closes=mkt_data_dict["close"]
        df_rets=mkt_data_dict["ret"]
        #input(f'df_closes: {df_closes}')
        trade_dates_df=portfolio_df[["datetime"]]
        trade_dates_df.set_index("datetime",inplace=True)
        df_closes=trade_dates_df.join(df_closes)
        df_rets=trade_dates_df.join(df_rets)
        #input(f'df_closes: {df_closes}')
        np_close=df_closes.to_numpy()
        np_rets=df_rets.to_numpy()
        np_weights=df_weights.iloc[:,1:].to_numpy()
        #input(f'np_weights: {np_weights}')
        #leverage should be equal to sum of weights?
        #nominal return is 
        #np_positions=positions.iloc[:,1:].to_numpy()
        #(np_positions.shape,np_close.shape,np_rets.shape,np_weights.shape)
        day_pnl=np.empty(np_weights.shape[0])
        day_pnl[1:]=(np_weights[:-1,:]*np_rets[1:,:]).sum(axis=1)
        #input(f'np_weights[:-1,:].shape: {np_weights[:-1,:].shape}')
        np_positions_out=np.empty_like(np_weights)
        np_positions_out[0,:]=np_weights[0,:]*portfolio_df.at[0,"capital"]/np_close[0,:]
        #input(f'np_positions_out[0,:]: {np_positions_out[0,:]}')
        for i in range(1,np_weights.shape[0]):
            portfolio_df["day_pnl2"].at[i]=day_pnl[i]*portfolio_df.at[i-1,"capital"]
            portfolio_df["day_pnl"].at[i]=np_positions_out[i-1,:].dot(np_close[i,:]-np_close[i-1,:])
            portfolio_df["capital"].at[i]=portfolio_df.at[i-1,"capital"]+portfolio_df["day_pnl"].at[i]
            portfolio_df["nominal"].at[i]=portfolio_df.at[i,"capital"]# this will be set outside eventually
            portfolio_df.at[i,"leverage"]=portfolio_df.at[i,"nominal"]/portfolio_df.at[i,"capital"]#eventually this will be set outside
            np_positions_out[i,:]=np_weights[i,:]*portfolio_df.at[i,"capital"]/np_close[i,:]
        #day_pnl=np.concatenate(np.array([0.0]),day_pnl)
        portfolio_df["nominal_ret"]=portfolio_df["day_pnl"]/portfolio_df["nominal"].shift(1)
        portfolio_df["capital_ret"]=portfolio_df["nominal_ret"]*portfolio_df["leverage"]
        portfolio_df
        positions_out=pd.DataFrame(data=np_positions_out,columns=tickers,index=df_weights["datetime"])
        return (portfolio_df,positions_out) 



    

    #staticmdethod
    def init_portfolio_settings(trade_range):
        portfolio_df = pd.DataFrame(index=trade_range)\
            .reset_index()\
            .rename(columns={"index":"datetime"})
        portfolio_df.at[0,"capital"] = 10000
        portfolio_df.at[0,"day_pnl"] = 0.0
        portfolio_df.at[0,"day_pnl2"] = 0.0
        portfolio_df.at[0,"capital_ret"] = 0.0
        portfolio_df.at[0,"nominal_ret"] = 0.0
        return portfolio_df



def create_synthetic_closes(tickers,dates):
    num_dates=dates.shape[0]
    num_tickers=len(tickers)

    rand_returns=np.random.uniform(-0.01,0.01,(num_dates,num_tickers))
    closes=np.zeros_like(rand_returns)
    closes[0,:]=100
    for i in range(1,num_dates):
        closes[i,:]=closes[i-1,:]*(1+rand_returns[i,:])
    closes=pd.DataFrame(closes,columns=tickers,index=dates)
    closes.index.name="datetime"
    return closes

def create_random_positions(tickers,dates):
    num_dates=dates.shape[0]
    num_tickers=len(tickers)
    rand_positions=np.random.uniform(-1,1,(num_dates,num_tickers))*1000
    port_positions=pd.DataFrame(data=rand_positions,index=dates,columns=tickers)
    port_positions.reset_index(inplace=True)
    port_positions=port_positions.rename(columns={"index":"datetime"})
    return port_positions


def calculate_weights(port_positions,closes):
    port_weights=deepcopy(port_positions)
    port_weights.iloc[:,1:]=(port_positions.iloc[:,1:]*closes).div((port_positions.iloc[:,1:].abs()*closes).sum(axis=1),axis=0)
    return port_weights    

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

def output_dict_as_xlsx(dict_data, path):
    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
        for sheet_name, df in dict_data.items():
            #if (df.columns[0]=="datetime"):
            #    df=YahooDataConverter.make_timezone_naive(df)
            print(f'sheet_name: {sheet_name}')
            print(f'df.index: {df.index}')
            print(f'type(df.index): {type(df.index)}')
            if(type(df.index)==pd.DatetimeIndex):
                df.index=df.index.tz_localize(None)
                df.index.name='datetime'
            if("datetime" in df.columns):
                df["datetime"]=df["datetime"].dt.tz_localize(None)
            df.to_excel(writer, sheet_name=sheet_name, index=True)  
            
def main():
    from datetime import datetime
    import pytz
    #tickers=["AAPL","MSFT","GOGL","AMZN","FB","TSLA","NVDA","INTC","CSCO","ADBE"]
    tickers=["AAPL","MSFT","GOGL","AMZN"]
    start=datetime(2010,1,1, tzinfo=pytz.utc)
    #end=datetime.now(pytz.utc)
    end=datetime(2011,1,1, tzinfo=pytz.utc)
    trade_dates=pd.date_range(start=start,end=end,freq="d")
    #the index column is numbers and there is a separate datetime column
    #presumably that will make it easier to work with different trading frequencies

    closes=create_synthetic_closes(tickers,trade_dates)
    dict_data={"close":closes}
    df_data=ReAlpha.compute_meta_info(start,end,dict_data)

    
    #input(f'closes: {closes}')
    df_port=ReAlpha.init_portfolio_settings(trade_range=trade_dates)

    (forecasts,forecast_chips)=ReAlpha.compute_signal_distribution(df_data,tickers)

    weights=forecasts.div(forecast_chips,axis=0)
    weights.reset_index(inplace=True)
    weights.rename(columns={"index":"datetime"},inplace=True)
    input(f'forecasts: {forecasts}')

    (portfolio_df, positions_out)=ReAlpha.compute_daily_pnl(df_data,tickers,df_port,weights)

    input(f'portfolio_df: {portfolio_df}')
    input(f'positions_out: {positions_out}')
    dict_out=deepcopy(df_data)
    #    dict_out["positions"]=port_positions
    dict_out["weights"]=weights
    dict_out["positions_out"]=positions_out
    dict_out["portfolio"]=portfolio_df
    dict_out["forecasts"]=forecasts
    ###dict_out["forecast_chips"]=forecast_chips
   
    output_dict_as_xlsx(dict_out,"output1.xlsx")

class ReAlpha1(Alpha):
    @staticmethod
    def compute_meta_info(start,end,dfs):
        return ReAlpha.compute_meta_info(start,end,dfs)
    
    @staticmethod
    def compute_meta_info(start,end,dfs):
        dict_out= ReAlpha.compute_meta_info(start,end,dfs)
        dict_out["low_raw"]=dict_out["low"]
        dict_out["high_raw"]=dict_out["high"]
        dict_out["volume_raw"]=dict_out["volume"]

        dict_out["low"]=dict_out["low"].fillna(method="ffill").fillna(method="bfill")
        dict_out["high"]=dict_out["high"].fillna(method="ffill").fillna(method="bfill")
        dict_out["volume"]=dict_out["volume"].fillna(method="ffill").fillna(method="bfill")

        dict_out["op1"]=dict_out["volume"]
        dict_out["op2"]=(dict_out["close"]-dict_out["low"])-(dict_out["high"]-dict_out["close"])
        dict_out["op3"]=dict_out["high"]-dict_out["low"]
        dict_out["op4"]=dict_out["op1"]*dict_out["op2"]/dict_out["op3"]
        dict_out["op4"]=dict_out["op4"].replace(np.inf,0).replace(-np.inf,0)
        zscore=lambda x: (x-np.mean(x))/np.std(x)
        cszcre_df=dict_out["op4"].fillna(method="ffill").apply(zscore,axis=1)
        dict_out["zscore"]=cszcre_df
        dict_out["alpha"]=cszcre_df.rolling(12).mean()*-1
        dict_out["eligible"]=dict_out["eligible"]&(~pd.isna(dict_out["alpha"]))
        return dict_out





if __name__ == "__main__":
    main()
"""
def get_pnl_stats(date, prev, portfolio_df, insts, idx, dfs):
    day_pnl = 0
    nominal_ret = 0
    for inst in insts:
        units = portfolio_df.loc[idx - 1, "{} units".format(inst)]
        if units != 0:
            delta = dfs[inst].loc[date,"close"] - dfs[inst].loc[prev,"close"]
            inst_pnl = delta * units
            day_pnl += inst_pnl
            nominal_ret += portfolio_df.loc[idx - 1, "{} w".format(inst)] * dfs[inst].loc[date, "ret"]
    capital_ret = nominal_ret * portfolio_df.loc[idx - 1, "leverage"]
    portfolio_df.loc[idx,"capital"] = portfolio_df.loc[idx - 1,"capital"] + day_pnl
    portfolio_df.loc[idx,"day_pnl"] = day_pnl
    portfolio_df.loc[idx,"nominal_ret"] = nominal_ret
    portfolio_df.loc[idx,"capital_ret"] = capital_ret
    return day_pnl, capital_ret
"""
