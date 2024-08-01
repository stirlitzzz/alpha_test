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

    #staticmethod
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
    