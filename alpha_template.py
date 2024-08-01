import pandas as pd

class Alpha_template():

    def __init__(self,insts,dfs,start,ends):
        """
        insts: list of strings
        dfs: dictionary of dataframes
        start: datetime
        end: datetime
        """
        self.insts=insts
        self.dfs=dfs
        self.start=start
        self.end=ends
    
    def init_portfolio_settings(self,date_range):
        portfolio_df=pd.DataFrame(index=date_range)\
            .reset_index()\
            .rename(columns={'index':'datetime'})
        portfolio_df.loc[0,'capital']=1000000
        print(f'portfolio_df: {portfolio_df}')
        input("bla")
        return portfolio_df
        pass

    
    def compute_meta_informaiton(self,date_range):
        print(f'self.dfs {self.dfs}')
        input("start of compute_meta_informaiton")
        for inst in self.insts:
            df=pd.DataFrame(index=date_range)
            df=df.join(self.dfs[inst]).fillna(method='ffill').fillna(method='bfill')
            self.dfs[inst]=df
            self.dfs[inst]['ret']=self.dfs[inst]['close']/self.dfs[inst]['close'].shift(1)-1
            sampled=self.dfs[inst]["close"]!=self.dfs[inst]["close"].shift(1).fillna(method='bfill')
            eligible=sampled.rolling(5).sum()>0 # 5 days
            #input(f'self.dfs[inst]["close"]: {self.dfs[inst]["close"]}')
            eligible2=self.dfs[inst]["close"]>0
            self.dfs[inst]["eligible"]=eligible.astype(int)&eligible2.astype(int)
            #print(f'df: {df}')
            print(f'inst: {self.dfs[inst]}')
        pass

    def run_sumulation(self):
        date_range=pd.date_range(start=self.start,end=self.end,freq='D')
        portfolio_df=self.init_portfolio_settings(date_range)
        self.compute_meta_informaiton(date_range=date_range)
        for i in portfolio_df.index:
            date=portfolio_df.loc[i,'datetime']
            eligibles=[inst for inst in self.insts if self.dfs[inst].loc[date,'eligible']==1]
            non_eligibles=[inst for inst in self.insts if inst not in eligibles]
            if i!=0:
                date_prev=portfolio_df.loc[i-1,'datetime']
                day_pnl,capital_return=get_pnl_stats(
                    date=date,
                    date_prev=date_prev,
                    portfolio_df=portfolio_df,
                    insts=self.insts,
                    idx=i,
                    dfs=self.dfs
                )
                pass


            alpha_scores={}
            import random
            for inst in eligibles:
                alpha_scores[inst]=random.uniform(0,1)
        
            alpha_scores={k: v for k, v in sorted(alpha_scores.items(), key=lambda item: item[1],reverse=True)}
            longs=list(alpha_scores.keys())[:int(len(eligibles)/4)]
            shorts=list(alpha_scores.keys())[-int(len(eligibles)/4):]
            print(f'longs: {longs}')
            print(f'shorts: {shorts}')
            input(f'alpha_scores: {alpha_scores}')
            for inst in non_eligibles:
                portfolio_df.loc[i,"{} w".format(inst)]=0
                portfolio_df.loc[i,"{} units".format(inst)]=0

            nominal_tot=0
            for inst in eligibles:
                forecast=1 if inst in longs else -1 if inst in shorts else 0
                dollar_allocation=portfolio_df.loc[i,"capital"]/(len(longs)+len(shorts))
                position=forecast*dollar_allocation/self.dfs[inst].loc[date,"close"]
                portfolio_df.loc[i,"{} units".format(inst)]=position
                nominal_tot+=abs(position)*self.dfs[inst].loc[date,"close"]

            for inst in eligibles:
                units=portfolio_df.loc[i,"{} units".format(inst)]
                nominal_inst=units*self.dfs[inst].loc[date,"close"]
                portfolio_df.loc[i,"{} w".format(inst)]=nominal_inst/nominal_tot
            portfolio_df.loc[i,"nominal_tot"]=nominal_tot
            portfolio_df.loc[i,"leverage"]=nominal_tot/portfolio_df.loc[i,"capital"]
            if i%100 == 0: print(portfolio_df.loc[i])

    
            input(f'portfolio_df: {portfolio_df.loc[i]}')
        #print(f'date_range: {date_range}')
        pass
   