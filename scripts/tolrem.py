import shutil
import tweepy
import os, sys
import requests

import urllib.request
import hashlib, hmac, urllib3

from config import *
from glob import glob
from time import sleep, time

import pickle
import telegram
import numpy as np
import polars as pl
import pandas as pd
import random as rd

import datetime
from datetime import timedelta
from dateutil.parser import parse
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# from keras.models import load_model
# from tensorflow import keras
import json

# from kucoin_futures.client import Market
from pybit import usdt_perpetual
from apollox.rest_api import Client as ApolloxC

import pathlib
# import dropbox
# from dropbox.exceptions import AuthError

def truncate(f, n):
    return np.floor(f * 10 ** n) / 10 ** n

def perc_diff(b_arr, a_rr, lvl):
    b_prices_2x, a_prices_2x = b_arr[:lvl], a_rr[:lvl]
    diff_bid_ = 100*(b_prices_2x[-1]-b_prices_2x[0])/b_prices_2x[0] ### bid perc var
    diff_ask_ = 100*(a_prices_2x[-1]-a_prices_2x[0])/a_prices_2x[0] ### ask perc var
    return diff_bid_, diff_ask_

def norm_order_book(depth):
    
    wtf = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250]
    
    ask_ = np.array(depth['asks'], dtype=np.float64)
    bid_ = np.array(depth['bids'], dtype=np.float64)

    a_prices_ = ask_[:, 0]
    b_prices_ = bid_[:, 0]

    a_qtys_ = ask_[:, 1]
    b_qtys_ = bid_[:, 1]
    
    WW = []
    for ww in wtf:
        dbid, dask = perc_diff(b_prices_, a_prices_, ww)
        WW.append([dbid, dask])

    fa, la = a_prices_[0], a_prices_[-1]
    fb, lb = b_prices_[0], b_prices_[-1]
    # fb, lb, fa, la
    p_middle = (max(b_prices_) + min(a_prices_))/2 ### middle price | price between ask and bid boundary

    c_prices = np.concatenate([b_prices_, a_prices_])
    c_prices = np.sort(c_prices)

    serieC2 = c_prices-p_middle

    maxC2 = max(abs(serieC2))

    AB_norm = serieC2/maxC2
    ask_ab = AB_norm[AB_norm>0]
    
    bid_ab = AB_norm[AB_norm<0]
    bid_ab2 = abs(bid_ab[:][::-1]) ### normalized levels
    
    B = np.array([bid_ab2, b_qtys_]).reshape(2, -1).T
    A = np.array([ask_ab, a_qtys_]).reshape(2, -1).T

    # return B, A, diff_bid, diff_ask
    return B, A, WW
    
def real_levels2(bookB, bookA, levels=100):
    arr_r = np.arange(0,  levels)/levels
    # arr_range_ = list(arr_r)
    
    zero_b_ = np.zeros(bookB.shape)
    zero_a_ = np.zeros(bookB.shape)

    if levels>100:
        decimmal = 3
        # arr_range_ = [truncate(i, decimmal) for i in arr_range_]
        arr_range_ = truncate(arr_r, decimmal)
    else:
        decimmal = 2
        # arr_range_ = [truncate(i, decimmal) for i in arr_range_]
        arr_range_ = truncate(arr_r, decimmal)

    for i, (placeB, qtyB) in enumerate(bookB):
        placeA, qtyA = bookA[i]

        jeje1 = arr_r>placeB
        jeje2 = arr_r>placeA

        suma1 = jeje1.sum()
        suma2 = jeje2.sum()

        if suma1!=0:
            indiceB = jeje1.argmax()
            zero_b_[indiceB][1]+=qtyB
        else:
            zero_b_[-1][1]+=qtyB
    
        if suma2!=0:
            indiceA = jeje2.argmax()
            zero_a_[indiceA][1]+=qtyA
        else:
            zero_a_[-1][1]+=qtyA
        
    zero_b_[:, 0] = arr_r
    zero_a_[:, 0] = arr_r

    return zero_b_, zero_a_

def pump_it_up(B, A, K):
    B_p = B[:, 0].astype(np.float64)[:K]
    B_o = B[:, 1].astype(np.float64)[:K]
    # B_k = B_o.sum()
    B_vol = (B_o*B_p).sum()
    
    A_p = A[:, 0].astype(np.float64)[:K]
    A_o = A[:, 1].astype(np.float64)[:K]
    # A_k = A_o.sum()
    A_vol = (A_o*A_p).sum()
    return B_vol, A_vol#, B_k, A_k

def cheeseBinance(depth,
            # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 40, 45, 50, 100, 250] | 24
            players = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200], ### | 13
            limit=500
            # limit=250
            ):
    
    # bid_, ask_, dlevels = norm_order_book(depth) ### (b, a), (%b, %a; at 250 levels) ### dlevels => [:250]
    # bid, ask = real_levels2(bid_, ask_, levels=limit)
    ### [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 40, 45, 50, 100, 250]
    
    bid = np.array(depth['bids']).astype(np.float64)
    ask = np.array(depth['asks']).astype(np.float64)
    a_prices_ = ask[:, 0]
    b_prices_ = bid[:, 0]
    
    spread = 100*(a_prices_[0] - b_prices_[0])/b_prices_[0]
    
    dlevels = [];wtf = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250]
    for ww in wtf:
        dbid, dask = perc_diff(b_prices_, a_prices_, ww) ### -% | +%
        dlevels.append([dbid, dask])
    
    tt = depth['T']
    ### order book
    linea = ""
    for i, p in enumerate(players):
        B_vol, A_vol = pump_it_up(bid, ask, p)
        b_i, a_i = dlevels[i]
        linea += f"{B_vol}|{A_vol}|{b_i}|{a_i}|"
        
        # total = B_vol + A_vol
        # relB, relA = B_vol/total, A_vol/total
        # linea += f"{relB}|{relA}|{b_i}|{a_i}|"
    
    B_vol, A_vol = pump_it_up(bid, ask, 250)
    b250, a250 = dlevels[-1]
    linea += f"{B_vol}|{A_vol}|{b250}|{a250}\n"
    
    # total = B_vol + A_vol
    # relB, relA = B_vol/total, A_vol/total
    # linea += f"{relB}|{relA}|{b250}|{a250}\n"
    linea = f"{tt}|{spread}|{linea}"
    return tt, linea, 'Binance'

def cheeseOKX_Kucoin(depth,
            # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 40, 45, 50, 100, 250] | 24
            players = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200], ### | 13
            limit=500,
            ms=None,
            cdx=None
            # limit=250
            ):
    w=str(cdx)
    # bid_, ask_, dlevels = norm_order_book(depth) ### (b, a), (%b, %a; at 250 levels) ### dlevels => [:250]
    # bid, ask = real_levels2(bid_, ask_, levels=limit)
    ### [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 40, 45, 50, 100, 250]
    
    bid = np.array(depth['bids']).astype(np.float64)
    ask = np.array(depth['asks']).astype(np.float64)
    a_prices_ = ask[:, 0]
    b_prices_ = bid[:, 0]
    
    spread = 100*(a_prices_[0] - b_prices_[0])/b_prices_[0]

    dlevels = [];wtf = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250]
    for ww in wtf:
        dbid, dask = perc_diff(b_prices_, a_prices_, ww) ### -% | +%
        dlevels.append([dbid, dask])
    
    # tt = depth['T']
    # tt = ms
    dex = ''
    tt = depth['ts']
    if len(str(tt))>16:
        dex = 'Kucoin'
    else:
        dex = 'OKX'
    ### order book
    linea = ""
    # print(players)
    for i, p in enumerate(players):
        B_vol, A_vol = pump_it_up(bid, ask, p)
        b_i, a_i = dlevels[i]
        linea += f"{B_vol}|{A_vol}|{b_i}|{a_i}|"
        # print(i)
    
    B_vol, A_vol = pump_it_up(bid, ask, 250)
    b250, a250 = dlevels[-1]
    linea += f"{B_vol}|{A_vol}|{b250}|{a250}\n"
    linea = f"{tt}|{spread}|{linea}"
    return tt, linea, dex

def cheeseBitstamp(depth,
            # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 40, 45, 50, 100, 250] | 24
            players = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200], ### | 13
            limit=500,
            ms=None
            # limit=250
            ):
    
    # bid_, ask_, dlevels = norm_order_book(depth) ### (b, a), (%b, %a; at 250 levels) ### dlevels => [:250]
    # bid, ask = real_levels2(bid_, ask_, levels=limit)
    ### [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 40, 45, 50, 100, 250]
    
    bid = np.array(depth['bids']).astype(np.float64)
    ask = np.array(depth['asks']).astype(np.float64)
    a_prices_ = ask[:, 0]
    b_prices_ = bid[:, 0]
    
    spread = 100*(a_prices_[0] - b_prices_[0])/b_prices_[0]

    dlevels = [];wtf = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250]
    for ww in wtf:
        dbid, dask = perc_diff(b_prices_, a_prices_, ww) ### -% | +%
        dlevels.append([dbid, dask])
    
    # tt = depth['T']
    # tt = ms
    tt = depth['timestamp']
    ### order book
    linea = ""
    # print(players)
    for i, p in enumerate(players):
        B_vol, A_vol = pump_it_up(bid, ask, p)
        b_i, a_i = dlevels[i]
        linea += f"{B_vol}|{A_vol}|{b_i}|{a_i}|"
        # print(i)
    
    B_vol, A_vol = pump_it_up(bid, ask, 250)
    b250, a250 = dlevels[-1]
    linea += f"{B_vol}|{A_vol}|{b250}|{a250}\n"
    linea = f"{tt}|{spread}|{linea}"
    return tt, linea, "Bitstamp"

def cheeseCOIN(depth,
            # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 40, 45, 50, 100, 250] | 24
            players = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200], ### | 13
            limit=500,
            ms=None
            # limit=250
            ):
    
    # bid_, ask_, dlevels = norm_order_book(depth) ### (b, a), (%b, %a; at 250 levels) ### dlevels => [:250]
    # bid, ask = real_levels2(bid_, ask_, levels=limit)
    ### [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 40, 45, 50, 100, 250]
    
    bid = np.array(depth['bids']).astype(np.float64)
    ask = np.array(depth['asks']).astype(np.float64)
    a_prices_ = ask[:, 0]
    b_prices_ = bid[:, 0]
    
    spread = 100*(a_prices_[0] - b_prices_[0])/b_prices_[0]
    
    dlevels = [];wtf = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250]
    for ww in wtf:
        dbid, dask = perc_diff(b_prices_, a_prices_, ww) ### -% | +%
        dlevels.append([dbid, dask])
    
    # tt = depth['T']
    tt = time()
    ### order book
    linea = ""
    # print(players)
    for i, p in enumerate(players):
        B_vol, A_vol = pump_it_up(bid, ask, p)
        b_i, a_i = dlevels[i]
        linea += f"{B_vol}|{A_vol}|{b_i}|{a_i}|"
        # print(i)
    
    B_vol, A_vol = pump_it_up(bid, ask, 250)
    b250, a250 = dlevels[-1]
    linea += f"{B_vol}|{A_vol}|{b250}|{a250}\n"
    linea = f"{tt}|{spread}|{linea}"
    return tt, linea, 'Coinbase'

def cheeseBybit(jjbook,
            # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 40, 45, 50, 100, 250] | 24
            players = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200], ### | 13
            limit=500,
            ms=None
            # limit=250
            ):
    
    # bid_, ask_, dlevels = norm_order_book(depth) ### (b, a), (%b, %a; at 250 levels) ### dlevels => [:250]
    # bid, ask = real_levels2(bid_, ask_, levels=limit)
    ### [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 40, 45, 50, 100, 250]
    # bbid = [[i['price'], i['size']] for i in jjbook['data'] if i['side']=='Buy']
    # aask = [[i['price'], i['size']] for i in jjbook['data'] if i['side']=='Sell']
    # depth = jjbook['result']
    bid = np.array(jjbook['b']).astype(np.float64)
    ask = np.array(jjbook['a']).astype(np.float64)
    # bid = np.array(depth['bids']).astype(np.float64)
    # ask = np.array(depth['asks']).astype(np.float64)
    a_prices_ = ask[:, 0]
    b_prices_ = bid[:, 0]
    
    spread = 100*(a_prices_[0] - b_prices_[0])/b_prices_[0]
    
    dlevels = [];wtf = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250]
    for ww in wtf:
        dbid, dask = perc_diff(b_prices_, a_prices_, ww) ### -% | +%
        dlevels.append([dbid, dask])
    
    # tt = depth['T']
    tt = jjbook['ts']
    ### order book
    linea = ""
    # print(players)
    for i, p in enumerate(players):
        B_vol, A_vol = pump_it_up(bid, ask, p)
        b_i, a_i = dlevels[i]
        linea += f"{B_vol}|{A_vol}|{b_i}|{a_i}|"
        # print(i)
    
    B_vol, A_vol = pump_it_up(bid, ask, 250)
    b250, a250 = dlevels[-1]
    linea += f"{B_vol}|{A_vol}|{b250}|{a250}\n"
    linea = f"{tt}|{spread}|{linea}"
    return tt, linea, "Bybit"

def stts(dw, 
        # rangos=[10, 25, 50, 100, 250]
        # rangos=[5, 10, 15, 25, 50, 100, 250]
        # rangos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 40, 45, 50, 100, 250]
        rangos = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250]
        ):
    for r in rangos:
        t_vol_10 = dw[f'volb_{r}']+dw[f'vola_{r}']    
        dw[f'rb_{r}'] = dw[f'volb_{r}']/t_vol_10
        dw[f'ra_{r}'] = dw[f'vola_{r}']/t_vol_10
    return dw

def heikin_ashi(df):
    heikin_ashi_df = pd.DataFrame(index=df.index.values, columns=['Open', 'High', 'Low', 'Close'])
    heikin_ashi_df['Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    for i in range(len(df)):
        if i == 0:
            heikin_ashi_df.iat[0, 0] = df['Open'].iloc[0]
        else:
            heikin_ashi_df.iat[i, 0] = (heikin_ashi_df.iat[i-1, 0] + heikin_ashi_df.iat[i-1, 3]) / 2
    heikin_ashi_df['High'] = heikin_ashi_df.loc[:, ['Open', 'Close']].join(df['High']).max(axis=1)
    heikin_ashi_df['Low'] = heikin_ashi_df.loc[:, ['Open', 'Close']].join(df['Low']).min(axis=1)
    return heikin_ashi_df

def ds_heikin_zz(failneim, 
                 client, ### apollox
                 dicc={1: '1m', 3: '3m', 5: '5m', 15: '15m', 30: '30m', 60: '1h', 120: '2h', 240: '4h', 360: '6h', 480: '8h', 720: '12h'}, 
                 frec=1,
                 dex='',
                 client2=None ### binance
                 ):
    
    drr = pl.read_csv(failneim, sep='|')
    dr = drr.to_pandas()
    spreadd = dr['spread'].to_list()[-1]
    timesx = dr['dtimes'].to_list()
    init, finit = timesx[0], timesx[-1]
    # lols = drr['dtimes'].tolist()
    
    ### esta linea mamaweba leia un dtimes todo atrofiado 
    ### por una vaina destructiva que yo mismo hice
    ### linea del diablo, coño!
    dr.index = (1e6*dr['dtimes']).astype('datetime64[ns]') ### aqui antes se tenia una serie de spreads en lugar de dtimes, vrg
    # drt = dr['dtimes'].tolist()
    dr = dr.drop(columns=['dtimes'])#.resample('60s').sum() ### resample
    
    stable = 'USDT'
    elec = failneim.split("/")[0].replace("coin", "")
    
    # L3m = client.futures_klines(symbol=f'{elec}{stable}',interval="1m",limit=11) ### binance api
    if dex=='Binance':
        # L3m = client2.futures_klines(symbol=f'{elec}{stable}',interval="1m",limit=11) ### binance api
        # L3m = client2.get_klines(symbol=f'{elec}{stable}', interval="1m", limit=11) ### binance api
        # L3m = client2.get_historical_klines(symbol=f'{elec}{stable}', interval="1m", limit=1500, start_str=init, end_str=finit) ### binance api | spot
        L3m = client2.futures_historical_klines(symbol=f'{elec}{stable}', interval="1m", 
                                                # limit=25, 
                                                start_str=init, 
                                                end_str=finit
                                                ) ### binance api | futures
    else:
        L3m = client.klines(symbol=f'{elec}{stable}', interval='1m') ### apollox api
    
    arr = np.array(L3m, dtype=np.float)
    index = (1e6*arr[:, 0]).astype('datetime64[ns]') ### 2
    do = pd.DataFrame(arr[:, 1:5], columns=['Open', 'High', 'Low', 'Close'], index=index)
    # dh = heikin_ashi(do) ### aki kline
    
    levels = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250]
    vol_c = ''
    vor_c = ''
    r_c = ''
    for lev in levels:
        vol_c+= f'volb_{lev}|vola_{lev}|'
        r_c+= f'rb_{lev}|ra_{lev}|'
        vor_c+= f'vb_{lev}|va_{lev}|'
    
    col_v = vol_c[:-1].split("|")
    c_r = r_c[:-1].split("|")
    cor_v = vor_c[:-1].split("|")
    
    dvol = stts(dr[col_v].resample('60s').sum())[c_r]
    dvor = dr[cor_v].resample('60s').mean()
    # dspread = dr[['spread']].resample('60s').mean().to_list()[-1]
    dspread = dr[['spread']].resample('60s').mean().values[-1][0]
    
    dw = pd.concat([do, dvol, dvor], axis=1)
    
    return dw, do, dspread

def get_precision(info, symbol):
    for x in info['symbols']:
        if x['symbol'] == symbol:
            return x['quantityPrecision']

def m_G_5X(Dataframe, Dataframe2, Q): ### dha, da
    
    dataframe, dataframe2 = Dataframe.iloc[-Q:], Dataframe2.iloc[-Q:] ### (h, a)
    
    n = dataframe.shape[0]
    
    d50 = dataframe[['rb_5',   'ra_5',   'rb_10',  'ra_10',  'rb_15',  'ra_15',  'rb_20',  'ra_20', 'rb_25', 'ra_25',
                     'rb_30',  'ra_30',  'rb_40',  'ra_40',  'rb_50',  'ra_50',  'rb_75',  'ra_75',
                     'rb_100', 'ra_100', 'rb_150', 'ra_150', 'rb_200', 'ra_200', 'rb_250', 'ra_250']]
    
    d51 = dataframe[['vb_5',   'va_5',   'vb_10',  'va_10',  'vb_15',  'va_15',  'vb_20',  'va_20', 'vb_25', 'va_25',
                     'vb_30',  'va_30',  'vb_40',  'va_40',  'vb_50',  'va_50',  'vb_75',  'va_75',
                     'vb_100', 'va_100', 'vb_150', 'va_150', 'vb_200', 'va_200', 'vb_250', 'va_250']]
    
    w, h = d50.shape ### rows and columns
    
    SS, SS2, diff_dha, diff_ha = [], [], [], []
    SS3 = []
    
    for jk in range(w):
        
        ### price variations
        # dff_h = 100*(dataframe.iloc[jk].Close-dataframe.iloc[jk].Open)/dataframe.iloc[jk].Open ### dha variation
        dff_a = 100*(dataframe2.iloc[jk].Close-dataframe2.iloc[jk].Open)/dataframe2.iloc[jk].Open ### da variation
        
        # diff_dha.append(dff_h)
        diff_ha.append(dff_a)
        
        ### vol and vor information
        mozz0 = d50.iloc[jk].to_list() ### vol
        mozz1 = d51.iloc[jk].to_list() ### vor | :)
        
        SS+=mozz0   ### [vol0, vol1, vol2, ..., vol_n]
        SS2+=mozz1  ### [vor0, vor1, vor2, ..., vor_n]
        SS3+=mozz0+mozz1
        
    # total_var_Q = sum(diff_ha)        ### +/- %  all Q values
    # total_var_3 = sum(diff_ha[-3:])   ### +/- % last 3 values
    
    # j_Q = {True:1, False:0}
    # Q_Q3 = j_Q[abs(total_var_Q)>abs(total_var_3)] ### bool: variacion total de las Q velas vs variacion de las 3 ultimas velas
    # Q_QX = j_Q[total_var_Q>=0.20] ### float: variacion porcentual 
    # tupla = (Q_Q3, Q_QX, total_var_Q, total_var_3, diff_ha[-1]) ### (bool, float)
    
    vv = diff_ha + SS + SS2 ### 2022-10-04
    # vv = diff_dha + diff_ha + SS + SS2 ### 2022-05-20 | 1st try with 6 minutes data & 7 minutes collecting -> so far so good | currently using 5 minutes
    # vv = diff_dha + diff_ha + SS2 ### 2022-09-11 | currently using 9 minutes
    # vv = diff_dha + diff_ha + SS3 ### 2022-05-20 | 1st try with 6 minutes data & 7 minutes collecting -> so far so good | currently using 5 minutes

    V = np.array(vv)
    linea = ",".join(V.astype(np.str))
    # return V, linea, tupla
    return V, linea

def m_G5X(Dataframe, Dataframe2, Q): ### dha, da
    
    dataframe, dataframe2 = Dataframe.iloc[-Q:], Dataframe2.iloc[-Q:] ### (h, a)
    
    n = dataframe.shape[0]
    
    d50 = dataframe[['rb_5',   'ra_5',   'rb_10',  'ra_10',  'rb_15',  'ra_15',  'rb_20',  'ra_20', 'rb_25', 'ra_25',
                     'rb_30',  'ra_30',  'rb_40',  'ra_40',  'rb_50',  'ra_50',  'rb_75',  'ra_75',
                     'rb_100', 'ra_100', 'rb_150', 'ra_150', 'rb_200', 'ra_200', 'rb_250', 'ra_250']]
    
    d51 = dataframe[['vb_5',   'va_5',   'vb_10',  'va_10',  'vb_15',  'va_15',  'vb_20',  'va_20', 'vb_25', 'va_25',
                     'vb_30',  'va_30',  'vb_40',  'va_40',  'vb_50',  'va_50',  'vb_75',  'va_75',
                     'vb_100', 'va_100', 'vb_150', 'va_150', 'vb_200', 'va_200', 'vb_250', 'va_250']]
    
    w, h = d50.shape ### rows and columns
    
    SS, SS2, diff_dha, diff_ha = [], [], [], []
    SS3 = []
    
    for jk in range(w):
        
        ### price variations
        # dff_h = 100*(dataframe.iloc[jk].Close-dataframe.iloc[jk].Open)/dataframe.iloc[jk].Open ### dha variation
        dff_a = 100*(dataframe2.iloc[jk].Close-dataframe2.iloc[jk].Open)/dataframe2.iloc[jk].Open ### da variation
        
        # diff_dha.append(dff_h)
        diff_ha.append(dff_a)
        
        ### vol and vor information
        mozz0 = d50.iloc[jk].to_list() ### vol
        mozz1 = d51.iloc[jk].to_list() ### vor | :)
        
        SS+=mozz0   ### [vol0, vol1, vol2, ..., vol_n]
        SS2+=mozz1  ### [vor0, vor1, vor2, ..., vor_n]
        SS3+=mozz0+mozz1
        
    vv = diff_ha + SS + SS2 ### 2022-10-04
    
    V = np.array(vv)
    linea = ",".join(V.astype(np.str))
    # return V, linea, tupla
    return V, linea

def inference_456X(dataframe, dataframe2, minuto): ### (dataframe hakin en minutos, dataframe base en minutos, indices, scaler_norm, model)
    
    vector, _ = m_G5X(dataframe, dataframe2, Q=minuto)    ### last Q minutos
    # vector, linea, tupla_ = m_G5(dataframe, dataframe2, Q=minuto)    ### last Q minutos
    vector_str = ",".join(vector.astype(np.str_))                                                  ### original vector
    return vector, vector_str, minuto

def send_activity(coin, msg): ### group orders report
    
    WR={}
    
    channelID='-'
    
    aol = []
    
    token = np.random.choice(aol)
    bot = telegram.Bot(token=token) ### 1/8 bots 0.125 | to avoid to get timeout error
    bot.send_message(chat_id=channelID, text=msg)

def alertaaaaa(bot='',
               mensaje='Done!'
                ):
    bot = telegram.Bot(token=bot)
    que = bot.send_message(chat_id=, text=mensaje)

async def send_activity_2(msg): ### errors report
    tokens = []
              
    token = np.random.choice(tokens)
    bot2 = telegram.Bot(token=token)
    await bot2.send_message(chat_id='-', text=msg)
    
def send_activity2(msg): ### errors report
    tokens = []
              
    token = np.random.choice(tokens)
    bot2 = telegram.Bot(token=token)
    bot2.send_message(chat_id='-', text=msg)

def current_5p(client, par='USDT'):
    asset = [i for i in client.futures_account_balance() if par in list(i.values())][0]
    balance = float(asset['balance'])
    return balance*.05

### encargado de definir el monto a especular
def current_p(client, p, par='USDT'):
    
    ### retornar una seleccion porcentual del balance total
    # asset = [i for i in client.futures_account_balance() if par in list(i.values())][0]
    # balance = float(asset['balance'])
    # inv = balance*p
    # return inv                 ### seleccion de % a especular
    #
    t = (client, p, par)
    return 90 ### retorna una seleccion especifica de busd / usdt

### encargado de calcular el monto ajustado al apalancamiento
def to_speculate(qty, curr_price, leverage=10):
    quantity = (qty/curr_price)*leverage
    return quantity, qty*leverage


### responsable de firmar ordenes
def sign_order0(coin, y_pred, QTY, clientf, par='BUSD'):
    if y_pred==1:
        order = clientf.new_order(
                        symbol = f'{coin}{par}',
                        side = 'BUY',
                        orderType = 'MARKET',
                        quantity = QTY, ### editar cantidad
                    )
        return order, 'BUY', 'SELL'
    elif y_pred==2:
        order = clientf.new_order(
                        symbol = f'{coin}{par}',
                        side = 'SELL',
                        orderType = 'MARKET',
                        quantity = QTY, ### editar cantidad
                    )
        return order, 'SELL', 'BUY'

def get_min_dec(coin, session):
    # coin = 'DOT'
    stable = 'USDT'
    trading_info = [i for i in session.query_symbol()['result'] if f'{coin}{stable}' in i['name']][0]
    
    price_m = trading_info['price_filter']
    prices_m = [price_m['max_price'], price_m['min_price']]
    dcc = []
    for pp in prices_m:
        if "." in pp:
            div = pp.split(".")
            dccs = len(div[-1])
            dcc.append(dccs)
        else:
            dcc.append(0)
    
    sl_ro = min(dcc)
    # sl_ro -> minima cantidad de decimales a redondear en el sl
    
    minQty = trading_info['lot_size_filter']['min_trading_qty']
    
    minQtyStr = str(minQty)
    if "." in minQtyStr:
        minDec = minQtyStr.split(".")[-1]
        minDecL = len(minDec)

    else:
        minDecL = 1 - len(minQtyStr)
    
    if minDecL<0:
        minDecL=0
    
    return minQty, minDecL, sl_ro

def calc_liq_price(coin, entryPrice, leverage, side, p=0.125):
    coins = ['BTC', 'ETH']
    if coin in coins:
        comission = 0.5
    else:
        comission = 1.0
    p_0 = 100/leverage
    p_1 = p_0 - comission
    if side=='long':
        p_2 = 100 - p_1
        p_3 = p_2 / 100
        p_4 = entryPrice*p_3
        w = entryPrice - p_4
        vv = w*(1-p)
        sl = p_4 + vv
        # print(p_2, p_3, p_4, w, vv, sl)
        # print(p_4, w, vv, sl)
        # print(p_4, vv, sl)
        return p_4, sl
        
    elif side=='short':
        p_2 = 100 + p_1
        p_3 = p_2 / 100
        p_4 = entryPrice*p_3
        u = p_4 - entryPrice
        vv = u*(1-p)
        sl = p_4 - vv
        # print(p_2, p_3, p_4, u, vv, sl)
        # print(p_4, u, vv, sl)
        # print(p_4, vv, sl)
        return p_4, sl

def sign_order(coin, y_pred, QTY, SL, session, par='USDT'):
    if y_pred==1:
        # print("1 OJO:", coin, y_pred, QTY, SL)
        order = session.place_active_order(symbol=f'{coin}{par}',
                           side="Buy",
                           order_type="Market",
                           qty=QTY, ### same shit of binanace
                           time_in_force="GoodTillCancel",
                           reduce_only=False,
                           close_on_trigger=False,
                           stop_loss=SL)
        # print(1, order)
        return order, 'BUY', 'SELL'
    
    elif y_pred==2:
        # print("2 OJO:", coin, y_pred, QTY, SL)
        order = session.place_active_order(symbol=f'{coin}{par}',
                           side="Sell",
                           order_type="Market",
                           qty=QTY, ### same shit of binanace
                           time_in_force="GoodTillCancel",
                           reduce_only=False,
                           close_on_trigger=False,
                           stop_loss=SL)
        # print(2, order)
        return order, 'SELL', 'BUY'

### encargado de establecer las cantidades para una apertura/cierre exitosa de ordenes
def open_order_(coin, y_pred, client, clientf, p=0.10, leverage=10, par='USDT', dec=0):
    L = []
    value = current_p(client, p, par) ### 10% budget
    precio_actual = float(client.futures_symbol_ticker(symbol=f'{coin}{par}')['price'])
    cantidad, c_apal = to_speculate_(qty=value,
                                    curr_price=precio_actual,
                                    leverage=leverage
                                    ) ### with decimals
    
    r_qty = round(cantidad, dec) ### cantidad crypto redondeada | proxima a validarse
    dicc, toopen, toclose = sign_order_(coin, y_pred, r_qty, clientf, par)
    llaves = list(dicc.keys())
    
    if 'orderId' in llaves: ### valida el exito de la transaccion de existir el orderid
        if y_pred==1:
            return dicc, 'BUY', 'SELL', r_qty, c_apal, precio_actual, dec
        elif y_pred==2:
            return dicc, 'SELL', 'BUY', r_qty, c_apal, precio_actual, dec

### encargado de establecer las cantidades para una apertura/cierre exitosa de ordenes
def open_order0(coin, y_pred, client, clientf, p=0.10, leverage=10, par='USDT', dec=0):
    L = []
    value = current_p(client, p, par) ### 10% budget
    # precio_actual = float(client.futures_symbol_ticker(symbol=f'{coin}{par}')['price'])
    symbolInfo = session.latest_information_for_symbol(symbol=f'{coin}USDT')
    coinInfo = symbolInfo['result'][0]
    arr_BA = np.array([coinInfo['bid_price'], coinInfo['ask_price']], dtype=np.float64)
    if y_pred==1:
        precio_actual = arr_BA[0]
    elif y_pred==0:
        precio_actual = arr_BA[1]
    
    cantidad, c_apal = to_speculate0(qty=value,
                                    curr_price=precio_actual,
                                    leverage=leverage
                                    ) ### with decimals
    # print(cantidad, c_apal)
    decimales = [5, 4, 3, 2, 1, 0]
    
    for d in decimales:
        r_qty = round(cantidad, d) ### cantidad crypto redondeada | proxima a validarse
        # print(r_qty)
        dicc, toopen, toclose = sign_order_(coin, y_pred, r_qty, clientf, par)
        # print(dicc)
        llaves = list(dicc.keys())
        # print()
        if 'orderId' in llaves: ### valida el exito de la transaccion de existir el orderid
            if y_pred==1:
                return dicc, 'BUY', 'SELL', r_qty, c_apal, precio_actual, d
            elif y_pred==2:
                return dicc, 'SELL', 'BUY', r_qty, c_apal, precio_actual, d

def open_order(coin, y_pred, client, session, p=0.125, leverage=10, par='USDT', dec=0, value=0):
    """
    Creates and configures a trading order based on prediction and market data.
    
    Parameters:
        coin (str): The cryptocurrency symbol (e.g., 'BTC', 'ETH')
        y_pred (int): Prediction value (1 for long position, 2 for short position)
        client: Trading client instance for API interactions
        session: Session object for market data access
        p (float): Price percentage offset for liquidation calculation (default: 0.125)
        leverage (int): Leverage multiplier for position (default: 10)
        par (str): Quote currency pair (default: 'USDT')
        dec (int): Decimal precision override (default: 0)
        value (float): Dollar value to speculate (default: 0)
    
    Returns:
        tuple: Contains order dictionary and position details:
            - dicc: Order configuration dictionary
            - entry_side: 'BUY' or 'SELL' for entry position
            - exit_side: 'SELL' or 'BUY' for exit position
            - r_qty: Rounded quantity to trade
            - c_apal: Calculated leveraged amount
            - precio_actual: Current market price
            - minDecL: Minimum decimal places for quantity
    
    Raises:
        None explicitly, but may propagate exceptions from called functions
    """
    
    L = []  # Unused list, possibly for future logging
    
    # Fetch current market data for the symbol
    symbolInfo = session.latest_information_for_symbol(symbol=f'{coin}{par}')
    coinInfo = symbolInfo['result'][0]
    arr_BA = np.array([coinInfo['bid_price'], coinInfo['ask_price']], dtype=np.float64)
    
    # Determine position and price based on prediction
    precio_actual = 0
    possi = ''
    if y_pred == 1:
        precio_actual += arr_BA[0]  # Use bid price for long position
        possi = 'long'
    elif y_pred == 2:
        precio_actual += arr_BA[1]  # Use ask price for short position
        possi = 'short'
    
    # Calculate speculation amount and leveraged quantity
    cantidad, c_apal = to_speculate(qty=value,          # Dollar amount to trade
                                  curr_price=precio_actual,  # Current price
                                  leverage=leverage)        # Leverage multiplier
    
    # Get minimum quantity and decimal requirements for the coin
    minQty, minDecL, sl_ro = get_min_dec(coin, session)  # Returns qty, decimals, and rounding
    
    # Round quantity to coin-specific decimal places
    r_qty = round(cantidad, minDecL)
    
    # Calculate liquidation price and limit
    liqPrice, limitLiq = calc_liq_price(coin, precio_actual, leverage, possi, p=p)
    sl = round(limitLiq, sl_ro)  # Round liquidation price
    
    # Generate order configuration
    dicc, toopen, toclose = sign_order(coin, y_pred, r_qty, sl, session, par)
    
    # Return appropriate order details based on position type
    if y_pred == 1:
        return dicc, 'BUY', 'SELL', r_qty, c_apal, precio_actual, minDecL  # Long position
    elif y_pred == 2:
        return dicc, 'SELL', 'BUY', r_qty, c_apal, precio_actual, minDecL  # Short position

def cerrar_orden0(coin, cerrar, QTY, par, clientf):
    order = clientf.new_order(
                        symbol = f'{coin}{par}',
                        side = cerrar,
                        orderType = 'MARKET',
                        quantity = QTY
                        )
    return order, 0
                        
def cerrar_orden1(coin, cerrar, QTY, par, session):

    order = session.place_active_order(symbol=f'{coin}{par}',
                               side=cerrar.title(),
                               order_type="Market",
                               qty=QTY, ### same shit of binanace
                               time_in_force="GoodTillCancel",
                               reduce_only=True,
                               close_on_trigger=False,
                               # stop_loss=19001
                               )
    return order, 0

def cerrar_orden(coin, cerrar, QTY, par, session):
    order = session.close_position(symbol=f'{coin}{par}')
    precio_actual = current_position(session, coin)['result']['data'][0]['last_exec_price']
    return order, precio_actual

def current_position(session, coin, stable='USDT'):
    opss = session.get_active_order(symbol=f'{coin}{stable}')['result']['data'] ### request info
    if opss!=None: ### there is information
        pricee = opss[0]['stop_loss'] ### desired sl price
        if pricee!=0: ### operation in progress
            return opss ### current operation
        else:
            return [] ### no operation
    else:
        return [] ### no information hence no operation

def current_position0(session, coin, stable='USDT'):
    # opss = session.get_active_order(symbol=f'{coin}{stable}')['result']['data'][0]\
    opss = session.get_active_order(symbol=f'{coin}{stable}')['result']['data']
    if opss!=None:
        return [opss[0]]
    else:
        return []

### encargado de la logistica de informacion para dar apertura a longs/shorts
def open_long(coin, ypredd, apertura, client, session, lvg, json_dec={}, stable='BUSD', seconds=43, api=None, on=False, value=0):
    
    ## abrir long
    QTYs, apertura = [], []
    
    decimal = 0
    qty = 0
    orden = {}
    if on:
        orden, orderType, orderClose, qty, \
        lvg_qty, entryPrice, decimal = open_order(coin=coin,
                            y_pred=1, ### BUY
                            client=client,
                            session=session,
                            p=0.0725, # % stop loss
                            leverage=lvg,
                            par=stable,
                            value=value
                            ) ### time*
        QTYs.append(qty)
    
    nao2 = datetime.datetime.today() - datetime.timedelta(hours=5)
    datte = nao2.strftime("%Y-%m-%d %H:%M:%S")
    side = 'long'
    
    amount = float(qty)
    
    if amount==0:
        symbolInfo = session.latest_information_for_symbol(symbol=f'{coin}{stable}')
        coinInfo = symbolInfo['result'][0]
        # arr_BA = np.array([coinInfo['bid_price'], coinInfo['ask_price']], dtype=np.float64)
        entryPrice = float(coinInfo['bid_price'])
    
    apertura.append(entryPrice) ### opening price
    ypredd_ = str(float(ypredd)).split(".")
    isz = ypredd_[0]
    isz2 = ypredd_[-1][:4]
    decc = f"{isz}.{isz2}"
    
    info = f'{datte} | open @ {entryPrice} | ${coin.replace("1000", "")} | {amount}{coin.replace("1000", "")} ~ {decc} - {lvg}x | {side}'
    info2 = f'{datte} | Open long @ {round(entryPrice, 4)} | ${coin.replace("1000", "")}\n#{json_dec[coin]} #long #algotrade #stewiebot'
    print(info)
    send_activity(coin, info)
    
    if True:
        tweet = post_tweet(info2, api)

    operation, longg = 1, 1
    # sleep(0.1)
    wall = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(seconds=seconds)
    return side, wall, apertura, operation, longg, QTYs, decimal, tweet

#
def open_short(coin, ypredd, apertura, client, session, lvg, json_dec={}, stable='BUSD', seconds=43, api=None, on=False, value=0):
    
    ## abrir short
    QTYs, apertura = [], []
    
    decimal = 0
    qty = 0
    orden = {}
    if on:
        orden, orderType, orderClose, qty, \
        lvg_qty, entryPrice, decimal = open_order(coin=coin,
                            y_pred=2, ### SELL
                            client=client,
                            session=session,
                            p=0.0725, # % stop loss
                            leverage=lvg,
                            par=stable,
                            value=value
                            ) ### time*
        QTYs.append(qty)
    
    nao2 = datetime.datetime.today() - datetime.timedelta(hours=5)
    datte = nao2.strftime("%Y-%m-%d %H:%M:%S")
    side = 'short'
    
    amount = float(qty)
    
    if amount==0:
        symbolInfo = session.latest_information_for_symbol(symbol=f'{coin}{stable}')
        coinInfo = symbolInfo['result'][0]
        # arr_BA = np.array([coinInfo['bid_price'], coinInfo['ask_price']], dtype=np.float64)
        entryPrice = float(coinInfo['ask_price'])
    
    apertura.append(entryPrice)
    ypredd_ = str(float(ypredd)).split(".")
    isz = ypredd_[0]
    isz2 = ypredd_[-1][:4]
    decc = f"{isz}.{isz2}"
    
    info = f'{datte} | open @ {entryPrice} | ${coin.replace("1000", "")} | {amount}{coin.replace("1000", "")} ~ {decc} - {lvg}x | {side}'
    info2 = f'{datte} | Open short @ {round(entryPrice, 4)} | ${coin.replace("1000", "")}\n#{json_dec[coin]} #short #algotrade #stewiebot'
    print(info)
    send_activity(coin, info)
    
    if True:
        tweet = post_tweet(info2, api)

    operation, short = 1, 1
    wall = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(seconds=seconds)
    return side, wall, apertura, operation, short, QTYs, decimal, tweet

### encargado de la logistica de informacion para dar cierre a longs/shorts
def close_short(coin, client, session, apertura, QTYs, hist, stable='BUSD', seconds=43, lv=20, amount=0, yyprd=0.5, tweet="", api=None, json_dec=None, on=False, sl_q=True, value=1e-6):
    ### close long | sleep 15~20 minutes
    side = 'short'
    
    nao2 = datetime.datetime.today() - datetime.timedelta(hours=5)
    datte = nao2.strftime("%Y-%m-%d %H:%M:%S")
    
    if len(QTYs)>0:
        amount = QTYs[0]
        entryPrice = current_position(session, coin)[0]['last_exec_price']
        symbolInfo = session.latest_information_for_symbol(symbol=f'{coin}{stable}')
        coinInfo = symbolInfo['result'][0]
        current_price = float(coinInfo['last_price'])
        hist.append(current_price)
    else:
        amount = 0
        entryPrice = apertura[0] ### precio de entrada
        symbolInfo = session.latest_information_for_symbol(symbol=f'{coin}{stable}')
        coinInfo = symbolInfo['result'][0]
        current_price = float(coinInfo['last_price'])
        hist.append(current_price)
    
    if on and len(QTYs)>0 and sl_q: ### encendido & active qty & habilitado el cierre
        orden, operation = cerrar_orden(coin, 'BUY', amount, stable, session)          ### time*
    
    dif_ = -1*(current_price - entryPrice)/entryPrice
    dif_ = round(100*lv*dif_, 4)
    if dif_>0:perccc = f"+{dif_}%"
    else:perccc = f"{dif_}%"
    
    dif_2 = -1*(max(hist) - entryPrice)/entryPrice
    dif_2 = round(100*lv*dif_2, 4)
    if dif_2>0:perccc2 = f"+{dif_2}%"
    else:perccc2 = f"{dif_2}%"
    
    dif_3 = -1*(min(hist) - entryPrice)/entryPrice
    dif_3 = round(100*lv*dif_3, 4)
    if dif_3>0:perccc3 = f"+{dif_3}%"
    else:perccc3 = f"{dif_3}%"
    inf0 = f' | max: {perccc3} ({min(hist)}) - min: {perccc2} ({max(hist)}) theoretical: {perccc}'
    
    typr = str(float(yyprd)).split(".")
    iz = typr[0]
    dr = typr[-1][:3]
    ypd = f"{iz}.{dr}"
    
    if on:
        L = session.user_trade_records(symbol=f'{coin}{stable}')['result']['data']
        if L!=None and on:netIncome = L[0]['exec_value']-L[1]['exec_value']
        else:netIncome = 0
        
        v1 = float(value)
        v2 = float(netIncome)
        v3 = v1+v2
        
        diff0 = round(100*(v3-v1)/v1, 4)
        
        if diff0>0:strr = f"+{diff0}%"
        else:strr = f"{diff0}%"
    else:
        netIncome = 0
        diff0 = 0.0
        strr = f"{diff0}%"
    
    info = f'{datte} | close @ {round(entryPrice, 4)} - {round(current_price, 4)} | ${coin.replace("1000", "")} | {amount}{coin.replace("1000", "")} - {strr} - ${round(netIncome, 4)} - {lv}x - {ypd}p ~ market{inf0} | {side}'
    info2 = f'{datte} | close short @ {round(current_price, 4)}{inf0} | ${coin.replace("1000", "")}\n#{json_dec[coin]} #short #algotrade #stewiebot'
    print(info)
    print();
    # print(hist, "\n")
    send_activity(coin, info)
    
    if True:
        post_reply(info2, tweet, api)
    
    operation, short, QTYs, apertura = 0, 0, [], []
    wall = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(seconds=seconds)
    return wall, side, operation, short, apertura, QTYs

def close_long(coin, client, session, apertura, QTYs, hist, stable='BUSD', seconds=43, lv=20, amount=0, yyprd=0.5, tweet="", api=None, json_dec=None, on=False, sl_q=True, value=1e-6):
    ### close long | sleep 15~20 minutes
    side = 'long'
    
    nao2 = datetime.datetime.today() - datetime.timedelta(hours=5)
    datte = nao2.strftime("%Y-%m-%d %H:%M:%S")
    
    if len(QTYs)>0:
        amount = QTYs[0]
        entryPrice = current_position(session, coin)[0]['last_exec_price']
        symbolInfo = session.latest_information_for_symbol(symbol=f'{coin}{stable}')
        coinInfo = symbolInfo['result'][0]
        current_price = float(coinInfo['last_price'])
        hist.append(current_price)
    else:
        amount = 0
        entryPrice = apertura[0] ### precio de entrada
        symbolInfo = session.latest_information_for_symbol(symbol=f'{coin}{stable}')
        coinInfo = symbolInfo['result'][0]
        current_price = float(coinInfo['last_price'])
        hist.append(current_price)
    
    if on and len(QTYs)>0 and sl_q: ### encendido & active qty & habilitado el cierre
        orden, current_price = cerrar_orden(coin, 'SELL', amount, stable, session)          ### time*
    
    dif_ = (current_price - entryPrice)/entryPrice
    dif_ = round(100*lv*dif_, 4)
    if dif_>0:perccc = f"+{dif_}%"
    else:perccc = f"{dif_}%"
        
    dif_2 = (max(hist) - entryPrice)/entryPrice
    dif_2 = round(100*lv*dif_2, 4)
    if dif_>0:perccc2 = f"+{dif_2}%"
    else:perccc2 = f"{dif_2}%"
        
    dif_3 = (min(hist) - entryPrice)/entryPrice
    dif_3 = round(100*lv*dif_3, 4)
    if dif_3>0:perccc3 = f"+{dif_3}%"
    else:perccc3 = f"{dif_3}%"
    inf0 = f' | max: {perccc2} ({max(hist)}) -  min: {perccc3} ({min(hist)}) theoretical: {perccc}'
    
    typr = str(float(yyprd)).split(".")
    iz = typr[0]
    dr = typr[-1][:3]
    ypd = f"{iz}.{dr}"
    
    if on:
        L = session.user_trade_records(symbol=f'{coin}{stable}')['result']['data']
        if L!=None and on:netIncome = L[0]['exec_value']-L[1]['exec_value']
        else:netIncome = 0
        
        v1 = float(value)
        v2 = float(netIncome)
        v3 = v1+v2
        
        diff0 = round(100*(v3-v1)/v1, 4)
        if diff0>0:strr = f"+{diff0}%"
        else:strr = f"{diff0}%"
    else:
        netIncome = 0
        diff0 = 0.0
        strr = f"{diff0}%"
    
    info = f'{datte} | close @ {round(entryPrice, 4)} - {round(current_price, 4)} | ${coin.replace("1000", "")} | {amount}{coin.replace("1000", "")} - {strr} - ${round(netIncome, 4)} - {lv}x - {ypd}p ~ market{inf0} | {side}'
    info2 = f'{datte} | close long @ {round(current_price, 4)}{inf0} | ${coin.replace("1000", "")}\n#{json_dec[coin]} #long #algotrade #stewiebot'
    print(info)
    print()
    # print(hist, "\n")
    send_activity(coin, info)
    
    if True:
        post_reply(info2, tweet, api)
    
    operation, longg, QTYs, apertura = 0, 0, [], []    
    wall = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(seconds=seconds)
    return wall, side, operation, longg, apertura, QTYs
    
def verina(value):
    val_t = pd.isna(value)
    if val_t:
        return True
    else:
        return False

def get_position_json(json_position):
    for i in json_position:
        if abs(float(i['positionAmt']))>0:
            return i
    
def post_tweet(cadena, api, post=0):
    if post==1:
        original_tweet = api.update_status(status=cadena)
        return original_tweet
    else:
        return np.nan

def post_reply(cadena, prev_tweet, api, post=0):
    if post==1:
        reply_tweet = api.update_status(status=cadena, 
                                        in_reply_to_status_id=prev_tweet.id, 
                                        auto_populate_reply_metadata=True)
        return reply_tweet
    else:
        return np.nan

def get_coins_info():
    try:
        with open('../coins_data.json') as json_file:
            data = json.load(json_file)
            data2 = json.loads(data)
            json_file.close()
    except:
        with open('coins_data.json') as json_file:
            data = json.load(json_file)
            data2 = json.loads(data)
            json_file.close()
        
    return data2

def get_pond_arr(vals_p=5):
    arr_pond = np.linspace(0, 1, vals_p+2)[1:-1]
    suma_pond = arr_pond.sum()
    return arr_pond/suma_pond

def get_pond_m(info_arr):
    arr_s = np.array(info_arr)
    nl = arr_s.shape[0]
    arr = arr_s*get_pond_arr(nl)
    return arr.sum()

def get_date_f(mmss):
    date_ = datetime.datetime.fromtimestamp(mmss/1000.0)
    date__ = datetime.datetime.fromtimestamp(mmss/1000.0) - timedelta(hours=5)
    return ":".join(str(date_).split(":")[:-1]) + ":00", ":".join(str(date__).split(":")[:-1]) + ":00"

def get_low_high(df, clusters=12, maxim=9, c_pass=6):
    low_clusters = KMeans(n_clusters=clusters, init='k-means++', max_iter=300, n_init=10, random_state=None, verbose=0).fit(df[['Close']])
    # values = low_clusters.inertia_
    # low_clusters.get_params()
    values = low_clusters.cluster_centers_.T[0] ### transpose and 
    frec = low_clusters.labels_
    # print(values, "\n")
    # print(values, "\n", frec, "\n")

    ### record levels
    R = {}
    for i in low_clusters.labels_:
        # price = str(values[i])
        price = values[i]
        if price not in R:
            R[price]=1
        else:
            R[price]+=1

    ### filter frecquency
    vals = []
    LO = []
    for price_ in values:
        frecc = R[price_]
        if frecc>maxim:
            LO.append([price_, frecc])
            vals.append(price_)
    
    ### filter k neighbors
    # lowss = np.array(get_low_high(data, clusters=12))
    last_price = df['Close'].iloc[-1]
    lowss = np.array(vals)
    difff = last_price - lowss
    lools = []
    for i, diif in enumerate(difff):
        lools.append([abs(diif), i])
    lools.sort()
    levelss = []
    for i in range(c_pass):
        idx_p = lools[i][-1]
        levelss.append(lowss[idx_p])
    
    arr_dif = 100*(df['Close'].iloc[-1] - np.array(levelss))/np.array(levelss)
    return levelss, arr_dif
    
def order_book_filee(coin, bp, client,
                    filename1='',  ### temporal current
                    filename2='',  ### complete current
                    limit=500,
                    dex='',
                    client2=None):
    """
    Retrieves and processes order book data from various cryptocurrency exchanges.
    
    Parameters:
        coin (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        bp (str): Base pair/currency type (e.g., 'USDT', 'USD_PERP')
        client: Primary exchange client instance for API interactions
        filename1 (str): Path to temporary output file (default: '')
        filename2 (str): Path to complete output file (default: '')
        limit (int): Number of order book entries to retrieve (default: 500)
        dex (str): Exchange identifier ('Binance', 'OKX', etc.) (default: '')
        client2: Secondary client instance for specific exchanges (default: None)
    
    Returns:
        tuple: Processed order book information containing:
            - tt: Raw timestamp
            - dtime_str: Formatted datetime string
            - linex: Processed order book data string
            - exch: Exchange identifier
    
    Raises:
        requests.exceptions.RequestException: If API requests fail
        KeyError: If expected response data structure is invalid
    """
    
    # Exchange-specific order book retrieval
    if dex == 'Binance':
        if bp == 'USD_PERP':
            depth = client.futures_coin_order_book(symbol=f'{coin}{bp}')  # Fetch coin-margined futures
        elif bp in ['USDT', 'BUSD']:
            depth = client.futures_order_book(symbol=f'{coin}{bp}')  # Fetch USD-margined futures
        
        # Process Binance order book data
        tt, linex, exch = cheeseBinance(depth, limit=limit)  # Transform raw data
        residuo = float(tt) % 60000
        tt2 = float(tt) - residuo
        dtime_str = str(np.array([tt2*1e6]).astype('datetime64[ns]')[0])
    
    elif dex == 'OKX':
        # Fetch and process OKX order book
        dicc = requests.get(f'https://www.okex.com/api/v5/market/books?instId={coin}-USDT-SWAP&sz=300').json()
        depth = dicc['data'][0]
        tt, linex, exch = cheeseOKX_Kucoin(depth, cdx='OKX')
        exch = 'OKX'
    
    elif dex == 'Bitstamp':
        # Fetch and process Bitstamp order book
        depth2 = requests.get(f'https://www.bitstamp.net/api/v2/order_book/{coin.lower()}usd').json()
        tt, linex, exch = cheeseBitstamp(depth2)
    
    elif dex == 'Coinbase':
        # Fetch and process Coinbase order book
        dicc = requests.get(f"https://api.pro.coinbase.com/products/{coin}-USD/book?level=2").json()
        tt, linex, exch = cheeseCOIN(dicc)
    
    elif dex == 'Bybit':
        # Fetch and process Bybit order book
        url = f'https://api.bybit.com/derivatives/v3/public/order-book/L2?category=linear&symbol={coin}USDT&limit=200'
        depth = requests.get(url).json()['result']
        tt, linex, exch = cheeseBybit(depth)
    
    # Write to temporary file if specified
    if filename1 != '':
        with open(filename1, "a+") as arch:
            arch.write(linex)
    
    # Write to complete file if specified
    if filename2 != '':
        with open(filename2, "a+") as arch:
            arch.write(linex)
    
    return tt, dtime_str, linex, exch
    
def order_book_file(coin, bp, client,
                            filename1='', ### temporal current
                            filename2='', ### complete current
                            limit=500,
                            dex='Binance'
                            ):
    
    if dex=='Binance':
        if bp=='USD_PERP':
            depth = client.futures_coin_order_book(symbol=f'{coin}{bp}') ### order book | COIN
        elif bp=='USDT' or bp=='BUSD':
            depth = client.futures_order_book(symbol=f'{coin}{bp}') ### order book | USD
        
        ### prepo de cheese | current
        tt, linex, exch = cheeseBinance(depth, limit=limit) ### current
        residuo = float(tt)%60000
        tt2 = float(tt) - residuo
        # dtime_str = str(np.array([tt2*1e6]).astype('datetime64[ns]')[0])
        dtime_str = str(np.array([tt2*1e6]).astype('datetime64[ns]')[0])
    
    # return tt, linea, tt2, linea2
    return tt, dtime_str, linex, exch
    
def dict_data(tt, linea, dG):
    ### ORDER BOOK - MINIMIZE PROCESSING - DATA BLOCK
    ##
    #
    # levels = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250]
    #
    ##

    # data_vl = [float(i) for i in linea.split("|")] ### split line
    arr_ = np.array(linea.split("|"), dtype=np.float64).reshape(-1, 4) ### order book line | [volb,vola,vorb,vora]
    data_vl = arr_[:, :2].flatten().tolist() ### market actores | no stts()
    data_vr = arr_[:, 2:].flatten().tolist() ### market changes
    # print(data_vl)
    # print(data_vr)
    f_str, f_str_5 = get_date_f(tt) ### UTC date | local date

    ### registro de un nuevo bloque de mvol por cada minuto | 'minus':'{minuto}':"{[valores_vol]}"
    ### registro de un nuevo bloque de mvor por cada minuto | 'plus' :'{minuto}':"{[valores_vor]}"
    if f_str_5 not in dG['minus']: ### constancia de la existencia del registro por minuto | cada iteracion | primer registro
        
        dG['all'].append([data_vl])  ### adicion de un nuevo bloque vol para el minuto de la iteracion en curso | lista general => lista de cada minuto => lista de cada iteracion
        dG['all2'].append([data_vr]) ### adicion de un nuevo bloque vor para el minuto de la iteracion en curso | lista general => lista de cada minuto => lista de cada iteracion
        
        dG['minus'][f_str_5] = data_vl ### first vol block (minute) | creacion del vol diccionario interno por minuto | primer bloque O suma del unico vector
        dG['plus'][f_str_5]  = data_vr ### first vor block (minute) | creacion del vor diccionario interno por minuto | primer bloque O media del unico vector
        dG['tts']+=[tt]
    
    else: ### existencia del minuto de la iteracion en curso | registro de cada nueva iteracion sobre el minuto existente
        dG['all'][-1].append(data_vl) ### adicion de valores al nuevo bloque del minuto de la iteracion en curso | lista de cada minuto appends lista de cada iteracion
        dG['all2'][-1].append(data_vr) ### adicion de valores al nuevo bloque del minuto de la iteracion en curso | lista de cada minuto appends lista de cada iteracion
        
        ### => bloque del json previo al dataframe:
        
        dG['minus'][f_str_5] = np.sum(dG['all'][-1], axis=0).tolist() ### obtecion de los valores del bloque vol para la obtencion de la suma y su registro
        dG['plus'][f_str_5]  = np.mean(dG['all2'][-1], axis=0).tolist() ### obtecion de los valores del bloque vor para la obtencion de la media y su registro
        dG['tts']+=[tt]
    #
    ##
    ### ORDER BOOK - MINIMIZE PROCESSING - END BLOCK
    return dG

def get_utils(coin, minutos=9, fecha=None, y=None):
    
    # minutos = 9
    
    sclrs = glob(f"mM*/*/*{fecha}*");sclrs = glob(f"mM*/*/*{fecha}*")
    scalers = sorted(glob(f"mM*/{coin}/*{fecha}*{minutos}j*csv")) ### csv files
    
    dates = [i.split(f"{coin}_")[-1].split(f"_{minutos}j")[0].split("_") for i in scalers]
    dates2 = ["-".join(i[:3][::-1])+" "+":".join(i[-3:]) for i in dates]
    dates3 = [parse(o) for o in dates2]
    idMx = dates3.index(max(dates3))

    sc_file = scalers[idMx]
    dscaler = pl.read_csv(sc_file, sep="|")
    scaler = StandardScaler().fit(dscaler.to_numpy()) ### scaler

    date_str = "_".join(dates[idMx])
    model = glob(f"mM*/{coin}/*{date_str}*{y}*h5")[0] ### h5 files
    loaded_model = load_model(model) ### model
    
    return (loaded_model, scaler), (model, sc_file)
    
def bybit_request(**kwargs):
    
    sessionn = kwargs['sessionn']
    path = kwargs['path']
    metodo = kwargs['method']
    
    del kwargs['sessionn']
    del kwargs['path']
    del kwargs['method']
    
    api_key = sessionn.api_key
    # print(api_key)
    if type(api_key)==tuple:
        api_key = api_key[0]
    secretKey = bytes(sessionn.api_secret, 'utf-8')
    
    timestamp = int(time() * 10 ** 3)
    urllib3.disable_warnings()
    s = requests.session()
    s.keep_alive = False
    
    params = {
        "api_key": api_key,
        "timestamp": str(timestamp)
    }
    
    params = {**params, **kwargs}
    
    sign = ''
    for key in sorted(params.keys()):
        v = params[key]
        if isinstance(params[key], bool):
            if params[key]:
                v = 'true'
            else :
                v = 'false'
        sign += key + '=' + f'{v}' + '&'
    
    sign = sign[:-1]
    hashh = hmac.new(secretKey, sign.encode("utf-8"), hashlib.sha256)
    signature = hashh.hexdigest()
    
    params['sign'] = signature
    
    del params['api_key']
    # print(params)
    
    cadeena = ''
    for clave in params:
        valor = params[clave]
        cadeena += f"&{clave}={valor}"
    
    url = 'https://api.bybit.com' + path
    for clave in params:
        valor = params[clave]
        if type(valor)==int:
            params[clave] = str(valor)
    raw = json.dumps(params)
    # print(raw)
    if metodo=='get':
        full_url = url + '?api_key=' + api_key + cadeena
        # print(full_url)
        response = requests.get(full_url)
    elif metodo=='post':
        full_url = url + '?api_key=' + api_key + raw
        # print(full_url)
        response = requests.get(full_url, data=raw)
    return json.loads(response.text), response

def update_sl(session, coin, new_sl, flag, stable='USDT'):
    side0 = {'green':'Buy', 'red':'Sell'}
    side_w = side0[flag]
    order2 = session.set_trading_stop(symbol=f"{coin}{stable}", 
                                  side=side_w, stop_loss=new_sl)
    return 0
    
def upload_data(dpbx, filee, dst):
    if dst.startswith("/"):dest_path=str(dst)
    else:dest_path=f"/{dst}"
    with open(filee, 'rb') as f:
        dpbx.files_upload(f=f.read(), path=dest_path, mode=dropbox.files.WriteMode.overwrite, mute=True)

def downl_data(dpbx, src, dst):
    dpbx.files_download_to_file(download_path=dst, path=src, rev=rev)
    
def get_vekctor(dicc, llave, headder, _col_v, _c_r, _cor_v):
    """
    Computes mathematical vectors from order book data for analysis.
    
    This function processes raw data into three key vectors: spread, volume relations,
    and price variations, performing normalization and aggregation operations.
    
    Parameters:
        dicc (dict): Input dictionary containing order book data
        llave (str): Key to access specific data within dicc['data']
        headder (list): Column names for initial DataFrame
        _col_v (list): Column names for volume-related data
        _c_r (list): Column names for relation vector output
        _cor_v (list): Column names for variation vector data
    
    Returns:
        tuple: Mathematical vectors and components:
            - V (np.ndarray): Concatenated vector of spread, relations, and variations
            - _sspread (float): Mean spread value
            - l_r (list): Volume relations vector as list
            - l_v (list): Price variations vector as list
    
    Notes:
        - Assumes input data contains pipe-delimited strings
        - Timestamps are converted from microseconds to nanoseconds
        - Volume relations are normalized against total sums
    """
    
    # Parse and structure raw data into DataFrame
    dff = pd.DataFrame(dicc['data'][llave])[0].str.split("|", expand=True).astype(float)
    dff.columns = headder
    # Convert timestamp from microseconds to nanoseconds for datetime
    dff['dtimes'] = (1e6 * dff['dtimes']).astype('datetime64[ns]')

    # Calculate mean spread vector (single value)
    _sspread = float(dff['spread'].mean(axis=0))
    _vector_s = [_sspread]  # Spread vector initialization

    # Compute volume relations vector
    _resh = dff[_col_v]  # Extract volume-related columns
    _resh2 = _resh.sum(axis=0)  # Sum across rows
    _resh3 = _resh2.values.reshape(-1, 2)  # Reshape into pairs
    _sume = _resh3.sum(axis=1).reshape(-1, 1)  # Total sum per pair for normalization
    _rell = _resh3 / _sume  # Normalize volumes against totals
    _df_rel = pd.DataFrame(_rell.reshape(1, -1), columns=_c_r)  # Format as DataFrame
    _vector_r = _df_rel.values[0]  # Extract relation vector

    # Compute variations vector
    _df_vr = dff[_cor_v].mean(axis=0)  # Mean of variation columns
    _vector_v = _df_vr.values  # Extract variation vector

    # Convert to lists for concatenation
    l_r = _vector_r.tolist()
    l_v = _vector_v.tolist()
    
    # Combine all vectors into final mathematical representation
    V = np.array(_vector_s + l_r + l_v)
    
    return V, _sspread, l_r, l_v
    
def vektor_line(coin, dicc, last_xtime, nokeys, allkeys, c_head, col_v, c_r, cor_v, xtaim, minutos, client):
    V0, spr0, lr0, lv0 = get_vekctor(dicc, last_xtime, c_head, col_v, c_r, cor_v)

    curr_price = dicc['klines'][last_xtime][0] ### actual open
    stcker = client.futures_symbol_ticker(symbol=f'{coin}USDT')
    price = float(stcker['price'])
    diff0 = 100*(price-curr_price)/curr_price


    RR, VV, DD = [], [], []
    for i, key_ in enumerate(nokeys[1:]): ### [1, 2, 3, ..., 9)
        _, rr, vv = dicc['avg'][key_]
        RR += rr
        VV += vv

    for i in range(len(allkeys)-1):
        key_0 = allkeys[i]
        key_1 = allkeys[i+1]
        p_0 = dicc['klines'][key_0][0] ### initial
        p_1 = dicc['klines'][key_1][0] ### final
        diff_x = 100*(p_1-p_0)/p_0
        DD.append(diff_x)

    RR += lr0
    VV += lv0
    DD.append(diff0)

    L0 = [xtaim, coin, minutos, spr0]
    L1 = RR + VV
    # L1 = [spr0] + RR + VV ### soon
    L2 = L0 + DD + L1
    arr_l = np.array(L2, dtype=str)
    arr_v = np.array(L1)
    s_cadena = ",".join(arr_l)
    return arr_v, s_cadena

class vannamei():

    def __white__(params):
        
        auth = tweepy.OAuthHandler(apikey,apisecrets)
        # auth.set_access_token(accesstoken,accesssecret)
        # api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)
        api = tweepy.API(auth)
        # client_k = Market(url='https://api-futures.kucoin.com')
        clientt = ApolloxC()
        
        # try:
        #     dat = 'sl.--'
        #     dbx = dropbox.Dropbox(dat)
        #     dbx.users_get_current_account()
        #     # print(dbx.users_get_current_account())
         
        #     dropbox_user_name = dbx.users_get_current_account().name.display_name
        #     # dropbox_email =      dbx.users_get_current_account().email
         
        #     print("----------------------------------------------")
        #     print(f'AutenticaciÃ³n Dropbox: \nPropiedad de: {dropbox_user_name}')
        #     print("----------------------------------------------")
         
        # except dropbox.auth.AuthError as err:
        #     print(err)
        
        client = params['client'] ### BINANCE CLIENT
        W = params['W']
        coin = params['coin']
        book_pair = params['pair'] ### USD_PERP / USDT
        
        ### scaler and model
        loaded_model = params['model']
        scaler = params['scaler']
        dex = params['exchange']
        
        # clientf = params['clientf']
        # bot = params['bot']
        # lvg = params['lvg']
        
        ### baibit
        api_key_b = ""
        api_secret_b = ""
        session = usdt_perpetual.HTTP(
            endpoint="https://api.bybit.com",
            api_key=api_key_b,
            api_secret=api_secret_b
        )
        
        lvg = 4 ### leverage | CAUTION!!!
        
        coins_json = get_coins_info()
        
        slpwdefr = glob('vector*/*')
        
        CCC = ['AVAX', 'FTM', 'FIL', 'MANA', 'EOS', 'LINK', 'UNI', 
               'BNB', 'NEAR', 'XRP', 'MATIC', 'DOT', 'BTC', 'EGLD', 
               'LUNA', 'GALA', 'ETC', 'THETA', 'ETH', 'ATOM', 'BCH', 
               'DOGE', 'SOL', 'XLM', 'SAND', 'TRX', 'LTC', 'ADA',
               'SC', 'YFI', 'APE', 'GMT', 'KLAY', 'FTT', 'XEM']
        
        lbggh = 20 ### leverage | 20x top w/ BUSD | +25x with USDT
        
        lvrg = {}
        for ccc in CCC:
            lvrg[ccc] = lbggh ### leverage
        
        stablepair = { 
                       ### BUSD
                    #   'BTC':'BUSD', 'ETH':'BUSD', 'BNB':'BUSD', 'DOGE':'BUSD',
                    #   'SOL':'BUSD', 'XRP':'BUSD', 'ADA':'BUSD', 'FTT' :'BUSD',
                      
                       ### USDT
                      'BTC':'USDT', 'ETH':'USDT', 'BNB':'USDT', 'DOGE':'USDT',
                      'SOL':'USDT', 'XRP':'USDT', 'ADA':'USDT', 'FTT' :'USDT',
                       
                       ### USDT
                      'MANA':'USDT', 'THETA':'USDT', 'XLM' :'USDT', 'BCH'  :'USDT',
                      'EGLD':'USDT', 'SAND' :'USDT', 'GALA':'USDT', 'MATIC':'USDT',
                      'KLAY':'USDT', 'TRX':  'USDT', 'LTC': 'USDT', 'FIL':  'USDT',
                      'EOS': 'USDT', 'LINK': 'USDT', 'UNI': 'USDT', 'DOT':  'USDT',
                      'ETC': 'USDT', 'ATOM': 'USDT', 'LUNA':'USDT',
                      
                       ### USDT || USD$
                      'SC':  'USDT', 'YFI':'USDT', 'APE': 'USDT', 'GMT':'USDT',
                      'KLAY':'USDT', 'XEM':'USDT', 'AVAX':'USDT', 'FTM':'USDT',
                      'NEAR':'USDT', 
                     }
        
        
        stable = 'USDT'
        pair = f"{coin}{stable}"
        
        # lbuy, lsell = session.my_position(symbol=f'{coin}{stable}')['result']
        dicc, response = bybit_request(sessionn=session,
                                        method='get',
                                        path='/contract/v3/private/position/list',
                                        symbol='DOTUSDT'
                                       )
        # lbuy, lsell = dicc['result']['list']
        # ng1, ng2 =  np.array([lbuy['leverage'], lsell['leverage']], dtype=np.float64)
        # # ng1, ng2 =  lbuy['leverage'], lsell['leverage']
        
        # if ng1!=lvg and ng2!=lvg:
        #     session.set_leverage(symbol=f'{coin}{stable}', # symbol
        #                          buy_leverage=lvg,  ### long leverage
        #                          sell_leverage=lvg) ### short leverage
        
        os.makedirs(f"coin{coin}", exist_ok=True)
        os.makedirs(f"vectors", exist_ok=True)

        diff = datetime.datetime.today() - datetime.timedelta(hours=5)
        now = diff.strftime("%d_%m_%Y_%H_%M_%S")
        
        nFile1 = f"coin{coin}/db4_{coin}_{now}_temp.csv" ### order book dataset | my life | temporal dataset
        # nFile1_x = f"coin{coin}/db4_{coin}_{now}.csv" ### order book dataset | my life | nFile2 | complete dataset
        print(nFile1) ### temporal file
        arch1 = open(nFile1, "w") ### temporal dataset
        
        # arch.write("dtimes|volb_10|vola_10|bk_10|ak_10|volb_500|vola_500|bk_500|ak_500\n") ### 23-09-2021
        # arch.write("dtimes|volb_10|vola_10|volb_250|vola_250\n") ### 2 | 23-09-2021
        # arch1.write("dtimes|volb_10|vola_10|volb_25|vola_25|volb_50|vola_50|volb_100|vola_100|volb_250|vola_250\n") ### 5 | 11-11-2021
        # arch2.write("dtimes|volb_5|vola_5|volb_10|vola_10|volb_15|vola_15|volb_25|vola_25|volb_50|vola_50|volb_100|vola_100|volb_250|vola_250\n") ### 7 | 14-04-2022 | original
        
        levels = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250] ### | 13 | 18-06-2022 ### current
        headder = ""
        for lvl in levels:
            minic = f"volb_{lvl}|vola_{lvl}|vb_{lvl}|va_{lvl}|"
            headder+=minic
        headder = headder[:-1]
        # headerr = f"dtimes|{headder}\n" ### 18-06-2022
        # headerr = f"dtimes|{headder}|b250|a250\n" ### 17-07-2022 03:27:33
        # headerr = f"dtimes|{headder}\n" ### 17-07-2022 06:38:24
        headerr = f"dtimes|spread|{headder}\n" ### 05-12-2022 22:51:59
        
        arch1.write(headerr) ### 13 | temporal
        # arch1x.write(headerr) ### 13 | complete
        
        arch1.close() ### current
        # arch1x.close() ### current
        
        ### vector file
        name_v = f"vectors/v2{coin}_{now}.csv" ### binance
        # name_v = f"vectors2/v2{coin}_{now}.csv"
        # name_v = f"tensors/v2{coin}_{now}.csv"
        registro_v = open(name_v, "w")
        registro_v.close()
        print(name_v, '\n')
        
        operation, short, longg = 0, 0, 0 ### state of the operation gg
        CC = []
        tok = 0
        
        ### colors and flags variables
        dates = []
        duett, duett2, duett3 = [], [], [] ### lista larga de predicciones
        cuett, cuett2, cuett3 = [], [], []
        
        peperepe = []
        
        pG,pR = 0,0
        cG,cR = 0,0
        cGL,cRL = [], []
        op = 0
        
        lim_0,lim_1 = [0.5]*2
        
        ###
        
        zec_0 = 5.05 ### RECENT TRADES
        zec_9 = 5.05 ### PREDS
        zec = 5.05 ### market lecture | orders
        
        first = 0
        
        pps = [0.06, 0.06, 0.06, 0.06, 0.08, 0.08, 0.08, 0.08, 0.08]
        vals = {"qty":1, 5:0, 10:0, 15:0, 20:0, 25:0, 30:0} ### checkpoint
        
        apertura = [] ### ENTRY PRICE LIST
        precios = [] ### precios xd
        sls = [0]
        sls2 = [0]
        
        flag = ''
        na_str = ''
        
        mins_dict = set() ### registro de minutos procesados
        
        multi, plus= 11, 1
        # kw = 5
        
        wall = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(minutes=multi)
        wall2 = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(minutes=multi)
        wall0 = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(minutes=multi)
        wall9 = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(minutes=multi)
        wall975 = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(minutes=multi)
        wawo = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(minutes=multi)
        
        wol = datetime.datetime.today() + datetime.timedelta(seconds=900) ### ORDER BOOK FILE
        
        # arch = open(nFile, "a+")
        d_general = {"tree":{}, 'minus':{}, 'plus':{}, 'all':[], 'all2':[], 'conteo':0, 'tts':[], 'ms_utc':[], 'ms_local':[]} ### DICCIONARIO ORDER BOOK ### DICCIONARIO ORDER BOOK
        
        check = ''
        check_sl = ''
        
        encendido = False
        tolrem = 20
        
        while 1:
            
            try:
                
                tok+=1
                
                # tt, linea = order_book_file_(coin=coin, bp=book_pair, filename=nFile, client=client) ### uptdate file | resize order book
                tt, _, exchange = order_book_file(coin=coin, 
                                        bp=book_pair, 
                                        filename1=nFile1, ### temporal dataset
                                        # filename1=nFile1, ### temporal dataset
                                        # filename2=nFile1x, ### complete dataset
                                        filename2='', ### complete dataset
                                        client=client,
                                        # client2 = client_k,
                                        dex=dex
                                        ) ### uptdate file | resize order book
                
                #
                ##
                ### inferences | data
                noa = datetime.datetime.today() - datetime.timedelta(hours=5)
                if noa>wall9:
                    
                    ### current
                    df_3, dfn, spreadd  = ds_heikin_zz(failneim=nFile1, ### temporal file | process into to get real file
                                              client=clientt,    ### APOLLOX CLIENT API
                                              dex=dex,
                                              client2=client    ### BINANCE CLIENT API
                                              ) ### lectura del archivo en tiempo real | polars reading | temp dset
                    
                    y_pred_s = [] ### lista temporal de predicciones
                    
                    ### get x set
                    dtime = tt
                    minuto = 9
                    V, V_str, minz = inference_456X(dataframe=df_3,  ### merge | ohlc o-klines + vol + vor
                                                    dataframe2=dfn,  ### ohlc o-klines
                                                    minuto=minuto    ### minute
                                                    )
                    
                    name_w = name_v.replace("_", f"_{exchange}_{minuto}j_", 1) ### assign minute & exchange
                    registro_v = open(name_w, "a+")
                    registro_v.write(f"{tt},{coin},{minuto},{spreadd},{V_str}\n")
                    registro_v.close()
                    
                    if tok%850==0:
                        shutil.copy(name_w, name_w.replace(".", "_X.")) # prds
                    
                    #  BLOCK TO VECTORS
                    ##
                    ### the wall of time RISES
                    wall9 = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(seconds=zec_9)
                    ### the wall of time FALLS
                    ##
                    #
                    
                #     #
                #     ## AI inference section
                #     vector_inf = scaler.transform([V[-minuto*52:]]) ### scaled vector
                #     preds = loaded_model.predict(vector_inf, verbose=0) ### prediction vector
                #     y_pred = preds.max(axis=1)[0] ### max color probability -> float
                #     c_pred = preds.argmax(axis=1)[0] ### color position -> int
                #     ## AI inference section finish
                #     #
                    
                #     if verina(y_pred)==0:      ### primera etapa | se llena no se modifica
                #         y_pred_s.append(y_pred)
                #         duett.append(y_pred)
                #         cuett.append(c_pred)
                #         peperepe.append([dtime, y_pred])
                    
                #     if len(duett)>0: ### filter inference values | duett2 & cuett2
                #         val_m, col_m = duett[-1], cuett[-1]
                #         if val_m<0.850 and col_m!=2:col_m = 2      ### 0.666 -> 2 (GRAY) NOPE | 0.850 -> 2 (GRAY)
                #         duett2.append(val_m);cuett2.append(col_m); ### segunda etapa | no se modifica la probabilidad - se modifica el color
                    
                #     kw = 2 ### min number of samples
                #     if len(duett2)>=kw: ### new prediction list | duett3 & cuett3 | se promedia | N-1
                #         date = dtime
                #         pred = np.mean(duett2[-kw:]) ### MEDIA NORMAL
                #         # pred = get_pond_m(duett2[-kw:]) ### MEDIA PONDERDADA
                #         duett3.append(pred) ### nueva serie
                        
                #         if duett2[-1]>duett2[-2]:cuett3.append(cuett[-1])
                #         else:cuett3.append(cuett[-2])
                    
                #     ### the wall of time RISES
                #     wall9 = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(seconds=zec_9)
                #     ### the wall of time FALLS
                
                # #
                # ##
                # ### BOT
                # nao = datetime.datetime.today() - datetime.timedelta(hours=5)
                
                # K_z = 10 ### blocks
                
                # if nao>wall and len(duett3)>=K_z: ### i didnt know i was lost... until now
                    
                #     ### Decision box
                #     ### Prediction processing | "automate human activity" | resampleo de valores a X segundos (ej: 10s)
                #     ### INICIO DEL PROCESAMIENTO
                    
                #     arr_d = np.array(duett3[-K_z:]) ### last k probabilities
                #     arr_c = np.array(cuett3[-K_z:]) ### last k colors
                #     c1, c2, c3 = arr_c[-3:]
                    
                #     prob_1 = 0.9825
                    
                #     yesOnoD = (arr_d>=prob_1).mean() ### last k probabilities mean
                #     yesOnoC = arr_c.mean() ### last k colors mean
                    
                #     #
                #     ##
                #     ### begin
                #     ### OPEN/CLOSE OPERATIONS BLOCK
                    
                #     if (yesOnoD==1 ### minimun n confirmations
                #         and yesOnoC==1 ### last colors | GREEN
                #         and flag=='' ### no operation alive
                #         and (check!='up') ### no long alive
                #         and (check_sl!='g') ### previous stop loss
                #         ): ### OPEN LONG
                        
                #         ### long opening (first long)
                #         side, wall, apertura, operation, longg, QTYs, dec, tweet = \
                #         open_long(coin=coin, ypredd=pred, apertura=apertura, client=client, session = \
                #         session, lvg=lvg, stable=stable, seconds=zec, api=api, json_dec=coins_json, on=encendido, value=tolrem)
                        
                #         if len(QTYs)>0:
                #             vals['qty'] = QTYs[0]
                        
                #         op+=1
                #         first+=1
                #         wall = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(seconds=zec)
                #         wallz = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(seconds=zec)
                        
                #         check = 'up'
                #         check_sl = ''
                #         flag='green' ### operation alive: green
                        
                #         if len(apertura)>0:
                #             precios.append(apertura[0])
                    
                #     elif (yesOnoD==1 ### last seconds
                #         and yesOnoC==0 ### last colors | RED
                #         and flag=='' ### no operation alive
                #         and (check!='down') ### no short alive
                #         and (check_sl!='r') ### previous stop loss
                #         ): ### OPEN SHORT
                        
                #         ### short opening (first short)
                #         side, wall, apertura, operation, short, QTYs, dec, tweet = \
                #         open_short(coin=coin, ypredd=pred, apertura=apertura, client=client, session = \
                #         session, lvg=lvg, stable=stable, seconds=zec, api=api, json_dec=coins_json, on=encendido, value=tolrem)
                        
                #         if len(QTYs)>0:
                #             vals['qty'] = QTYs[0]
                        
                #         op+=1
                #         first+=1
                #         wall = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(seconds=zec)
                #         wallz = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(seconds=zec)
                        
                #         check = 'down'
                #         check_sl = ''
                #         flag='red' ### operation alive: red
                        
                #         if len(apertura)>0:
                #             precios.append(apertura[0])
                    
                #     elif (flag=='green' 
                #         and ((c1!=1 and c2!=1 and c3!=1))
                #         ): ### CLOSE LONG | when I get both red or gray colors
                #         #  and (m2<0.70 and m3<0.70 and m4<0.70)): ### CLOSE LONG
                        
                #         if len(QTYs)>0:
                #             amount = QTYs[0]
                #         else:
                #             amount = 0
                            
                #         c_op = current_position(session, coin, stable='USDT') ### HAY O NO HAY!?
                        
                #         hay = True
                #         if len(c_op)==0:
                #             hay = False
                        
                #         ### close long
                #         wall, side, operation, longg, apertura, QTYs = \
                #         close_long(coin=coin, client=client, session=session, apertura=apertura, QTYs=QTYs, hist=precios,
                #         stable=stable, seconds=zec, lv=lvg, amount=amount, yyprd=pred, tweet=tweet, api=api, json_dec=coins_json, on=encendido, value=tolrem, sl_q=hay)
                        
                #         flag='' ### new operation alive: none
                #         check = ''
                #         precios = []
                #         sls = [0]
                #         sls2 = [0]
                    
                #     elif (flag=='red'
                #         and (c1!=0 and c2!=0 and c3!=0)
                #         ): ### CLOSE SHORT | when I get both green or gray colors
                #         #  and (m2>0.30 and m3>0.30 and m4>0.30)): ### CLOSE SHORT
                        
                #         if len(QTYs)>0:
                #             amount = QTYs[0]
                #         else:
                #             amount = 0
                            
                #         c_op = current_position(session, coin, stable='USDT') ### HAY O NO HAY!?
                        
                #         hay = True
                #         if len(c_op)==0:
                #             hay = False
                        
                #         ### close short
                #         wall, side, operation, short, apertura, QTYs = \
                #         close_short(coin=coin, client=client, session=session, apertura=apertura, QTYs=QTYs, hist=precios,
                #         stable=stable, seconds=zec, lv=lvg, amount=amount, yyprd=pred, tweet=tweet, api=api, json_dec=coins_json, on=encendido, value=tolrem, sl_q=hay)
                        
                #         flag='' ### new operation alive: none
                #         check = ''
                #         precios = []
                #         sls = [0]
                #         sls2 = [0]
                    
                #     else:
                        
                #         if flag=='green':
                #             symbolInfo = session.latest_information_for_symbol(symbol=f'{coin}{stable}')
                #             coinInfo = symbolInfo['result'][0]
                #             # arr_BA = np.array([coinInfo['bid_price'], coinInfo['ask_price']], dtype=np.float64)
                #             # precioxd = arr_BA[0]
                #             precioxd = float(coinInfo['bid_price'])
                #             precios.append(precioxd)
                        
                #         elif flag=='red':
                #             symbolInfo = session.latest_information_for_symbol(symbol=f'{coin}{stable}')
                #             coinInfo = symbolInfo['result'][0]
                #             # arr_BA = np.array([coinInfo['bid_price'], coinInfo['ask_price']], dtype=np.float64)
                #             # precioxd = arr_BA[1]
                #             precioxd = float(coinInfo['ask_price'])
                #             precios.append(precioxd)
                            
                        
                #         wall = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(seconds=zec) ### MARKET INFERENCE SECONDS
                
                # ### OPEN/CLOSE OPERATIONS BLOCK
                # ##
                # # end
                
                # #
                # ## COMMENT TO TEST
                # ## UNCOMMENT TO DEPLOY
                # ### begin
                # ### STOP LOSS MONITOR BLOCK
                # nao = datetime.datetime.today() - datetime.timedelta(hours=5)
                # if (nao>wall975    ### time wall
                #     and flag!=''   ### an operation alive
                #     and encendido  ### botcito funcionando
                #     ):
                    
                #     c_op = current_position(session, coin, stable='USDT') ### HAY O NO HAY!?
                    
                #     if len(QTYs)>0:
                #         amount = QTYs[0]
                #     else:
                #         amount = 0
                    
                #     if flag=='green':
                #         diff_123 = 100*(precios[-1] - apertura[0])/apertura[0] ### current profit state
                #         sls.append(diff_123)
                #     elif flag=='red':
                #         diff_123 = -100*(precios[-1] - apertura[0])/apertura[0] ### current profit state
                #         sls.append(diff_123)
                    
                #     STAP = -0.3*lvg
                #     varr = 0.375
                #     MAX_sl = max(sls) ### max profit reached | current %
                #     if MAX_sl>=(varr*lvg):STAP = MAX_sl*0.454545 ### set new stop loss | 33% lower current profit reached | STAP
                #     if STAP not in sls2 and max(sls2)<STAP: ### new limit line
                #         sls2.append(STAP) ### add to make future comparisons
                #         if len(c_op)>0:
                #             cero = update_sl(session, coin, new_sl=STAP, flag=flag, stable='USDT') ### update sl in the exchange position
                    
                #     ### to close in the book the closed order by the exchange
                #     if (flag=='green' ### orden registrada
                #         and len(c_op)==0 ### cero ordenes abiertas | cierra la position formalmente en el registro
                #         ): ### CLOSE GHOST LONG OPERATION
                        
                #         side = 'close'
                #         ### close long
                #         wall, side, operation, longg, apertura, QTYs = \
                #         close_long(coin=coin, client=client, session=session, apertura=apertura, QTYs=QTYs, hist=precios,
                #         stable=stable, seconds=zec, lv=lvg, amount=amount, yyprd=pred, tweet=tweet, api=api, json_dec=coins_json, on=encendido, value=tolrem, sl_q=False)
                        
                #         flag='' ### new operation alive: none
                #         check = ''
                #         check_sl = 'g'
                #         precios = []
                #         sls = [0]
                #         sls2 = [0]
                        
                #     elif (flag=='red' ### orden registrada
                #         and len(c_op)==0 ### cero ordenes abiertas | cierra la position formalmente en el registro
                #         ): ### CLOSE GHOST SHORT OPERATION
                        
                #         side = 'close'
                #         ### close short
                #         wall, side, operation, short, apertura, QTYs = \
                #         close_short(coin=coin, client=client, session=session, apertura=apertura, QTYs=QTYs, hist=precios,
                #         stable=stable, seconds=zec, lv=lvg, amount=amount, yyprd=pred, tweet=tweet, api=api, json_dec=coins_json, on=encendido, value=tolrem, sl_q=False)
                        
                #         flag='' ### new operation alive: none
                #         check = ''
                #         check_sl = 'r'
                #         precios = []
                #         sls = [0]
                #         sls2 = [0]
                        
                #     else:
                #         ### keep OPERATION ALIVE
                #         wall975 = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(seconds=zec) ###  STOP LOSS SECONDS
                #         wallz = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(seconds=zec)
                
                # ### STOP LOSS MONITOR BLOCK
                # ### end
                # ##
                # #
                
                
                
                #
                ##
                ### FILE LOGISTIC: save a copy of the stored data
                
                noa = datetime.datetime.today()
                if noa>wol: ### temporal dataset | 900 segundos | 15 minutos
                    
                    shutil.copy(nFile1, nFile1.replace(".", "_Y.")) # temporal dataset
                    shutil.copy(name_v, name_v.replace(".", "_X.")) # vctrs
                    
                    A = open(nFile1, "r") # temp
                    lineas = A.readlines()[1:] ### avoid to get the first line
                    cadenota = "".join(lineas[-6250:]) ### string to place new file
                    A.close()
                    
                    # diff_x = datetime.datetime.today() - datetime.timedelta(hours=5)
                    # now_x = diff_x.strftime("%d_%m_%Y_%H_%M_%S")
                    # nFile1 = f"coin{coin}/db4_{coin}_{now_x}.csv" ### order book dataset | my life | temporal dataset | new name every x time
                    
                    arch1 = open(nFile1, "w") ### current | recreate the object
                    arch1.write(headerr) ### 13 values | current
                    arch1.write(cadenota) ### last 6000 lines from last file from last iteration
                    arch1.close() ### current
                    
                    wol = datetime.datetime.today() + datetime.timedelta(seconds=840) ### ORDER BOOK FILE | 14 minutes
            
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                error_l = exc_tb.tb_lineno
                cadena_error = str(exc_type) + " => " + str(exc_obj)
                
                if ('onnection aborted' not in cadena_error
                    and 'onnection reset' not in cadena_error
                    and '.com' not in cadena_error
                    and 'APIError(code=0): Invalid JSON error message from Binance' not in cadena_error
                    ): ### no ignorar iteracion
                    
                    diff = datetime.datetime.today() - datetime.timedelta(hours=5)
                    
                    ### screen message
                    # print(cadena_error)
                    # print("zzz |", diff.ctime())
                    # print()
                    
                    if 'OSError' in cadena_error:
                        
                        c_op = current_position(session, coin, stable='USDT')
                        
                        ### CLOSE CURRENT OPERATION BLOCK
                        if (flag=='green' 
                            and len(c_op)>0 
                            and encendido): ### CLOSE LONG
                            
                            if len(QTYs)>0:
                                # uP = client.futures_position_information(symbol=f'{coin}{stable}')[0] # time*
                                # uP = get_position_json(pos_info)
                                # amount = abs(float(uP['positionAmt'])) ### qty in current position
                                amount = QTYs[0]
                            else:
                                amount = 0
                            
                            ### close long | sl_q doesnt close operation bc there is no operation alive
                            wall, side, operation, longg, apertura, QTYs = \
                            close_long(coin=coin, client=client, session=session, apertura=apertura, QTYs=QTYs, hist=precios,
                            stable=stable, seconds=zec, lv=lvg, amount=amount, yyprd=pred, tweet=tweet, api=api, json_dec=coins_json, on=encendido)
                            flag='' ### new operation alive: none
                        
                        elif (flag=='red' 
                            and len(c_op)>0 
                            and encendido): ### CLOSE SHORT
                            
                            if len(QTYs)>0:
                                # uP = client.futures_position_information(symbol=f'{coin}{stable}')[0] # time*
                                # uP = get_position_json(pos_info)
                                # amount = abs(float(uP['positionAmt'])) ### qty in current position
                                amount = QTYs[0]
                            else:
                                amount = 0
                            
                            ### close short | sl_q doesnt close operation bc there is no operation alive
                            wall, side, operation, short, apertura, QTYs = \
                            close_short(coin=coin, client=client, session=session, apertura=apertura, QTYs=QTYs, hist=precios,
                            stable=stable, seconds=zec, lv=lvg, amount=amount, yyprd=pred, tweet=tweet, api=api, json_dec=coins_json, on=encendido)
                            flag='' ### new operation alive: none
                        ### END BLOCK
                        
                        send_activity2(f"{coin} session: Dead") ### enviar reporte de cierre
                        print(cadena_error)
                        print("zzz |", diff.ctime())
                        print(f"{coin} session: Dead")
                        break
                    
                    else:
                        send_activity2(f"Revisar {coin} | {cadena_error} | linea {error_l}") ### enviar reporte de error
                        print(cadena_error)
                        print("zzz |", diff.ctime())
                sleep(0.1)
                
    def __lotus__(params):
        
        auth = tweepy.OAuthHandler(apikey, apisecrets)
        api = tweepy.API(auth)
        # client_k = Market(url='https://api-futures.kucoin.com')
        clientt = ApolloxC()
        
        client = params['client'] ### BINANCE CLIENT
        W = params['W']
        coin = params['coin']
        book_pair = params['pair'] ### USD_PERP / USDT
        
        ### scaler and model
        loaded_model = params['model']
        scaler = params['scaler']
        dex = params['exchange']
        
        ### baibit
        api_key_b = ""
        api_secret_b = ""
        session = usdt_perpetual.HTTP(
            endpoint="https://api.bybit.com",
            api_key=api_key_b,
            api_secret=api_secret_b
        )
        
        lvg = 4 ### leverage | CAUTION!!!
        
        coins_json = get_coins_info()
        
        # slpwdefr = glob('vector*/*');slpwdefr = glob('vector*/*');slpwdefr = glob('vector*/*')
        # slpwdefr = glob('vector*/*');slpwdefr = glob('vector*/*');slpwdefr = glob('vector*/*')
        # slpwdefr = glob('vector*/*');slpwdefr = glob('vector*/*');slpwdefr = glob('vector*/*')
        
        for i in range(100):
            slpwdefr = glob('vector*/*')
            if len(slpwdefr)>0:
                print(i)
                break
        
        stable = 'USDT'
        pair = f"{coin}{stable}"
        
        dicc, response = bybit_request(sessionn=session,
                                        method='get',
                                        path='/contract/v3/private/position/list',
                                        symbol='DOTUSDT'
                                       )
        # lbuy, lsell = dicc['result']['list']
        # ng1, ng2 =  np.array([lbuy['leverage'], lsell['leverage']], dtype=np.float64)
        
        # if ng1!=lvg and ng2!=lvg:
        #     session.set_leverage(symbol=f'{coin}{stable}', # symbol
        #                          buy_leverage=lvg,  ### long leverage
        #                          sell_leverage=lvg) ### short leverage
        
        os.makedirs(f"coin{coin}", exist_ok=True)
        os.makedirs(f"vectors", exist_ok=True)

        diff = datetime.datetime.today() - datetime.timedelta(hours=5)
        now = diff.strftime("%d_%m_%Y_%H_%M_%S")
        
        levels = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250] ### | 13 | 18-06-2022 ### current
        headder = ""
        for lvl in levels:
            minic = f"volb_{lvl}|vola_{lvl}|vb_{lvl}|va_{lvl}|"
            headder+=minic
        headder = headder[:-1]
        headerr = f"dtimes|spread|{headder}\n" ### 17-07-2022 06:38:24
        
        ### vector file
        name_v = f"vectors/v2{coin}_{now}.csv" ### vectors x 
        registro_v = open(name_v, "w")
        registro_v.close()
        print(name_v, '\n')
        
        operation, short, longg = 0, 0, 0 ### state of the operation gg
        CC = []
        tok = 0
        
        ### colors and flags variables
        dates = []
        duett, duett2, duett3 = [], [], [] ### lista larga de predicciones
        cuett, cuett2, cuett3 = [], [], []
        
        pepereperepe = []
        
        pG,pR = 0,0
        cG,cR = 0,0
        cGL,cRL = [], []
        op = 0
        
        lim_0, lim_1 = [0.5]*2
        
        ###
        zec_9, zec = [4.95]*2  ### PREDS & market lecture | orders
        
        first = 0
        
        apertura = [] ### ENTRY PRICE LIST
        precios = [] ### precios xd
        sls = [0]
        sls2 = [0]
        
        flag = ''
        na_str = ''
        
        mins_dict = set() ### registro de minutos procesados
        
        multi, plus= 11, 1
        # kw = 5
        
        wall = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(minutes=multi)
        wall9 = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(minutes=multi)
        wall0 = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(minutes=1440) ### 24 hours
        
        wol = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(seconds=810) ### ORDER BOOK FILE
        
        d_general = {"tree":{}, 'minus':{}, 'plus':{}, 'all':[], 'all2':[], 'conteo':0, 'tts':[], 'ms_utc':[], 'ms_local':[]} ### DICCIONARIO ORDER BOOK ### DICCIONARIO ORDER BOOK
        D = {}
        check, check_sl = '', ''
        
        encendido = False
        tolrem = 20
        
        levels = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250] ### 13 values
        headder = ""
        vol_c = ''
        vor_c = ''
        r_c = ''
        for lev in levels:
            headder += f"volb_{lev}|vola_{lev}|vb_{lev}|va_{lev}|" ### header
            vol_c+= f'volb_{lev}|vola_{lev}|' ### volume total
            r_c+= f'rb_{lev}|ra_{lev}|'       ### volume relation
            vor_c+= f'vb_{lev}|va_{lev}|'     ### variation prices
        
        col_v = vol_c[:-1].split("|")
        c_r = r_c[:-1].split("|")
        cor_v = vor_c[:-1].split("|")
        headder = headder[:-1]
        headerr = f"dtimes|spread|{headder}" ### 05-12-2022 22:51:59
        c_head = headerr.split("|")
        
        D = {'data':{},
             'avg':{},
            #  'keys':[],
             'keys_str':[],
             'klines':{}
             }
        
        minutos = 9 ### temporalidad
        minuto = 9 ### temporalidad
        
        while 1:
            
            try:
                
                tok+=1
                
                # tt, tt_str, _, exchange = order_book_file(coin=coin, 
                xtime, xtime_str, linea, exchange = order_book_file(coin=coin, 
                                        bp=book_pair, 
                                        client=client,
                                        dex=dex
                                        ) ### uptdate file | resize order book
                                        
                light = 0
                if xtime_str not in D['data']:
                    D['data'][xtime_str] = [linea]
                    D['keys_str'].append(xtime_str)
                    light = 1
                    
                    stcker = client.futures_symbol_ticker(symbol='BTCUSDT')
                    price = float(stcker['price'])
                    D['klines'][xtime_str] = [price]
                    
                else:
                    D['data'][xtime_str].append(linea)
                
                nokeys = D['keys_str'][-minutos-1:-1] ### length: n minutos (9)
                allkeys = D['keys_str'][-minutos:]
                last_xtime = D['keys_str'][-1]
                avgs = D['avg']
                for nk in nokeys: ### length: minutos
                    if nk not in avgs: ### each minute
                        V, espred, lr, lv = get_vekctor(D, nk, c_head, col_v, c_r, cor_v) ### 
                        D['avg'][nk] = [espred, lr, lv] ### [spread, lista_relation, lista_variation]
                
                quis = D['data']
                if len(quis)>=30:
                    lista_q = list(quis.keys())
                    for l in lista_q[:-15]:
                        del D['data'][l]
                
                #
                ##
                ### inferences | data
                noa = datetime.datetime.today() - datetime.timedelta(hours=5)
                if noa>wall9:
                    
                    ### current
                    arr_v, s_cadena = vektor_line(coin, D, last_xtime, nokeys, allkeys, c_head, col_v, c_r, cor_v, xtime, minutos, client)
                    
                    # y_pred_s = [] ### lista temporal de predicciones
                    
                    name_w = name_v.replace("_", f"_{exchange}_{minuto}j_", 1) ### assign minute & exchange
                    registro_v = open(name_w, "a+")
                    # registro_v.write(f"{tt},{coin},{minuto},{spreadd},{V_str}\n")
                    registro_v.write(f"{s_cadena}\n")
                    registro_v.close()
                    
                    if tok%1000==0:
                        shutil.copy(name_w, name_w.replace(".", "_X.")) # prds
                    
                    #  BLOCK TO VECTORS
                    ##
                    ### the wall of time RISES
                    wall9 = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(seconds=zec_9)
                    ### the wall of time FALLS
                    ##
                    #
                
                #
                ##
                ### FILE LOGISTIC: save a copy of the stored data || just need to save the info about the last minute; not all minutes before
                noa = datetime.datetime.today() - datetime.timedelta(hours=5)
                if noa>wol: ### temporal dataset | 810 segundos | 14 minutos
                    
                    # shutil.copy(nFile1, nFile1.replace(".", "_Y.")) # temporal dataset
                    shutil.copy(name_v, name_v.replace(".", "_X.")) # vctrs
                    
                    wol = datetime.datetime.today() - datetime.timedelta(hours=5) + datetime.timedelta(seconds=810) ### ORDER BOOK FILE | 14 minutes
            
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                error_l = exc_tb.tb_lineno
                cadena_error = str(exc_type) + " => " + str(exc_obj)
                
                if ('onnection aborted' not in cadena_error
                    and 'onnection reset' not in cadena_error
                    and '.com' not in cadena_error
                    and 'APIError(code=0): Invalid JSON error message from Binance' not in cadena_error
                    ): ### no ignorar iteracion
                    
                    diff = datetime.datetime.today() - datetime.timedelta(hours=5)
                    
                    if 'OSError' in cadena_error:
                        
                        send_activity2(f"{coin} session: Dead") ### enviar reporte de cierre
                        print(cadena_error)
                        print("zzz |", diff.ctime())
                        print(f"{coin} session: Dead")
                        break
                    
                    else:
                        send_activity2(f"Revisar {coin} | {cadena_error} | linea {error_l}") ### enviar reporte de error
                        print(cadena_error)
                        print("zzz |", diff.ctime())
                sleep(0.1)


def scaler_data(filename):
    dscaler = pl.read_csv(filename.replace("model_", 'scaler_').replace('h5', 'csv'), sep="|")
    # arr_sc = np.array(dscaler)
    scaler = StandardScaler().fit(dscaler.to_numpy())
    model = load_model(filename)
    # div = filename.replace("_2.", ".").split("_")[-1].split(".")[0]
    div = filename.split("_")[-1].split(".")[0]
    # print(div)
    nm = int(div.replace('m', ""))
    div_ = filename.split("futures_")[-1].split(".")[0]
    return model, scaler, nm, div_

def get_utilsss():
    ms3, scs3, tfs3, colss3 = [], [], [], []
    ### ai975 like the first love | ai976 maybe, the love of my life... | ai978 motomami | ai984 n de ni se te ocurra ni pensarlo | ai985 este si final final | ai986 just the two of us
    ### ai988 Tunometecabrazaramanbiquen't? | ai991 ule nwantiti uleeee | ai993 joder | ai1018  Ainsi bas la vida aaainsiii
    
    # for m in glob(f"models/*ai1030*m.h5"): ### ai993" | "ai1018
    # for m in glob(f"models/*ai993*m.h5") + glob(f"models/*ai1018*m.h5"):
    jjk = glob(f"models/*ai1049*m.h5")
    k_m = np.array([i.split("ai")[-1].split("v")[0] for i in jjk], dtype=np.int_).max()
    
    for m in glob(f"models/*ai{k_m}*m.h5"):
        # print(m)
        m, s, tx, cc = scaler_data(m)
        ms3.append(m)
        scs3.append(s)
        tfs3.append(tx)
        colss3.append(cc)
    return ms3, scs3, tfs3, colss3

# class PerformancePlotCallback(keras.callbacks.Callback):
#     def __init__(self, x_test, y_test, model_name):
#         self.x_test = x_test
#         self.y_test = y_test
#         self.model_name = model_name
    
#     def on_epoch_end(self, epoch, logs={}):
        
#         y_pred = self.model.predict(self.x_test) ### model performance
        
#         fig, ax = plt.subplots(figsize=(8,4))
        
#         plt.plot(y_pred[y_pred>=.5], "^", c='g', markersize=3.5)
#         plt.plot(y_pred[y_pred<.5], "v", c='r', markersize=3.5)
#         plt.plot([0, max([len(y_pred[y_pred>=.5]), len(y_pred[y_pred<.5])])], [0.5, 0.5])
#         plt.yticks([i/20 for i in range(0, 21)])
#         plt.grid()
        
#         plt.title(f'Prediction Visualization Keras Callback - Epoch: {epoch}')
#         plt.savefig('model_train_images/'+self.model_name+"_"+str(epoch))
#         plt.close()

def get_cols(MINUTTOS):
    # vectors
    levels = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250]

    vor_c, r_c = '', ''
    for lev in levels:r_c+= f'rb_{lev}|ra_{lev}|';vor_c+= f'vb_{lev}|va_{lev}|'

    c_r = r_c[:-1].split("|")
    cor_v = vor_c[:-1].split("|")

    rbra = [[i+f"_{j}" for i in c_r] for j in range(1, MINUTTOS+1)]
    vorba = [[i+f"_{j}" for i in cor_v] for j in range(1, MINUTTOS+1)]

    cols0 = ['dtimes', 'coin', 'minuto', 'sumaT', 'suma3', 'lastK']
    cols1 = [f"diff_ha_{i}" for i in range(1, MINUTTOS+1)] + [f"diff_a_{i}" for i in range(1, MINUTTOS+1)]

    lol = []
    for i in rbra+vorba:
        lol += i

    vCol = cols0+cols1+lol
    # len(vCol)
    return vCol
    
def get_cols2(MINUTTOS):
    # vectors
    levels = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250]

    vor_c, r_c = '', ''
    for lev in levels:r_c+= f'rb_{lev}|ra_{lev}|';vor_c+= f'vb_{lev}|va_{lev}|'

    c_r = r_c[:-1].split("|")
    cor_v = vor_c[:-1].split("|")

    rbra = [[i+f"_{j}" for i in c_r] for j in range(1, MINUTTOS+1)]
    vorba = [[i+f"_{j}" for i in cor_v] for j in range(1, MINUTTOS+1)]

    cols0 = ['dtimes', 'coin', 'minuto']
    cols1 = [f"diff_a_{i}" for i in range(1, MINUTTOS+1)]

    lol = []
    for i in rbra+vorba:
        lol += i

    vCol = cols0+cols1+lol
    # len(vCol)
    return vCol

def get_new_columns(aa=0, bb=9):
    # aa, bb = rango_m[0], rango_m[1]+1
    difffs = ['dtimes'] + [f'diff_a_{i}' for i in range(aa, bb)]
    
    levels = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250]
    vor_c = ''
    for lev in levels:vor_c+= f'vb_{lev}|va_{lev}|'
    cor_v = vor_c[:-1].split("|")
    vorba = [[i+f"_{j}" for i in cor_v] for j in range(aa, bb+1)]
    return difffs + np.array(vorba).flatten().tolist()
