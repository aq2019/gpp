import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller



from dataset import PriceDataset

path = "./datacopy/"
gold_symbol = "^xauusd"
symbol_lst_vol = ["^btcusd", "shv",
                  "tlt"] #gold etf and bond etf "gld", "iei", "shy","ief",
symbol_lst = ["^xagusd", "^xptusd", "dxy", #indices
              "vix", "spx"] #gold etf and bond etf "dowi", "nasx", 
bond_etf = ['shy', 'ief', 'tlt'] # 'iei' only has data since 2007 so not included
stock_index = ['dowi', 'spx', 'nasx']

xauusd = pd.read_csv(path+'^xauusd.csv')

def gold_bond():

    
    # xauusd and bond etf
    _merge_temp = xauusd[:-1]
    for symb in bond_etf:
        _temp = pd.read_csv(path+symb+'.csv')[:-1].copy(deep=True)
        _merge_temp = pd.merge(_merge_temp, _temp, on='Time', suffixes=('', '_'+symb), how='inner')
        _merge_temp['Date'] = pd.to_datetime(_merge_temp['Time'])
        _merge_temp.set_index(_merge_temp['Date'])

    fig1 = make_subplots(specs=[[{"secondary_y":True}]])
    fig1.add_trace(
        go.Scatter(x = _merge_temp['Date'], y = _merge_temp['Last'], name='xau'), secondary_y=False,
    )
    fig1.add_trace(
        go.Scatter(x = _merge_temp['Date'], y = _merge_temp['Last_shy'], name='shy (1-3 yrs)'), secondary_y=True,
    )
    fig1.add_trace(
        go.Scatter(x = _merge_temp['Date'], y = _merge_temp['Last_ief'], name='ief (7-10 yrs)'), secondary_y=True,
    )
    fig1.add_trace(
        go.Scatter(x = _merge_temp['Date'], y = _merge_temp['Last_tlt'], name='tlt (20+ yrs)'), secondary_y=True,
    )
    fig1.update_layout(
        title_text="Gold price and bond ETFs"
    )
    # Set x-axis title
    fig1.update_xaxes(title_text="date",
                    rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=2, label="2y", step="year", stepmode='backward'),
                dict(count=5, label="5y", step="year", stepmode='backward'),
                dict(step="all")
            ])
        ))
    # Set y-axes titles
    fig1.update_yaxes(title_text="<b>xauusd</b> ", secondary_y=False)
    fig1.update_yaxes(title_text="<b>bond etf</b>", secondary_y=True)
    return fig1


def gold_stock():
    _merge_stock = xauusd[:-1]
    for symb in stock_index:
        _temp_stock = pd.read_csv(path+symb+'.csv')[:-1].copy(deep=True)
        _merge_stock = pd.merge(_merge_stock, _temp_stock, on='Time', suffixes=('', '_'+symb), how='inner')
        _merge_stock['Date'] = pd.to_datetime(_merge_stock['Time'])
        _merge_stock.set_index(_merge_stock['Date'])

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x = _merge_stock['Date'], y = _merge_stock['Last'], name='xau', 
    ))


    fig2.add_trace(go.Scatter(
        x = _merge_stock['Date'], y = _merge_stock['Last_dowi'], name='dowi', yaxis = 'y2'
    ))

    fig2.add_trace(go.Scatter(
        x = _merge_stock['Date'], y = _merge_stock['Last_spx'], name='spx', yaxis = 'y3'
    ))

    fig2.add_trace(go.Scatter(
        x = _merge_stock['Date'], y = _merge_stock['Last_nasx'], name='nasx',
        yaxis="y4"
    ))


    # Create axis objects
    fig2.update_layout(
        xaxis=dict(
            domain=[0.2, 0.8]
        ),
        yaxis=dict(
            title="xauusd",
            titlefont=dict(
                #color="#1f77b4"
                color = 'blue'
            ),
            tickfont=dict(
                #color="#1f77b4"
                color = 'blue'
            )
        ),
        yaxis2=dict(
            title="dowi",
            titlefont=dict(
                #color="#ff7f0e"
                color = 'red'
            ),
            tickfont=dict(
                #color="#ff7f0e"
                color = 'red'
            ),
            anchor="free",
            overlaying="y",
            side="left",
            position=0.1
        ),
        yaxis3=dict(
            title="spx",
            titlefont=dict(
                #color="#d62728"
                color = 'green'
            ),
            tickfont=dict(
                #color="#d62728"
                color = 'green'
            ),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        yaxis4=dict(
            title="nasx",
            titlefont=dict(
                color="#9467bd"
            ),
            tickfont=dict(
                color="#9467bd"
            ),
            anchor="free",
            overlaying="y",
            side="right",
            position=0.9
        )
    )

    # Update layout properties
    fig2.update_layout(
        title_text="gold price and stock indices",
        width=1200,
    )

    fig2.update_xaxes(title_text="date",
                    rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=2, label="2y", step="year", stepmode='backward'),
                dict(count=5, label="5y", step="year", stepmode='backward'),
                dict(step="all")
            ])
        ))
    return fig2

def gold_vix():
    vix = pd.read_csv(path+'vix.csv')
    xau_vix = pd.merge(xauusd, vix, on='Time', suffixes=('', '_vix'))
    xau_vix['Date'] = pd.to_datetime(xau_vix['Time'])
    xau_vix.set_index(xau_vix['Date'])


    fig3 = make_subplots(specs=[[{"secondary_y":True}]])
    fig3.add_trace(
        go.Scatter(x = xau_vix['Date'], y = xau_vix['Last'], name='xau'), secondary_y=False,
    )
    fig3.add_trace(
        go.Scatter(x = xau_vix['Date'], y = xau_vix['Last_vix'], name='vix'), secondary_y=True,
    )
    fig3.update_layout(
        title_text="Gold price and CBOE volatility index"
    )


    # Set x-axis title
    fig3.update_xaxes(title_text="date",
                    rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=2, label="2y", step="year", stepmode='backward'),
                dict(count=5, label="5y", step="year", stepmode='backward'),
                dict(step="all")
            ])
        ))
    # Set y-axes titles
    fig3.update_yaxes(title_text="<b>xauusd</b> ", secondary_y=False)
    fig3.update_yaxes(title_text="<b>vix</b>", secondary_y=True)
    return fig3

def gold_dxy():
    ## xauusd and US dollar index
    dxy = pd.read_csv(path+'dxy.csv')
    xau_dxy = pd.merge(xauusd, dxy, on='Time', suffixes=('', '_dxy'))
    xau_dxy['Date'] = pd.to_datetime(xau_dxy['Time'])
    xau_dxy.set_index(xau_dxy['Date'])

    fig4 = make_subplots(specs=[[{"secondary_y":True}]])
    fig4.add_trace(
        go.Scatter(x = xau_dxy['Date'], y = xau_dxy['Last'], name='xau'), secondary_y=False,
    )
    fig4.add_trace(
        go.Scatter(x = xau_dxy['Date'], y = xau_dxy['Last_dxy'], name='dxy'), secondary_y=True,
    )
    fig4.update_layout(
        title_text="Gold price and U.S. Dollar Index"
    )
    # Set x-axis title
    fig4.update_xaxes(title_text="date",
                    rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=2, label="2y", step="year", stepmode='backward'),
                dict(count=5, label="5y", step="year", stepmode='backward'),
                dict(step="all")
            ])
        ))
    # Set y-axes titles
    fig4.update_yaxes(title_text="<b>xauusd</b> ", secondary_y=False)
    fig4.update_yaxes(title_text="<b>dxy</b>", secondary_y=True)
    return fig4