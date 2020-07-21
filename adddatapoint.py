import pandas as pd

path = './datacopy/'

def add_row(symbol, Time, Open, High, Low, Last, Change, Volume):
    df = pd.read_csv(path+symbol+'.csv')
    if Time in df['Time'].values:
        raise Exception('Date {0} already exists'.format(Time))
    top_row = pd.DataFrame({'Time':[Time], 'Open':[Open], 'High':[High], 'Low':[Low], 'Last':[Last], 'Change':[Change], 'Volume':[Volume]})
    print(top_row)
    print(df.shape)
    df = pd.concat([top_row, df]).reset_index(drop=True)
    print(df.shape)
    print(df.head())
    df.to_csv(path+symbol+'.csv', index=False)

    
symb = '^xauusd'
date = '07/20/2020'

Open = 1809.34
High = 1819.09
Low = 1805.91
Last = 1816.78
Change = +7.45
Volume = 6842





add_row(symb, date, Open, High, Low, Last, Change, Volume)
