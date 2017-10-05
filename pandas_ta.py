import pandas as pd

def EMA(df, n, column='Close'):
    """
    Exponential Moving Average
    """
    result=pd.Series(df[column].ewm(span=n,min_periods=n - 1,adjust=True,ignore_na=False).mean(), name='EMA_' + str(n))
    return df.join(result)

