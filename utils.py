import numpy as np
import pandas as pd  

def diff_next_n_bar(df, n, column='Close', cat_diff=True, cat_diff_group_by_percent=[0]):
    name = "DIFF_NEXT_" + str(n)
    if not cat_diff:
        result = pd.Series(-df[column].diff(-n), name=name)
    else:
        result = pd.Series(100*(-df[column].diff(-n)/df[column]), name=name)
        bin_map = [-np.inf] + cat_diff_group_by_percent + [np.inf]
        result = pd.cut(result, bins=bin_map, labels=range(len(bin_map)-1))
    return df.join(result)