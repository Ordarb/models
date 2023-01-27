import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from evohub.utils import ChartPresets

PATH2DATA = r'D:\OneDrive\Aionite\Advisory\2212 IST FX\valuation.xlsx'


class CreditValuation(object):

    def run(self, window=30):
        data_raw = self.load_data()                         # load data for excel sheets
        data = pd.DataFrame({'ret': data_raw.iloc[:, 0],
                             'lvl': (data_raw.iloc[:, 1]/10000.),
                             'steep': (data_raw.iloc[:, 1] - data_raw.iloc[:, 2])/10000.,
                             'vol': (data_raw.iloc[:, 1]/10000.).rolling(180).std()*np.sqrt(360)})

        # return preparation
        data.ret = np.log(data.ret/data.ret.shift(1))       # calc log returns of cdx tr index
        data['fwd_return'] = data.ret.rolling(window=window, min_periods=int(window/2)).sum()  # calc fwd looking window
        data.fwd_return = data.fwd_return.shift(-window)  # shift return for prediction

        # clean entire dataframe
        data.dropna(inplace=True)

        # prepare model inputs
        y = data.fwd_return
        x = data[['lvl', 'steep', 'vol']]
        x = sm.add_constant(x)

        # train the model
        results = sm.OLS(y, x).fit()
        print(results.summary())
        pred = results.fittedvalues
        zscore = self.transform2signal(pred)
        zscore = zscore.rolling(30).mean()

        chrt = ChartPresets()
        _, ax = chrt.get_chart(2)
        zscore.plot(ax=ax, label='Credit Value Indicator')
        ax2 = ax.twinx()
        data.ret.cumsum().plot(ax=ax2, label='Total Return Credit')
        plt.legend()
        plt.show()

        chrt = ChartPresets()
        _, ax = chrt.get_chart(2)
        data.ret.shift(-1)[zscore>=0].cumsum().plot(label='LONG Value')
        data.ret.shift(-1)[zscore<-0].cumsum().plot(label='SHORT Value')
        plt.title('Credit Value')
        plt.legend()
        plt.show()

    def transform2signal(self, data):
        """floors all indicators, so it is not perfectly zero, and looks nicer"""
        zscore = (data-0) / np.std(data)
        zscore_trimmed = np.clip(zscore, a_max=2, a_min=-2)
        return zscore_trimmed


    def load_data(self):
        data = pd.read_excel(PATH2DATA, sheet_name='cds', skiprows=1)
        data.set_index(data.Datum, inplace=True)
        data= data.drop('Datum', axis=1)
        data.ffill(inplace=True)
        data.dropna(inplace=True)
        return data


if __name__ == "__main__":

    obj = CreditValuation()
    obj.run(window=30)