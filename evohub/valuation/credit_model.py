import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from evohub.utils import _BaseIndicator
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler

PATH2DATA = r'D:\OneDrive\Aionite\Advisory\2212 IST FX\valuation.xlsx'
# Use EURUSD - PPP Rate = Result
# Use USDCHF - PPP Rate = Result * -1


class CreditValuation(object):

    def run(self):
        data = self.load_data()


        window = 60
        y = pd.DataFrame({'ret': data.iloc[:, 0]})      # return data to predict
        y = np.log(y/y.shift(1))                        # calc daily log returns
        y = y.rolling(window=window, min_periods=int(window/2)).sum()    # calc ret over a period
        y = y.shift(-window)                             # shift return for prediction

        x = pd.DataFrame({'lvl': data.iloc[:, 1], 'steep': (data.iloc[:, 1] - data.iloc[:, 2])})
        x = x / 10000.

        df = pd.concat([y, x], axis=1)
        df.dropna(inplace=True)

        y = df.iloc[:, 0]
        x = df.iloc[:, 1:]
        x = sm.add_constant(x)


        results = sm.OLS(y, x).fit()
        print(results.summary())
        y_hat = results.fittedvalues
        y.plot(color='lightgrey')
        y_hat.plot(color='black')
        plt.show()

        score = y_hat / np.std(y_hat)

        scaler= MinMaxScaler(feature_range=(-1,1))
        scaler.fit(results.fittedvalues.values)
        scaler.transform(results.fittedvalues.values)

        model = LinearRegression(fit_intercept=False)
        model.fit(df.iloc[:, 1:], df.iloc[:, 0])

        output = pd.DataFrame({'y': df.iloc[:, 0], 'y_hat': model.predict(df.iloc[:, 1])})
        output.plot()
        plt.show()



    def load_data(self):
        data = pd.read_excel(PATH2DATA, sheet_name='cds', skiprows=1)
        data.set_index(data.Datum, inplace=True)
        data= data.drop('Datum', axis=1)
        data.ffill(inplace=True)
        data.dropna(inplace=True)
        return data

if __name__ == "__main__":

    obj = CreditValuation()
    obj.run()