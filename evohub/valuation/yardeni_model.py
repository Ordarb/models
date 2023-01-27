import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from evohub.utils import ChartPresets
from evohub.utils import PATH2SASKIA

PATH2DATA = r'C:\Users\senns\Documents\Aionite Capital\valuation.xlsx'


class YardeniModel(object):

    def run(self):

        data = self.load_data()
        data.columns = ['SPXIndexYield', 'USGG10YRIndex', 'LD06TRUUIndex', 'H15T1YIndex']
        data.SPXIndexYield = (1. / data.SPXIndexYield*100.) #earnings yield (E/P), the inverse of the PE ratio
        RP = pd.DataFrame((data.LD06TRUUIndex - data.USGG10YRIndex), columns=['RP']) #risk premium
        EGP = pd.DataFrame((data.USGG10YRIndex - data.H15T1YIndex), columns=['EGP']) #long-term expected earnings growth proxy
        df = pd.concat([data.USGG10YRIndex, RP, EGP], axis=1)

        a = 0
        b = 1
        c = 1
        d = 1
        CEY = pd.DataFrame((a + b * df.USGG10YRIndex + c * df.RP - d * df.EGP), columns=['EarningsYield']) #current earnings yield
        data.insert(4, 'EarningsYield', CEY, True)

        indicator_raw = pd.DataFrame((data.EarningsYield - data.SPXIndexYield), columns=['Indicator'])
        indicator = (indicator_raw - indicator_raw.mean()) / indicator_raw.std()
        output = pd.concat([data.EarningsYield, data.SPXIndexYield, indicator], axis=1)

        self.viz_check(output)


        return output



    def viz_check(self, output):
        """ Plotting for checking """
        chrt = ChartPresets()
        _, ax = chrt.get_chart(2)
        output.plot(ax=ax, title='Yardeni Indicator for S&P500')
        chrt.despine_default(ax=ax)
        plt.show()


    def load_data(self):
        """ interim data loader """
        data = pd.read_excel(PATH2DATA, sheet_name='yd', skiprows=1)
        data.set_index(data.Datum, inplace=True)
        data = data.drop('Datum', axis=1)
        data.ffill(inplace=True)
        data.dropna(inplace=True)
        return data


if __name__ == "__main__":

    obj = YardeniModel()
    obj.run()
