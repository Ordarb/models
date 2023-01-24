import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from evohub.utils import ChartPresets
from evohub.utils import PATH2SASKIA

PATH2DATA = r'C:\Users\senns\Documents\Aionite Capital\valuation.xlsx'


class BuffettModel(object):

    def run(self):

        data = self.load_data()

        data.columns = ['StockMarket', 'NomGDP']        # name the data
        data.StockMarket = data.StockMarket / 1000000.  # bring data to billions
        data.NomGDP = data.NomGDP / 1000.               # bring data to billions
        indicator_raw = pd.DataFrame((data.StockMarket / data.NomGDP), columns=['indicator_raw'])
        tl = self.exponential_trendline(indicator_raw)
        df = pd.concat([indicator_raw, tl], axis=1)
        indicator = pd.DataFrame((df.indicator_raw - df.trend), columns=['buffet_indicator'])
        output = pd.concat([df, indicator], axis=1)

        self.viz_check(output)

        return output

    def exponential_trendline(self, indicator_raw):

        y = np.array(indicator_raw)
        x = np.arange(0, len(indicator_raw), 1)
        fit = np.polyfit(x, np.log(y), 1)

        tl = pd.DataFrame(np.exp(fit[1] + x*fit[0]),
                          index=indicator_raw.index,
                          columns=['trend'])
        return tl

    def viz_check(self, df):
        """ Plotting for checking """
        chrt = ChartPresets()
        _, ax = chrt.get_chart(2)
        df.plot(ax=ax, title='Buffet Indicator for S&P500')
        chrt.despine_default(ax=ax)
        plt.show()

    def load_data(self):
        """ interim data loader """
        data = pd.read_excel(PATH2DATA, sheet_name='buffett', skiprows=1)
        data.set_index(data.Datum, inplace=True)
        data = data.drop('Datum', axis=1)
        data.ffill(inplace=True)
        data.dropna(inplace=True)
        return data


if __name__ == "__main__":

    obj = BuffettModel()
    obj.run()
