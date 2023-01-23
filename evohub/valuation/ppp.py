import pandas as pd
import matplotlib.pyplot as plt

PATH2DATA = r'D:\OneDrive\Aionite\Advisory\2212 IST FX\valuation.xlsx'


class PurchasingPowerParity(object):

    def run(self):

        data = self.load_data()

        matching = {'EUR': {'cross': 'EURUSD Curncy', 'ppp': 'BPPPCPEU Index'},
                    'CHF': {'cross': 'USDCHF Curncy', 'ppp': 'BPPPCPCH Index'}}

        output = []
        for key in matching:

            cross = data[matching[key]['cross']]
            ppp_raw = data[matching[key]['ppp']]

            if matching[key]['cross'][:3] == key:
                ppp = 1. / ppp_raw
                spread = (cross - ppp)
                value = spread / ppp
            else:
                ppp = ppp_raw
                spread = (cross - ppp)
                value = (spread / ppp) * -1

            self.viz_check(key, cross, ppp, value)

            output.append(pd.DataFrame({'{}_cross'.format(key): cross,
                                        '{}_ppp'.format(key): ppp,
                                        '{}_value'.format(key): value}))
        return pd.concat(output, axis=1)

    def viz_check(self, key, cross, ppp, value):
        """ Plotting for checking """
        cross.plot(label='{}-cross'.format(key))
        ppp.plot(label='{}-PPP'.format(key))
        plt.legend()
        plt.title('{} vs PPP (CPI)'.format(key))
        plt.show()

        value.plot(label='PPP (CPI)'.format(key))
        plt.hlines(y=0, xmin=value.index.min(), xmax=value.index.max())
        plt.legend()
        plt.title('Overvaluation of {}'.format(key[:3]))
        plt.show()

    def load_data(self):
        """ interim data loader """
        data = pd.read_excel(PATH2DATA, sheet_name='ppp', skiprows=1)
        data.set_index(data.Datum, inplace=True)
        data= data.drop('Datum', axis=1)
        data.ffill(inplace=True)
        data.dropna(inplace=True)
        return data


if __name__ == "__main__":

    obj = PurchasingPowerParity()
    obj.run()

