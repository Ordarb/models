import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from evohub.utils import ChartPresets

PATH2DATA = r'D:\OneDrive\Aionite\Advisory\2212 IST FX\valuation.xlsx'


class TaylorRule(object):

    def rule(self, data):
        """ Bullard version of Taylor Rule """

        real_rate = data.RealRate
        inflation_target = 2
        reaction_function = 1.25
        inflation_gap = (data.Infl-inflation_target)
        output_gap = data.OutGap
        output_gap[output_gap >= 0] = 0.

        return real_rate + inflation_target + reaction_function*(inflation_gap) + data.OutGap

    def simple_approx(self, data):
        number = data.shape[0]
        start = 2.0
        end = -0.0
        data['RealRate'] = np.arange(start, end, -(start-end)/float(number))
        return data

    def run(self):

        data = self.load_data()
        data.columns = ['Infl', 'FFR', 'OutGap', 'UST2Y']        # name the data

        data = self.simple_approx(data)
        policy_rate = self.rule(data)

        chrt = ChartPresets()
        _, ax = chrt.get_chart(2)
        policy_rate.plot(ax=ax, label='Bullard-Rule')
        data.UST2Y.plot(ax=ax, label='Policy Rate (ink. Fwd Guidance')
        chrt.despine_default(ax=ax)
        plt.show()

        bullard = (data.UST2Y - policy_rate).rolling(30).mean()
        bullard = np.clip(bullard, a_max=2, a_min=-2)

        chrt = ChartPresets()
        _, ax = chrt.get_chart(2)
        bullard.plot(ax=ax, label='Bullard Indicator')
        data.UST2Y.plot(label='2y UST')
        plt.hlines(0, xmin=bullard.index.min(), xmax=bullard.index.max())
        plt.legend()
        plt.title('Taylor (Bullard) Rule - Overvaluation')
        chrt.despine_default(ax=ax)
        plt.show()

    def load_data(self):
        """ interim data loader """
        data = pd.read_excel(PATH2DATA, sheet_name='taylor', skiprows=1)
        data.set_index(data.Datum, inplace=True)
        data = data.drop('Datum', axis=1)
        data.ffill(inplace=True)
        data.dropna(inplace=True)
        return data


if __name__ == "__main__":

    obj = TaylorRule()
    obj.run()
