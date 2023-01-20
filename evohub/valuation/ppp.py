from evohub.utils import _BaseIndicator


# Use EURUSD - PPP Rate = Result
# Use USDCHF - PPP Rate = Result * -1


class PurchasingPowerParity(object):

    def __init__(self):
        print('Saskia')
        print('Sandro')

    def run(self):
        data = self.load_data()
        eur = ['EURUSD Cunncy', 'BPPPCPEU Index',]
        chf = ['USDCHF Curncy', 'BPPPCPCH Index']

        ppp_eur = data[eur[0]- data[eur[1]]]
        ppp_chf = (data[chf[0]- data[chf[1]]]) * -1

    def load_data(self):
        data = None
        return data

