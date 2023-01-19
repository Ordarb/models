# Valuation Models
Collection of valuation models for different asset classes.


## Equity Markets

### Equity Risk Premium (Demand & Supply Model)

### Buffet Rule

### Yardeni Model


## Currency Markets

### Purchasing Power Parity (PPP)

PPP is a theory that suggests that in the long run, exchange rates between two countries 
should adjust so that the cost of a basket of goods and services is the same in both countries, after accounting for 
exchange rate changes. PPP states that the exchange rate between two countries should be such that it would cost the 
same amount to purchase the same basket of goods and services in either country, after accounting for the exchange rate.

PPP theory suggests that if a basket of goods costs $100 in the United States, and the exchange rate is 1.5 US dollars 
to 1 Euro, then the same basket of goods should cost 150 Euros in Europe. If it does not, then there is an opportunity 
for arbitrage, and market forces should drive the exchange rate towards the PPP rate.

To calculate we follow the Long-Run Averaging methodology as proposed from McKinnon & Ohno (1990) (For the case rate in CPI, 
the average of the time period 1982 to 2000 is taken):
```math  
PPP = Average Exchange Rate * [(Foreign CPI[t]/Foreign CPI[average]) /(Base CPI[t] /Base CPI[average])] 
```
For the inflation adjustment, we take consumer prices (CPI) as well as producer prices (PPI) and take the average of
the two measures.

- Frequency: monthly frequency update in CPI/PPI, daily update in actual nominal exchange rate.
- Type: USD is taken as base currency and is therefore always "fairly valued" with a score of 1
- Valuation range: Valuation ranges of +/- 20% are considered as significant, +/- 10% are in "fair"
- Source: Bloomberg
- EUR: BPPPCOEU Index (EUR CPI), BPPPPOEU Index (EUR PPI) 
- CHF: BPPPCOCH Index (CHF CPI), BPPPPOCH Index (CHF PPI)

### Real Effective Exchange Rate (REER)

The REER is a measure of the value of a country's currency relative to the currencies of its major trading partners, 
adjusted for inflation. It is calculated by taking the nominal exchange rate, which is the rate at which a currency 
can be exchanged for another, and adjusting it for differences in inflation between the countries. The REER is used to 
measure the competitiveness of a country's exports, as well as the overall level of economic activity.

The REER reflects not just the price of a single good or service, but the price of all goods and services that are 
traded in the international market. It can also be used as an indicator of a country's overall economic performance, 
as a higher REER can indicate that a country's exports are becoming less competitive and its imports are becoming more 
expensive.

A country with a rising REER may be experiencing appreciation of its currency, making its exports more expensive and 
less competitive, which can lead to a decrease in the trade balance and a decrease in the overall economic activity. 
Conversely, a country with a falling REER may be experiencing depreciation of its currency, making its exports cheaper 
and more competitive, which can lead to an increase in the trade balance and an increase in the overall economic 
activity.

The difference between purchasing power parity (PPP) and the real effective exchange rate (REER) is that PPP is a 
theoretical concept that compares the relative purchasing power of different currencies, while the REER is a measure 
of how the value of a currency changes against a basket of other currencies.

- Frequency: monthly frequency update in CPI/PPI, daily update in actual nominal exchange rate.
- Type: USD is taken as base currency and is therefore always "fairly valued" with a score of 1
- Base: 2010 = 100
- Source: REER from BIS
- Valuation range: Valuation ranges of +/- 20% are considered as significant, +/- 10% are in "fair"
- Source: Bloomberg
- Indices: BISNUSR Index (USD), BISNCHR Index (CHF), BISNEUR Index (EUR)

## Interest Rate Markets

### Taylor Rule

## Credit Markets

### Credit Risk Premium
https://academic.oup.com/rof/article/22/2/419/4828075


### Valuation Model
Running a regression of spreadlevels + Steepness on future returns to estimate expected returns