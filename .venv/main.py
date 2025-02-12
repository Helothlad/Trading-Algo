# Required Libraries
from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta
import warnings
from sklearn.cluster import KMeans
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import matplotlib.ticker as mtick

warnings.filterwarnings('ignore')

# Load S&P 500 Symbols
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
symbols_list = sp500['Symbol'].unique().tolist()

# Handle Failed Yahoo Finance Downloads
failed_tickers = ['SOLV', 'VLTO', 'SW', 'GEV']
symbols_list = [symbol for symbol in symbols_list if symbol not in failed_tickers]

# Set Date Range
end_date = '2024-12-31'
start_date = pd.to_datetime(end_date) - pd.DateOffset(365 * 8)

# Download Stock Data
df = yf.download(tickers=symbols_list, start=start_date, end=end_date).stack()
df.index.names = ['date', 'ticker']
df.columns = df.columns.str.lower()

# Feature Engineering
df['garman_klass_vol'] = ((np.log(df['high']) - np.log(df['low'])) ** 2) / 2 - (2 * np.log(2) - 1) * (
        (np.log(df['adj close']) - np.log(df['open'])) ** 2)

df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))

df['bb_low'] = df.groupby(level=1)['adj close'].transform(
    lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 0])
df['bb_mid'] = df.groupby(level=1)['adj close'].transform(
    lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 1])
df['bb_high'] = df.groupby(level=1)['adj close'].transform(
    lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 2])


# ATR Calculation
def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'],
                        low=stock_data['low'],
                        close=stock_data['close'],
                        length=14)
    return atr.sub(atr.mean()).div(atr.std())


df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)


# MACD Calculation
def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20).iloc[:, 0]
    return macd.sub(macd.mean()).div(macd.std())


df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)

# Calculate Dollar Volume
df['dollar_volume'] = (df['adj close'] * df['volume']) / 1e6

# Selecting Relevant Columns
last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open', 'high', 'low', 'close']]

# Creating Monthly Data
data = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
                   df.unstack()[last_cols].resample('M').last().stack('ticker')],
                  axis=1)).dropna()

# Rank and Filter by Liquidity
data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker').rolling(5 * 12, min_periods=12).mean().stack())
data['dollar_vol_rank'] = data.groupby('date')['dollar_volume'].rank(ascending=False)
data = data[data['dollar_vol_rank'] < 150].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)


# Calculate Returns
def calculate_returns(df):
    outlier_cutoff = 0.005
    lags = [1, 2, 3, 6, 9, 12]
    for lag in lags:
        df[f'return_{lag}m'] = (df['adj close']
                                .pct_change(lag)
                                .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                       upper=x.quantile(1 - outlier_cutoff)))
                                .add(1)
                                .pow(1 / lag)
                                .sub(1))
    return df


data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()

# Load Fama-French Factor Data
factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                             'famafrench',
                             start='2010')[0].drop('RF', axis=1)
factor_data.index = factor_data.index.to_timestamp()
factor_data = factor_data.resample('M').last().div(100)
factor_data.index.name = 'date'

# Merge Factor Data
factor_data = factor_data.join(data['return_1m']).sort_index()

# Compute Factor Betas
observations = factor_data.groupby(level=1).size()
valid_stocks = observations[observations >= 10]
factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]

betas = (factor_data.groupby(level=1, group_keys=False)
         .apply(lambda x: RollingOLS(endog=x['return_1m'],
                                     exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                     window=min(24, x.shape[0]),
                                     min_nobs=len(x.columns) + 1)
                .fit(params_only=True)
                .params
                .drop('const', axis=1)))

# Merge Data and Betas
factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
data = data.join(betas.groupby('ticker').shift())
data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))
data = data.drop('adj close', axis=1)
data = data.dropna()

# Check Data
print(data.info())

# Unsupervised Learning: K-Means Clustering
if 'cluster' in data.columns:
    data = data.drop('cluster', axis=1)


def get_clusters(df):
    df = df.copy()
    df['cluster'] = KMeans(n_clusters=4, random_state=0).fit(df).labels_
    return df


data = data.dropna().groupby('date', group_keys=False).apply(get_clusters)


# Function to Plot Clusters
def plot_clusters(data):
    for i in range(4):
        cluster = data[data['cluster'] == i]
        plt.scatter(cluster.iloc[:, 0], cluster.iloc[:, 6], label=f'Cluster {i}')
    plt.legend()
    plt.show()


# Plot Clusters for Each Date
plt.style.use('ggplot')
for date in data.index.get_level_values('date').unique().tolist():
    g = data.xs(date, level=0)
    plt.title(f'Date {date}')
    plot_clusters(g)

# Portfolio Construction
target_rsi_values = [30, 45, 55, 70]
initial_centroids = np.zeros((len(target_rsi_values), 18))
initial_centroids[:, 6] = target_rsi_values

filtered_df = data[data['cluster'] == 3].copy()
filtered_df = filtered_df.reset_index(level=1)
filtered_df.index = filtered_df.index + pd.DateOffset(1)
filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])
dates = filtered_df.index.get_level_values('date').unique().tolist()
fixed_dates = {}
for d in dates:
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()

stocks = data.index.get_level_values('ticker').unique().tolist()

# Download updated stock data
new_df = yf.download(tickers=stocks,
                     start=data.index.get_level_values('date').unique()[0] - pd.DateOffset(months=12),
                     end=data.index.get_level_values('date').unique()[-1])

# Handle MultiIndex columns
if isinstance(new_df.columns, pd.MultiIndex):
    adj_close = new_df.xs('Adj Close', level=0, axis=1)
else:
    adj_close = new_df['Adj Close']

returns_dataframe = np.log(adj_close).diff()

# Download SPY data for benchmark
spy = yf.download('SPY', start=start_date, end=end_date)[['Adj Close']].rename(columns={'Adj Close': 'SPY'})
spy_ret = np.log(spy).diff().resample('M').sum()

# Portfolio Optimization Function
def optimize_weights(prices, lower_bound=0.01):
    try:
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)
        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda w: w >= lower_bound)
        ef.max_sharpe()
        return ef.clean_weights()
    except Exception as e:
        print(f"Optimization error: {str(e)}")
        return None

# Build portfolio
portfolio_df = pd.DataFrame()

for start_date in fixed_dates.keys():
    try:
        end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
        cols = fixed_dates[start_date]

        optimization_start = (pd.to_datetime(start_date) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
        optimization_end = (pd.to_datetime(start_date) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')

        print(f"Processing {start_date} to {end_date}")
        print(f"Stocks: {cols}")
        print(f"Optimization window: {optimization_start} to {optimization_end}")

        # Filter columns that exist in our data
        valid_cols = [col for col in cols if col in adj_close.columns]
        if not valid_cols:
            print(f"Skipping {start_date} - no valid tickers")
            continue

        optimization_df = adj_close[optimization_start:optimization_end][valid_cols]

        if optimization_df.empty or len(optimization_df) < 20:
            print(f"Skipping {start_date} - insufficient data")
            continue

        weights = optimize_weights(optimization_df)
        if weights is None:
            print(f"Using equal weights for {start_date}")
            weights = {col: 1 / len(valid_cols) for col in valid_cols}

        weights = pd.Series(weights)

        temp_df = returns_dataframe[start_date:end_date][valid_cols].mul(weights).sum(axis=1).to_frame(
            'Strategy Return')
        portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)

    except Exception as e:
        print(f"Error processing {start_date}: {str(e)}")

# Ensure datetime index
portfolio_df.index = pd.to_datetime(portfolio_df.index)
spy_ret.index = pd.to_datetime(spy_ret.index)

# Merge with SPY returns
portfolio_df = portfolio_df.merge(spy_ret, left_index=True, right_index=True, how='left')
portfolio_df = portfolio_df.dropna()

# Calculate cumulative returns
cumulative_returns = np.exp(np.log1p(portfolio_df).cumsum()) - 1

# Plot results
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(16, 6))

cumulative_returns['Strategy Return'].plot(ax=ax, label='Strategy')
cumulative_returns['SPY'].plot(ax=ax, label='S&P 500', linestyle='--')

ax.set_title('Strategy vs S&P 500 Performance')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylabel('Cumulative Returns')
ax.legend()

plt.show()
