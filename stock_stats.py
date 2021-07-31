import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt

def plot_gme_stats(start_date, end_date, tracking_symbol="AG"):
    ticker = yf.Ticker(tracking_symbol)
    df = ticker.history(start=start_date, end=end_date, interval="1d")
    x = list(range((end_date - start_date).days))
    prints = "Close"
    print(df[prints])
    plt.title(f'{tracking_symbol} {prints} 1d Interval from {str(start_date.date())} to {str(end_date.date())}', fontsize=10)
    df[prints].plot()
    plt.show()
    # y = list(df["Volume"])
    # print(y)
    # fig, ax = plt.subplots()
    # ax.plot(x, y)
    # ax.set_title(f'Volume 1d Interval from {str(start_date)} to {str(end_date)}')
    # plt.show()



start_date = dt.datetime(2021, 1, 1).replace(tzinfo=dt.timezone.utc)
end_date = dt.datetime(2021, 2, 3).replace(tzinfo=dt.timezone.utc)
plot_gme_stats(start_date, end_date)
