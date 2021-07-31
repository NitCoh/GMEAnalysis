import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_now(exps, date, num_posts):
    print("=" * 80)
    print(exps)
    print(date)
    print(f"max: {max(exps)}, min: {min(exps)}")
    print(f"num of related posts: {len(exps)}")
    print(f"total num of posts {num_posts}")
    print(f"percent related: {(len(exps) / num_posts) * 100}%")
    print("=" * 80)
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 1000]
    kwargs = dict(histtype='stepfilled', alpha=0.4, density=True, bins=bins)
    plt.title(f"Single Post's Exposure Frequency Dist of AG at {date}")
    plt.hist(exps, **kwargs)
    plt.show()


def plot_different_exposure(filename):
    date_rx = r"Date: (2021-01-2[0-9]) 00:00:00\+00:00"
    rx_gme = r"GME-exposure-\[([0-9]+(?:\.[0-9]+)?)\]"
    rx_ag = r"AG-exposure-\[([0-9]+(?:\.[0-9]+)?)\]"

    exposures_gme = []
    exposures_ag = []
    acc_gme = 0
    acc_ag = 0
    number_of_posts = 0
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            found_date = re.search(date_rx, line)
            if found_date is not None:
                plot_now(exposures_gme, found_date.group(1), number_of_posts)
                plot_now(exposures_ag, found_date.group(1), number_of_posts)
                exposures_gme = []
                exposures_ag = []
                acc_gme = 0
                acc_ag = 0
                number_of_posts = 0

            gme_m_g = re.search(rx_gme, line)
            if gme_m_g is not None:
                acc_exp = float(gme_m_g.group(1))
                exp = acc_exp - acc_gme
                acc_gme = acc_exp
                if exp > 0.0: exposures_gme.append(exp)

            ag_m_g = re.search(rx_ag, line)
            if ag_m_g is not None:
                acc_exp = float(ag_m_g.group(1))
                exp = acc_exp - acc_ag
                acc_ag = acc_exp
                if exp > 0.0: exposures_ag.append(exp)

            if ag_m_g is not None or gme_m_g is not None:
                number_of_posts += 1


def plot_exposure_hist(filename):
    date_rx = r"Date: (2021-01-2[0-9]) 00:00:00\+00:00"
    rx = r"exposure-\[([0-9]+(?:\.[0-9]+)?)\]"

    exposures = []
    number_of_posts = 0
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            found_date = re.search(date_rx, line)
            if found_date is not None:
                plot_now(exposures, found_date.group(1), number_of_posts)
                exposures = []
                number_of_posts = 0

            m_g = re.search(rx, line)
            if m_g is not None:
                exp = float(m_g.group(1))
                number_of_posts += 1
                if exp > 0.0: exposures.append(exp)


# plot_exposure_hist('./output-2021-07-23 22:49:59.555285.log') # gme

# plot_exposure_hist('./output-2021-07-24 23:20:28.680997.log') # ag

# plot_different_exposure('./output-2021-07-30 09:41:00.330006.log')


def plot_exps():
    import datetime
    import pandas as pd
    gme_exps = [7000.265377344789, 4621.704191972047, 16740.03492280733, 39005.219974445696, 26272.915514198758]
    ag_exps = [3.0, 2.0, 57.46703917210822, 133.13861978062786, 71.27140496138989]
    # dates = ["01/25/2021", "01/26/2021", "01/27/2021", "01/28/2021", "01/29/2021"]

    # dates = ['2021-01-25', '2021-01-26', '2021-01-27', '2021-01-28', '2021-01-29']
    # xdates = [datetime.datetime.strptime(str(int(date)), '%Y-%m-%d') for date in dates]

    dates = [datetime.datetime(2021, 1, 25) + datetime.timedelta(days=i) for i in range(5)]

    plt.plot_date(dates, gme_exps, linestyle='dashdot', label='GME')

    plt.gcf().autofmt_xdate()
    myFmt = mdates.DateFormatter('%Y-%m-%d')
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.grid()
    plt.xlabel('Date')
    plt.ylabel('Exposure')
    plt.title('GME Stock Exposure')
    plt.legend(loc=4)
    plt.xticks(rotation=45)
    plt.show()


plot_exps()


#
# def plot_exposure_by_symbol():
#     """
#     Plot the exposure over a defined period of a time.
#     :return:
#     """
#     start_date = dt.datetime(2021, 1, 25).replace(tzinfo=dt.timezone.utc)
#     end_date = dt.datetime(2021, 1, 30).replace(tzinfo=dt.timezone.utc)
#
#     tracking_symbol = "AG"
#     logger.debug(f"Start computing on symbol {tracking_symbol} from {start_date.date()} to {end_date.date()}")
#     total_days = (end_date - start_date).days
#     total_posts = 0
#
#     enum_days = list(range(total_days))
#     exposure_by_day = []
#
#     for i in range(total_days):
#         total_exposure = 0
#         start_of_day = (start_date + dt.timedelta(days=i)).replace(tzinfo=dt.timezone.utc)  # datetime object
#         end_of_day = (start_date + dt.timedelta(days=i + 1)).replace(tzinfo=dt.timezone.utc)  # end of the day
#         gen = get_all_posts_by_interval(start_of_day, end_of_day)
#         for post in gen:
#             try:
#                 G = build_tree_by_bfs_traversal(post)
#                 exposure = compute_exposure(G, start_of_day, end_of_day, tracking_symbol)
#             except Exception:
#                 continue
#             try:
#                 if exposure >= 5:
#                     plot_post(G)
#             except Exception:
#                 logger.error("couldn't plot graph")
#             total_exposure += exposure
#             logger.debug(
#                 f"[{total_posts}]: id-[{post.id}], Title-[{post.title}], exposure-[{exposure}], url-[{post.permalink}]")
#             total_posts += 1
#
#         logger.debug("=" * 80)
#         logger.debug(f"Date: {start_of_day}, Total Exposure {total_exposure}")
#         exposure_by_day.append(total_exposure)
#         logger.debug(enum_days)
#         logger.debug(exposure_by_day)
#         logger.debug("=" * 80)
#
#     assert len(enum_days) == len(exposure_by_day)
#
#     logger.debug(enum_days)
#     logger.debug(exposure_by_day)
#     plt.scatter(enum_days, exposure_by_day)
#     plt.ylabel('Exposure')
#     plt.xlabel('Enumerated Days')
#     plt.title(f"GME Exposure from {start_date.date()} to {end_date.date()}")
#     plt.show()
#
#
# def compute_exposure(G, start_of_day, end_of_day, symbol):
#     local_popularity = {}  # convention: save only string upper form keys
#     new_comments = 0
#
#     root = 0
#     edges = nx.bfs_edges(G, source=root)
#     bfs_nodes = [root] + [v for u, v in edges]  # traversing the nodes
#     for u in bfs_nodes:
#         node = G.nodes[u]
#         created_date = dt.datetime.utcfromtimestamp(node["created"]).replace(tzinfo=dt.timezone.utc)
#         text = node["comment"]
#         if "title" in node:  # root case
#             text += (" " + node["title"])
#
#         found_symbols = find_symbols_ner(text)
#         found_symbols = [x.upper() for x in found_symbols]
#         if len(found_symbols) > 0 and created_date < end_of_day:  # LP is cumulative
#             LP_tag = (2 ** (-node["depth"])) * node["score"]  # compute LP' for a comment.
#             for ticker in found_symbols:
#                 if ticker not in local_popularity:
#                     local_popularity[ticker] = LP_tag
#                 else:
#                     local_popularity[ticker] += LP_tag
#
#             # exposure is defined on a day
#             if created_date >= start_of_day and symbol in found_symbols:
#                 new_comments += 1
#
#     probs = softmax(list(local_popularity.values()))
#     tickers = list(local_popularity.keys())
#     local_popularity = {k: v for k, v in zip(tickers, probs)}
#
#     return new_comments * local_popularity[symbol] if symbol in local_popularity else 0