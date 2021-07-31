import praw
import yfinance as yf
import networkx as nx
import datetime as dt
from psaw import PushshiftAPI
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib as mpl
import numpy as np
import string
import spacy
from spacy.tokenizer import Tokenizer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
import time
import logging
import sys
# This module is broken at the current version, please refer to the issues section in the github repo to examine the fix.
from get_all_tickers import get_tickers as gt

nlp = spacy.load('en_core_web_lg')
tokenizer = Tokenizer(nlp.vocab)

list_of_tickers = gt.get_tickers()
black_list = ['lmao', 'fda', 'covid', 'nyse', 'nasdaq', 'spac', 'ev', 'yolo', 'fomo']

logger = logging.getLogger('GMEAnalysis')
logger.setLevel("DEBUG")
output_file_handler = logging.FileHandler(f"output-{str(dt.datetime.now())}.log")
stdout_handler = logging.StreamHandler(sys.stdout)

logger.addHandler(output_file_handler)
logger.addHandler(stdout_handler)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if len(x) <= 0:
        return x
    if isinstance(x, list):
        x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return list(e_x / e_x.sum(axis=0))


def plot_post(G):
    """
    Plots the constructed tree (=G).
    Assuming nodes have "sentiment score" and a "score".
    Plot Attributes:
    1. The size of each node determined by the comment's score
        Note: The size is a relative-size, scaled by min-max according to all scores in the submission
    2. The color of each node determined by the sentiment score given to the comment.
        Blue = Negative, Gray = Neutral, Red = Positive
    :param G:
    :param exposure:
    :return:
    """

    def scale(arr):
        min_max_scaler = MinMaxScaler(feature_range=(1, 20))
        narr = np.array(arr).reshape(-1, 1)
        return min_max_scaler.fit_transform(narr).squeeze(-1).tolist()

    labels = nx.get_node_attributes(G, "comment")
    ax = plt.gca()
    title = G.nodes[0]["title"]
    ax.set_title(title)
    sentiment_dict = nx.get_node_attributes(G, "sentiment")

    norm = mpl.colors.Normalize(vmin=-1, vmax=1, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)
    scores_dict = nx.get_node_attributes(G, "score")
    scores_norm = scale(list(scores_dict.values()))  # scale the scores of comments relative to the post
    pos = graphviz_layout(G, prog='dot')
    nx.draw(G, pos=pos,
            node_size=[x * 100 for x in scores_norm],
            node_color=[mapper.to_rgba(i) for i in sentiment_dict.values()]
            )
    plt.savefig(f"figs_trial/{title.replace(' ', '-')}.png")
    plt.show(block=False)
    plt.clf()


def build_tree_by_bfs_traversal(submission):
    """
    Build a networkx Graph object from a submission by BFS traversal.
    Efficiency Notes:
    Assumes a submission has tree structure.
    Sentiment Score is provided by VADER.
    VADER (Valence Aware Dictionary and sEntiment Reasoner)
    is a lexicon and rule-based sentiment analysis tool
    that is specifically attuned to sentiments expressed in social media.
    :param submission:
    :return:
    """
    G = nx.Graph()
    sid = SentimentIntensityAnalyzer()
    i = 0
    submission.comments.replace_more()  # transforming all MoreComments instances to Comments.
    # Since we assume a submission is a tree, we don't need a "visited" array.

    # flatten_comments = submission.comments.list()
    # visited = [False] * (len(flatten_comments) + 1)

    title_score = (sid.polarity_scores(submission.title)["compound"])
    text_score = (sid.polarity_scores(submission.selftext)["compound"])
    sentiment_score = (title_score + text_score) / 2
    G.add_node(i, **{"comment": submission.selftext, "title": submission.title, "score": submission.score,
                     "awards": submission.total_awards_received, "url": submission.url,
                     "created": submission.created_utc,
                     "flair": submission.link_flair_richtext, "depth": 0, "sentiment": sentiment_score})

    # start init
    queue = []
    submission.node_num = i
    submission.depth_level = 0
    queue.append(submission)
    # visited[i] = True
    i += 1

    while queue:

        s = queue.pop(0)
        if hasattr(s, 'comments'):
            forest = s.comments
        elif hasattr(s, 'replies'):
            forest = s.replies
        else:
            logger.debug("something wrong")
            break

        forest.replace_more()

        for comment in forest:
            # if not visited[i]:
            queue.append(comment)
            # visited[i] = True
            comment.node_num = i
            comment.depth_level = s.depth_level + 1
            sentiment_score = sid.polarity_scores(comment.body)["compound"]
            G.add_node(i, **{"comment": comment.body, "score": comment.score, "created": comment.created_utc,
                             "depth": comment.depth_level, "sentiment": sentiment_score})
            G.add_edge(*(s.node_num, i))

            i += 1

    return G


def get_all_posts_by_interval(start, end):
    """
    Get all posts at "wallstreetbets" in a given time interval.
    :param start: Datettime object
    :param end: Datettime object
    :return: Generator(Reddit Submission Object)
    """
    reddit = praw.Reddit("bot1", user_agent="GME Analyzer Bot")
    api = PushshiftAPI(reddit)
    return api.search_submissions(
        after=int(start.timestamp()),
        before=int(end.timestamp()),
        subreddit='wallstreetbets',
        sort_type='created_utc',
        sort='asc',
    )


def plot_exposures(tracking_symbols=("GME", "AG")):
    """
    Plot the exposure of all tracking symbols over a pre-defined period of a time.
    :return:
    """
    start_date = dt.datetime(2021, 1, 25).replace(tzinfo=dt.timezone.utc)
    end_date = dt.datetime(2021, 1, 31).replace(tzinfo=dt.timezone.utc)

    logger.debug(f"Start computing on symbols {tracking_symbols} from {start_date.date()} to {end_date.date()}")
    total_days = (end_date - start_date).days
    total_posts = 0

    enum_days = list(range(total_days))
    exposure_by_day = []

    for i in range(total_days):
        total_exposure = {k: v for k, v in zip(tracking_symbols, [0] * len(tracking_symbols))}

        start_of_day = (start_date + dt.timedelta(days=i)).replace(tzinfo=dt.timezone.utc)  # datetime object
        end_of_day = (start_date + dt.timedelta(days=i + 1)).replace(tzinfo=dt.timezone.utc)  # end of the day
        gen = get_all_posts_by_interval(start_of_day, end_of_day)
        for post in gen:
            try:
                G = build_tree_by_bfs_traversal(post)
                exposures = compute_exposures(G, start_of_day, end_of_day, tracking_symbols)
            except Exception:
                continue
            try:
                if any([x >= 5 for x in exposures.values()]):
                    plot_post(G)
            except Exception:
                logger.error("couldn't plot graph")

            new_exposures = [sum(x) for x in zip(total_exposure.values(), exposures.values())]
            total_exposure = {k: v for k, v in zip(total_exposure.keys(), new_exposures)}
            s = f"[{total_posts}]: id-[{post.id}], Title-[{post.title}], url-[{post.permalink}], date-[{str(start_date.date())}"
            for k, v in total_exposure.items():
                s += f", {k}-exposure-[{v}]"
            logger.debug(s)
            total_posts += 1

        logger.debug("=" * 80)
        logger.debug(f"Date: {start_of_day}, Total Exposure {total_exposure}")
        exposure_by_day.append(total_exposure)
        logger.debug(enum_days)
        logger.debug(exposure_by_day)
        logger.debug("=" * 80)

    logger.debug(enum_days)
    logger.debug(exposure_by_day)

    for symbol in tracking_symbols:
        exposures_list = [x[symbol] for x in exposure_by_day]
        assert len(enum_days) == len(exposures_list)
        plt.scatter(enum_days, exposures_list)
        plt.ylabel('Exposure')
        plt.xlabel('Enumerated Days')
        plt.title(f"{symbol} Exposure from {start_date.date()} to {end_date.date()}")

    plt.show()


def compute_exposures(G, start_of_day, end_of_day, symbols):
    """
    Compute exposure_tag in a certain day for each symbol in a post (modeled by graph G)
    :param G:
    :param start_of_day:
    :param end_of_day:
    :param symbols:
    :return: dict(symbol->exposure')
    """
    local_popularity = {}  # convention: save only string upper form keys

    new_comments = {k: v for k, v in zip(symbols, [0] * len(symbols))}

    root = 0
    edges = nx.bfs_edges(G, source=root)
    bfs_nodes = [root] + [v for u, v in edges]  # traversing the nodes
    for u in bfs_nodes:
        node = G.nodes[u]
        created_date = dt.datetime.utcfromtimestamp(node["created"]).replace(tzinfo=dt.timezone.utc)
        text = node["comment"]
        if "title" in node:  # root case
            text += (" " + node["title"])

        found_symbols = find_symbols_ner(text)
        found_symbols = [x.upper() for x in found_symbols]
        if len(found_symbols) > 0 and created_date < end_of_day:  # LP is cumulative
            LP_tag = (2 ** (-node["depth"])) * node["score"]  # compute LP' for a comment.
            for ticker in found_symbols:
                if ticker not in local_popularity:
                    local_popularity[ticker] = LP_tag
                else:
                    local_popularity[ticker] += LP_tag

            # exposure is defined on a day
            if created_date >= start_of_day:
                for symbol in symbols:
                    if symbol in found_symbols:
                        new_comments[symbol] += 1

    probs = softmax(list(local_popularity.values()))
    tickers = list(local_popularity.keys())
    local_popularity = {k: v for k, v in zip(tickers, probs)}

    exposures = {}
    for symbol in symbols:
        if symbol in local_popularity:
            exposures[symbol] = new_comments[symbol] * local_popularity[symbol]
        else:
            exposures[symbol] = 0

    return exposures


def find_symbols_ner(text):
    """
    Find all the tickers (=symbols) in a given text
    using spaCy pipeline, contains NER model.
    :param text:
    :return:
    """
    text = text.translate(str.maketrans('', '', string.punctuation))
    doc = nlp(text)  # spaCy pipeline activated
    tickers = []
    for entity in doc.ents:
        if entity.label_ == "ORG" and entity.text.lower() not in black_list:
            tickers.append(entity.text)

    # same ticker could be added multiple times
    # validate symbols
    tickers = list(set(tickers))
    return list(filter(is_symbol, tickers))


def is_symbol(candidate):
    """
    Predicate to check if a certain candidate is a valid ticker in the american stock market.
    :param candidate:
    :return:
    """
    return isinstance(candidate, str) and len(candidate) <= 5 and candidate in list_of_tickers


def check_how_many(tracking_symbol="GME"):
    """
    A tool to check how many posts in a given time interval.
    :param tracking_symbol:
    :return:
    """
    start_date = dt.datetime(2021, 1, 28).replace(tzinfo=dt.timezone.utc)
    end_date = dt.datetime(2021, 1, 29).replace(tzinfo=dt.timezone.utc)
    logger.debug(
        f"Evaluating number of posts on symbol {tracking_symbol} from {start_date.date()} to {end_date.date()}")
    total_days = (end_date - start_date).days

    for i in range(total_days):
        start_of_day = (start_date + dt.timedelta(days=i)).replace(tzinfo=dt.timezone.utc)  # datetime object
        end_of_day = (start_date + dt.timedelta(days=i + 1)).replace(tzinfo=dt.timezone.utc)  # end of the day
        gen = get_all_posts_by_interval(start_of_day, end_of_day)
        print(len(list(gen)))


plot_exposures()
