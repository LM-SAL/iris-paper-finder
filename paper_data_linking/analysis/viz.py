import argparse
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from paper_data_linking.data.download_text import read_metadata
from paper_data_linking.enums import Locations

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

sns.set_theme(style="whitegrid")


def plot_pub_open(df, n=None, log_scale=False, failure=False):
    if failure:
        gb = df.groupby(["pub", "failure_type"])
    else:
        gb = df.groupby("pub")
    open_df = gb.aggregate({"is_open": "sum"})
    open_df["total"] = gb['is_open'].count()
    top_pubs_df = open_df.sort_values("total", ascending=False).reset_index()

    if n is not None:
        top_pubs_df = top_pubs_df[0:n]

    f, ax = plt.subplots()

    sns.set_color_codes("pastel")
    sns.barplot(x="total", y="pub", data=top_pubs_df,
                label="Total", color="b")

    sns.set_color_codes("muted")
    sns.barplot(x="is_open", y="pub", data=top_pubs_df,
                label="Open", color="b")

    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(ylabel="",
           xlabel="SOHO publications")
    if log_scale:
        ax.set_xscale("log")
    sns.despine(left=True, bottom=True)
    return f


def main():
    df = read_metadata()
    open_articles = df['is_open'].sum()
    total_articles = df['is_open'].shape[0]
    frac = open_articles / total_articles
    LOG.info(f"{open_articles}/{total_articles}={frac}")

    fig = plot_pub_open(df, n=15, log_scale=False)
    # plot labels are currently being cut off.
    LOG.info(f"Writing plot to {Locations.PUB_PLOT.value}")
    fig.savefig(Locations.PUB_PLOT.value)


if __name__ == "__main__":
    main()
