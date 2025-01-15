import logging

import pandas as pd
import seaborn as sns
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from enums import Locations, DataFields

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def read_texts_to_df():
    records = []
    for l in tqdm(Locations.TEXTS.value.iterdir()):
        with open(l, 'r') as f0:
            d = {
                DataFields.BIBCODE.value: l.stem,
                DataFields.TEXT.value: f0.read().strip(),
            }
            records.append(d)
    df = pd.DataFrame(records)
    return df


def count_tokens(txt):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # using Fast because specificity not important here
    # only using this for approximate token counts, not for IDs
    tokens = tokenizer(txt)['input_ids']
    num_tokens = len(tokens)
    return num_tokens


def no_captcha(txt):
    if "ShieldSquare Captcha" in txt:
        return False
    else:
        return True


def get_df_with_counts():
    df = read_texts_to_df()
    df[DataFields.TOKEN_COUNT.value] = (
        df[DataFields.TEXT.value].apply(count_tokens)
    )
    df.to_csv(Locations.COUNT_DF.value)
    return df


def load_counts_df():
    df = pd.read_csv(Locations.COUNT_DF.value, index_col=0)
    return df


def main():
    # 1. df = get_df_with_counts()
    df = load_counts_df()
    df[DataFields.VALID.value] = df[DataFields.TEXT.value].apply(no_captcha)
    df = df.loc[df[DataFields.VALID.value]]
    LOG.info(f"{len(df)} valid papers")

    mn = df[DataFields.TOKEN_COUNT.value].mean()
    LOG.info(f"Mean document token number: {mn}")
    val = mn * 6000 * (0.02 / 1000)
    LOG.info(f"Expected cost for 6000 documents: ${val:.2f}")

    plot = sns.displot(df, x=DataFields.TOKEN_COUNT.value)
    plot.savefig(Locations.COUNT_PLOT.value)


if __name__ == "__main__":
    main()
