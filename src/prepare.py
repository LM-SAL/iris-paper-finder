import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from enum import Enum


class Columns(Enum):
    BIBCODE = "bibcode"
    INSTRUMENTS = "instruments"
    PATH = "path"
    SOHO = "soho"
    TOPIC = "topic"


def get_path(b, already_have: dict):
    if isinstance(b, str):
        b = b.strip()
        if b in already_have:
            return already_have[b]
    return None


def get_already_have_bibs(bibcodes, soho_pdf_dir):
    count = 0
    already_have = {}
    for p in (pbar := tqdm(soho_pdf_dir.iterdir())):
        if p.stem in bibcodes:
            already_have[p.stem] = p
            count += 1
            pbar.set_description(f"Already have {count} papers.")
    return already_have


def format_instruments(instrument_str):
    if isinstance(instrument_str, str):
        instruments = list(set(s.strip() for s in instrument_str.split(",")))
        return instruments
    return []


def get_data_df(data_dir: Path, soho: int, topic: str):
    if soho not in [0, 1]:
        raise ValueError("Label for SOHO must be 0 or 1.")
    records = []
    for p in data_dir.iterdir():
        r = {
            Columns.BIBCODE.value: p.stem.strip(),
            Columns.INSTRUMENTS.value: [],
            Columns.PATH.value: p,
            Columns.SOHO.value: 0,
            Columns.TOPIC.value: topic,
        }
        records.append(r)
    df = pd.DataFrame(records)
    return df


def get_soho_df(in_multilabel, soho_pdf_dir):
    keep_cols = [Columns.INSTRUMENTS.value, Columns.BIBCODE.value, Columns.PATH.value]
    df = pd.read_csv(in_multilabel)
    df = df.drop(df.filter(regex="Unnamed").columns, axis=1)
    df = df.rename(
        columns={
            "INTRUMENT": Columns.INSTRUMENTS.value,
            "BIBCODE": Columns.BIBCODE.value,
        }
    )

    bibcodes = df[Columns.BIBCODE.value].str.strip().unique()
    already_have = get_already_have_bibs(bibcodes, soho_pdf_dir)
    df[Columns.PATH.value] = df[Columns.BIBCODE.value].apply(
        get_path, already_have=already_have
    )
    df[Columns.INSTRUMENTS.value] = df[Columns.INSTRUMENTS.value].apply(
        format_instruments
    )

    ah_df = df[~df[Columns.PATH.value].isna()].copy()
    ah_df = ah_df.loc[:, keep_cols]
    ah_df[Columns.SOHO.value] = 1
    ah_df[Columns.TOPIC.value] = "solar"
    return ah_df


def main(in_multilabel: Path, data_dir: Path, out_jsonl: Path, seed=42):
    print(f"Reading from {data_dir}")

    soho_pdf_dir = data_dir / "soho" / "pdf"
    not_soho_solar_pdf_dir = data_dir / "non_soho_solar" / "pdf"
    cosmology_pdf_dir = data_dir / "non_soho_cosmology" / "pdf"

    ah_df = get_soho_df(in_multilabel, soho_pdf_dir)
    not_soho_solar_df = get_data_df(not_soho_solar_pdf_dir, soho=0, topic="solar")
    cosmology_df = get_data_df(cosmology_pdf_dir, soho=0, topic="cosmology")

    num_soho = ah_df.shape[0]
    solar_df = not_soho_solar_df.sample(num_soho, random_state=seed)
    cosmo_df = cosmology_df.sample(num_soho, random_state=seed)

    combined_df = pd.concat([solar_df, cosmo_df, ah_df], ignore_index=True)
    combined_df[Columns.PATH.value] = combined_df[Columns.PATH.value].apply(
        lambda x: str(x)
    )
    print(f"Writing to {out_jsonl}")
    combined_df.to_json(out_jsonl, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for testing.")
    parser.add_argument(
        "--in_multilabel", type=Path, required=True, help="CSV with labeled data"
    )
    parser.add_argument(
        "--data_dir", type=Path, required=True, help="Directory containing the data"
    )
    parser.add_argument(
        "--out_jsonl", type=Path, required=True, help="Path to output jsonlines file"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    args = parser.parse_args()
    main(args.in_multilabel, args.data_dir, args.out_jsonl, args.seed)
