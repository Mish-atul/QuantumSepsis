"""Generate real-data window tensors from processed parquet splits."""

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import get_default_config
from src.data.windowing import WindowGenerator


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = get_default_config()
    generator = WindowGenerator(config)

    cohort = pd.read_csv(
        "data/processed/cohort.csv",
        parse_dates=["intime", "outtime", "sepsis_onset_time"],
    )

    results = {}
    summaries = {}

    for split in ["train", "val", "test"]:
        features = pd.read_parquet(f"data/processed/{split}_features.parquet")
        split_stays = set(features["stay_id"].unique())
        split_cohort = cohort[cohort["stay_id"].isin(split_stays)]

        X, y, meta = generator.generate_windows(features, split_cohort, split)
        results[split] = (X, y, meta)

        summaries[split] = {
            "shape": [int(v) for v in X.shape],
            "positives": int(y.sum()),
            "positive_rate": float(y.mean()),
            "n_stays": int(meta["stay_id"].nunique()) if len(meta) else 0,
        }

    generator.save_to_hdf5(
        results["train"][0],
        results["train"][1],
        results["val"][0],
        results["val"][1],
        results["test"][0],
        results["test"][1],
        results["train"][2],
        results["val"][2],
        results["test"][2],
    )

    out = Path("data/processed/window_summary.json")
    out.write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()