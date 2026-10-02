import argparse, sys
from pathlib import Path

import pandas as pd


def main(raw_path: Path) -> None:
    df = pd.read_parquet(raw_path)
    expect_cols = {"final_state", "n_3_1_8", "n_4_1_8"}
    missing = expect_cols - set(df.columns)
    if missing:
        sys.exit(
            f"Ошибка: в df_raw.parquet нет колонок {missing}.\n"
            "Пересчитайте data_load.py v4, чтобы появился n_3_1_8 и n_4_1_8."
        )

    total = len(df)
    print(df["n_3_1_8"])
    print(df["n_4_1_8"])
    cond_A = ~df["final_state"].isin(["отчислен", "академический отпуск"])
    cond_B = df["n_3_1_8"] == 0
    cond_C = df["n_4_1_8"] < 15

    print(f"\nВсего студентов: {total:,}\n")
    for name, cond in {
        "A. not expelled / not academic leave": cond_A,
        "B. no 3s (n_3_1_8 < 5)":            cond_B,
        "C. < 25 fours (n_4_1_8 < 15)":        cond_C,
    }.items():
        n = cond.sum()
        print(f"{name:<40}  {n:>7,}  ({n/total:6.2%})")

    both_AB = cond_A & cond_B
    all_ABC = cond_A & cond_B & cond_C
    print("\nКомбинации:")
    print(f"A & B                         {both_AB.sum():>7,}  ({both_AB.mean():6.2%})")
    print(
        f"A & B & C (строгий success)   "
        f"{all_ABC.sum():>7,}  ({all_ABC.mean():6.2%})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--raw", "-r",
        default="data/df_raw.parquet",
        help="путь до df_raw.parquet"
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        pass

    raw_path = Path(args.raw)
    main(raw_path)