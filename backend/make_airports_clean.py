from pathlib import Path
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

RAW_PATH = DATA_DIR / "airports.csv"
OUT_PATH = DATA_DIR / "airports_clean.csv"


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw airports.csv not found at: {RAW_PATH}\n"
            f"Put your raw file here and rerun."
        )

    df = pd.read_csv(RAW_PATH)

    required = {"iata_code", "name", "municipality", "iso_country"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Your airports.csv does not look like OurAirports format.\n"
            f"Missing columns: {sorted(list(missing))}\n"
            f"Found: {list(df.columns)[:40]}"
        )

    out = pd.DataFrame()
    out["iata"] = df["iata_code"].astype(str).str.strip().str.upper()
    out["name"] = df["name"].astype(str).str.strip()
    out["city"] = df["municipality"].astype(str).str.strip()
    out["country_code"] = df["iso_country"].astype(str).str.strip().str.upper()

    # optional useful columns
    out["type"] = df.get("type", "")
    out["scheduled_service"] = df.get("scheduled_service", "")
    out["iso_region"] = df.get("iso_region", "")
    out["ident"] = df.get("ident", "")
    out["latitude_deg"] = df.get("latitude_deg", pd.NA)
    out["longitude_deg"] = df.get("longitude_deg", pd.NA)

    # keep valid IATA only
    out = out[out["iata"].str.match(r"^[A-Z]{3}$", na=False)].copy()

    # label for UI
    out["label"] = out.apply(
        lambda r: f"{r.get('city','') or 'Unknown city'} — {r.get('name','Unknown airport')} ({r.get('iata','')}) [{r.get('country_code','')}]",
        axis=1
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved clean file: {OUT_PATH} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
