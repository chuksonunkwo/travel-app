# backend/app.py
import os
import re
import json
from pathlib import Path
from datetime import datetime, date, timedelta
from urllib.parse import quote

import pandas as pd
import streamlit as st

# Optional deps (do not break app if missing)
try:
    import requests  # for Airtable
except Exception:
    requests = None


# =========================
# Config
# =========================
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

CANDIDATE_CSVS = [
    DATA_DIR / "airports_clean.csv",
    DATA_DIR / "airports.csv",
]

MIN_CHARS = 2
DEFAULT_LIMIT = 30

# local fallback storage (dev only; in production prefer Airtable)
BOOKING_STORE_PATH = DATA_DIR / "booking_requests.jsonl"


# =========================
# Small utilities
# =========================
def _safe_get_secrets(key: str, default=None):
    """
    IMPORTANT: accessing st.secrets can raise StreamlitSecretNotFoundError
    if there is no secrets.toml. We must never break the app because of it.
    """
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def get_config(key: str, default=None):
    """
    Prefer env vars (Render-friendly), fall back to st.secrets (optional).
    """
    v = os.getenv(key)
    if v is not None and str(v).strip() != "":
        return v
    return _safe_get_secrets(key, default)


def normalize_text(x: str) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def file_exists(p: Path) -> bool:
    try:
        return p.exists() and p.is_file()
    except Exception:
        return False


def find_airports_csv() -> Path:
    checked = []
    for p in CANDIDATE_CSVS:
        checked.append(str(p))
        if file_exists(p):
            return p
    raise FileNotFoundError("Airports CSV not found. Checked:\n" + "\n".join(checked))


def standardize_airports_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supports:
      A) clean format: iata, name, city, country_code, ...
      B) OurAirports raw: iata_code, municipality, iso_country, ...
    Produces canonical:
      iata, name, city, country_code, type, scheduled_service, iso_region, ident,
      latitude_deg, longitude_deg, label, search_*
    """
    cols = set(df.columns)

    if "iata" in cols:
        out = df.copy()
        if "country_code" not in out.columns and "iso_country" in out.columns:
            out["country_code"] = out["iso_country"]
        if "city" not in out.columns and "municipality" in out.columns:
            out["city"] = out["municipality"]
        if "iata_code" in out.columns and "iata" not in out.columns:
            out["iata"] = out["iata_code"]

    elif "iata_code" in cols:
        out = pd.DataFrame()
        out["iata"] = df.get("iata_code", "")
        out["name"] = df.get("name", "")
        out["city"] = df.get("municipality", "")
        out["country_code"] = df.get("iso_country", "")
        out["type"] = df.get("type", "")
        out["scheduled_service"] = df.get("scheduled_service", "")
        out["iso_region"] = df.get("iso_region", "")
        out["ident"] = df.get("ident", "")
        out["latitude_deg"] = df.get("latitude_deg", pd.NA)
        out["longitude_deg"] = df.get("longitude_deg", pd.NA)

    else:
        raise ValueError(f"Unsupported airports CSV format. Found columns: {list(df.columns)[:40]}")

    for c in [
        "iata",
        "name",
        "city",
        "country_code",
        "type",
        "scheduled_service",
        "iso_region",
        "ident",
        "latitude_deg",
        "longitude_deg",
    ]:
        if c not in out.columns:
            out[c] = ""

    out["iata"] = out["iata"].astype(str).str.strip().str.upper()
    out["name"] = out["name"].astype(str).str.strip()
    out["city"] = out["city"].astype(str).str.strip()
    out["country_code"] = out["country_code"].astype(str).str.strip().str.upper()
    out["type"] = out["type"].astype(str).str.strip()
    out["scheduled_service"] = out["scheduled_service"].astype(str).str.strip()
    out["iso_region"] = out["iso_region"].astype(str).str.strip()
    out["ident"] = out["ident"].astype(str).str.strip()

    # Keep valid IATA only: exactly 3 letters
    out = out[out["iata"].str.match(r"^[A-Z]{3}$", na=False)].copy()

    out["label"] = out.apply(
        lambda r: (
            f"{r.get('city','') or 'Unknown city'} — "
            f"{r.get('name','Unknown airport')} "
            f"({r.get('iata','')}) [{r.get('country_code','')}]"
        ),
        axis=1,
    )

    out["search_iata"] = out["iata"].map(normalize_text)
    out["search_name"] = out["name"].map(normalize_text)
    out["search_city"] = out["city"].map(normalize_text)
    out["search_country"] = out["country_code"].map(normalize_text)

    return out.reset_index(drop=True)


def score_row(row: pd.Series, q: str) -> int:
    qi = q.strip().lower()
    if not qi:
        return 0

    iata = row.get("search_iata", "")
    name = row.get("search_name", "")
    city = row.get("search_city", "")
    country = row.get("search_country", "")

    score = 0
    if qi == iata:
        score += 100
    if iata.startswith(qi):
        score += 60
    if city.startswith(qi):
        score += 40
    if name.startswith(qi):
        score += 30
    if qi in city:
        score += 25
    if qi in name:
        score += 15
    if qi in country:
        score += 5

    for t in qi.split():
        if t and t in city:
            score += 8
        if t and t in name:
            score += 5

    return score


def rank_airports(df: pd.DataFrame, query: str, limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
    q = normalize_text(query)
    if not q:
        return df.head(0).copy()

    scored = df.copy()
    scored["score"] = scored.apply(lambda r: score_row(r, q), axis=1)
    scored = scored[scored["score"] > 0].sort_values(["score", "iata"], ascending=[False, True])
    return scored.head(limit).reset_index(drop=True)


def clean_filter_values(series: pd.Series) -> list:
    if series is None or series.empty:
        return []
    vals = []
    for v in series.dropna().tolist():
        s = str(v).strip()
        if not s or s.lower() == "nan":
            continue
        vals.append(s)
    return sorted(set(vals))


def airport_payload(row: pd.Series) -> dict:
    return {
        "iata": row.get("iata", ""),
        "airport_name": row.get("name", ""),
        "city": row.get("city", ""),
        "country_code": row.get("country_code", ""),
        "type": row.get("type", ""),
        "scheduled_service": row.get("scheduled_service", ""),
        "iso_region": row.get("iso_region", ""),
        "ident": row.get("ident", ""),
        "latitude_deg": row.get("latitude_deg", None),
        "longitude_deg": row.get("longitude_deg", None),
        "label": row.get("label", ""),
    }


def whatsapp_link(message: str, phone_e164: str = "") -> str:
    encoded = quote(message)
    phone = re.sub(r"[^\d+]", "", phone_e164).replace("+", "")
    if phone:
        return f"https://wa.me/{phone}?text={encoded}"
    return f"https://wa.me/?text={encoded}"


def append_local_jsonl(payload: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False)
    with open(BOOKING_STORE_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def airtable_enabled() -> bool:
    return bool(get_config("AIRTABLE_API_KEY")) and bool(get_config("AIRTABLE_BASE_ID")) and bool(get_config("AIRTABLE_TABLE"))


def save_to_airtable(payload: dict) -> tuple[bool, str]:
    """
    Saves payload to Airtable if configured.
    Returns (ok, message).
    """
    if requests is None:
        return False, "requests not installed. Install it or use local storage."
    api_key = get_config("AIRTABLE_API_KEY")
    base_id = get_config("AIRTABLE_BASE_ID")
    table = get_config("AIRTABLE_TABLE")
    if not api_key or not base_id or not table:
        return False, "Airtable not configured."

    url = f"https://api.airtable.com/v0/{base_id}/{quote(str(table))}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    passengers = payload.get("passengers", [])
    passenger_names = ", ".join([p.get("name", "").strip() for p in passengers if p.get("name", "").strip()])

    fields = {
        "created_at": payload.get("created_at", ""),
        "origin_iata": payload.get("origin", {}).get("iata", ""),
        "destination_iata": payload.get("destination", {}).get("iata", ""),
        "origin_city": payload.get("origin", {}).get("city", ""),
        "destination_city": payload.get("destination", {}).get("city", ""),
        "trip_type": payload.get("trip_type", ""),
        "depart_date": payload.get("depart_date", ""),
        "return_date": payload.get("return_date", ""),
        "pax": payload.get("pax", 1),
        "cabin": payload.get("cabin", ""),
        "passenger_names": passenger_names,
        "budget": payload.get("budget", ""),
        "airline_pref": payload.get("airline_pref", ""),
        "flexible_dates": payload.get("flexible_dates", ""),
        "notes": payload.get("notes", ""),
        "status": payload.get("status", "NEW"),
        "handoff_message": payload.get("handoff_message", ""),
        "raw_json": json.dumps(payload, ensure_ascii=False),
    }

    body = {"records": [{"fields": fields}]}

    try:
        r = requests.post(url, headers=headers, json=body, timeout=20)
        if 200 <= r.status_code < 300:
            return True, "Saved to Airtable."
        return False, f"Airtable error {r.status_code}: {r.text[:500]}"
    except Exception as e:
        return False, f"Airtable request failed: {e}"


def build_handoff_message(payload: dict) -> str:
    """
    Professional + WhatsApp-safe (emojis but clean).
    """
    origin = payload.get("origin", {})
    dest = payload.get("destination", {})
    trip_type = payload.get("trip_type", "")
    depart_date = payload.get("depart_date", "")
    return_date = payload.get("return_date", "")

    pax = payload.get("pax", 1)
    cabin = payload.get("cabin", "")

    lines = [
        "🛫 *Travel App — Flight Booking Request*",
        "",
        f"📍 *Origin:* {origin.get('city','')} ({origin.get('iata','')})",
        f"🎯 *Destination:* {dest.get('city','')} ({dest.get('iata','')})",
        "",
        f"🧳 *Passengers:* {pax}",
        f"💺 *Cabin:* {cabin}",
        f"🔁 *Trip:* {trip_type}",
        f"📅 *Departure:* {depart_date}",
    ]
    if trip_type == "Return":
        lines.append(f"📅 *Return:* {return_date or '-'}")

    flex = payload.get("flexible_dates", "")
    if flex:
        lines += ["", f"📆 *Flexible:* {flex}"]

    airline = payload.get("airline_pref", "")
    if airline:
        lines += ["", f"✈️ *Airline preference:* {airline}"]

    budget = payload.get("budget", "")
    if budget:
        lines += ["", f"💰 *Budget:* {budget}"]

    passengers = payload.get("passengers", [])
    if passengers:
        lines += ["", "👥 *Passenger details:*"]
        for i, p in enumerate(passengers, start=1):
            nm = p.get("name", "").strip() or f"Passenger {i}"
            ph = p.get("whatsapp", "").strip()
            if ph:
                lines.append(f"  • {nm} — {ph}")
            else:
                lines.append(f"  • {nm}")

    notes = payload.get("notes", "")
    if notes:
        lines += ["", f"📝 *Notes:* {notes}"]

    lines += ["", "✅ Please share best available options + total price.", "🙏 Thank you!"]
    return "\n".join(lines)


def create_booking_pdf_bytes(booking_payload: dict) -> bytes | None:
    """
    Creates a professional PDF summary.
    Requires reportlab. If missing, returns None (no breakage).
    """
    try:
        from io import BytesIO
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
    except Exception:
        return None

    origin = booking_payload.get("origin", {})
    dest = booking_payload.get("destination", {})
    passengers = booking_payload.get("passengers", [])

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    title = Paragraph("<b>Travel App — Booking Request Summary</b>", styles["Title"])
    created = booking_payload.get("created_at", "")

    def p(txt: str):
        return Paragraph(txt, styles["BodyText"])

    elements = [
        title,
        Spacer(1, 10),
        p(f"<b>Created:</b> {created}"),
        Spacer(1, 10),
        p(f"<b>Origin:</b> {origin.get('city','')} ({origin.get('iata','')}) — {origin.get('airport_name','')}"),
        p(f"<b>Destination:</b> {dest.get('city','')} ({dest.get('iata','')}) — {dest.get('airport_name','')}"),
        Spacer(1, 12),
    ]

    rows = [
        ["Passengers", str(booking_payload.get("pax", ""))],
        ["Cabin", booking_payload.get("cabin", "")],
        ["Trip type", booking_payload.get("trip_type", "")],
        ["Departure", booking_payload.get("depart_date", "")],
        ["Return", booking_payload.get("return_date", "") or "-"],
        ["Flexible", booking_payload.get("flexible_dates", "") or "-"],
        ["Airline preference", booking_payload.get("airline_pref", "") or "-"],
        ["Budget", booking_payload.get("budget", "") or "-"],
        ["Notes", booking_payload.get("notes", "") or "-"],
        ["Status", booking_payload.get("status", "")],
    ]

    if passengers:
        passenger_lines = []
        for i, psg in enumerate(passengers, start=1):
            nm = (psg.get("name", "") or f"Passenger {i}").strip()
            wa = (psg.get("whatsapp", "") or "").strip()
            passenger_lines.append(f"{i}. {nm}" + (f" ({wa})" if wa else ""))
        rows.insert(2, ["Passenger details", "\n".join(passenger_lines)])

    table = Table(rows, colWidths=[140, 360])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#FAFAFA")]),
            ]
        )
    )
    elements += [table, Spacer(1, 14)]

    elements += [p("<b>WhatsApp / Message Handoff</b>"), Spacer(1, 6)]
    msg = booking_payload.get("handoff_message", "").replace("\n", "<br/>")
    elements += [p(msg)]

    doc.build(elements)
    return buf.getvalue()


# =========================
# Airline “logos” (SVG badges)
# - avoids blurry external images
# - avoids logo mix-ups due to caching/URL issues
# =========================
AIRLINE_STYLES = {
    "KLM": {"a": "#0ea5e9", "b": "#2563eb"},
    "Virgin Atlantic": {"a": "#f43f5e", "b": "#7c3aed"},
    "Lufthansa": {"a": "#0f172a", "b": "#2563eb"},
    "Emirates": {"a": "#ef4444", "b": "#b91c1c"},
    "SkyWays": {"a": "#22c55e", "b": "#14b8a6"},
    "AeroLink": {"a": "#a855f7", "b": "#6366f1"},
    "CloudAir": {"a": "#06b6d4", "b": "#3b82f6"},
    "NovaJet": {"a": "#f59e0b", "b": "#ef4444"},
}


def airline_badge_svg(airline: str, width: int = 140, height: int = 44) -> str:
    style = AIRLINE_STYLES.get(airline, {"a": "#111827", "b": "#374151"})
    a, b = style["a"], style["b"]
    initials = "".join([w[0] for w in airline.split() if w])[:3].upper() or "AIR"
    safe_airline = (airline or "Airline").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    return f"""
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{safe_airline}">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="{a}"/>
          <stop offset="100%" stop-color="{b}"/>
        </linearGradient>
        <filter id="s" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#000000" flood-opacity="0.18"/>
        </filter>
      </defs>
      <rect x="0" y="0" rx="12" ry="12" width="{width}" height="{height}" fill="url(#g)" filter="url(#s)"/>
      <text x="14" y="28" font-family="Inter, Arial, sans-serif" font-size="16" font-weight="700" fill="#ffffff">{initials}</text>
      <text x="56" y="28" font-family="Inter, Arial, sans-serif" font-size="14" font-weight="600" fill="#ffffff">{safe_airline}</text>
    </svg>
    """


def mock_flight_search_results(
    origin_iata: str,
    dest_iata: str,
    depart_date: date,
    pax: int,
    cabin: str,
    trip_type: str,
    flex_days: int = 0,
) -> list[dict]:
    """
    Non-breaking placeholder results (no external API).
    Returns list of dict cards so we can render badges nicely.
    """
    seed = sum(ord(c) for c in (origin_iata + dest_iata)) + int(depart_date.strftime("%Y%m%d")) + pax + flex_days
    base = 180 + (seed % 220)
    cabin_mult = {"Economy": 1.0, "Premium Economy": 1.35, "Business": 2.3, "First": 3.2}.get(cabin, 1.0)
    trip_mult = 1.0 if trip_type == "One-way" else 1.75

    def price(i: int) -> int:
        return int((base + i * 35) * cabin_mult * trip_mult)

    airlines = ["KLM", "Virgin Atlantic", "Lufthansa", "Emirates"]
    stops = ["Non-stop", "1 stop", "1 stop", "2 stops"]
    durations = ["6h 20m", "8h 05m", "9h 10m", "11h 40m"]
    notes = ["Best overall", "Value option", "Flexible change", "Lowest fare"]

    cards = []
    for i in range(4):
        cards.append(
            {
                "airline": airlines[i],
                "stops": stops[i],
                "duration": durations[i],
                "price": f"${price(i)}",
                "note": notes[i],
            }
        )
    return cards


# =========================
# Data Load (cached)
# =========================
@st.cache_data(show_spinner=False)
def load_airports() -> pd.DataFrame:
    csv_path = find_airports_csv()
    df_raw = pd.read_csv(csv_path)
    return standardize_airports_df(df_raw)


# =========================
# Theme (System default)
# =========================
def apply_theme(mode: str):
    """
    System: do nothing (Streamlit follows environment / browser).
    Light/Dark: apply minimal CSS overrides.
    """
    if mode == "System":
        return

    if mode == "Light":
        st.markdown(
            """
            <style>
              html, body, [class*="stApp"] { background: #ffffff !important; }
              [data-testid="stSidebar"] { background: #f6f7fb !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        return

    if mode == "Dark":
        # "very black" but only when user explicitly selects Dark
        st.markdown(
            """
            <style>
              html, body, [class*="stApp"] { background: #070707 !important; color: #ffffff !important; }
              [data-testid="stSidebar"] { background: #0b0b0b !important; }
              .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
                background: #0f0f0f !important; color: #ffffff !important;
              }
            </style>
            """,
            unsafe_allow_html=True,
        )
        return


# =========================
# UI
# =========================
st.set_page_config(page_title="Travel App", page_icon="🧭", layout="wide")

st.markdown(
    """
    <style>
      .block-container { max-width: 1200px; padding-top: 1.4rem; }
      .card {
        border: 1px solid rgba(17,24,39,.08);
        border-radius: 16px;
        padding: 14px 16px;
        background: #ffffff;
        box-shadow: 0 6px 18px rgba(17,24,39,.06);
      }
      .muted { color: rgba(17,24,39,.65); }
      .hstack { display:flex; align-items:center; gap:12px; }
      .pill {
        display:inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(17,24,39,.06);
        font-weight: 600;
        font-size: 0.85rem;
      }
      .section-title { font-size: 1.2rem; font-weight: 800; margin: 0.25rem 0 0.75rem; }
      .flight-card {
        border: 1px solid rgba(17,24,39,.10);
        border-radius: 16px;
        padding: 14px 14px;
        background: #ffffff;
        box-shadow: 0 4px 14px rgba(17,24,39,.05);
        margin-bottom: 12px;
      }
      .flight-kv { color: rgba(17,24,39,.75); font-size: 0.95rem; }
      .price { font-size: 1.35rem; font-weight: 900; }
      .divider { height: 1px; background: rgba(17,24,39,.10); margin: 10px 0; }
      .highlight {
        border-left: 6px solid rgba(59,130,246,.75);
        padding-left: 12px;
        margin-top: 8px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar: theme + filters
st.sidebar.header("Settings")
theme_mode = st.sidebar.selectbox("Theme", ["System", "Light", "Dark"], index=0, help="System follows your OS/browser.")
apply_theme(theme_mode)

st.sidebar.markdown("---")
st.sidebar.header("Filters")

# Load airports
try:
    airports = load_airports()
except Exception as e:
    st.error("Could not load airports database.")
    st.code(str(e))
    st.stop()

# Filters (robust; never greyed out)
types = clean_filter_values(airports.get("type", pd.Series(dtype=str)))
countries = clean_filter_values(airports.get("country_code", pd.Series(dtype=str)))
scheduled = clean_filter_values(airports.get("scheduled_service", pd.Series(dtype=str)))

default_types = [t for t in types if t in ("large_airport", "medium_airport", "small_airport")]
if not default_types and types:
    default_types = types[:]

selected_types = st.sidebar.multiselect("Airport type", options=types, default=default_types) if types else []
if not types:
    st.sidebar.info("Airport types not available in this CSV.")

selected_countries = st.sidebar.multiselect("Country (ISO)", options=countries, default=[]) if countries else []
if not countries:
    st.sidebar.info("Country codes not available in this CSV.")

selected_scheduled = st.sidebar.multiselect("Scheduled service", options=scheduled, default=[]) if scheduled else []
if not scheduled:
    st.sidebar.info("Scheduled service not available in this CSV.")

limit = st.sidebar.slider("Max suggestions", min_value=10, max_value=100, value=DEFAULT_LIMIT, step=5)

# Apply filters
filtered = airports.copy()
if selected_types:
    filtered = filtered[filtered["type"].isin(selected_types)]
if selected_countries:
    filtered = filtered[filtered["country_code"].isin(selected_countries)]
if selected_scheduled:
    filtered = filtered[filtered["scheduled_service"].isin(selected_scheduled)]
filtered = filtered.reset_index(drop=True)

# Header
st.markdown(
    """
    <div class="card">
      <div style="font-size:2.0rem;font-weight:900;margin:0;">Travel App</div>
      <div class="muted" style="margin-top:4px;">
        Select origin and destination, then search flights and create a WhatsApp/Email booking handoff.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Session state init
defaults = {
    "origin_airport": None,
    "dest_airport": None,
    "origin_q": "",
    "dest_q": "",
    "selector_nonce": 0,
    # Persist results so they remain visible after booking:
    "last_search_cards": None,
    "last_search_meta": None,
    "last_selected_airline": "",
    # Passenger details: list of dicts
    "passengers": [{"name": "", "whatsapp": ""}],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def ensure_passenger_list_length(n: int):
    n = max(1, int(n))
    cur = st.session_state.get("passengers", [])
    if not isinstance(cur, list):
        cur = []
    while len(cur) < n:
        cur.append({"name": "", "whatsapp": ""})
    while len(cur) > n:
        cur.pop()
    st.session_state["passengers"] = cur


def airport_selector_block(title: str, key_prefix: str):
    st.subheader(title)

    q_key = f"{key_prefix}_q"
    sel_key = f"{key_prefix}_select_{st.session_state['selector_nonce']}"

    q = st.text_input(
        "Search (airport / city / IATA)",
        key=q_key,
        placeholder="e.g., lagos, LOS, heathrow, london...",
    )

    if len(q.strip()) < MIN_CHARS:
        st.info(f"Type at least {MIN_CHARS} characters to see suggestions.")
        return None

    results = rank_airports(filtered, q, limit=limit)
    if results.empty:
        st.warning("No matches. Try city name, airport name, or a 3-letter IATA code.")
        return None

    options = results["label"].tolist()
    sel = st.selectbox("Choose an airport", options=options, key=sel_key)

    selected_rows = results[results["label"] == sel]
    if selected_rows.empty:
        st.warning("Selection not found. Try again.")
        return None

    row = selected_rows.iloc[0]
    payload = airport_payload(row)

    st.success(f"Selected: {payload['iata']}")
    st.markdown(
        f"""
        <div class="hstack">
          <span class="pill">IATA: {payload['iata']}</span>
          <span class="pill">City: {payload['city']}</span>
          <span class="pill">Country: {payload['country_code']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Details (JSON)"):
        st.json(payload)

    return payload


# Airport selectors
colA, colB = st.columns(2, gap="large")
with colA:
    origin = airport_selector_block("Origin", "origin")
    if origin:
        st.session_state["origin_airport"] = origin

with colB:
    dest = airport_selector_block("Destination", "dest")
    if dest:
        st.session_state["dest_airport"] = dest

# Swap / Clear (safe: nonce bumps so widgets rebuild and never break)
btn1, btn2, btn3 = st.columns([1, 1, 2])
with btn1:
    if st.button("🔁 Swap origin/destination"):
        st.session_state["origin_airport"], st.session_state["dest_airport"] = (
            st.session_state["dest_airport"],
            st.session_state["origin_airport"],
        )

        oa = st.session_state["origin_airport"] or {}
        da = st.session_state["dest_airport"] or {}
        st.session_state["origin_q"] = (oa.get("iata") or "").lower()
        st.session_state["dest_q"] = (da.get("iata") or "").lower()

        st.session_state["selector_nonce"] += 1
        st.toast("Swapped origin and destination.")
        st.rerun()

with btn2:
    if st.button("🧹 Clear selections"):
        st.session_state["origin_airport"] = None
        st.session_state["dest_airport"] = None
        st.session_state["origin_q"] = ""
        st.session_state["dest_q"] = ""
        st.session_state["selector_nonce"] += 1
        st.toast("Cleared selections.")
        st.rerun()

with btn3:
    st.caption("Tip: select both airports, then set trip details below.")

st.divider()

# Trip details (NOT a form) -> return date shows immediately
st.markdown('<div class="highlight"><div class="section-title">Trip details</div></div>', unsafe_allow_html=True)

origin_airport = st.session_state.get("origin_airport")
dest_airport = st.session_state.get("dest_airport")

if not origin_airport or not dest_airport:
    st.warning("Select both origin and destination airports to continue.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
with c1:
    trip_type = st.selectbox("Trip type", ["One-way", "Return"], index=1)
with c2:
    depart_date = st.date_input("Departure date", value=date.today())
with c3:
    return_date = None
    if trip_type == "Return":
        return_date = st.date_input("Return date", value=date.today())
    else:
        st.caption("Return date not required for One-way.")
with c4:
    cabin = st.selectbox("Cabin", ["Economy", "Premium Economy", "Business", "First"], index=0)

c5, c6, c7, c8 = st.columns(4)
with c5:
    pax = st.number_input("Passengers", min_value=1, max_value=9, value=1, step=1)
with c6:
    flexible = st.checkbox("Flexible dates (±3 days)", value=False)
with c7:
    budget = st.text_input("Budget (optional)", placeholder="e.g., $800 max / flexible")
with c8:
    airline_pref = st.text_input("Airline preference (optional)", placeholder="e.g., KLM, Virgin, Lufthansa")

ensure_passenger_list_length(int(pax))

with st.expander("Passenger names & WhatsApp (per passenger)", expanded=True):
    for i in range(int(pax)):
        p = st.session_state["passengers"][i]
        cc1, cc2 = st.columns([1, 1])
        with cc1:
            p["name"] = st.text_input(f"Passenger {i+1} name", value=p.get("name", ""), key=f"pax_name_{i}")
        with cc2:
            p["whatsapp"] = st.text_input(
                f"Passenger {i+1} WhatsApp (optional)",
                value=p.get("whatsapp", ""),
                placeholder="+234xxxxxxxxxx",
                key=f"pax_wa_{i}",
            )
        st.session_state["passengers"][i] = p

notes = st.text_area("Notes / preferences (optional)", placeholder="Baggage, time preference, avoid long layovers, etc.")
phone_e164 = st.text_input("Send WhatsApp to (optional)", placeholder="+234xxxxxxxxxx (agent or yourself)")
email_to = st.text_input("Email (optional)", placeholder="agent@company.com or your email")

# Validation
if trip_type == "Return" and return_date and return_date < depart_date:
    st.error("Return date cannot be earlier than departure date.")
    st.stop()

st.divider()

# Buttons
b1, b2 = st.columns([1, 1])
do_search = b1.button("🔎 Search flights", use_container_width=True)
do_booking = b2.button("✅ Create booking handoff", use_container_width=True)

# Search handler: persist results in session_state
if do_search:
    flex_days = 3 if flexible else 0

    cards = mock_flight_search_results(
        origin_iata=origin_airport["iata"],
        dest_iata=dest_airport["iata"],
        depart_date=depart_date,
        pax=int(pax),
        cabin=cabin,
        trip_type=trip_type,
        flex_days=flex_days,
    )

    st.session_state["last_search_cards"] = cards
    st.session_state["last_search_meta"] = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "origin": origin_airport["iata"],
        "dest": dest_airport["iata"],
        "depart_date": str(depart_date),
        "return_date": str(return_date) if (trip_type == "Return" and return_date) else "",
        "trip_type": trip_type,
        "cabin": cabin,
        "pax": int(pax),
        "flexible": bool(flexible),
    }

# Flight results (render whenever we have last_search_cards)
if st.session_state.get("last_search_cards"):
    meta = st.session_state.get("last_search_meta") or {}
    st.markdown('<div class="highlight"><div class="section-title">Flight results</div></div>', unsafe_allow_html=True)

    if meta.get("flexible"):
        st.info("Showing flexible-date estimates (±3 days). Live providers will return real date variants.")
    else:
        st.info("Showing mock results (no external API). Live booking comes next.")

    # Render cards (badge SVG prevents logo mix-ups)
    cards = st.session_state["last_search_cards"]
    for idx, r in enumerate(cards):
        with st.container():
            st.markdown('<div class="flight-card">', unsafe_allow_html=True)
            left, mid, right = st.columns([1.2, 2.2, 1.1])

            with left:
                svg = airline_badge_svg(r["airline"], width=180, height=50)
                st.markdown(svg, unsafe_allow_html=True)

            with mid:
                st.markdown(f"**{r['airline']}**")
                st.markdown(
                    f"<div class='flight-kv'>"
                    f"<b>Stops:</b> {r['stops']} &nbsp; • &nbsp; "
                    f"<b>Duration:</b> {r['duration']} &nbsp; • &nbsp; "
                    f"<b>Note:</b> {r['note']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with right:
                st.markdown(f"<div class='price'>{r['price']}</div>", unsafe_allow_html=True)
                if st.button(
                    f"Select",
                    key=f"sel_air_{idx}_{r['airline']}_{meta.get('ts','')}",
                    help="Mock selection (live booking next).",
                    use_container_width=True,
                ):
                    st.session_state["last_selected_airline"] = r["airline"]
                    st.toast(f"Selected {r['airline']}")

            st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# Booking handoff (always below flight results; flight results remain visible)
st.markdown('<div class="highlight"><div class="section-title">Booking handoff</div></div>', unsafe_allow_html=True)

if do_booking:
    flex_label = "±3 days" if flexible else ""
    passengers_clean = []
    for i in range(int(pax)):
        p = st.session_state["passengers"][i]
        passengers_clean.append(
            {
                "name": (p.get("name") or "").strip(),
                "whatsapp": (p.get("whatsapp") or "").strip(),
            }
        )

    booking_payload = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "origin": origin_airport,
        "destination": dest_airport,
        "pax": int(pax),
        "cabin": cabin,
        "trip_type": trip_type,
        "depart_date": str(depart_date),
        "return_date": str(return_date) if (trip_type == "Return" and return_date) else "",
        "flexible_dates": flex_label,
        "budget": budget.strip(),
        "airline_pref": airline_pref.strip() or st.session_state.get("last_selected_airline", ""),
        "notes": notes.strip(),
        "passengers": passengers_clean,
        "status": "NEW",
    }

    # Message
    handoff_message = build_handoff_message(booking_payload)
    booking_payload["handoff_message"] = handoff_message

    # Save: Airtable if configured, else local JSONL
    saved_msg = ""
    if airtable_enabled():
        ok, msg = save_to_airtable(booking_payload)
        if ok:
            st.success(msg)
        else:
            st.warning("Airtable configured but save failed; falling back to local file.")
            st.code(msg)
            try:
                append_local_jsonl(booking_payload)
                st.success("Saved locally as fallback.")
                saved_msg = f"Saved to local file: {BOOKING_STORE_PATH}"
            except Exception as e:
                st.error("Local save failed.")
                st.code(str(e))
    else:
        try:
            append_local_jsonl(booking_payload)
            st.success("Booking request created (stored locally).")
            saved_msg = f"Saved to local file: {BOOKING_STORE_PATH}"
        except Exception as e:
            st.error("Could not save booking request locally.")
            st.code(str(e))

    if saved_msg:
        st.caption(saved_msg)

    # Professional summary (instead of just JSON)
    o = booking_payload["origin"]
    d = booking_payload["destination"]
    st.markdown("### Booking summary")
    st.write(
        f"**Route:** {o.get('city','')} ({o.get('iata','')}) → {d.get('city','')} ({d.get('iata','')})\n\n"
        f"**Trip:** {trip_type}  •  **Cabin:** {cabin}  •  **Passengers:** {int(pax)}\n\n"
        f"**Departure:** {booking_payload['depart_date']}  •  **Return:** {booking_payload['return_date'] or '-'}  •  **Flexible:** {booking_payload['flexible_dates'] or '-'}\n\n"
        f"**Preferred airline:** {booking_payload.get('airline_pref') or '-'}"
    )

    st.markdown("### WhatsApp-ready message")
    st.code(handoff_message)

    wa_url = whatsapp_link(handoff_message, phone_e164=phone_e164.strip())
    st.markdown(f"**Open WhatsApp:** {wa_url}")

    # Per-passenger WhatsApp links (optional)
    with st.expander("Per-passenger WhatsApp links"):
        any_links = False
        for i, psg in enumerate(passengers_clean, start=1):
            ph = (psg.get("whatsapp") or "").strip()
            nm = (psg.get("name") or f"Passenger {i}").strip()
            if ph:
                any_links = True
                st.markdown(f"- {nm}: {whatsapp_link(handoff_message, phone_e164=ph)}")
        if not any_links:
            st.caption("No passenger WhatsApp numbers provided.")

    # Email handoff
    subj = quote(f"Flight booking request: {o.get('iata','')} → {d.get('iata','')}")
    body = quote(handoff_message)
    mailto = f"mailto:{email_to.strip()}?subject={subj}&body={body}" if email_to.strip() else f"mailto:?subject={subj}&body={body}"
    st.markdown(f"**Email handoff (mailto):** {mailto}")

    # PDF
    pdf_bytes = create_booking_pdf_bytes(booking_payload)
    if pdf_bytes:
        fname = f"booking_{o.get('iata','')}_{d.get('iata','')}_{booking_payload['depart_date']}.pdf"
        st.download_button(
            "📄 Download booking PDF",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.info("PDF export not available (reportlab not installed).")

    # Keep JSON (but not primary)
    with st.expander("Technical payload (JSON)"):
        st.json(booking_payload)
else:
    st.caption("Click **✅ Create booking handoff** to generate the WhatsApp message, save the request, and download a PDF.")
