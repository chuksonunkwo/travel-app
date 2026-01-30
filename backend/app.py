# backend/app.py
import os
import re
import json
from pathlib import Path
from datetime import datetime, date
from urllib.parse import quote

import pandas as pd
import streamlit as st

# Optional deps (do not break app if missing)
try:
    import requests  # Airtable
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

# local fallback storage
BOOKING_STORE_PATH = DATA_DIR / "booking_requests.jsonl"


# =========================
# Utilities
# =========================
def _safe_get_secrets(key: str, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def get_config(key: str, default=None):
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
    if s in ("nan", "none", "null"):
        return ""
    s = re.sub(r"\s+", " ", s)
    return s


def clean_str(x) -> str:
    s = str(x).strip()
    if s.lower() in ("nan", "none", "null"):
        return ""
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
        "iata", "name", "city", "country_code",
        "type", "scheduled_service", "iso_region", "ident",
        "latitude_deg", "longitude_deg"
    ]:
        if c not in out.columns:
            out[c] = ""

    out["iata"] = out["iata"].map(clean_str).str.upper()
    out["name"] = out["name"].map(clean_str)
    out["city"] = out["city"].map(clean_str)
    out["country_code"] = out["country_code"].map(clean_str).str.upper()
    out["type"] = out["type"].map(clean_str)
    out["scheduled_service"] = out["scheduled_service"].map(clean_str)
    out["iso_region"] = out["iso_region"].map(clean_str)
    out["ident"] = out["ident"].map(clean_str)

    out = out[out["iata"].astype(str).str.match(r"^[A-Z]{3}$", na=False)].copy()

    def _label(r):
        city = r.get("city", "") or "Unknown city"
        name = r.get("name", "") or "Unknown airport"
        iata = r.get("iata", "") or ""
        cc = r.get("country_code", "") or ""
        return f"{city} — {name} ({iata}) [{cc}]"

    out["label"] = out.apply(_label, axis=1)

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
        s = clean_str(v)
        if not s:
            continue
        vals.append(s)
    return sorted(set(vals))


def airport_payload(row: pd.Series) -> dict:
    def _n(x):
        if isinstance(x, float) and pd.isna(x):
            return None
        return x

    return {
        "iata": clean_str(row.get("iata", "")),
        "airport_name": clean_str(row.get("name", "")),
        "city": clean_str(row.get("city", "")),
        "country_code": clean_str(row.get("country_code", "")),
        "type": clean_str(row.get("type", "")),
        "scheduled_service": clean_str(row.get("scheduled_service", "")),
        "iso_region": clean_str(row.get("iso_region", "")),
        "ident": clean_str(row.get("ident", "")),
        "latitude_deg": _n(row.get("latitude_deg", None)),
        "longitude_deg": _n(row.get("longitude_deg", None)),
        "label": clean_str(row.get("label", "")),
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


# =========================
# Airtable (compatible envs)
# =========================
def _truthy(v: str) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "y", "on")


def airtable_config():
    # Support both naming schemes
    api_key = get_config("AIRTABLE_API_KEY") or get_config("AIRTABLE_PAT")
    base_id = get_config("AIRTABLE_BASE_ID")
    table = get_config("AIRTABLE_TABLE") or get_config("AIRTABLE_TABLE_NAME")
    enabled_flag = get_config("USE_AIRTABLE") or get_config("AIRTABLE_ENABLE")
    enabled = _truthy(enabled_flag) if enabled_flag is not None else bool(api_key and base_id and table)
    return enabled, api_key, base_id, table


def save_to_airtable(payload: dict) -> tuple[bool, str]:
    if requests is None:
        return False, "requests not installed. Install it or use local storage."

    enabled, api_key, base_id, table = airtable_config()
    if not enabled or not api_key or not base_id or not table:
        return False, "Airtable not configured."

    url = f"https://api.airtable.com/v0/{base_id}/{quote(str(table))}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    passengers = payload.get("passengers", [])
    pax_names = ", ".join([p.get("name", "").strip() for p in passengers if p.get("name", "").strip()])
    pax_whatsapps = ", ".join([p.get("whatsapp", "").strip() for p in passengers if p.get("whatsapp", "").strip()])
    traveler_name = pax_names.split(",")[0].strip() if pax_names else payload.get("traveler_name", "")

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
        "traveler_name": traveler_name,
        "pax_names": pax_names,
        "pax_whatsapp": pax_whatsapps,
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


# =========================
# Handoff + PDF
# =========================
def build_handoff_message(payload: dict) -> str:
    origin = payload.get("origin", {})
    dest = payload.get("destination", {})
    trip_type = payload.get("trip_type", "")
    depart_date = payload.get("depart_date", "")
    return_date = payload.get("return_date", "")

    passengers = payload.get("passengers", [])
    pax = payload.get("pax", 1)

    lines = [
        "🛫 *Travel App — Flight Booking Request*",
        "",
        f"📍 *Origin:* {origin.get('city','')} ({origin.get('iata','')})",
        f"🎯 *Destination:* {dest.get('city','')} ({dest.get('iata','')})",
        "",
        f"🧳 *Passengers:* {pax}",
        f"💺 *Cabin:* {payload.get('cabin','')}",
        f"🔁 *Trip:* {trip_type}",
        f"📅 *Departure:* {depart_date}",
    ]
    if trip_type == "Return":
        lines.append(f"📅 *Return:* {return_date or '-'}")

    flex = payload.get("flexible_dates", "")
    if flex:
        lines += ["", f"📆 *Flexible:* {flex}"]

    if passengers:
        lines += ["", "👤 *Passengers details:*"]
        for i, p in enumerate(passengers, start=1):
            nm = (p.get("name") or "").strip() or "-"
            wa = (p.get("whatsapp") or "").strip() or "-"
            lines.append(f"  {i}. {nm}  |  WhatsApp: {wa}")

    budget = payload.get("budget", "")
    if budget:
        lines += ["", f"💰 *Budget:* {budget}"]

    airline = payload.get("airline_pref", "")
    if airline:
        lines += ["", f"✈️ *Airline preference:* {airline}"]

    notes = payload.get("notes", "")
    if notes:
        lines += ["", f"📝 *Notes:* {notes}"]

    lines += ["", "✅ Please share best available options + total price.", "🙏 Thank you!"]
    return "\n".join(lines)


def create_booking_pdf_bytes(booking_payload: dict) -> bytes | None:
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

    def p(txt: str):
        return Paragraph(txt, styles["BodyText"])

    elements = [
        Paragraph("<b>Travel App — Booking Request Summary</b>", styles["Title"]),
        Spacer(1, 10),
        p(f"<b>Created:</b> {booking_payload.get('created_at','')}"),
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
        ["Budget", booking_payload.get("budget", "") or "-"],
        ["Airline preference", booking_payload.get("airline_pref", "") or "-"],
        ["Notes", booking_payload.get("notes", "") or "-"],
        ["Status", booking_payload.get("status", "")],
    ]

    if passengers:
        pax_lines = []
        for i, px in enumerate(passengers, start=1):
            pax_lines.append(f"{i}. {px.get('name','-')} | {px.get('whatsapp','-')}")
        rows.insert(2, ["Passenger list", "<br/>".join(pax_lines)])

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
# Mock flight data + logos
# =========================
AIRLINE_LOGOS = {
    "Virgin Atlantic": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Virgin_Atlantic_logo.svg/320px-Virgin_Atlantic_logo.svg.png",
    "KLM": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/KLM_logo.svg/320px-KLM_logo.svg.png",
    "Lufthansa": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/Lufthansa_Logo_2018.svg/320px-Lufthansa_Logo_2018.svg.png",
    "Emirates": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Emirates_logo.svg/320px-Emirates_logo.svg.png",
    "British Airways": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/British_Airways_Logo.svg/320px-British_Airways_Logo.svg.png",
}


def mock_flight_search_results(
    origin_iata: str,
    dest_iata: str,
    depart_date: date,
    pax: int,
    cabin: str,
    trip_type: str,
    flex_days: int = 0,
) -> list[dict]:
    seed = sum(ord(c) for c in (origin_iata + dest_iata)) + int(depart_date.strftime("%Y%m%d")) + pax + flex_days
    base = 180 + (seed % 220)
    cabin_mult = {"Economy": 1.0, "Premium Economy": 1.35, "Business": 2.3, "First": 3.2}.get(cabin, 1.0)
    trip_mult = 1.0 if trip_type == "One-way" else 1.75

    def price(i: int) -> int:
        return int((base + i * 35) * cabin_mult * trip_mult)

    airlines = ["Virgin Atlantic", "KLM", "British Airways", "Lufthansa"]
    stops = ["Non-stop", "1 stop", "1 stop", "2 stops"]
    durations = ["6h 20m", "7h 55m", "8h 10m", "11h 40m"]
    notes = ["Best overall", "Great timing", "Flexible change", "Lowest fare"]

    cards = []
    for i in range(4):
        cards.append(
            {
                "airline": airlines[i],
                "stops": stops[i],
                "duration": durations[i],
                "price_usd": price(i),
                "note": notes[i],
            }
        )
    return cards


# =========================
# Data load
# =========================
@st.cache_data(show_spinner=False)
def load_airports() -> pd.DataFrame:
    csv_path = find_airports_csv()
    df_raw = pd.read_csv(csv_path)
    return standardize_airports_df(df_raw)


# =========================
# Production UI (Light only)
# =========================
def apply_production_light_ui():
    st.markdown(
        """
        <style>
          html, body, [class*="stApp"] { background: #ffffff !important; color: #0f172a !important; }
          [data-testid="stSidebar"] { background: #f8fafc !important; border-right: 1px solid #e5e7eb; }
          .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
          h1, h2, h3 { letter-spacing: -0.02em; }
          .ta-hero {
            background: linear-gradient(90deg, #eef2ff 0%, #ecfeff 100%);
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 18px 18px;
          }
          .ta-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 14px 14px;
            box-shadow: 0 2px 10px rgba(2,6,23,0.06);
          }
          .ta-muted { color: #475569; }
          .ta-chip {
            display: inline-block;
            padding: 4px 10px;
            margin-right: 8px;
            border-radius: 999px;
            border: 1px solid #e5e7eb;
            background: #f8fafc;
            font-size: 12px;
            color: #0f172a;
          }
          .ta-green {
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
            border-radius: 12px;
            padding: 10px 12px;
          }
          .ta-blue {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 12px;
            padding: 10px 12px;
          }
          .ta-rowgap { margin-top: 10px; }
          /* Make inputs clean in Light */
          .stTextInput input, .stTextArea textarea {
            background: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 10px !important;
          }
          .stSelectbox div[data-baseweb="select"] {
            background: #ffffff !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 10px !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# App
# =========================
st.set_page_config(page_title="Travel App", page_icon="🧭", layout="wide")
apply_production_light_ui()

# Load airports
try:
    airports = load_airports()
except Exception as e:
    st.error("Could not load airports database.")
    st.code(str(e))
    st.stop()

# Sidebar (keep simple + stable)
st.sidebar.header("Filters")
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

# Session state init
defaults = {
    "origin_airport": None,
    "dest_airport": None,
    "origin_q": "",
    "dest_q": "",
    "origin_select": None,
    "dest_select": None,
    "search_cards": None,        # persists after booking
    "booking_payload_last": None # persists after actions
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# Callbacks (SAFE for Streamlit; avoids widget mutation errors)
def do_swap():
    oa = st.session_state.get("origin_airport")
    da = st.session_state.get("dest_airport")

    st.session_state["origin_airport"], st.session_state["dest_airport"] = da, oa

    # swap queries too (so UI reflects it)
    oq, dq = st.session_state.get("origin_q", ""), st.session_state.get("dest_q", "")
    st.session_state["origin_q"], st.session_state["dest_q"] = dq, oq

    # swap selected labels
    osel, dsel = st.session_state.get("origin_select"), st.session_state.get("dest_select")
    st.session_state["origin_select"], st.session_state["dest_select"] = dsel, osel

    # keep prior search results; do not clear
    st.toast("Swapped origin and destination.")


def do_clear():
    st.session_state["origin_airport"] = None
    st.session_state["dest_airport"] = None
    st.session_state["origin_q"] = ""
    st.session_state["dest_q"] = ""
    st.session_state["origin_select"] = None
    st.session_state["dest_select"] = None
    # do not clear search_cards automatically (user can still see last results if they want)
    st.toast("Cleared selections.")


# Header
st.markdown(
    """
    <div class="ta-hero">
      <h1 style="margin:0;">Travel App</h1>
      <div class="ta-muted" style="margin-top:6px;">
        Select origin and destination, then search flights and create a WhatsApp/Email booking handoff.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")


def airport_selector_block(title: str, key_prefix: str):
    st.subheader(title)

    q_key = f"{key_prefix}_q"
    sel_key = f"{key_prefix}_select"

    q = st.text_input(
        "Search (airport / city / IATA)",
        key=q_key,
        placeholder="e.g., lagos, LOS, heathrow, london...",
    )

    if len(q.strip()) < MIN_CHARS:
        st.markdown(f'<div class="ta-blue">Type at least {MIN_CHARS} characters to see suggestions.</div>', unsafe_allow_html=True)
        return None

    results = rank_airports(filtered, q, limit=limit)
    if results.empty:
        st.warning("No matches. Try city name, airport name, or a 3-letter IATA code.")
        return None

    options = results["label"].tolist()

    # Keep selection stable if present in options
    current = st.session_state.get(sel_key)
    if current in options:
        idx = options.index(current)
    else:
        idx = 0

    sel = st.selectbox("Choose an airport", options=options, index=idx, key=sel_key)
    st.session_state[sel_key] = sel  # explicit persist

    selected_rows = results[results["label"] == sel]
    if selected_rows.empty:
        st.warning("Selection not found. Try again.")
        return None

    row = selected_rows.iloc[0]
    payload = airport_payload(row)

    st.markdown(f'<div class="ta-green"><b>Selected:</b> {payload["iata"]}</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="ta-rowgap">
          <span class="ta-chip">IATA: {payload["iata"]}</span>
          <span class="ta-chip">City: {payload["city"] or "-"}</span>
          <span class="ta-chip">Country: {payload["country_code"] or "-"}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Details (JSON)"):
        st.json(payload)

    return payload


colA, colB = st.columns(2, gap="large")
with colA:
    origin = airport_selector_block("Origin", "origin")
    if origin:
        st.session_state["origin_airport"] = origin
with colB:
    dest = airport_selector_block("Destination", "dest")
    if dest:
        st.session_state["dest_airport"] = dest

# Actions
a1, a2, a3 = st.columns([1, 1, 2])
with a1:
    st.button("🔁 Swap origin/destination", on_click=do_swap, use_container_width=True)
with a2:
    st.button("🧹 Clear selections", on_click=do_clear, use_container_width=True)
with a3:
    st.caption("Tip: select both airports, then set trip details below.")

st.divider()

# Trip details
st.subheader("Trip details")

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
    return_date = st.date_input("Return date", value=date.today()) if trip_type == "Return" else None
    if trip_type != "Return":
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

st.markdown("#### Passenger details (per passenger)")
passengers = []
for i in range(int(pax)):
    p1, p2 = st.columns(2)
    with p1:
        name = st.text_input(f"Passenger {i+1} — Full name", key=f"pax_name_{i}", placeholder="e.g., Chukwuma Onunkwo")
    with p2:
        wa = st.text_input(f"Passenger {i+1} — WhatsApp (optional)", key=f"pax_wa_{i}", placeholder="+234xxxxxxxxxx")
    passengers.append({"name": name.strip(), "whatsapp": wa.strip()})

notes = st.text_area("Notes / preferences (optional)", placeholder="Baggage, time preference, avoid long layovers, etc.")
email_to = st.text_input("Email (optional)", placeholder="agent@company.com or your email")

# Validate dates
if trip_type == "Return" and return_date and return_date < depart_date:
    st.error("Return date cannot be earlier than departure date.")
    st.stop()

# Actions
b1, b2 = st.columns([1, 1])
do_search = b1.button("🔎 Search flights", use_container_width=True)
do_booking = b2.button("✅ Create booking handoff", use_container_width=True)

st.divider()

# Search results (persist in session_state so they remain when booking is created)
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
    st.session_state["search_cards"] = {
        "meta": {
            "origin_iata": origin_airport["iata"],
            "dest_iata": dest_airport["iata"],
            "depart_date": str(depart_date),
            "trip_type": trip_type,
            "cabin": cabin,
            "pax": int(pax),
            "flex_days": flex_days,
        },
        "cards": cards,
    }

search_state = st.session_state.get("search_cards")
if search_state:
    st.subheader("Flight results")
    if search_state["meta"].get("flex_days", 0) > 0:
        st.info("Showing flexible-date estimates (±3 days). Live providers will return real date variants.")
    else:
        st.info("Showing mock results (no external API). Live booking comes next.")

    for r in search_state["cards"]:
        logo_url = AIRLINE_LOGOS.get(r["airline"])
        price = f"${int(r['price_usd']):,}"

        st.markdown(
            f"""
            <div class="ta-card" style="display:flex; align-items:center; justify-content:space-between; gap:14px;">
              <div style="display:flex; align-items:center; gap:14px; min-width:0;">
                <div style="width:64px; height:40px; display:flex; align-items:center; justify-content:center;">
                  {"<img src='"+logo_url+"' style='max-height:40px; max-width:64px; object-fit:contain;'/>" if logo_url else "<b>"+r["airline"]+"</b>"}
                </div>
                <div style="min-width:0;">
                  <div style="font-weight:700; font-size:16px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{r["airline"]}</div>
                  <div class="ta-muted" style="font-size:13px;">
                    {r["stops"]} • {r["duration"]} • {r["note"]}
                  </div>
                </div>
              </div>
              <div style="text-align:right;">
                <div style="font-weight:800; font-size:18px;">{price}</div>
                <div class="ta-muted" style="font-size:12px;">est. total</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

# Booking handoff (render below results; results remain visible)
if do_booking:
    st.subheader("Booking handoff")

    flex_label = "±3 days" if flexible else ""
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
        "passengers": passengers,
        "traveler_name": passengers[0]["name"] if passengers and passengers[0]["name"] else "",
        "budget": budget.strip(),
        "airline_pref": airline_pref.strip(),
        "notes": notes.strip(),
        "status": "NEW",
    }

    handoff_message = build_handoff_message(booking_payload)
    booking_payload["handoff_message"] = handoff_message
    st.session_state["booking_payload_last"] = booking_payload

    # Save: Airtable if enabled else local JSONL
    saved_msg = ""
    enabled, *_ = airtable_config()
    if enabled:
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

    o = booking_payload["origin"]
    d = booking_payload["destination"]
    st.markdown("### Booking summary")
    st.write(
        f"**Route:** {o.get('city','')} ({o.get('iata','')}) → {d.get('city','')} ({d.get('iata','')})\n\n"
        f"**Trip:** {trip_type}  •  **Cabin:** {cabin}  •  **Passengers:** {int(pax)}\n\n"
        f"**Departure:** {booking_payload['depart_date']}  •  **Return:** {booking_payload['return_date'] or '-'}  •  **Flexible:** {booking_payload['flexible_dates'] or '-'}"
    )

    st.markdown("### WhatsApp-ready message")
    st.text_area("Message", value=handoff_message, height=220)

    # Use first passenger WhatsApp if provided, else blank
    preferred_phone = ""
    for p in passengers:
        if p.get("whatsapp", "").strip():
            preferred_phone = p["whatsapp"].strip()
            break

    wa_url = whatsapp_link(handoff_message, phone_e164=preferred_phone)
    st.link_button("Open WhatsApp", wa_url, use_container_width=True)

    subj = quote(f"Flight booking request: {o.get('iata','')} → {d.get('iata','')}")
    body = quote(handoff_message)
    mailto = f"mailto:{email_to.strip()}?subject={subj}&body={body}" if email_to.strip() else f"mailto:?subject={subj}&body={body}"
    st.link_button("Open Email draft (mailto)", mailto, use_container_width=True)

    pdf_bytes = create_booking_pdf_bytes(booking_payload)
    if pdf_bytes:
        fname = f"booking_{o.get('iata','')}_{d.get('iata','')}_{booking_payload['depart_date']}.pdf"
        st.download_button("📄 Download booking PDF", data=pdf_bytes, file_name=fname, mime="application/pdf", use_container_width=True)
    else:
        st.info("PDF export not available (reportlab not installed).")

    with st.expander("Technical payload (JSON)"):
        st.json(booking_payload)
