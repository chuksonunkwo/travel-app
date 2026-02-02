# backend/app.py
import os
import re
import json
from pathlib import Path
from datetime import datetime, date, timezone
from urllib.parse import quote

import pandas as pd
import streamlit as st

# Optional deps (do not break app if missing)
try:
    import requests  # Airtable
except Exception:
    requests = None


# =========================
# Paths / Config
# =========================
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
ASSETS_DIR = APP_DIR / "assets"
AIRLINE_SVG_DIR = ASSETS_DIR / "airlines"

CANDIDATE_CSVS = [
    DATA_DIR / "airports_clean.csv",
    DATA_DIR / "airports.csv",
]

MIN_CHARS = 2
DEFAULT_LIMIT = 30

# local fallback storage (dev; production containers can be ephemeral, prefer Airtable)
BOOKING_STORE_PATH = DATA_DIR / "booking_requests.jsonl"


# =========================
# Small utilities
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


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_text(x: str) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _clean_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
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

    out["iata"] = out["iata"].map(_clean_str).astype(str).str.upper()
    out["name"] = out["name"].map(_clean_str)
    out["city"] = out["city"].map(_clean_str)
    out["country_code"] = out["country_code"].map(_clean_str).astype(str).str.upper()
    out["type"] = out["type"].map(_clean_str)
    out["scheduled_service"] = out["scheduled_service"].map(_clean_str)
    out["iso_region"] = out["iso_region"].map(_clean_str)
    out["ident"] = out["ident"].map(_clean_str)

    # Keep valid IATA only: exactly 3 letters
    out = out[out["iata"].str.match(r"^[A-Z]{3}$", na=False)].copy()

    def _label(r):
        city = r.get("city", "") or "Unknown city"
        name = r.get("name", "") or "Unknown airport"
        iata = r.get("iata", "")
        cc = r.get("country_code", "")
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


# =========================
# Airline logo handling (repo SVG + fallback)
# =========================
def airline_badge_svg(code: str, style_seed: int = 0) -> str:
    code = (code or "XX")[:2].upper()
    gradients = [
        ("#0ea5e9", "#22c55e"),
        ("#a855f7", "#ec4899"),
        ("#f97316", "#facc15"),
        ("#14b8a6", "#3b82f6"),
        ("#ef4444", "#fb7185"),
    ]
    g1, g2 = gradients[style_seed % len(gradients)]
    return f"""
    <svg width="52" height="52" viewBox="0 0 52 52" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="{g1}"/>
          <stop offset="100%" stop-color="{g2}"/>
        </linearGradient>
      </defs>
      <rect x="3" y="3" rx="14" ry="14" width="46" height="46" fill="url(#g)"/>
      <rect x="3" y="3" rx="14" ry="14" width="46" height="46" fill="none" stroke="rgba(255,255,255,0.35)" stroke-width="1"/>
      <text x="26" y="32" text-anchor="middle" font-family="Arial" font-size="16" fill="#FFFFFF" font-weight="800">{code}</text>
    </svg>
    """


def load_airline_svg_from_repo(iata_code: str) -> str | None:
    code = (iata_code or "").strip().upper()
    if not code:
        return None
    p = AIRLINE_SVG_DIR / f"{code}.svg"
    if not file_exists(p):
        return None
    try:
        # Read raw SVG text (no external fetch, no CORS issues)
        return p.read_text(encoding="utf-8")
    except Exception:
        return None


def airline_logo_svg(iata_code: str, style_seed: int = 0) -> str:
    svg = load_airline_svg_from_repo(iata_code)
    if svg:
        return svg
    return airline_badge_svg(iata_code, style_seed=style_seed)


# =========================
# Mock flight results (still used until Amadeus is live)
# =========================
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

    airlines = [
        {"name": "Virgin Atlantic", "code": "VS"},
        {"name": "KLM", "code": "KL"},
        {"name": "British Airways", "code": "BA"},
        {"name": "Lufthansa", "code": "LH"},
    ]
    stops = ["Non-stop", "1 stop", "1 stop", "2 stops"]
    durations = ["6h 20m", "7h 55m", "8h 10m", "11h 40m"]
    notes = ["Best overall", "Great timing", "Flexible change", "Lowest fare"]

    cards = []
    for i in range(4):
        cards.append(
            {
                "airline": airlines[i]["name"],
                "airline_code": airlines[i]["code"],
                "stops": stops[i],
                "duration": durations[i],
                "price_value": price(i),
                "price": f"${price(i)}",
                "note": notes[i],
                "style_seed": (seed + i) % 5,
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
# Theme + UI polish
# =========================
def apply_theme(mode: str):
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1200px; }
          [data-testid="stSidebar"] { border-right: 1px solid rgba(15,23,42,0.08); }
          h1, h2, h3, h4, h5, h6 { letter-spacing: -0.02em; }
          .ow-muted { opacity: 0.78; }

          .ow-hero {
            border-radius: 22px;
            padding: 22px 22px;
            border: 1px solid rgba(15,23,42,0.08);
            box-shadow: 0 10px 28px rgba(2,6,23,0.06);
          }
          .ow-card {
            border-radius: 18px;
            padding: 18px 18px;
            border: 1px solid rgba(15,23,42,0.08);
            box-shadow: 0 8px 22px rgba(2,6,23,0.05);
          }
          .ow-flight {
            border-radius: 16px;
            padding: 14px 14px;
            border: 1px solid rgba(15,23,42,0.08);
            box-shadow: 0 8px 20px rgba(2,6,23,0.05);
            margin: 10px 0;
          }

          /* Make buttons clearly visible */
          .stButton button, .stDownloadButton button { border-radius: 12px !important; font-weight: 650 !important; }
          .stButton { margin-top: 0.2rem; margin-bottom: 0.2rem; }

          @media (prefers-color-scheme: light) {
            html, body, [class*="stApp"] { background: #f7f9fc !important; color: #0f172a !important; }
            [data-testid="stSidebar"] { background: #ffffff !important; }
            .ow-hero { background: linear-gradient(135deg, rgba(59,130,246,0.16), rgba(16,185,129,0.12)); }
            .ow-card, .ow-flight { background: #ffffff !important; }
          }

          @media (prefers-color-scheme: dark) {
            html, body, [class*="stApp"] { background: #05060a !important; color: #e5e7eb !important; }
            [data-testid="stSidebar"] { background: #070a12 !important; border-right: 1px solid rgba(148,163,184,0.14); }
            .ow-hero { background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(16,185,129,0.10)); border: 1px solid rgba(148,163,184,0.14); }
            .ow-card, .ow-flight { background: rgba(15,23,42,0.78) !important; border: 1px solid rgba(148,163,184,0.14) !important; }
            .ow-muted { opacity: 0.82; }
            .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
              background: rgba(2,6,23,0.55) !important; color: #e5e7eb !important;
              border-color: rgba(148,163,184,0.18) !important;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if mode == "System":
        return

    if mode == "Light":
        st.markdown(
            """
            <style>
              html, body, [class*="stApp"] { background: #f7f9fc !important; color: #0f172a !important; }
              [data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid rgba(15,23,42,0.08) !important; }
              .ow-hero { background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(16,185,129,0.14)) !important; }
              .ow-card, .ow-flight { background: #ffffff !important; }
              .ow-muted { color: rgba(15,23,42,0.78) !important; opacity: 1 !important; }
              .stMarkdown, .stText, p, span, div { color: inherit !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        return

    if mode == "Dark":
        st.markdown(
            """
            <style>
              html, body, [class*="stApp"] { background: #05060a !important; color: #e5e7eb !important; }
              [data-testid="stSidebar"] { background: #070a12 !important; border-right: 1px solid rgba(148,163,184,0.14) !important; }
              .ow-hero { background: linear-gradient(135deg, rgba(59,130,246,0.22), rgba(16,185,129,0.12)) !important; }
              .ow-card, .ow-flight { background: rgba(15,23,42,0.82) !important; border: 1px solid rgba(148,163,184,0.14) !important; }
              .ow-muted { color: rgba(226,232,240,0.85) !important; opacity: 1 !important; }
              .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
                background: rgba(2,6,23,0.55) !important; color: #e5e7eb !important;
              }
            </style>
            """,
            unsafe_allow_html=True,
        )
        return


# =========================
# Airtable (aligned to YOUR table columns)
# =========================
def _airtable_api_key() -> str:
    return (get_config("AIRTABLE_PAT") or get_config("AIRTABLE_API_KEY") or "").strip()


def _airtable_base_id() -> str:
    return (get_config("AIRTABLE_BASE_ID") or "").strip()


def _airtable_table_name() -> str:
    return (get_config("AIRTABLE_TABLE_NAME") or get_config("AIRTABLE_TABLE") or "").strip()


def airtable_enabled() -> bool:
    use_flag = (get_config("USE_AIRTABLE") or "").strip()
    if use_flag and use_flag not in ("1", "true", "True", "yes", "YES"):
        return False
    return bool(_airtable_api_key() and _airtable_base_id() and _airtable_table_name())


def save_to_airtable(payload: dict) -> tuple[bool, str]:
    if requests is None:
        return False, "requests not installed."

    api_key = _airtable_api_key()
    base_id = _airtable_base_id()
    table = _airtable_table_name()

    if not api_key or not base_id or not table:
        return False, "Airtable not configured (missing key/base/table)."

    url = f"https://api.airtable.com/v0/{base_id}/{quote(str(table))}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    pax_details = payload.get("passengers_details", []) or []
    pax_summary = " | ".join(
        [f"{i+1}. {d.get('name','-') or '-'} ({d.get('whatsapp','-') or '-'})" for i, d in enumerate(pax_details)]
    )

    # IMPORTANT: these field names match your CSV export columns
    fields = {
        "created_at": payload.get("created_at", ""),
        "origin_iata": payload.get("origin", {}).get("iata", ""),
        "destination_iata": payload.get("destination", {}).get("iata", ""),
        "origin_city": payload.get("origin", {}).get("city", ""),
        "destination_city": payload.get("destination", {}).get("city", ""),
        "trip_type": payload.get("trip_type", ""),
        "departure_date": payload.get("depart_date", ""),
        "return_date": payload.get("return_date", ""),
        "passengers": int(payload.get("pax", 1)),
        "cabin_class": payload.get("cabin", ""),
        "budget": payload.get("budget", ""),
        "airline_preference": payload.get("airline_pref", ""),
        "flexible_dates": payload.get("flexible_dates", ""),
        "notes": payload.get("notes", ""),
        "status": payload.get("status", "NEW"),
        "handoff_message": payload.get("handoff_message", ""),
        "passengers_details": pax_summary,
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

    pax = payload.get("pax", 1)
    cabin = payload.get("cabin", "")
    flex = payload.get("flexible_dates", "")
    budget = payload.get("budget", "")
    airline = payload.get("airline_pref", "")
    notes = payload.get("notes", "")

    pax_details = payload.get("passengers_details", []) or []
    pax_lines = []
    for i, d in enumerate(pax_details, start=1):
        nm = d.get("name", "") or "-"
        wa = d.get("whatsapp", "") or "-"
        pax_lines.append(f"{i}. {nm} — {wa}")

    lines = [
        "🧭 *Travel App — Flight Booking Request*",
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

    if flex:
        lines += ["", f"📆 *Flexible:* {flex}"]
    if pax_lines:
        lines += ["", "👥 *Passenger details:*"] + pax_lines
    if budget:
        lines += ["", f"💰 *Budget:* {budget}"]
    if airline:
        lines += ["", f"✈️ *Airline preference:* {airline}"]
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

    pax_details = booking_payload.get("passengers_details", []) or []
    pax_text = ""
    if pax_details:
        lines = []
        for i, pxd in enumerate(pax_details, start=1):
            nm = pxd.get("name", "") or "-"
            wa = pxd.get("whatsapp", "") or "-"
            lines.append(f"{i}. {nm} — {wa}")
        pax_text = "<br/>".join(lines)

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
        ["Passenger details", pax_text or "-"],
        ["Status", booking_payload.get("status", "")],
    ]

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
    msg = (booking_payload.get("handoff_message", "") or "").replace("\n", "<br/>")
    elements += [p(msg)]

    doc.build(elements)
    return buf.getvalue()


# =========================
# Streamlit page config
# =========================
st.set_page_config(page_title="Travel App", page_icon="🧭", layout="wide")


# =========================
# Session state
# =========================
DEFAULTS = {
    "origin_airport": None,
    "dest_airport": None,
    "origin_q": "",
    "dest_q": "",
    "last_search_cards": None,
    "last_search_meta": None,
    "last_booking_payload": None,
    "passengers_details": [],
    "_pending_action": None,  # "swap" | "clear"
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _ensure_pax_details(pax: int):
    pax = int(pax)
    current = st.session_state.get("passengers_details") or []
    if len(current) < pax:
        current = current + [{"name": "", "whatsapp": ""} for _ in range(pax - len(current))]
    elif len(current) > pax:
        current = current[:pax]
    st.session_state["passengers_details"] = current


def _apply_pending_actions_before_widgets():
    action = st.session_state.get("_pending_action")

    if action == "swap":
        st.session_state["_pending_action"] = None
        oa = st.session_state.get("origin_airport")
        da = st.session_state.get("dest_airport")
        st.session_state["origin_airport"], st.session_state["dest_airport"] = da, oa
        st.session_state["origin_q"] = ((da or {}).get("iata") or "").lower()
        st.session_state["dest_q"] = ((oa or {}).get("iata") or "").lower()
        st.rerun()

    if action == "clear":
        st.session_state["_pending_action"] = None
        st.session_state["origin_airport"] = None
        st.session_state["dest_airport"] = None
        st.session_state["origin_q"] = ""
        st.session_state["dest_q"] = ""
        st.rerun()


_apply_pending_actions_before_widgets()


# =========================
# Sidebar: theme + status + filters
# =========================
st.sidebar.header("Settings")
theme_mode = st.sidebar.selectbox("Theme", ["System", "Light", "Dark"], index=0)
apply_theme(theme_mode)

st.sidebar.markdown("---")
st.sidebar.subheader("Status")
st.sidebar.write(f"**Airtable enabled:** {'Yes' if airtable_enabled() else 'No'}")
st.sidebar.write(f"**Airline SVG folder:** {'OK' if AIRLINE_SVG_DIR.exists() else 'Missing'}")
st.sidebar.caption("Tip: On Render, set env vars in the service settings.")

st.sidebar.markdown("---")
st.sidebar.header("Filters")

try:
    airports = load_airports()
except Exception as e:
    st.error("Could not load airports database.")
    st.code(str(e))
    st.stop()

types = clean_filter_values(airports.get("type", pd.Series(dtype=str)))
countries = clean_filter_values(airports.get("country_code", pd.Series(dtype=str)))
scheduled = clean_filter_values(airports.get("scheduled_service", pd.Series(dtype=str)))

default_types = [t for t in types if t in ("large_airport", "medium_airport", "small_airport")]
if not default_types and types:
    default_types = types[:]

selected_types = st.sidebar.multiselect("Airport type", options=types, default=default_types) if types else []
selected_countries = st.sidebar.multiselect("Country (ISO)", options=countries, default=[]) if countries else []
selected_scheduled = st.sidebar.multiselect("Scheduled service", options=scheduled, default=[]) if scheduled else []
limit = st.sidebar.slider("Max suggestions", min_value=10, max_value=100, value=DEFAULT_LIMIT, step=5)

filtered = airports.copy()
if selected_types:
    filtered = filtered[filtered["type"].isin(selected_types)]
if selected_countries:
    filtered = filtered[filtered["country_code"].isin(selected_countries)]
if selected_scheduled:
    filtered = filtered[filtered["scheduled_service"].isin(selected_scheduled)]
filtered = filtered.reset_index(drop=True)


# =========================
# Header
# =========================
st.markdown(
    """
    <div class="ow-hero">
      <div style="display:flex; align-items:flex-end; justify-content:space-between; gap:12px;">
        <div>
          <h1 style="margin:0; line-height:1.05;">Travel App</h1>
          <div class="ow-muted" style="margin-top:8px;">
            Select origin and destination, search flights, then create a WhatsApp/Email booking handoff + PDF.
          </div>
        </div>
        <div class="ow-muted" style="text-align:right;">
          <div style="font-size:12px;">Demo-ready • API-safe • Production-friendly</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")


# =========================
# Airport selector blocks
# =========================
def airport_selector_block(title: str, key_prefix: str, preselected: dict | None):
    st.markdown("<div class='ow-card'>", unsafe_allow_html=True)
    st.subheader(title)

    q_key = f"{key_prefix}_q"
    if q_key not in st.session_state:
        st.session_state[q_key] = ""

    if key_prefix == "origin" and st.session_state.get("origin_q") and not st.session_state.get(q_key):
        st.session_state[q_key] = st.session_state.get("origin_q", "")
    if key_prefix == "dest" and st.session_state.get("dest_q") and not st.session_state.get(q_key):
        st.session_state[q_key] = st.session_state.get("dest_q", "")

    q = st.text_input(
        "Search (airport / city / IATA)",
        key=q_key,
        placeholder="e.g., lagos, LOS, heathrow, london...",
    )

    if len(q.strip()) < MIN_CHARS:
        st.info(f"Type at least {MIN_CHARS} characters to see suggestions.")
        st.markdown("</div>", unsafe_allow_html=True)
        return None

    results = rank_airports(filtered, q, limit=limit)
    if results.empty:
        st.warning("No matches. Try city name, airport name, or a 3-letter IATA code.")
        st.markdown("</div>", unsafe_allow_html=True)
        return None

    options = results["label"].tolist()

    idx = 0
    if preselected and preselected.get("iata"):
        iata = preselected["iata"]
        match = results[results["iata"] == iata]
        if not match.empty:
            wanted_label = match.iloc[0]["label"]
            try:
                idx = options.index(wanted_label)
            except ValueError:
                idx = 0

    sel = st.selectbox("Choose an airport", options=options, index=idx, key=f"{key_prefix}_select")

    selected_rows = results[results["label"] == sel]
    if selected_rows.empty:
        st.warning("Selection not found. Try again.")
        st.markdown("</div>", unsafe_allow_html=True)
        return None

    row = selected_rows.iloc[0]
    payload = airport_payload(row)

    st.success(f"Selected: {payload['iata']}")
    st.markdown(
        f"**IATA:** {payload['iata']} &nbsp;&nbsp;•&nbsp;&nbsp; "
        f"**City:** {payload['city'] or 'Unknown'} &nbsp;&nbsp;•&nbsp;&nbsp; "
        f"**Country:** {payload['country_code'] or '-'}"
    )

    st.markdown("</div>", unsafe_allow_html=True)
    return payload


ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
with ctrl1:
    if st.button("🔁 Swap origin/destination", use_container_width=True):
        st.session_state["_pending_action"] = "swap"
        st.rerun()
with ctrl2:
    if st.button("🧹 Clear selections", use_container_width=True):
        st.session_state["_pending_action"] = "clear"
        st.rerun()
with ctrl3:
    st.caption("Tip: select both airports, then set trip details below.")

colA, colB = st.columns(2, gap="large")
with colA:
    origin = airport_selector_block("Origin", "origin", st.session_state.get("origin_airport"))
    if origin:
        st.session_state["origin_airport"] = origin
with colB:
    dest = airport_selector_block("Destination", "dest", st.session_state.get("dest_airport"))
    if dest:
        st.session_state["dest_airport"] = dest

st.divider()


# =========================
# Trip details
# =========================
st.markdown("<div class='ow-card'>", unsafe_allow_html=True)
st.subheader("Trip details")

origin_airport = st.session_state.get("origin_airport")
dest_airport = st.session_state.get("dest_airport")

if not origin_airport or not dest_airport:
    st.warning("Select both origin and destination airports to continue.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

c1, c2, c3, c4 = st.columns(4)
with c1:
    trip_type = st.selectbox("Trip type", ["One-way", "Return"], index=1)
with c2:
    depart_date = st.date_input("Departure date", value=date.today())
with c3:
    if trip_type == "Return":
        return_date = st.date_input("Return date", value=date.today())
    else:
        return_date = None
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

if trip_type == "Return" and return_date and return_date < depart_date:
    st.error("Return date cannot be earlier than departure date.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# Buttons always visible (fixes “missing search button”)
b1, b2 = st.columns([1, 1])
do_search = b1.button("🔎 Search flights", use_container_width=True)
do_booking = b2.button("✅ Create booking handoff", use_container_width=True)

_ensure_pax_details(int(pax))
with st.expander("Passenger details (per passenger)", expanded=False):
    st.caption("Enter name + WhatsApp per passenger. These are included in the booking payload + PDF.")
    for i in range(int(pax)):
        pcol1, pcol2 = st.columns([2, 2])
        with pcol1:
            st.session_state["passengers_details"][i]["name"] = st.text_input(
                f"Passenger {i+1} name",
                value=st.session_state["passengers_details"][i].get("name", ""),
                key=f"pax_name_{i}",
            )
        with pcol2:
            st.session_state["passengers_details"][i]["whatsapp"] = st.text_input(
                f"Passenger {i+1} WhatsApp",
                value=st.session_state["passengers_details"][i].get("whatsapp", ""),
                key=f"pax_wa_{i}",
                placeholder="+234xxxxxxxxxx",
            )

notes = st.text_area("Notes / preferences (optional)", placeholder="Baggage, time preference, avoid long layovers, etc.")
email_to = st.text_input("Email (optional)", placeholder="agent@company.com or your email")

st.markdown("</div>", unsafe_allow_html=True)
st.write("")


# Search (mock for now)
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
        "origin": origin_airport,
        "destination": dest_airport,
        "depart_date": str(depart_date),
        "return_date": str(return_date) if (trip_type == "Return" and return_date) else "",
        "trip_type": trip_type,
        "cabin": cabin,
        "pax": int(pax),
        "flexible": bool(flexible),
    }

cards = st.session_state.get("last_search_cards")
meta = st.session_state.get("last_search_meta")

if cards:
    st.markdown("<div class='ow-card'>", unsafe_allow_html=True)
    st.subheader("Flight results")
    st.info("Showing mock results (no external API). Live booking comes next.")

    for idx, r in enumerate(cards):
        st.markdown("<div class='ow-flight'>", unsafe_allow_html=True)
        left, mid, right = st.columns([1, 3, 1])
        with left:
            st.markdown(airline_logo_svg(r.get("airline_code", "XX"), style_seed=r.get("style_seed", idx)), unsafe_allow_html=True)
        with mid:
            st.markdown(f"**{r['airline']}**")
            st.caption(f"{r['stops']} • {r['duration']} • {r['note']}")
        with right:
            st.markdown(f"### {r['price']}")
            st.button(
                f"Select {r['airline']}",
                key=f"sel_{idx}_{r.get('airline_code','XX')}",
                help="Mock selection (live booking next).",
                use_container_width=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")


# Booking handoff + save
if do_booking:
    flex_label = "±3 days" if flexible else ""
    pax_details = st.session_state.get("passengers_details") or []

    booking_payload = {
        "created_at": now_utc_iso(),
        "origin": origin_airport,
        "destination": dest_airport,
        "pax": int(pax),
        "cabin": cabin,
        "trip_type": trip_type,
        "depart_date": str(depart_date),
        "return_date": str(return_date) if (trip_type == "Return" and return_date) else "",
        "flexible_dates": flex_label,
        "passengers_details": pax_details,
        "traveler_name": pax_details[0].get("name", "") if pax_details else "",
        "budget": (budget or "").strip(),
        "airline_pref": (airline_pref or "").strip(),
        "notes": (notes or "").strip(),
        "status": "NEW",
    }

    handoff_message = build_handoff_message(booking_payload)
    booking_payload["handoff_message"] = handoff_message
    st.session_state["last_booking_payload"] = booking_payload

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
            except Exception as e:
                st.error("Local save failed.")
                st.code(str(e))
    else:
        try:
            append_local_jsonl(booking_payload)
            st.success("Booking request created (stored locally).")
        except Exception as e:
            st.error("Could not save booking request locally.")
            st.code(str(e))


# Always render last booking
last_booking = st.session_state.get("last_booking_payload")
if last_booking:
    st.markdown("<div class='ow-card'>", unsafe_allow_html=True)
    st.subheader("Booking handoff")

    o = last_booking.get("origin", {})
    d = last_booking.get("destination", {})
    pax_details = last_booking.get("passengers_details", []) or []

    st.markdown("### Booking summary")
    st.write(
        f"**Route:** {o.get('city','')} ({o.get('iata','')}) → {d.get('city','')} ({d.get('iata','')})\n\n"
        f"**Trip:** {last_booking.get('trip_type','')}  •  **Cabin:** {last_booking.get('cabin','')}  •  "
        f"**Passengers:** {last_booking.get('pax',1)}\n\n"
        f"**Departure:** {last_booking.get('depart_date','')}  •  "
        f"**Return:** {last_booking.get('return_date','-') or '-'}  •  "
        f"**Flexible:** {last_booking.get('flexible_dates','-') or '-'}"
    )

    st.markdown("### Passenger details")
    for i, pxd in enumerate(pax_details, start=1):
        st.write(f"{i}. **{pxd.get('name','-') or '-'}** — {pxd.get('whatsapp','-') or '-'}")

    st.markdown("### WhatsApp-ready message")
    st.code(last_booking.get("handoff_message", ""))

    default_phone = ""
    if pax_details and pax_details[0].get("whatsapp"):
        default_phone = pax_details[0].get("whatsapp", "")
    wa_url = whatsapp_link(last_booking.get("handoff_message", ""), phone_e164=(default_phone or "").strip())
    try:
        st.link_button("Open WhatsApp", wa_url, use_container_width=True)
    except Exception:
        st.markdown(f"**Open WhatsApp:** {wa_url}")

    subj = quote(f"Flight booking request: {o.get('iata','')} → {d.get('iata','')}")
    body = quote(last_booking.get("handoff_message", ""))
    to = (email_to or "").strip()
    mailto = f"mailto:{to}?subject={subj}&body={body}" if to else f"mailto:?subject={subj}&body={body}"
    try:
        st.link_button("Open email draft", mailto, use_container_width=True)
    except Exception:
        st.markdown(f"**Email handoff (mailto):** {mailto}")

    pdf_bytes = create_booking_pdf_bytes(last_booking)
    if pdf_bytes:
        fname = f"booking_{o.get('iata','')}_{d.get('iata','')}_{last_booking.get('depart_date','')}.pdf"
        st.download_button(
            "📄 Download booking PDF",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.info("PDF export not available (reportlab not installed).")

    st.markdown("</div>", unsafe_allow_html=True)
