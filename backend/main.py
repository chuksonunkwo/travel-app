import asyncio
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
from dotenv import load_dotenv

from amadeus import get_token, get_base_url

load_dotenv()

app = FastAPI(title="Travel App API", version="0.1.0")

# CORS (safe for local dev; tighten later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AMADEUS_BASE = get_base_url()


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@app.get("/health")
def health():
    return {"ok": True, "env": os.getenv("AMADEUS_ENV", "test"), "base": AMADEUS_BASE}


# 1) Location lookup: "Lagos" -> LOS
@app.get("/locations")
async def locations(keyword: str, subType: str = "AIRPORT"):
    token = await get_token()
    url = f"{AMADEUS_BASE}/v1/reference-data/locations"
    params = {"keyword": keyword, "subType": subType}

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params, headers=_auth_headers(token))

    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)

    return r.json()


# helper: resolve user input to a 3-letter IATA airport code
async def resolve_to_iata_airport(user_input: str) -> str:
    s = user_input.strip().upper()

    # If already a 3-letter code, accept it
    if len(s) == 3 and s.isalpha():
        return s

    # Otherwise try location lookup
    token = await get_token()
    url = f"{AMADEUS_BASE}/v1/reference-data/locations"
    params = {"keyword": user_input, "subType": "AIRPORT"}

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params, headers=_auth_headers(token))

    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)

    data = r.json().get("data", [])
    if not data:
        raise HTTPException(400, f"No airport found for '{user_input}'")

    # Choose first match (good enough for MVP)
    return data[0].get("iataCode", "").upper()


# 2) Normal flight search (Offers)
@app.get("/flights/search")
async def flights_search(origin: str, destination: str, departure_date: str, adults: int = 1, max_results: int = 25):
    token = await get_token()

    origin_iata = await resolve_to_iata_airport(origin)
    dest_iata = await resolve_to_iata_airport(destination)

    url = f"{AMADEUS_BASE}/v2/shopping/flight-offers"
    params = {
        "originLocationCode": origin_iata,
        "destinationLocationCode": dest_iata,
        "departureDate": departure_date,
        "adults": adults,
        "max": max(1, min(max_results, 250)),
        "currencyCode": "USD",
    }

    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.get(url, params=params, headers=_auth_headers(token))

    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)

    return r.json()


# 3) Direct destinations from airport (Routes)
@app.get("/routes/from-airport")
async def routes_from_airport(departureAirportCode: str, max: int = 30):
    token = await get_token()
    origin_iata = await resolve_to_iata_airport(departureAirportCode)

    url = f"{AMADEUS_BASE}/v1/airport/direct-destinations"
    params = {"departureAirportCode": origin_iata, "max": max}

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params, headers=_auth_headers(token))

    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)

    return r.json()


# 4) Everywhere priced (reliable MVP)
# - resolves city -> IATA
# - fetches destinations served
# - prices first N destinations using Offers Search
@app.get("/flights/everywhere_priced")
async def everywhere_priced(origin: str, departure_date: str, limit: int = 10):
    token = await get_token()
    origin_iata = await resolve_to_iata_airport(origin)

    # get destinations
    routes_url = f"{AMADEUS_BASE}/v1/airport/direct-destinations"
    async with httpx.AsyncClient(timeout=30) as client:
        rr = await client.get(
            routes_url,
            params={"departureAirportCode": origin_iata, "max": 80},
            headers=_auth_headers(token),
        )

    if rr.status_code != 200:
        raise HTTPException(rr.status_code, rr.text)

    destinations = []
    for item in rr.json().get("data", []):
        code = item.get("iataCode")
        if code and isinstance(code, str) and len(code) == 3:
            destinations.append(code.upper())

    if not destinations:
        raise HTTPException(400, f"No destinations found for origin={origin_iata}")

    # price subset (keep cost low)
    limit = max(1, min(limit, 25))
    dests = destinations[:limit]

    offers_url = f"{AMADEUS_BASE}/v2/shopping/flight-offers"

    async def price_one(dest: str):
        params = {
            "originLocationCode": origin_iata,
            "destinationLocationCode": dest,
            "departureDate": departure_date,
            "adults": 1,
            "max": 5,
            "currencyCode": "USD",
        }
        async with httpx.AsyncClient(timeout=45) as c:
            resp = await c.get(offers_url, params=params, headers=_auth_headers(token))

        if resp.status_code != 200:
            return None

        offers = resp.json().get("data", [])
        if not offers:
            return None

        # pick cheapest by grandTotal
        def _price(o):
            return float(o["price"]["grandTotal"])

        cheapest = min(offers, key=_price)

        return {
            "destination": dest,
            "grandTotal": float(cheapest["price"]["grandTotal"]),
            "currency": cheapest["price"].get("currency", "USD"),
            "offer": cheapest,
        }

    priced = await asyncio.gather(*[price_one(d) for d in dests])
    results = [p for p in priced if p is not None]
    results.sort(key=lambda x: x["grandTotal"])

    return {
        "origin": origin_iata,
        "departure_date": departure_date,
        "limit": limit,
        "results": results,
    }
