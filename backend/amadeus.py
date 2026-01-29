import os
import time
import httpx

# Base URL controlled by env: test or production
def get_base_url() -> str:
    env = os.getenv("AMADEUS_ENV", "test").strip().lower()
    if env in ("prod", "production", "live"):
        return "https://api.amadeus.com"
    return "https://test.api.amadeus.com"


# Simple in-memory token cache
_token: str | None = None
_token_exp: int = 0  # unix timestamp


async def get_token() -> str:
    """
    Fetch (and cache) OAuth2 token for Amadeus.
    Reads:
      AMADEUS_CLIENT_ID
      AMADEUS_CLIENT_SECRET
      AMADEUS_ENV (test/prod)
    """
    global _token, _token_exp

    now = int(time.time())
    if _token and now < (_token_exp - 30):
        return _token

    client_id = os.getenv("AMADEUS_CLIENT_ID")
    client_secret = os.getenv("AMADEUS_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError("Missing AMADEUS_CLIENT_ID or AMADEUS_CLIENT_SECRET in .env")

    base = get_base_url()
    url = f"{base}/v1/security/oauth2/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, data=data)

    # If error, raise with details
    if r.status_code != 200:
        raise RuntimeError(f"Amadeus token error {r.status_code}: {r.text}")

    payload = r.json()
    _token = payload["access_token"]
    expires_in = int(payload.get("expires_in", 1800))
    _token_exp = now + expires_in
    return _token
