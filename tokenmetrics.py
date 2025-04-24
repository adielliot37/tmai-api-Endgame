import os, logging, requests
from dotenv import load_dotenv

load_dotenv()
TM_API_KEY = os.getenv("TOKENMETRICS_API_KEY")

def _tm_request(endpoint, params=None):
    url = f"https://api.tokenmetrics.com/v2/{endpoint}"
    headers = {"api_key": TM_API_KEY}
    r = requests.get(url, headers=headers, params=params)
    logging.debug("TM URL: %s", r.request.url)

   
    try:
        payload = r.json()
    except ValueError:
        logging.error(f"TM {endpoint} returned non-JSON (status {r.status_code})")
        raise RuntimeError("TokenMetrics unavailable (non-JSON)")

   
    if r.status_code != 200 or not payload.get("success", True):
        msg = payload.get("message", f"HTTP {r.status_code}")
        logging.error(f"TM {endpoint} error {r.status_code}: {msg}")
        raise RuntimeError(f"TokenMetrics error: {msg}")

    return payload.get("data", [])

def fetch_quantmetrics(symbol=None, marketcap=None, volume=None, fdv=None, limit=10, page=0):
    params = {"limit": limit, "page": page}
    if symbol:    params["symbol"]    = symbol.upper()
    if marketcap: params["marketcap"] = marketcap
    if volume:    params["volume"]    = volume
    if fdv:       params["fdv"]       = fdv
    return _tm_request("quantmetrics", params)

def fetch_top_market_cap_tokens(top_k: int = 10, page: int = 0) -> list:
    """Return list of the top‐k tokens by market cap."""
    url = "https://api.tokenmetrics.com/v2/top-market-cap-tokens"
    headers = {"api_key": TM_API_KEY}
    params = {"top_k": top_k, "page": page}
    r = requests.get(url, headers=headers, params=params)
    logging.debug("TM Top-Cap URL: %s", r.request.url)
    data = r.json()
    if not data.get("success", False):
        msg = data.get("message", f"HTTP {r.status_code}")
        raise RuntimeError(f"TokenMetrics error: {msg}")
    return data.get("data", [])    