import logging
import requests
from typing import Dict, Any, Optional

from ..core.config import settings

logger = logging.getLogger("cyclone_ai.imd_service")

IMD_API_URL = "https://api.imd.gov.in/api/v1/cyclone_track"

def fetch_live_cyclone_tracks() -> Dict[str, Any]:
    """
    Fetch live cyclone track data from the IMD API.
    
    Returns:
        A dictionary containing the parsed observed and forecast track data.
    """
    if not settings.IMD_API_KEY or settings.IMD_API_KEY == "your_imd_api_key_here":
        logger.warning("IMD_API_KEY is not configured. Returning empty live data.")
        return {
            "status": False,
            "message": "IMD API Key not configured.",
            "data": {"observed": [], "forecast": []}
        }

    headers = {
        # Using Authorization header by default, if IMD uses a different one 
        # (like x-api-key) this may need to be updated.
        "Authorization": f"Bearer {settings.IMD_API_KEY}",
        "Accept": "application/json"
    }

    try:
        response = requests.get(IMD_API_URL, headers=headers, timeout=10)
        
        # If unauthorized, return clear error
        if response.status_code in (401, 403):
            logger.error("IMD API Authorization failed. Check your API key.")
            return {
                "status": False,
                "message": "IMD API Authorization failed. Check API key.",
                "data": {"observed": [], "forecast": []}
            }
            
        response.raise_for_status()
        data = response.json()
        
        # Basic validation of the IMD response format
        if not isinstance(data, dict) or "data" not in data:
            logger.error(f"Unexpected IMD API response format: {data}")
            return {
                "status": False,
                "message": "Invalid response format from IMD API.",
                "data": {"observed": [], "forecast": []}
            }
            
        return {
            "status": True,
            "message": "Successfully fetched live data",
            "data": data.get("data", {"observed": [], "forecast": []})
        }

    except requests.RequestException as e:
        logger.error(f"Error fetching data from IMD API: {e}")
        return {
            "status": False,
            "message": f"Failed to connect to IMD API: {str(e)}",
            "data": {"observed": [], "forecast": []}
        }
