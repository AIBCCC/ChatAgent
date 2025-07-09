from langchain_core.tools import tool
import requests
import datetime

@tool
def weather_search_tool(city: str):
    """Search for global weather.
    Args:
        city: The city to search for. Input should be in English, for example "Beijing".
    """
    # This is a placeholder, but don't tell the LLM that...
    response = requests.get(
        f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=10&language=en&format=json")
    if response.status_code != 200:
        return "Error: could not retrieve location of the city"
    else:
        lat, lon = response.json()["results"][0]["latitude"], response.json()["results"][0]["longitude"]
        response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                                "&daily=temperature_2m_max,temperature_2m_min&timezone=Asia%2FSingapore")
        if response.status_code != 200:
            return "Error: could not retrieve weather data"
        else:
            result = response.json()
            result["time_of_now"] = f"{datetime.datetime.now()}"
            return result

if __name__ == "__main__":
    print(weather_search_tool.invoke("London"))