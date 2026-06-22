import requests

def get_weather_data(location):
    api_key = 'b22f24aaa09f545a8630d4fc23648686'
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}"
    response = requests.get(base_url)
    return response.json()
