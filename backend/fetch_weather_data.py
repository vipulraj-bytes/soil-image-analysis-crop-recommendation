import requests

def get_weather_data(location):
    api_key = "b22f24aaa09f545a8630d4fc23648686"  # Replace this with your actual OpenWeather API key
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}"
    
    try:
        response = requests.get(base_url)
        response.raise_for_status()  # Raise error if request fails
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Failed to fetch data:", e)
        return None

# Test the function
if __name__ == "__main__":
    location = "Coimbatore"
    data = get_weather_data(location)
    print(data)
