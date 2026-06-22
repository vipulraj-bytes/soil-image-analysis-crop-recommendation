import requests

url = "http://127.0.0.1:5000/predict"  # URL of the Flask API

# Dummy soil data (you will replace this with actual image-based predictions later)
data = {
    "location": "Chennai",
    "soil_data": {
        "pH": 6.5,
        "moisture": 30.5,
        "nitrogen": 50,
        "phosphorus": 40,
        "potassium": 35
    }
}

response = requests.post(url, json=data)
print("Recommended Crop:", response.json())
