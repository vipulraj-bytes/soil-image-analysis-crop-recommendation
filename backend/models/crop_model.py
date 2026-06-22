# models/crop_model.py

def predict_crop_yield(weather_data):
    """
    Predict the crop based on the weather data (e.g., temperature, humidity).
    :param weather_data: dict containing weather parameters like temperature, humidity.
    :return: Predicted crop (string).
    """
    temperature = weather_data.get("temperature")
    humidity = weather_data.get("humidity")
    
    # Example rule-based prediction (you can improve this with a real model)
    if temperature > 30 and humidity > 60:
        return "Rice"  # Rice is suitable for hot, humid conditions
    elif temperature < 25 and humidity < 50:
        return "Wheat"  # Wheat is suitable for cooler and dry conditions
    else:
        return "Corn"  # Corn can grow in moderate conditions
