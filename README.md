# Soil Image Analysis & Crop Recommendation System

An AI-powered system that analyzes soil images to determine soil type and recommends suitable crops with weather-based insights and market prices.

## Features

- 🌱 Soil type classification from images using CNN/ResNet models
- 🌾 Crop recommendations based on soil type
- 🌤️ Weather-based crop suitability analysis
- 💰 Market price information for recommended crops
- 🤖 Interactive chatbot for agricultural queries
- 📅 Crop calendar with planting schedules

## Technology Stack

- **Backend**: Python, Flask
- **ML Models**: PyTorch, scikit-learn
- **Frontend**: HTML, CSS, JavaScript
- **APIs**: Weather data integration

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/vipulraj-bytes/soil-image-analysis-crop-recommendation.git
   cd soil-image-analysis-crop-recommendation
   ```

2. **Create virtual environment**
   ```bash
   python -m venv env
   
   # On Windows
   env\Scripts\activate
   
   # On Linux/Mac
   source env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. **Download Model Files** ⚠️
   
   The trained model files are not included in the repository due to size constraints. You have two options:
   
   **Option A: Download Pre-trained Models**
   - Contact the repository owner for model files
   - Place the following files in the `backend/` directory:
     - `best_soil_cnn_model.pkl`
     - `best_soil_resnet18_finetuned.pth`
     - `label_encoder.pkl`
     - `soil_model.pkl`
   
   **Option B: Train Models Yourself**
   - Download the soil dataset (contact owner or use your own)
   - Place dataset in `backend/soil_dataset/`
   - Run training scripts:
     ```bash
     python train_cnn.py
     ```

5. **Run the application**
   ```bash
   python app.py
   ```
   
   The application will start at `http://localhost:5000`

## Usage

1. **Upload Soil Image**: Navigate to the upload page and select a soil image
2. **Get Analysis**: The system will classify the soil type
3. **View Recommendations**: See recommended crops based on your soil type
4. **Check Weather**: Input location for weather-based crop suitability
5. **Market Prices**: View current vegetable prices in Tamil Nadu

## Project Structure

```
soil-image-analysis-crop-recommendation/
├── backend/
│   ├── models/              # Model definitions
│   ├── app.py              # Flask application
│   ├── predict.py          # Prediction logic
│   ├── train_cnn.py        # Model training
│   ├── requirements.txt    # Python dependencies
│   └── crop_guide.json     # Crop information database
├── templates/              # HTML templates
├── static/                 # CSS, JS, images
└── README.md
```

## Model Information

- **Soil Classification**: CNN and ResNet-18 models trained on soil images
- **Soil Types Supported**: Alluvial, Black, Clay, Loamy, Red, Saline, Sandy
- **Accuracy**: Model performance metrics available after training

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Contact

For model files or questions, please open an issue on GitHub.

## Acknowledgments

- Dataset sources: [               ]
- Weather API: [             ]

---

**Note**: This project requires large model files (~500MB total) that are not included in the repository. Please follow the installation instructions to obtain them.
