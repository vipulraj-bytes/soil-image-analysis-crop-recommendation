# backend/app.py

# 1. Imports
import os
import numpy as np
import traceback
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session # Import session
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import datetime
import csv
import random
import json
from dotenv import load_dotenv
import sys
import re # Import regular expression library for Jinja filter

# 2. Configuration
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(BASE_DIR, "best_soil_resnet18_finetuned.pth")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.txt")
CSV_FILENAME = "vegetable_prices_tamilnadu.csv"
MARKET_DATA_CSV = os.path.join(BASE_DIR, CSV_FILENAME)
TIPS_FILE_PATH = os.path.join(BASE_DIR, "farming_tips.txt")
CROP_GUIDE_JSON_PATH = os.path.join(BASE_DIR, "crop_guide.json") # Path to JSON file
CHATBOT_TEMPLATE_PATH = os.path.join(PROJECT_ROOT, 'templates', 'chatbot.html')

print("--- Checking Required Files ---")
required_files = { "Model": MODEL_PATH, "Class Names": CLASS_NAMES_PATH, }
optional_files = { "Market Data CSV": MARKET_DATA_CSV, "Farming Tips": TIPS_FILE_PATH, "Crop Guide JSON": CROP_GUIDE_JSON_PATH, "Chatbot Template": CHATBOT_TEMPLATE_PATH, }
for name, path in required_files.items():
    if not os.path.exists(path): print(f"CRITICAL ERROR: Required file '{name}' not found at: {path}"); raise FileNotFoundError(f"Required file '{name}' not found: {path}")
    else: print(f"OK: Found {name} at {path}")
for name, path in optional_files.items():
    if not os.path.exists(path): print(f"WARNING: Optional file '{name}' not found at: {path}")
    else: print(f"OK: Found {name} at {path}")
print("--- File Checks Complete ---")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_WIDTH = 224; IMAGE_HEIGHT = 224; IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

# --- Load Class Names ---
print(f"INFO: Loading class names from: {CLASS_NAMES_PATH}")
try:
    with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f: class_names = [line.strip() for line in f if line.strip()]
    NUM_CLASSES = len(class_names);
    if NUM_CLASSES == 0: raise ValueError("Class names file is empty.")
    print(f"INFO: Loaded {NUM_CLASSES} class names: {class_names}")
except Exception as e: print(f"CRITICAL ERROR loading class names: {e}"); traceback.print_exc(); raise SystemExit(f"Failed to load class names: {e}")


# --- Climate Normals Data ---
climate_normals_tn = { "chennai": { 1: {"temp": 25, "rain": 25}, 2: {"temp": 27, "rain": 10}, 3: {"temp": 29, "rain": 15}, 4: {"temp": 31, "rain": 20}, 5: {"temp": 33, "rain": 40}, 6: {"temp": 32, "rain": 55}, 7: {"temp": 31, "rain": 100}, 8: {"temp": 30, "rain": 120}, 9: {"temp": 30, "rain": 140}, 10: {"temp": 28, "rain": 270}, 11: {"temp": 26, "rain": 350}, 12: {"temp": 25, "rain": 140} }, "coimbatore": { 1: {"temp": 25, "rain": 5}, 2: {"temp": 27, "rain": 10}, 3: {"temp": 29, "rain": 15}, 4: {"temp": 30, "rain": 50}, 5: {"temp": 29, "rain": 70}, 6: {"temp": 27, "rain": 40}, 7: {"temp": 26, "rain": 50}, 8: {"temp": 26, "rain": 40}, 9: {"temp": 27, "rain": 70}, 10: {"temp": 26, "rain": 150}, 11: {"temp": 25, "rain": 130}, 12: {"temp": 24, "rain": 40} }, "madurai": { 1: {"temp": 26, "rain": 10}, 2: {"temp": 28, "rain": 15}, 3: {"temp": 30, "rain": 25}, 4: {"temp": 32, "rain": 50}, 5: {"temp": 32, "rain": 70}, 6: {"temp": 31, "rain": 40}, 7: {"temp": 30, "rain": 50}, 8: {"temp": 30, "rain": 70}, 9: {"temp": 29, "rain": 100}, 10: {"temp": 28, "rain": 180}, 11: {"temp": 27, "rain": 140}, 12: {"temp": 26, "rain": 50} }, "default": { 1: {"temp": 25, "rain": 10}, 2: {"temp": 27, "rain": 10}, 3: {"temp": 30, "rain": 15}, 4: {"temp": 31, "rain": 40}, 5: {"temp": 31, "rain": 65}, 6: {"temp": 30, "rain": 50}, 7: {"temp": 29, "rain": 70}, 8: {"temp": 29, "rain": 80}, 9: {"temp": 29, "rain": 110}, 10: {"temp": 28, "rain": 180}, 11: {"temp": 26, "rain": 170}, 12: {"temp": 25, "rain": 60} } }
print("INFO: Climate normals data loaded.")

# --- Load Farming Tips ---
farming_tips = ["மாதிரி குறிப்பு: நீர்ப்பாசனம் செய்வதற்கு முன் மண் ஈரப்பதத்தை சரிபார்க்கவும்."]
try:
    if os.path.exists(TIPS_FILE_PATH):
        with open(TIPS_FILE_PATH, 'r', encoding='utf-8') as f: tips_from_file = [line.strip() for line in f if line.strip()]
        if tips_from_file: farming_tips = tips_from_file
except Exception as e: print(f"ERROR loading tips file '{TIPS_FILE_PATH}': {e}")
print(f"INFO: Loaded {len(farming_tips)} farming tips.")

# --- Hardcoded English to Tamil Vegetable Name Mapping (Used for Ticker only) ---
veg_english_to_tamil_map = {"tomato": "தக்காளி", "onion (big)": "வெங்காயம்", "potato": "உருளைக்கிழங்கு", "green chilli": "பச்சை மிளகாய்", "beetroot": "பீட்ரூட்", "carrot": "கேரட்", "cabbage": "முட்டைக்கோசு", "brinjal": "கத்திரிக்காய்", "ladies finger": "வெண்டைக்காய்", "radish": "முள்ளங்கி", "rice": "அரிசி"}
print("INFO: Hardcoded English-to-Tamil vegetable name map created.")

# ======================================================
# ===== MARKET DATA CSV LOADING SECTION ==============
# ======================================================
market_price_data = [] ; ticker_display_list = []
print(f"INFO: Attempting to load market prices from: {MARKET_DATA_CSV}")
try:
    if os.path.exists(MARKET_DATA_CSV):
        print(f"INFO: File found at {MARKET_DATA_CSV}. Proceeding to read...")
        with open(MARKET_DATA_CSV, mode='r', newline='', encoding='utf-8-sig') as csvfile:
            try: dialect = csv.Sniffer().sniff(csvfile.read(2048)); csvfile.seek(0); reader = csv.DictReader(csvfile, dialect=dialect); print(f"INFO: Detected CSV dialect: delimiter='{dialect.delimiter}', quotechar='{dialect.quotechar}'")
            except csv.Error as sniff_err: print(f"WARNING: Could not determine CSV dialect ({sniff_err}). Reading with default comma delimiter."); csvfile.seek(0); reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames; required_headers = ['English Name', 'Date', 'Wholesale Price (?/kg)', 'Retail Price Max (?/kg)']
            if not fieldnames: print("ERROR: CSV file appears to be empty or has no header row.")
            elif not all(header in fieldnames for header in required_headers): missing_headers = [h for h in required_headers if h not in fieldnames]; print(f"ERROR: Market price CSV missing required headers: {missing_headers}. Found headers: {fieldnames}")
            else:
                print(f"INFO: CSV Headers OK: {fieldnames}"); temp_data = {}
                processed_count = 0; error_count = 0; skipped_missing_data = 0; skipped_bad_date = 0
                for row_num, row in enumerate(reader, 1):
                    try:
                        cleaned_row = {k.strip(): v.strip() for k, v in row.items() if k is not None and k.strip() != ''}; veg = cleaned_row.get('English Name','').lower(); date_str = cleaned_row.get('Date')
                        if not veg or not date_str: skipped_missing_data += 1; continue
                        key = veg; current_date = None; date_formats = ['%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%m/%d/%Y']
                        for fmt in date_formats:
                            try: current_date = datetime.datetime.strptime(date_str, fmt); break
                            except ValueError: continue
                        if current_date is None: skipped_bad_date += 1; continue
                        existing_record = temp_data.get(key)
                        if not existing_record or current_date >= existing_record.get('parsed_date', datetime.datetime.min): cleaned_row['parsed_date'] = current_date; temp_data[key] = cleaned_row
                        processed_count += 1
                    except Exception as row_e: print(f"ERROR processing market data row {row_num}: {row_e} - Row content: {row}"); error_count += 1
                market_price_data = list(temp_data.values())
                print(f"INFO: Finished processing CSV. Processed rows: {processed_count}, Skipped (missing data): {skipped_missing_data}, Skipped (bad date): {skipped_bad_date}, Errors: {error_count}.")
                print(f"INFO: Stored {len(market_price_data)} unique latest market price records (one per vegetable).")
                print(f"INFO: Generating ticker display strings using hardcoded map..."); ticker_display_list = []
                for record in market_price_data:
                    try:
                        english_name_lower = record.get('English Name', '').lower();
                        if not english_name_lower: continue
                        tamil_name = veg_english_to_tamil_map.get(english_name_lower, english_name_lower.capitalize()); retail_price_str = record.get('Retail Price Max (?/kg)') or 'N/A'; wholesale_price_str = record.get('Wholesale Price (?/kg)') or 'N/A'
                        retail_price = retail_price_str.replace('?','').strip() if retail_price_str != 'N/A' else 'N/A'; wholesale_price = wholesale_price_str.replace('?','').strip() if wholesale_price_str != 'N/A' else 'N/A'
                        display_string = f"{tamil_name}: சில்லறை ₹{retail_price} | மொத்த ₹{wholesale_price}";
                        if retail_price != 'N/A' or wholesale_price != 'N/A': display_string += "/kg"
                        ticker_display_list.append(display_string)
                    except Exception as ticker_err: print(f"ERROR generating ticker string for record {record}: {ticker_err}")
                print(f"INFO: Generated {len(ticker_display_list)} ticker display strings."); print(f"INFO: Ticker Content: {ticker_display_list}")
    else: print(f"ERROR: Market price CSV file not found at the specified path: {MARKET_DATA_CSV}"); print("INFO: Market price data will be empty.")
except Exception as e: print(f"CRITICAL ERROR during market price CSV loading/processing: {e}"); traceback.print_exc(); print("INFO: Market price data loading failed."); market_price_data = []; ticker_display_list = []
# ======================================================
# === END OF DETAILED MARKET DATA CSV LOADING SECTION ===
# ======================================================

# ***** Load Crop Guide Data from JSON File *****
crop_guide_data = {}
print(f"INFO: Loading crop guide from: {CROP_GUIDE_JSON_PATH}")
if os.path.exists(CROP_GUIDE_JSON_PATH):
    try:
        with open(CROP_GUIDE_JSON_PATH, 'r', encoding='utf-8') as f:
            crop_guide_data = json.load(f)
        print(f"INFO: Loaded {len(crop_guide_data)} crop guide entries from JSON.")
    except json.JSONDecodeError as e: print(f"ERROR: Crop guide file '{CROP_GUIDE_JSON_PATH}' is not valid JSON: {e}")
    except Exception as e: print(f"ERROR loading crop guide json: {e}")
else: print(f"WARNING: Crop guide file not found at {CROP_GUIDE_JSON_PATH}. Chatbot crop info will be limited.")
# ***** END: Load Crop Guide Data *****

# ***** Farming Calendar Data *****
farming_calendar_data = {1: ["நெல் அறுவடைப் பணிகள் (சம்பா/தாளடி).","கரும்பு அறுவடை மற்றும் நடவு.","பயறு வகைகள் விதைப்பு (நிலக்கடலை, உளுந்து, பாசிப்பயறு).","குளிர்கால காய்கறிகள் பராமரிப்பு.","மரங்களுக்கு உரமிடுதல்."], 2: ["நெல் அறுவடை தொடர்ச்சி.","நிலக்கடலை, பயறு வகைகள் பராமரிப்பு.","கோடைக்கால காய்கறிப் பயிர்கள் நடவு (வெண்டை, கத்தரி, தக்காளி).","மாமரங்களில் பூ பூக்கும் பருவம் - பூச்சி மேலாண்மை.","நிலத்தை கோடை உழவுக்குத் தயார் செய்தல்."], 3: ["கோடை உழவு செய்தல்.","கோடைக்கால காய்கறிகள் பராமரிப்பு மற்றும் நீர்ப்பாசனம்.","பருத்தி விதைப்பு (சில பகுதிகளில்).","மாம்பழம் அறுவடை ஆரம்பம் (சில வகைகள்).","தென்னை மரங்களுக்கு உரமிடுதல்."], 4: ["சித்திரை முதல் பட்டத்திற்கான நெல் நடவு (சில பகுதிகளில்).","காய்கறிப் பயிர்களில் பூச்சி மற்றும் நோய் மேலாண்மை.","பழ மரங்களுக்கு நீர் பாய்ச்சுதல்.","நிலக்கடலை அறுவடை (கோடைப் பயிர்).","கால்நடைகளுக்கு பசுந்தீவன மேலாண்மை."], 5: ["கோடை மழை கிடைத்தால் நிலத்தை தயார் செய்தல்.","பருத்திப் பயிர் பராமரிப்பு.","காய்கறி அறுவடை.","மாம்பழம் அறுவடை உச்சம்.","அக்னி நட்சத்திரம் - பயிர்களுக்கு கூடுதல் கவனம் தேவை."], 6: ["தென்மேற்கு பருவமழை ஆரம்பம் - குருவை நெல் நடவு.","மக்காச்சோளம், கம்பு, கேழ்வரகு விதைப்பு.","பருத்தி, கரும்பு பராமரிப்பு.","நிலக்கடலை விதைப்பு (ஆடிப் பட்டம்).","மழைநீர் சேகரிப்பு கட்டமைப்புகளை சரிபார்த்தல்."], 7: ["குருவை நெல் பயிர் வளர்ச்சிப் பருவம் - உரமிடுதல், களை எடுத்தல்.","மக்காச்சோளம், சிறுதானியங்கள் பராமரிப்பு.","காய்கறிப் பயிர்கள் நடவு.","பயறு வகைகள் விதைப்பு.","மரக்கன்றுகள் நடுதல்."], 8: ["நெல் பயிரில் பூச்சி/நோய் கண்காணிப்பு.","சம்பா/தாளடி நெல் நடவுக்கான நாற்றங்கால் தயாரிப்பு.","கரும்பு பராமரிப்பு.","காய்கறி அறுவடை.","முக்கிய உரங்கள் இடுதல் (பயிர் தேவையைப் பொறுத்து)."], 9: ["சம்பா/தாளடி நெல் நடவுப் பணிகள்.","குருவை நெல் அறுவடைக்குத் தயாராகுதல்.","மக்காச்சோளம், கரும்பு பராமரிப்பு.","மழைக்கால காய்கறிகள் பராமரிப்பு.","நிலக்கடலைப் பயிரில் பூச்சி மேலாண்மை."], 10: ["வடகிழக்கு பருவமழை ஆரம்பம்.","சம்பா/தாளடி நெல் பயிர் பராமரிப்பு.","குருவை நெல் அறுவடை.","மழைக்கால பயறு வகைகள் விதைப்பு.","காய்கறிகள், பழ மரங்களுக்கு கூடுதல் கவனம்."], 11: ["நெல் பயிரில் நீர் மேலாண்மை (மழையைப் பொறுத்து).","பருத்தி அறுவடை ஆரம்பம்.","குளிர்கால காய்கறிகள் நடவு (வெங்காயம், உருளை).","கரும்பு வளர்ச்சிப் பருவம்.","மழைநீர் வடிகால் வசதிகளை உறுதி செய்தல்."], 12: ["சம்பா/தாளடி நெல் கதிர் வரும் பருவம்.","குளிர்கால காய்கறிகள் பராமரிப்பு.","பருத்தி அறுவடை தொடர்ச்சி.","அறுவடைக்கு பிந்தைய தொழில்நுட்பங்கள்.","அடுத்த பருவத்திற்கான திட்டமிடல்."]}
print("INFO: Farming calendar data loaded.")

# ***** Mapping for Display Names (Used to build crop_recommendations) *****
_crop_display_map_builder = {"paddy": "நெல் (Paddy)", "rice": "நெல் (Rice)", "wheat": "கோதுமை (Wheat)", "sugarcane": "சர்க்கரைக்கம்பு (Sugarcane)", "maize": "மக்காச்சோளம் (Maize)", "sorghum": "சோளம் (Sorghum)", "green_gram": "பாசிப்பயறு (Green Gram)", "red_gram": "துவரம்பருப்பு (Red Gram)", "black_gram": "உளுந்து (Black Gram)", "indian_gooseberry": "நெல்லிக்காய் (Indian Gooseberry)", "banana": "வாழை (Banana)", "coconut": "தேங்காய் (Coconut)", "guava": "கொய்யா (Guava)", "mango": "மாம்பழம் (Mango)", "papaya": "பப்பாளி (Papaya)", "ash_gourd": "பூசணிக்காய் (Ash Gourd)", "ladies_finger": "வெண்டைக்காய் (Ladies Finger)", "ridge_gourd": "பீர்க்கங்காய் (Ridge Gourd)", "peas": "பட்டாணி (Peas)", "cowpea": "தட்டைப்பயறு (Cowpea)", "garlic": "பூண்டு (Garlic)", "cotton": "பருத்தி (Cotton)", "pepper": "மிளகு (Pepper)", "black_pepper": "கரு மிளகு (Black Pepper)", "turmeric": "மஞ்சள் (Turmeric)", "ginger": "இஞ்சி (Ginger)", "butter_beans": "வெண்ணெய் அவரை (Butter Beans)", "onion": "வெங்காயம் (Onion)", "chilli": "மிளகாய் (Chilli)", "beans": "பீன்ஸ் (Beans)", "field_bean": "அவரைக்காய் (Field Bean)", "lablab": "மொச்சை (Lablab)", "lima_bean": "மொச்சை அவரை (Lima Bean)", "brinjal": "கத்திரிக்காய் (Brinjal)", "tea": "தேயிலை (Tea)", "coffee": "காபி (Coffee)", "coriander": "கொத்தமல்லி (Coriander)", "groundnut": "நிலக்கடலை (Groundnut)", "tapioca": "மரவள்ளி கிழங்கு (Tapioca)", "drumstick": "முருங்கை (Drumstick)", "spinach": "பசலைக்கீரை (Spinach)", "cumin": "சீரகம் (Cumin)", "fennel": "சோம்பு (Fennel)", "radish": "முள்ளங்கி (Radish)", "cassava": "மரவள்ளி கிழங்கு (Cassava)", "capsicum": "குடை மிளகாய் (Capsicum)", "mint": "புதினா (Mint)", "wild_turmeric": "கஸ்தூரி மஞ்சள் (Wild Turmeric)", "cashew": "முந்திரி (Cashew)", "jackfruit": "பலா (Jackfruit)", "rubber": "ரப்பர் (Rubber)", "arecanut": "பாக்கு (Arecanut)", "pineapple": "அன்னாசி (Pineapple)", "finger_millet": "கேழ்வரகு (Finger Millet)", "horse_gram": "கொள்ளு (Horse Gram)", "castor": "ஆமணக்கு (Castor)", "potato": "உருளைக்கிழங்கு (Potato)", "jute": "சணல் (Jute)", "sunflower": "சூரியகாந்தி (Sunflower)", "soybean": "சோயாபீன்ஸ் (Soybean)", "linseed": "ஆளிவிதை (Linseed)", "cabbage": "முட்டைக்கோசு (Cabbage)", "cauliflower": "பூக்கோசு (Cauliflower)", "watermelon": "தர்பூசணி (Watermelon)", "berseem": "பெர்சீம் (Berseem)", "sesame": "எள் (Sesame)", "casuarina": "சவுக்கு (Casuarina)", "carrot": "கேரட் (Carrot)", "cucumber": "வெள்ளரி (Cucumber)", "sugar_beet": "சர்க்கரை வள்ளிக்கிழங்கு (Sugar Beet)", "barley": "பார்லி (Barley)", "specific_rice_varieties": "உவர்நில நெல் வகைகள் (Specific Rice Varieties)", "sesbania": "செஸ்பானியா (Sesbania)", "moth_bean": "நரிப்பயறு (Moth Bean)", "guar": "கொத்தவரை (Guar)", "dates": "பேரீச்சை (Dates)", "cluster_beans": "கொத்தவரை (Cluster Beans)", "certain_vegetables": "சில காய்கறிகள் (Certain Vegetables)", "tobacco": "புகையிலை (Tobacco)", "millet": "தினை (Millet)", "millets": "சிறுதானியங்கள் (Millets)"}
def get_display_name(key): return _crop_display_map_builder.get(key.lower().replace(' ', '_').replace('(','').replace(')',''), key.capitalize())

# ***** EXPANDED CROP RECOMMENDATIONS (Uses Tamil + English Format) *****
crop_recommendations = {
    "alluvial": sorted([ get_display_name(k) for k in ["Rice", "Sugarcane", "Wheat", "Maize", "Banana", "Turmeric", "Groundnut", "Black Gram", "Green Gram", "Cowpea", "Sesame", "Brinjal", "Tomato", "Chilli", "Ladies Finger", "Gourds", "Jute", "Mustard", "Coriander", "Coconut"] ]),
    "black": sorted([ get_display_name(k) for k in ["Cotton", "Sugarcane", "Maize", "Sorghum", "Red Gram", "Black Gram", "Green Gram", "Paddy", "Pepper", "Turmeric", "Ginger", "Butter Beans", "Peas", "Cowpea", "Garlic", "Onion", "Chili", "Beans", "Field Bean", "Lablab", "Soybean", "Sunflower", "Wheat", "Linseed", "Gingelly", "Tobacco"] ]), # Combined and sorted
    "clay": sorted([ get_display_name(k) for k in ["Paddy", "Sugarcane", "Maize", "Sorghum", "Red Gram", "Black Gram", "Green Gram", "Pepper", "Turmeric", "Ginger", "Banana", "Coconut", "Mango", "Papaya", "Ash Gourd", "Ladies Finger", "Ridge Gourd", "Peas", "Cowpea", "Garlic", "Jute", "Cabbage", "Cauliflower", "Watermelon", "Mustard", "Soybean", "Berseem", "Wheat", "Tomato"] ]), # Combined and sorted
    "red": sorted([ get_display_name(k) for k in ["Paddy", "Maize", "Sorghum", "Green Gram", "Red Gram", "Black Gram", "Turmeric", "Ginger", "Black Pepper", "Banana", "Coconut", "Mango", "Papaya", "Ash Gourd", "Ladies Finger", "Ridge Gourd", "Peas", "Cowpea", "Garlic", "Onion", "Groundnut", "Finger Millet", "Horse Gram", "Cotton", "Castor", "Sesame", "Tapioca", "Guava", "Radish", "Potato", "Jackfruit", "Chilli", "Tobacco", "Millets"] ]), # Combined and sorted
    "laterite": sorted([ get_display_name(k) for k in ["Tea", "Coffee", "Turmeric", "Ginger", "Black Pepper", "Coriander", "Banana", "Coconut", "Mango", "Papaya", "Ash Gourd", "Ladies Finger", "Ridge Gourd", "Peas", "Cowpea", "Garlic", "Onion", "Chili", "Beans", "Field Bean", "Cashew", "Rubber", "Jackfruit", "Arecanut", "Pineapple", "Tapioca"] ]), # Combined and sorted
    "loamy": sorted([ get_display_name(k) for k in ["Paddy", "Wheat", "Sugarcane", "Maize", "Sorghum", "Green Gram", "Red Gram", "Black Gram", "Black Pepper", "Turmeric", "Ginger", "Banana", "Coconut", "Mango", "Papaya", "Ash Gourd", "Ladies Finger", "Ridge Gourd", "Peas", "Cowpea", "Cotton", "Groundnut", "Potato", "Onion", "Carrot", "Beans", "Cabbage", "Cauliflower", "Sunflower", "Barley", "Brinjal"] ]), # Combined and sorted
    "peaty": sorted([ get_display_name(k) for k in ["Paddy", "Green Gram", "Red Gram", "Black Gram", "Maize", "Sorghum", "Turmeric", "Ginger", "Black Pepper", "Banana", "Coconut", "Mango", "Papaya", "Ash Gourd", "Ladies Finger", "Ridge Gourd", "Peas", "Cowpea", "Garlic", "Onion", "Certain Vegetables"] ]), # Combined and sorted
    "saline": sorted([ get_display_name(k) for k in ["Paddy", "Castor", "Sorghum", "Maize", "Jute", "Spinach", "Tomato", "Onion", "Brinjal", "Ladies Finger", "Beans", "Drumstick", "Garlic", "Chili", "Cassava", "Ginger", "Radish", "Capsicum", "Mint", "Wild Turmeric", "Cotton", "Sugar Beet", "Barley", "Mustard", "Sunflower", "Specific Rice Varieties", "Sesbania", "Dates"] ]), # Combined and sorted
    "sandy": sorted([ get_display_name(k) for k in ["Groundnut", "Sorghum", "Maize", "Tapioca", "Ladies Finger", "Beans", "Drumstick", "Onion", "Garlic", "Chili", "Tomato", "Brinjal", "Spinach", "Cumin", "Fennel", "Ginger", "Turmeric", "Radish", "Coconut", "Banana", "Watermelon", "Cucumber", "Cashew", "Casuarina", "Sesame", "Horse Gram", "Carrot", "Pearl Millet"] ]), # Combined and sorted
    "desert": sorted([ get_display_name(k) for k in ["Brinjal", "Chili", "Peas", "Cowpea", "Garlic", "Onion", "Pepper", "Turmeric", "Ginger", "Banana", "Coconut", "Mango", "Papaya", "Ash Gourd", "Ladies Finger", "Ridge Gourd", "Field Bean", "Lablab", "Beans", "Lima Bean", "Pearl Millet", "Moth Bean", "Guar", "Dates", "Cluster Beans", "Castor", "Millet"] ]) # Combined and sorted
}
print("INFO: Expanded and sorted Crop Recommendations Loaded with Tamil (English) format.")


# 3. Flask App Initialization
app = Flask(__name__, template_folder=os.path.join(PROJECT_ROOT, 'templates'), static_folder=os.path.join(PROJECT_ROOT, 'static'))
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24)); CORS(app); print("INFO: Flask app initialized.") # Set secret key for sessions

# --- Custom Jinja Filter ---
def extract_eng_key(value):
    """Extracts the English part from 'Tamil (English)' format for image filename."""
    match = re.search(r'\(([^)]+)\)[^)]*$', str(value))
    if match:
        eng_name = match.group(1).strip()
        # Handle specific known cases explicitly for robustness
        if eng_name == "Paddy": return "rice"
        if eng_name == "Cassava": return "tapioca"
        if eng_name == "Gingelly": return "sesame"
        if eng_name == "Guar": return "cluster_beans" # Map Guar to cluster_beans image
        if eng_name == "Sorghum": return "sorghum"
        if eng_name == "Pearl Millet": return "pearl_millet"
        if eng_name == "Finger Millet": return "finger_millet"
        if eng_name == "Black Pepper": return "black_pepper"
        if eng_name == "Specific Rice Varieties": return "rice"
        if eng_name == "Certain Vegetables": return "default_crop"
        if eng_name == "Indian Gooseberry": return "indian_gooseberry"
        if eng_name == "Ladies Finger": return "ladies_finger"
        if eng_name == "Ridge Gourd": return "ridge_gourd"
        if eng_name == "Butter Beans": return "butter_beans"
        if eng_name == "Field Bean": return "field_bean"
        if eng_name == "Lima Bean": return "lima_bean"
        if eng_name == "Wild Turmeric": return "wild_turmeric"
        if eng_name == "Cluster Beans": return "cluster_beans"
        if eng_name == "Moth Bean": return "moth_bean"
        if eng_name == "Sugar Beet": return "sugar_beet"

        # General cleaning for other parenthesized names
        return eng_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    else:
        # Fallback if no parentheses found - clean the whole string
        cleaned = str(value).strip().lower().replace(' ', '_').replace('(', '').replace(')', '')
        return cleaned
app.jinja_env.filters['extract_eng_key'] = extract_eng_key
print("INFO: Custom Jinja filter 'extract_eng_key' registered.")

# 4. Load PyTorch Model
print("INFO: Setting up ResNet18 model structure...")
try: model = models.resnet18(weights=None); num_ftrs = model.fc.in_features; model.fc = nn.Linear(num_ftrs, NUM_CLASSES); model.load_state_dict(torch.load(MODEL_PATH, map_location=device)); model = model.to(device); model.eval(); print(f"INFO: Model loaded successfully to {device}.")
except Exception as e: print(f"CRITICAL ERROR loading PyTorch model: {e}"); traceback.print_exc(); raise SystemExit(f"Failed to load PyTorch model: {e}")

# 5. Define Image Preprocessing
imagenet_mean = [0.485, 0.456, 0.406]; imagenet_std = [0.229, 0.224, 0.225]; preprocess_transform = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor(), transforms.Normalize(imagenet_mean, imagenet_std)])
print("INFO: PyTorch preprocessing transforms defined.")
def preprocess_image_for_pytorch(image_bytes):
    try: img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB'); input_tensor = preprocess_transform(img_pil); return input_tensor.unsqueeze(0).to(device)
    except Exception as e: print(f"[PyTorch Preprocess] ERROR: {e}"); return None

# 6. Define Crop Data & Recommendations
# ***** EXPANDED Placeholder DB (NEEDS VALIDATION) *****
placeholder_crops_db = [
    # --- Existing & Updated ---
    {"crop_id": 1, "name": "கோதுமை (Wheat)", "english_name": "Wheat", "optimal_planting_start_month": 10, "optimal_planting_end_month": 11, "avg_growth_duration_days": 120, "optimal_temp_min_c": 10.0, "optimal_temp_max_c": 25.0, "optimal_rainfall_min_mm": 300},
    {"crop_id": 2, "name": "மக்காச்சோளம் (Maize)", "english_name": "Maize", "optimal_planting_start_month": 6, "optimal_planting_end_month": 8, "avg_growth_duration_days": 100, "optimal_temp_min_c": 20.0, "optimal_temp_max_c": 34.0, "optimal_rainfall_min_mm": 450},
    {"crop_id": 3, "name": "தக்காளி (Tomato)", "english_name": "Tomato", "optimal_planting_start_month": 1, "optimal_planting_end_month": 12, "avg_growth_duration_days": 85, "optimal_temp_min_c": 18.0, "optimal_temp_max_c": 32.0, "optimal_rainfall_min_mm": 400},
    {"crop_id": 4, "name": "பருத்தி (Cotton)", "english_name": "Cotton", "optimal_planting_start_month": 4, "optimal_planting_end_month": 9, "avg_growth_duration_days": 165, "optimal_temp_min_c": 21.0, "optimal_temp_max_c": 35.0, "optimal_rainfall_min_mm": 500},
    {"crop_id": 5, "name": "நெல் (Rice)", "english_name": "Rice", "optimal_planting_start_month": 6, "optimal_planting_end_month": 11, "avg_growth_duration_days": 130, "optimal_temp_min_c": 20.0, "optimal_temp_max_c": 37.0, "optimal_rainfall_min_mm": 1200},
    {"crop_id": 6, "name": "கடுகு (Mustard)", "english_name": "Mustard", "optimal_planting_start_month": 10, "optimal_planting_end_month": 11, "avg_growth_duration_days": 110, "optimal_temp_min_c": 10.0, "optimal_temp_max_c": 28.0, "optimal_rainfall_min_mm": 250},
    {"crop_id": 8, "name": "பார்லி (Barley)", "english_name": "Barley", "optimal_planting_start_month": 10, "optimal_planting_end_month": 12, "avg_growth_duration_days": 100, "optimal_temp_min_c": 5.0, "optimal_temp_max_c": 22.0, "optimal_rainfall_min_mm": 200},
    {"crop_id": 9, "name": "கரும்பு (Sugarcane)", "english_name": "Sugarcane", "optimal_planting_start_month": 12, "optimal_planting_end_month": 3, "avg_growth_duration_days": 360, "optimal_temp_min_c": 20.0, "optimal_temp_max_c": 35.0, "optimal_rainfall_min_mm": 1500},
    {"crop_id": 10, "name": "வாழை (Banana)", "english_name": "Banana", "optimal_planting_start_month": 1, "optimal_planting_end_month": 12, "avg_growth_duration_days": 330, "optimal_temp_min_c": 25.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 1800},
    {"crop_id": 11, "name": "மஞ்சள் (Turmeric)", "english_name": "Turmeric", "optimal_planting_start_month": 5, "optimal_planting_end_month": 6, "avg_growth_duration_days": 240, "optimal_temp_min_c": 20.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 1000},
    {"crop_id": 12, "name": "நிலக்கடலை (Groundnut)", "english_name": "Groundnut", "optimal_planting_start_month": 6, "optimal_planting_end_month": 7, "avg_growth_duration_days": 110, "optimal_temp_min_c": 25.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 500},
    {"crop_id": 13, "name": "உளுந்து (Black Gram)", "english_name": "Black Gram", "optimal_planting_start_month": 6, "optimal_planting_end_month": 9, "avg_growth_duration_days": 75, "optimal_temp_min_c": 25.0, "optimal_temp_max_c": 35.0, "optimal_rainfall_min_mm": 400},
    {"crop_id": 14, "name": "பாசிப்பயறு (Green Gram)", "english_name": "Green Gram", "optimal_planting_start_month": 6, "optimal_planting_end_month": 9, "avg_growth_duration_days": 70, "optimal_temp_min_c": 25.0, "optimal_temp_max_c": 35.0, "optimal_rainfall_min_mm": 350},
    {"crop_id": 15, "name": "தட்டைப்பயறு (Cowpea)", "english_name": "Cowpea", "optimal_planting_start_month": 6, "optimal_planting_end_month": 8, "avg_growth_duration_days": 80, "optimal_temp_min_c": 20.0, "optimal_temp_max_c": 35.0, "optimal_rainfall_min_mm": 400},
    {"crop_id": 16, "name": "எள் (Sesame)", "english_name": "Sesame", "optimal_planting_start_month": 6, "optimal_planting_end_month": 8, "avg_growth_duration_days": 90, "optimal_temp_min_c": 25.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 400},
    {"crop_id": 17, "name": "கத்திரிக்காய் (Brinjal)", "english_name": "Brinjal", "optimal_planting_start_month": 1, "optimal_planting_end_month": 12, "avg_growth_duration_days": 100, "optimal_temp_min_c": 21.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 500},
    {"crop_id": 18, "name": "மிளகாய் (Chilli)", "english_name": "Chilli", "optimal_planting_start_month": 6, "optimal_planting_end_month": 11, "avg_growth_duration_days": 150, "optimal_temp_min_c": 20.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 600},
    {"crop_id": 19, "name": "வெண்டைக்காய் (Ladies Finger)", "english_name": "Ladies Finger", "optimal_planting_start_month": 2, "optimal_planting_end_month": 8, "avg_growth_duration_days": 55, "optimal_temp_min_c": 24.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 400},
    {"crop_id": 20, "name": "சுரை/பீர்க்கன் (Gourds)", "english_name": "Gourds", "optimal_planting_start_month": 6, "optimal_planting_end_month": 7, "avg_growth_duration_days": 90, "optimal_temp_min_c": 24.0, "optimal_temp_max_c": 35.0, "optimal_rainfall_min_mm": 600}, # Placeholder
    {"crop_id": 21, "name": "சணல் (Jute)", "english_name": "Jute", "optimal_planting_start_month": 3, "optimal_planting_end_month": 5, "avg_growth_duration_days": 120, "optimal_temp_min_c": 24.0, "optimal_temp_max_c": 35.0, "optimal_rainfall_min_mm": 1500},
    {"crop_id": 22, "name": "கொத்தமல்லி (Coriander)", "english_name": "Coriander", "optimal_planting_start_month": 10, "optimal_planting_end_month": 11, "avg_growth_duration_days": 45, "optimal_temp_min_c": 15.0, "optimal_temp_max_c": 25.0, "optimal_rainfall_min_mm": 250},
    {"crop_id": 23, "name": "தென்னை (Coconut)", "english_name": "Coconut", "optimal_planting_start_month": 1, "optimal_planting_end_month": 12, "avg_growth_duration_days": 2000, "optimal_temp_min_c": 20.0, "optimal_temp_max_c": 32.0, "optimal_rainfall_min_mm": 1500},
    {"crop_id": 24, "name": "சோளம் (Sorghum)", "english_name": "Sorghum", "optimal_planting_start_month": 6, "optimal_planting_end_month": 8, "avg_growth_duration_days": 110, "optimal_temp_min_c": 25.0, "optimal_temp_max_c": 35.0, "optimal_rainfall_min_mm": 400},
    {"crop_id": 25, "name": "கம்பு (Pearl Millet)", "english_name": "Pearl Millet", "optimal_planting_start_month": 6, "optimal_planting_end_month": 8, "avg_growth_duration_days": 90, "optimal_temp_min_c": 25.0, "optimal_temp_max_c": 35.0, "optimal_rainfall_min_mm": 300},
    {"crop_id": 26, "name": "துவரை (Red Gram)", "english_name": "Red Gram", "optimal_planting_start_month": 6, "optimal_planting_end_month": 7, "avg_growth_duration_days": 180, "optimal_temp_min_c": 20.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 600},
    {"crop_id": 27, "name": "சோயாபீன்ஸ் (Soybean)", "english_name": "Soybean", "optimal_planting_start_month": 6, "optimal_planting_end_month": 7, "avg_growth_duration_days": 100, "optimal_temp_min_c": 20.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 600},
    {"crop_id": 28, "name": "சூரியகாந்தி (Sunflower)", "english_name": "Sunflower", "optimal_planting_start_month": 1, "optimal_planting_end_month": 12, "avg_growth_duration_days": 90, "optimal_temp_min_c": 18.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 500},
    {"crop_id": 29, "name": "வெங்காயம் (Onion)", "english_name": "Onion", "optimal_planting_start_month": 10, "optimal_planting_end_month": 11, "avg_growth_duration_days": 120, "optimal_temp_min_c": 15.0, "optimal_temp_max_c": 25.0, "optimal_rainfall_min_mm": 600},
    {"crop_id": 30, "name": "ஆளிவிதை (Linseed)", "english_name": "Linseed", "optimal_planting_start_month": 10, "optimal_planting_end_month": 11, "avg_growth_duration_days": 130, "optimal_temp_min_c": 10.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 400},
    {"crop_id": 31, "name": "கரு மிளகு (Black Pepper)", "english_name": "Black Pepper", "optimal_planting_start_month": 6, "optimal_planting_end_month": 7, "avg_growth_duration_days": 1500, "optimal_temp_min_c": 10.0, "optimal_temp_max_c": 40.0, "optimal_rainfall_min_mm": 2500},
    {"crop_id": 32, "name": "இஞ்சி (Ginger)", "english_name": "Ginger", "optimal_planting_start_month": 4, "optimal_planting_end_month": 5, "avg_growth_duration_days": 270, "optimal_temp_min_c": 20.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 1500},
    {"crop_id": 33, "name": "வெண்ணெய் அவரை (Butter Beans)", "english_name": "Butter Beans", "optimal_planting_start_month": 6, "optimal_planting_end_month": 7, "avg_growth_duration_days": 80, "optimal_temp_min_c": 18.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 500},
    {"crop_id": 34, "name": "பட்டாணி (Peas)", "english_name": "Peas", "optimal_planting_start_month": 10, "optimal_planting_end_month": 11, "avg_growth_duration_days": 70, "optimal_temp_min_c": 10.0, "optimal_temp_max_c": 20.0, "optimal_rainfall_min_mm": 350},
    {"crop_id": 35, "name": "பூண்டு (Garlic)", "english_name": "Garlic", "optimal_planting_start_month": 10, "optimal_planting_end_month": 11, "avg_growth_duration_days": 150, "optimal_temp_min_c": 12.0, "optimal_temp_max_c": 25.0, "optimal_rainfall_min_mm": 600},
    {"crop_id": 36, "name": "பீன்ஸ் (Beans)", "english_name": "Beans", "optimal_planting_start_month": 7, "optimal_planting_end_month": 11, "avg_growth_duration_days": 60, "optimal_temp_min_c": 18.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 400},
    {"crop_id": 37, "name": "அவரைக்காய் (Field Bean)", "english_name": "Field Bean", "optimal_planting_start_month": 9, "optimal_planting_end_month": 11, "avg_growth_duration_days": 100, "optimal_temp_min_c": 15.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 450},
    {"crop_id": 38, "name": "மொச்சை (Lablab)", "english_name": "Lablab", "optimal_planting_start_month": 9, "optimal_planting_end_month": 11, "avg_growth_duration_days": 120, "optimal_temp_min_c": 15.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 450},
    {"crop_id": 39, "name": "மொச்சை அவரை (Lima Bean)", "english_name": "Lima Bean", "optimal_planting_start_month": 6, "optimal_planting_end_month": 7, "avg_growth_duration_days": 80, "optimal_temp_min_c": 18.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 500},
    {"crop_id": 40, "name": "முட்டைக்கோசு (Cabbage)", "english_name": "Cabbage", "optimal_planting_start_month": 9, "optimal_planting_end_month": 11, "avg_growth_duration_days": 80, "optimal_temp_min_c": 15.0, "optimal_temp_max_c": 20.0, "optimal_rainfall_min_mm": 400},
    {"crop_id": 41, "name": "பூக்கோசு (Cauliflower)", "english_name": "Cauliflower", "optimal_planting_start_month": 9, "optimal_planting_end_month": 11, "avg_growth_duration_days": 90, "optimal_temp_min_c": 15.0, "optimal_temp_max_c": 20.0, "optimal_rainfall_min_mm": 500},
    {"crop_id": 42, "name": "தர்பூசணி (Watermelon)", "english_name": "Watermelon", "optimal_planting_start_month": 12, "optimal_planting_end_month": 2, "avg_growth_duration_days": 90, "optimal_temp_min_c": 25.0, "optimal_temp_max_c": 35.0, "optimal_rainfall_min_mm": 400},
    {"crop_id": 43, "name": "பெர்சீம் (Berseem)", "english_name": "Berseem", "optimal_planting_start_month": 10, "optimal_planting_end_month": 11, "avg_growth_duration_days": 150, "optimal_temp_min_c": 10.0, "optimal_temp_max_c": 25.0, "optimal_rainfall_min_mm": 600},
    {"crop_id": 44, "name": "கேழ்வரகு (Finger Millet)", "english_name": "Finger Millet", "optimal_planting_start_month": 6, "optimal_planting_end_month": 8, "avg_growth_duration_days": 100, "optimal_temp_min_c": 20.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 500},
    {"crop_id": 45, "name": "கொள்ளு (Horse Gram)", "english_name": "Horse Gram", "optimal_planting_start_month": 9, "optimal_planting_end_month": 10, "avg_growth_duration_days": 120, "optimal_temp_min_c": 20.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 300},
    {"crop_id": 46, "name": "ஆமணக்கு (Castor)", "english_name": "Castor", "optimal_planting_start_month": 6, "optimal_planting_end_month": 7, "avg_growth_duration_days": 160, "optimal_temp_min_c": 20.0, "optimal_temp_max_c": 35.0, "optimal_rainfall_min_mm": 500},
    {"crop_id": 47, "name": "மரவள்ளி கிழங்கு (Tapioca)", "english_name": "Tapioca", "optimal_planting_start_month": 6, "optimal_planting_end_month": 7, "avg_growth_duration_days": 270, "optimal_temp_min_c": 25.0, "optimal_temp_max_c": 32.0, "optimal_rainfall_min_mm": 1000},
    {"crop_id": 48, "name": "மா (Mango)", "english_name": "Mango", "optimal_planting_start_month": 6, "optimal_planting_end_month": 11, "avg_growth_duration_days": 1500, "optimal_temp_min_c": 24.0, "optimal_temp_max_c": 35.0, "optimal_rainfall_min_mm": 750},
    {"crop_id": 49, "name": "கொய்யா (Guava)", "english_name": "Guava", "optimal_planting_start_month": 6, "optimal_planting_end_month": 9, "avg_growth_duration_days": 1000, "optimal_temp_min_c": 15.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 1000},
    {"crop_id": 50, "name": "முள்ளங்கி (Radish)", "english_name": "Radish", "optimal_planting_start_month": 1, "optimal_planting_end_month": 12, "avg_growth_duration_days": 40, "optimal_temp_min_c": 15.0, "optimal_temp_max_c": 25.0, "optimal_rainfall_min_mm": 300},
    {"crop_id": 51, "name": "உருளைக்கிழங்கு (Potato)", "english_name": "Potato", "optimal_planting_start_month": 10, "optimal_planting_end_month": 11, "avg_growth_duration_days": 100, "optimal_temp_min_c": 15.0, "optimal_temp_max_c": 25.0, "optimal_rainfall_min_mm": 500},
    {"crop_id": 52, "name": "பலா (Jackfruit)", "english_name": "Jackfruit", "optimal_planting_start_month": 6, "optimal_planting_end_month": 8, "avg_growth_duration_days": 2000, "optimal_temp_min_c": 20.0, "optimal_temp_max_c": 35.0, "optimal_rainfall_min_mm": 1500},
    {"crop_id": 53, "name": "முந்திரி (Cashew)", "english_name": "Cashew", "optimal_planting_start_month": 6, "optimal_planting_end_month": 7, "avg_growth_duration_days": 1500, "optimal_temp_min_c": 20.0, "optimal_temp_max_c": 35.0, "optimal_rainfall_min_mm": 600},
    {"crop_id": 54, "name": "ரப்பர் (Rubber)", "english_name": "Rubber", "optimal_planting_start_month": 6, "optimal_planting_end_month": 7, "avg_growth_duration_days": 2500, "optimal_temp_min_c": 25.0, "optimal_temp_max_c": 34.0, "optimal_rainfall_min_mm": 2500},
    {"crop_id": 55, "name": "காபி (Coffee)", "english_name": "Coffee", "optimal_planting_start_month": 6, "optimal_planting_end_month": 7, "avg_growth_duration_days": 1500, "optimal_temp_min_c": 15.0, "optimal_temp_max_c": 25.0, "optimal_rainfall_min_mm": 1500},
    {"crop_id": 56, "name": "தேயிலை (Tea)", "english_name": "Tea", "optimal_planting_start_month": 1, "optimal_planting_end_month": 12, "avg_growth_duration_days": 1000, "optimal_temp_min_c": 12.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 1200},
    {"crop_id": 57, "name": "பாக்கு (Arecanut)", "english_name": "Arecanut", "optimal_planting_start_month": 6, "optimal_planting_end_month": 9, "avg_growth_duration_days": 2000, "optimal_temp_min_c": 14.0, "optimal_temp_max_c": 36.0, "optimal_rainfall_min_mm": 2250},
    {"crop_id": 58, "name": "அன்னாசி (Pineapple)", "english_name": "Pineapple", "optimal_planting_start_month": 5, "optimal_planting_end_month": 7, "avg_growth_duration_days": 500, "optimal_temp_min_c": 20.0, "optimal_temp_max_c": 32.0, "optimal_rainfall_min_mm": 1000},
    {"crop_id": 60, "name": "சவுக்கு (Casuarina)", "english_name": "Casuarina", "optimal_planting_start_month": 6, "optimal_planting_end_month": 9, "avg_growth_duration_days": 1500, "optimal_temp_min_c": 10.0, "optimal_temp_max_c": 40.0, "optimal_rainfall_min_mm": 500},
    {"crop_id": 61, "name": "கேரட் (Carrot)", "english_name": "Carrot", "optimal_planting_start_month": 8, "optimal_planting_end_month": 11, "avg_growth_duration_days": 80, "optimal_temp_min_c": 15.0, "optimal_temp_max_c": 20.0, "optimal_rainfall_min_mm": 400},
    {"crop_id": 62, "name": "வெள்ளரி (Cucumber)", "english_name": "Cucumber", "optimal_planting_start_month": 2, "optimal_planting_end_month": 6, "avg_growth_duration_days": 60, "optimal_temp_min_c": 18.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 500},
    {"crop_id": 64, "name": "பப்பாளி (Papaya)", "english_name": "Papaya", "optimal_planting_start_month": 1, "optimal_planting_end_month": 12, "avg_growth_duration_days": 300, "optimal_temp_min_c": 21.0, "optimal_temp_max_c": 33.0, "optimal_rainfall_min_mm": 1000},
    {"crop_id": 65, "name": "சர்க்கரை வள்ளிக்கிழங்கு (Sugar Beet)", "english_name": "Sugar Beet", "optimal_planting_start_month": 10, "optimal_planting_end_month": 11, "avg_growth_duration_days": 180, "optimal_temp_min_c": 15.0, "optimal_temp_max_c": 25.0, "optimal_rainfall_min_mm": 500},
    {"crop_id": 66, "name": "உவர்நில நெல் வகைகள் (Specific Rice Varieties)", "english_name": "Specific Rice Varieties", "optimal_planting_start_month": 6, "optimal_planting_end_month": 11, "avg_growth_duration_days": 130, "optimal_temp_min_c": 20.0, "optimal_temp_max_c": 37.0, "optimal_rainfall_min_mm": 1200},
    {"crop_id": 67, "name": "செஸ்பானியா (Sesbania)", "english_name": "Sesbania", "optimal_planting_start_month": 6, "optimal_planting_end_month": 8, "avg_growth_duration_days": 60, "optimal_temp_min_c": 20.0, "optimal_temp_max_c": 40.0, "optimal_rainfall_min_mm": 500},
    {"crop_id": 68, "name": "நரிப்பயறு (Moth Bean)", "english_name": "Moth Bean", "optimal_planting_start_month": 6, "optimal_planting_end_month": 7, "avg_growth_duration_days": 70, "optimal_temp_min_c": 25.0, "optimal_temp_max_c": 35.0, "optimal_rainfall_min_mm": 250},
    {"crop_id": 69, "name": "கொத்தவரை (Guar)", "english_name": "Guar", "optimal_planting_start_month": 6, "optimal_planting_end_month": 7, "avg_growth_duration_days": 100, "optimal_temp_min_c": 25.0, "optimal_temp_max_c": 35.0, "optimal_rainfall_min_mm": 300},
    {"crop_id": 70, "name": "பேரீச்சை (Dates)", "english_name": "Dates", "optimal_planting_start_month": 1, "optimal_planting_end_month": 12, "avg_growth_duration_days": 2000, "optimal_temp_min_c": 25.0, "optimal_temp_max_c": 45.0, "optimal_rainfall_min_mm": 100},
    {"crop_id": 71, "name": "கொத்தவரை (Cluster Beans)", "english_name": "Cluster Beans", "optimal_planting_start_month": 6, "optimal_planting_end_month": 7, "avg_growth_duration_days": 75, "optimal_temp_min_c": 25.0, "optimal_temp_max_c": 35.0, "optimal_rainfall_min_mm": 400},
    {"crop_id": 72, "name": "சில காய்கறிகள் (Certain Vegetables)", "english_name": "Certain Vegetables", "optimal_planting_start_month": 1, "optimal_planting_end_month": 12, "avg_growth_duration_days": 60, "optimal_temp_min_c": 15.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 400},
    {"crop_id": 73, "name": "புகையிலை (Tobacco)", "english_name": "Tobacco", "optimal_planting_start_month": 10, "optimal_planting_end_month": 12, "avg_growth_duration_days": 120, "optimal_temp_min_c": 20.0, "optimal_temp_max_c": 30.0, "optimal_rainfall_min_mm": 500},
    {"crop_id": 74, "name": "தினை (Millet)", "english_name": "Millet", "optimal_planting_start_month": 6, "optimal_planting_end_month": 8, "avg_growth_duration_days": 80, "optimal_temp_min_c": 20.0, "optimal_temp_max_c": 35.0, "optimal_rainfall_min_mm": 300},
    {"crop_id": 75, "name": "சிறுதானியங்கள் (Millets)", "english_name": "Millets", "optimal_planting_start_month": 6, "optimal_planting_end_month": 8, "avg_growth_duration_days": 90, "optimal_temp_min_c": 20.0, "optimal_temp_max_c": 35.0, "optimal_rainfall_min_mm": 400},
    {"crop_id": 76, "name": "பூசணிக்காய் (Ash Gourd)", "english_name": "Ash Gourd", "optimal_planting_start_month": 6, "optimal_planting_end_month": 7, "avg_growth_duration_days": 100, "optimal_temp_min_c": 24.0, "optimal_temp_max_c": 35.0, "optimal_rainfall_min_mm": 700},
    {"crop_id": 77, "name": "பீர்க்கங்காய் (Ridge Gourd)", "english_name": "Ridge Gourd", "optimal_planting_start_month": 6, "optimal_planting_end_month": 7, "avg_growth_duration_days": 60, "optimal_temp_min_c": 25.0, "optimal_temp_max_c": 35.0, "optimal_rainfall_min_mm": 500},
    # Added Ash Gourd, Ridge Gourd explicitly
    # Removed generic Pulses (ID 7) as specific ones are added
]
print(f"INFO: Expanded placeholder crop database defined with {len(placeholder_crops_db)} entries (VALIDATE VALUES!).")

# 7. ***** Define Routes *****
# (Keep all routes: home, upload, predict_soil, weather_input, weather_results, chat, chatbot, api routes)
# ... Routes remain the same as the previous full code block ...

# ***** Home Route *****
@app.route('/')
def home():
    """Renders the homepage, including the current month's calendar activities."""
    print("--- Rendering Homepage with Calendar ---"); current_month_num = None; current_activities = []
    try: now = datetime.datetime.now(); current_month_num = now.month
    except Exception as e: print(f"ERROR getting current month: {e}"); traceback.print_exc(); current_activities = ["காலண்டர் தரவைப் பெறுவதில் பிழை (மாதம் கண்டறியப்படவில்லை)."]
    if current_month_num is not None:
        try: current_activities = farming_calendar_data.get(current_month_num, [])
        except Exception as e: print(f"ERROR looking up calendar activities for month {current_month_num}: {e}"); traceback.print_exc(); current_activities = ["குறிப்பிட்ட மாதத்திற்கான காலண்டர் தரவைப் பெறுவதில் பிழை."]
    if not current_activities:
        print(f"WARNING: No calendar activities found or error occurred for month {current_month_num}.")
        if not any("பிழை" in str(msg) for msg in current_activities): current_activities = ["இந்த மாதத்திற்கான செயல்பாடுகள் எதுவும் தற்போது கிடைக்கவில்லை."]
    return render_template('home.html', current_month_num=current_month_num, calendar_activities=current_activities)

@app.route('/upload')
def upload_page(): return render_template('upload.html')

@app.route('/predict_soil', methods=['POST'])
def predict_soil():
    # (Uses the expanded crop_recommendations with Tamil format)
    print("\n=== Received POST request for /predict_soil ===")
    if 'file' not in request.files: flash("படக் கோப்பு எதுவும் தேர்ந்தெடுக்கப்படவில்லை.", "warning"); return redirect(url_for('upload_page'))
    file = request.files['file'];
    if not file or file.filename == '': flash("படக் கோப்பு எதுவும் தேர்ந்தெடுக்கப்படவில்லை.", "warning"); return redirect(url_for('upload_page'))
    allowed_extensions = {'png', 'jpg', 'jpeg'}; fname = file.filename
    if '.' not in fname or fname.rsplit('.', 1)[1].lower() not in allowed_extensions: flash(f"அனுமதிக்கப்பட்ட கோப்பு வகைகள்: {', '.join(allowed_extensions)}.", "danger"); return redirect(url_for('upload_page'))
    try:
        img_bytes = file.read(); input_tensor = preprocess_image_for_pytorch(img_bytes)
        if input_tensor is None: flash("படத்தைச் செயலாக்க முடியவில்லை.", "danger"); return redirect(url_for('upload_page'))
        with torch.no_grad(): outputs = model(input_tensor); probabilities = torch.softmax(outputs, dim=1)[0]; confidence, idx_tensor = torch.max(probabilities, 0); predicted_idx = idx_tensor.item(); predicted_confidence = confidence.item()
        print(f"INFO: Prediction index: {predicted_idx}, Confidence: {predicted_confidence:.4f}")
        CONFIDENCE_THRESHOLD = 0.60; soil_type_display = "வகைப்பாடு உறுதியற்றது"; recommended_crops_display = ["நம்பிக்கை குறைவாக இருப்பதால் பரிந்துரைகள் இல்லை."]
        if predicted_confidence >= CONFIDENCE_THRESHOLD:
            if 0 <= predicted_idx < len(class_names):
                soil_type_predicted_str = class_names[predicted_idx]; soil_type_display = soil_type_predicted_str.capitalize()
                recommended_crops_display = crop_recommendations.get(soil_type_predicted_str.lower(), []) # Get expanded list
                if not recommended_crops_display: recommended_crops_display = [f"'{soil_type_display}' மண்ணிற்கான பரிந்துரைகள் வரையறுக்கப்படவில்லை."]
            else: print(f"ERROR: Predicted index {predicted_idx} out of range"); flash("மாதிரி முரண்பாடு.", "error"); return redirect(url_for('upload_page'))
        else: print(f"WARNING: Confidence below threshold"); soil_type_display = f"வகைப்பாடு உறுதியற்றது (< {CONFIDENCE_THRESHOLD*100:.0f}%)"
        return render_template('results.html', soil_type=soil_type_display, confidence=f"{predicted_confidence:.2f}", recommended_crops=recommended_crops_display if recommended_crops_display else None)
    except Exception as e: print(f"CRITICAL ERROR in /predict_soil: {e}"); traceback.print_exc(); flash("படத்தை கணிக்கும்போது பிழை.", "danger"); return redirect(url_for('upload_page'))

@app.route('/weather_input')
def weather_input_page():
    # (weather_input logic remains the same)
    try: soil_types_available = sorted([s.capitalize() for s in crop_recommendations.keys()]); district_display_map = { "ariyalur": "அரியலூர் (Ariyalur)", "chengalpattu": "செங்கல்பட்டு (Chengalpattu)", "chennai": "சென்னை (Chennai)", "coimbatore": "கோயம்புத்தூர் (Coimbatore)", "cuddalore": "கடலூர் (Cuddalore)", "dharmapuri": "தர்மபுரி (Dharmapuri)", "dindigul": "திண்டுக்கல் (Dindigul)", "erode": "ஈரோடு (Erode)", "kallakurichi": "கள்ளக்குறிச்சி (Kallakurichi)", "kanchipuram": "காஞ்சிபுரம் (Kanchipuram)", "kanyakumari": "கன்னியாகுமரி (Kanyakumari)", "karur": "கரூர் (Karur)", "krishnagiri": "கிருஷ்ணகிரி (Krishnagiri)", "madurai": "மதுரை (Madurai)", "mayiladuthurai": "மயிலாடுதுறை (Mayiladuthurai)", "nagapattinam": "நாகப்பட்டினம் (Nagapattinam)", "namakkal": "நாமக்கல் (Namakkal)", "nilgiris": "நீலகிரி (Nilgiris)", "perambalur": "பெரம்பலூர் (Perambalur)", "pudukkottai": "புதுக்கோட்டை (Pudukkottai)", "ramanathapuram": "இராமநாதபுரம் (Ramanathapuram)", "ranipet": "இராணிப்பேட்டை (Ranipet)", "salem": "சேலம் (Salem)", "sivaganga": "சிவகங்கை (Sivaganga)", "tenkasi": "தென்காசி (Tenkasi)", "thanjavur": "தஞ்சாவூர் (Thanjavur)", "theni": "தேனி (Theni)", "thoothukudi": "தூத்துக்குடி (Thoothukudi)", "tiruchirappalli": "திருச்சிராப்பள்ளி (Tiruchirappalli)", "tirunelveli": "திருநெல்வேலி (Tirunelveli)", "tirupathur": "திருப்பத்தூர் (Tirupathur)", "tiruppur": "திருப்பூர் (Tiruppur)", "tiruvallur": "திருவள்ளூர் (Tiruvallur)", "tiruvannamalai": "திருவண்ணாமலை (Tiruvannamalai)", "tiruvarur": "திருவாரூர் (Tiruvarur)", "vellore": "வேலூர் (Vellore)", "viluppuram": "விழுப்புரம் (Viluppuram)", "virudhunagar": "விருதுநகர் (Virudhunagar)" }; districts_for_dropdown = {k: v for k, v in district_display_map.items() if k in climate_normals_tn}; sorted_districts = sorted(districts_for_dropdown.items(), key=lambda item: item[1])
    except Exception as e: print(f"ERROR preparing data for /weather_input: {e}"); flash("உள்ளீட்டுப் பக்கத்தைத் தயாரிப்பதில் பிழை.", "danger"); soil_types_available = []; sorted_districts = []
    return render_template('weather_input.html', soil_types=soil_types_available, districts=sorted_districts)

# =============================================================================
# ===== MODULE 2 - WEATHER RESULTS - CORE LOGIC & HELPERS ===============
# =============================================================================
# --- Helper Functions (UPDATED to work with new recommendation format) ---
def check_suitability_for_month(current_soil_lower, current_district_key, month_num_to_check):
    """ STRICT check: Requires BOTH temperature and rainfall to be suitable. """
    suitable_crops_details = []; recommended_display_names = crop_recommendations.get(current_soil_lower, []); district_normals = climate_normals_tn.get(current_district_key, climate_normals_tn.get("default"))
    if not recommended_display_names or not district_normals: return []
    recommended_english_keys = {extract_eng_key(name).replace('_', ' ') for name in recommended_display_names}

    for crop_data in placeholder_crops_db:
        crop_english_name_key = crop_data.get("english_name")
        if not crop_english_name_key or crop_english_name_key.lower() not in recommended_english_keys: continue
        start_plant = crop_data.get("optimal_planting_start_month"); end_plant = crop_data.get("optimal_planting_end_month"); month_is_valid = False
        if isinstance(start_plant, int) and isinstance(end_plant, int) and 1 <= start_plant <= 12 and 1 <= end_plant <= 12:
            if start_plant <= end_plant: month_is_valid = start_plant <= month_num_to_check <= end_plant
            else: month_is_valid = month_num_to_check >= start_plant or month_num_to_check <= end_plant
        if not month_is_valid: continue
        duration_days = crop_data.get("avg_growth_duration_days"); temp_min_req = crop_data.get("optimal_temp_min_c", -float('inf')); temp_max_req = crop_data.get("optimal_temp_max_c", float('inf')); rain_min_req = crop_data.get("optimal_rainfall_min_mm", 0)
        if not all(isinstance(req, (int, float)) for req in [temp_min_req, temp_max_req, rain_min_req]) or not isinstance(duration_days, int) or duration_days <= 0: continue
        duration_months = max(1, (duration_days + 29) // 30); total_rain_calc = 0.0; avg_temp_sum_calc = 0.0; months_counted = 0; valid_period = True; avg_norm_temp_period = "N/A"; total_norm_rain = "N/A"
        for i in range(duration_months):
            current_month = (month_num_to_check + i - 1) % 12 + 1; monthly_norm = district_normals.get(current_month)
            if monthly_norm and isinstance(monthly_norm.get("temp"), (int, float)) and isinstance(monthly_norm.get("rain"), (int, float)): avg_temp_sum_calc += monthly_norm["temp"]; total_rain_calc += monthly_norm["rain"]; months_counted += 1
            else: valid_period = False; break
        temp_norm_ok = False; rain_norm_ok = False
        if valid_period and months_counted > 0: avg_norm_temp_period = avg_temp_sum_calc / months_counted; total_norm_rain = total_rain_calc; temp_norm_ok = temp_min_req <= avg_norm_temp_period <= temp_max_req; rain_norm_ok = total_norm_rain >= rain_min_req
        if temp_norm_ok and rain_norm_ok:
             display_name_for_output = get_display_name(crop_english_name_key)
             suitable_crops_details.append({"name": display_name_for_output, "reason": "மண், நடவு மாதம் மற்றும் காலநிலை சாதகமானது.", "avg_temp": f"{avg_norm_temp_period:.1f}" if isinstance(avg_norm_temp_period, float) else avg_norm_temp_period, "temp_needed": f"{temp_min_req:.1f}°C - {temp_max_req:.1f}°C" if temp_min_req > -float('inf') and temp_max_req < float('inf') else "N/A", "temp_ok": True, "total_rain": f"{total_norm_rain:.0f}", "rain_needed": f"> {rain_min_req}mm", "rain_ok": True})
    return suitable_crops_details

def find_alternative_crops_relaxed(current_soil_lower, current_district_key, exclude_month):
    """ RELAXED check: Finds alternatives if EITHER temperature OR rainfall suitable. """
    alt_suggestions = {}; recommended_display_names = crop_recommendations.get(current_soil_lower, []); district_normals = climate_normals_tn.get(current_district_key, climate_normals_tn.get("default"))
    if not recommended_display_names or not district_normals: return None
    recommended_english_keys = {extract_eng_key(name).replace('_', ' ') for name in recommended_display_names}

    for month_to_try in range(1, 13):
        if month_to_try == exclude_month: continue
        monthly_crop_list = []
        for crop_data in placeholder_crops_db:
            crop_english_name_key = crop_data.get("english_name");
            if not crop_english_name_key or crop_english_name_key.lower() not in recommended_english_keys: continue
            start_plant = crop_data.get("optimal_planting_start_month"); end_plant = crop_data.get("optimal_planting_end_month"); month_is_valid = False
            if isinstance(start_plant, int) and isinstance(end_plant, int) and 1 <= start_plant <= 12 and 1 <= end_plant <= 12:
                if start_plant <= end_plant: month_is_valid = start_plant <= month_to_try <= end_plant
                else: month_is_valid = month_to_try >= start_plant or month_to_try <= end_plant
            if not month_is_valid: continue
            duration_days = crop_data.get("avg_growth_duration_days"); temp_min_req = crop_data.get("optimal_temp_min_c", -float('inf')); temp_max_req = crop_data.get("optimal_temp_max_c", float('inf')); rain_min_req = crop_data.get("optimal_rainfall_min_mm", 0)
            if not all(isinstance(req, (int, float)) for req in [temp_min_req, temp_max_req, rain_min_req]) or not isinstance(duration_days, int) or duration_days <= 0: continue
            duration_months = max(1, (duration_days + 29) // 30); total_rain_calc = 0.0; avg_temp_sum_calc = 0.0; months_counted = 0; valid_period = True; avg_norm_temp_period = "N/A"; total_norm_rain = "N/A"
            for i in range(duration_months):
                current_month = (month_to_try + i - 1) % 12 + 1; monthly_norm = district_normals.get(current_month)
                if monthly_norm and isinstance(monthly_norm.get("temp"), (int, float)) and isinstance(monthly_norm.get("rain"), (int, float)): avg_temp_sum_calc += monthly_norm["temp"]; total_rain_calc += monthly_norm["rain"]; months_counted += 1
                else: valid_period = False; break
            temp_norm_ok = False; rain_norm_ok = False
            if valid_period and months_counted > 0: avg_norm_temp_period = avg_temp_sum_calc / months_counted; total_norm_rain = total_rain_calc; temp_norm_ok = temp_min_req <= avg_norm_temp_period <= temp_max_req; rain_norm_ok = total_norm_rain >= rain_min_req
            if temp_norm_ok or rain_norm_ok:
                 reason = "";
                 if temp_norm_ok and rain_norm_ok: reason = "மண்/மாதம் சரி. காலநிலை: வெப்பநிலை & மழை இரண்டும் சாதகமானது."
                 elif temp_norm_ok: reason = "மண்/மாதம் சரி. காலநிலை: வெப்பநிலை மட்டும் ஓரளவு சாதகமானது."
                 elif rain_norm_ok: reason = "மண்/மாதம் சரி. காலநிலை: மழை மட்டும் ஓரளவு சாதகமானது."
                 display_name_for_output = get_display_name(crop_english_name_key)
                 monthly_crop_list.append({"name": display_name_for_output, "reason": reason, "avg_temp": f"{avg_norm_temp_period:.1f}" if isinstance(avg_norm_temp_period, float) else avg_norm_temp_period, "temp_needed": f"{temp_min_req:.1f}°C - {temp_max_req:.1f}°C" if temp_min_req > -float('inf') and temp_max_req < float('inf') else "N/A", "temp_ok": temp_norm_ok, "total_rain": f"{total_norm_rain:.0f}", "rain_needed": f"> {rain_min_req}mm", "rain_ok": rain_norm_ok}) # mm removed
        if monthly_crop_list:
            try: month_name = datetime.date(2024, month_to_try, 1).strftime('%B'); alt_suggestions[month_name] = monthly_crop_list
            except ValueError: pass
    return alt_suggestions if alt_suggestions else None

@app.route('/weather_results', methods=['POST'])
def weather_results():
    """Handles weather-based crop planning request and displays results."""
    # (Logic remains the same - uses updated helpers)
    print("\n=== Received POST request for /weather_results ===")
    try:
        soil_type_req = request.form.get('soil_type'); start_month_str = request.form.get('start_month'); district_req_key = request.form.get('location')
        print(f"[DEBUG] Raw form data: soil_type='{soil_type_req}', start_month='{start_month_str}', location_key='{district_req_key}'")
        if not soil_type_req or not start_month_str or not district_req_key: flash("அனைத்து புலங்களும் தேவை.", "warning"); return redirect(url_for('weather_input_page'))
        soil_lower = soil_type_req.lower(); district_key = district_req_key.lower()
        district_display_map = { "ariyalur": "அரியலூர் (Ariyalur)", "chengalpattu": "செங்கல்பட்டு (Chengalpattu)", "chennai": "சென்னை (Chennai)", "coimbatore": "கோயம்புத்தூர் (Coimbatore)", "cuddalore": "கடலூர் (Cuddalore)", "dharmapuri": "தர்மபுரி (Dharmapuri)", "dindigul": "திண்டுக்கல் (Dindigul)", "erode": "ஈரோடு (Erode)", "kallakurichi": "கள்ளக்குறிச்சி (Kallakurichi)", "kanchipuram": "காஞ்சிபுரம் (Kanchipuram)", "kanyakumari": "கன்னியாகுமரி (Kanyakumari)", "karur": "கரூர் (Karur)", "krishnagiri": "கிருஷ்ணகிரி (Krishnagiri)", "madurai": "மதுரை (Madurai)", "mayiladuthurai": "மயிலாடுதுறை (Mayiladuthurai)", "nagapattinam": "நாகப்பட்டினம் (Nagapattinam)", "namakkal": "நாமக்கல் (Namakkal)", "nilgiris": "நீலகிரி (Nilgiris)", "perambalur": "பெரம்பலூர் (Perambalur)", "pudukkottai": "புதுக்கோட்டை (Pudukkottai)", "ramanathapuram": "இராமநாதபுரம் (Ramanathapuram)", "ranipet": "இராணிப்பேட்டை (Ranipet)", "salem": "சேலம் (Salem)", "sivaganga": "சிவகங்கை (Sivaganga)", "tenkasi": "தென்காசி (Tenkasi)", "thanjavur": "தஞ்சாவூர் (Thanjavur)", "theni": "தேனி (Theni)", "thoothukudi": "தூத்துக்குடி (Thoothukudi)", "tiruchirappalli": "திருச்சிராப்பள்ளி (Tiruchirappalli)", "tirunelveli": "திருநெல்வேலி (Tirunelveli)", "tirupathur": "திருப்பத்தூர் (Tirupathur)", "tiruppur": "திருப்பூர் (Tiruppur)", "tiruvallur": "திருவள்ளூர் (Tiruvallur)", "tiruvannamalai": "திருவண்ணாமலை (Tiruvannamalai)", "tiruvarur": "திருவாரூர் (Tiruvarur)", "vellore": "வேலூர் (Vellore)", "viluppuram": "விழுப்புரம் (Viluppuram)", "virudhunagar": "விருதுநகர் (Virudhunagar)" }
        district_display_name = district_display_map.get(district_key, district_key.capitalize())
        if soil_lower not in crop_recommendations: flash(f"தவறான மண் வகை: {soil_type_req}", "danger"); return redirect(url_for('weather_input_page'))
        if district_key not in climate_normals_tn and "default" not in climate_normals_tn: flash(f"மாவட்டத்திற்கு காலநிலை தரவு இல்லை: {district_display_name}", "danger"); return redirect(url_for('weather_input_page'))
        try: year, selected_month_num = map(int, start_month_str.split('-'))
        except ValueError: flash(f"தவறான மாத வடிவம் (YYYY-MM தேவை): '{start_month_str}'", "danger"); return redirect(url_for('weather_input_page'))
        if not 1 <= selected_month_num <= 12: flash(f"தவறான மாத எண் (1-12 தேவை): {selected_month_num}", "danger"); return redirect(url_for('weather_input_page'))
        try: start_month_display = datetime.date(year, selected_month_num, 1).strftime('%B %Y')
        except ValueError as e: flash(f"தேதி உருவாக்கத்தில் பிழை: {e}", "danger"); return redirect(url_for('weather_input_page'))
        print(f"[DEBUG] Validated input: soil='{soil_lower}', district_key='{district_key}', selected_month_num={selected_month_num}")
        print(f"[DEBUG] === Calling STRICT check_suitability_for_month for selected month: {selected_month_num} ===")
        recommended_crops_list = check_suitability_for_month(soil_lower, district_key, selected_month_num)
        alternative_suggestions = None
        if not recommended_crops_list:
            print(f"[DEBUG] === No STRICT match found. Calling RELAXED find_alternative_crops_relaxed. ===")
            alternative_suggestions = find_alternative_crops_relaxed(soil_lower, district_key, selected_month_num)
            if alternative_suggestions: print(f"[DEBUG] === RELAXED Alternative suggestions FOUND for months: {list(alternative_suggestions.keys())}")
            else: print(f"[DEBUG] === NO alternative suggestions found even with RELAXED check.")
        else: print(f"[DEBUG] === Found {len(recommended_crops_list)} STRICTLY suitable crops. Skipping alternative search.")
        print(f"[DEBUG] === Rendering weather_results.html ===")
        return render_template('weather_results.html', soil_type=soil_type_req.capitalize(), district=district_display_name, start_month_display=start_month_display, recommended_crops=recommended_crops_list if recommended_crops_list else None, alternative_suggestions=alternative_suggestions )
    except Exception as e: print(f"CRITICAL ERROR in /weather_results: {e}"); traceback.print_exc(); flash("பரிந்துரைகளை உருவாக்கும்போது பிழை.", "danger"); return redirect(url_for('weather_input_page'))

# =============================================================================
# ================= END OF MODULE 2 =========================================
# =============================================================================

# --- API Routes ---
@app.route('/api/market_ticker_data')
def market_ticker_data():
    if not ticker_display_list:
        if not market_price_data and os.path.exists(MARKET_DATA_CSV): return jsonify({"error": "Ticker data not processed."}), 500
        elif not market_price_data: return jsonify({"error": "Market data source unavailable."}), 503
        else: return jsonify({"error": "No valid price data found."}), 404
    return jsonify({"ticker_items": ticker_display_list})

@app.route('/api/tip_of_day')
def tip_of_day():
     tip = random.choice(farming_tips) if farming_tips else "மண்ணை தவறாமல் சரிபார்க்கவும்."
     return jsonify({"tip": tip})

# --- Chat Page Route ---
@app.route('/chat')
def chatbot_page():
    session['chat_state'] = 'START'; session['selected_crop'] = None; print("INFO: Initializing chat session state to START.")
    if os.path.exists(CHATBOT_TEMPLATE_PATH): return render_template('chatbot.html')
    else: flash("அரட்டை இடைமுகம் கிடைக்கவில்லை.", "warning"); return redirect(url_for('home'))

# --- Chatbot API Route (State Machine Logic) ---
@app.route('/chatbot', methods=['POST'])
def chatbot():
    # (Chatbot logic remains the same, using loaded crop_guide_data)
    print("\n=== Received POST request for /chatbot ===")
    guide_data = crop_guide_data; tips = farming_tips; soil_crop_map = crop_recommendations
    try:
        data = request.get_json();
        if not data: return jsonify({"response": "தவறான கோரிக்கை.", "next_state": "START", "options": []}), 400
        user_input = data.get('message', '').strip(); current_state = data.get('state', 'START'); selected_crop = session.get('selected_crop', None)
        print(f"INFO: Received: input='{user_input}', state='{current_state}', selected_crop='{selected_crop}'")
        next_state = 'START'; response_text = "மன்னிக்கவும், ஏதோ தவறு நடந்துவிட்டது."; options = []
        if current_state == 'START' or user_input == 'start_over' or user_input == 'INIT_CHAT':
            response_text = "வணக்கம்! நான் உங்கள் விவசாய உதவியாளர். நான் எப்படி உதவ முடியும்?"; options = [{"label": "பயிர் பற்றிய தகவல்", "value": "ask_crop_info"}, {"label": "மண்ணுக்கேற்ற பயிர்", "value": "ask_soil_info"}, {"label": "விவசாய குறிப்பு", "value": "get_tip"}]; next_state = 'AWAITING_MAIN_CHOICE'; session['selected_crop'] = None
        elif current_state == 'AWAITING_MAIN_CHOICE':
            if user_input == 'ask_crop_info':
                response_text = "எந்த பயிர் பற்றி தெரிந்து கொள்ள விரும்புகிறீர்கள்?"; options = [{"label": v.get('name_ta', k.capitalize()), "value": k} for k, v in sorted(guide_data.items())]
                if not options: response_text = "மன்னிக்கவும், என்னிடம் தற்போது பயிர் விவரங்கள் இல்லை."; next_state = 'START'
                else: options.append({"label": "<-- முதன்மை பட்டி", "value": "back_to_start"}); next_state = 'AWAITING_CROP_CHOICE'
            elif user_input == 'ask_soil_info':
                response_text = "எந்த மண் வகை பற்றி தெரிந்து கொள்ள விரும்புகிறீர்கள்?"; options = [{"label": k.capitalize(), "value": k} for k in sorted(soil_crop_map.keys())]; options.append({"label": "<-- முதன்மை பட்டி", "value": "back_to_start"}); next_state = 'AWAITING_SOIL_CHOICE'
            elif user_input == 'get_tip': response_text = random.choice(tips) if tips else "தற்போது குறிப்புகள் எதுவும் இல்லை."; options = [{"label": "வேறு உதவி?", "value": "back_to_start"}, {"label": "நன்றி", "value": "end_chat"}]; next_state = 'SHOWING_INFO'
            elif user_input == 'back_to_start': response_text = "வேறு எப்படி உதவ முடியும்?"; options = [{"label": "பயிர் பற்றிய தகவல்", "value": "ask_crop_info"},{"label": "மண்ணுக்கேற்ற பயிர்", "value": "ask_soil_info"},{"label": "விவசாய குறிப்பு", "value": "get_tip"}]; next_state = 'AWAITING_MAIN_CHOICE'
            else: response_text = "தயவுசெய்து கீழே உள்ள விருப்பங்களில் ஒன்றைத் தேர்ந்தெடுக்கவும்."; options = [{"label": "பயிர் பற்றிய தகவல்", "value": "ask_crop_info"},{"label": "மண்ணுக்கேற்ற பயிர்", "value": "ask_soil_info"},{"label": "விவசாய குறிப்பு", "value": "get_tip"}]; next_state = 'AWAITING_MAIN_CHOICE'
        elif current_state == 'AWAITING_CROP_CHOICE':
            if user_input == 'back_to_start': response_text = "வேறு எப்படி உதவ முடியும்?"; options = [{"label": "பயிர் பற்றிய தகவல்", "value": "ask_crop_info"},{"label": "மண்ணுக்கேற்ற பயிர்", "value": "ask_soil_info"},{"label": "விவசாய குறிப்பு", "value": "get_tip"}]; next_state = 'AWAITING_MAIN_CHOICE'; session['selected_crop'] = None
            elif user_input in guide_data:
                selected_crop = user_input; session['selected_crop'] = selected_crop; crop_name_display = guide_data[selected_crop].get('name_ta', selected_crop.capitalize()); response_text = f"'{crop_name_display}' பற்றி என்ன தகவல் வேண்டும்?"
                available_topics = list(guide_data[selected_crop].keys()); options = []; topic_map = {"summary": "சுருக்கம்", "soil": "மண்", "water": "நீர் தேவை", "fertilizer": "உரம்", "pests_diseases": "பூச்சி/நோய்", "duration": "வளர்ச்சி காலம்", "planting_months": "நடவு மாதம்", "climate": "காலநிலை"}
                for topic_key, topic_label in topic_map.items():
                     if topic_key in available_topics and guide_data[selected_crop].get(topic_key): options.append({"label": topic_label, "value": topic_key})
                if not options: response_text = f"மன்னிக்கவும், {crop_name_display} பற்றி கூடுதல் விவரங்கள் இல்லை."; next_state = 'AWAITING_CROP_CHOICE'
                else: options.append({"label": "<-- வேறு பயிர்", "value": "ask_crop_info"}); next_state = 'AWAITING_TOPIC_CHOICE'
            else: response_text = "சரியான பயிரைத் தேர்ந்தெடுக்கவும்:"; options = [{"label": v.get('name_ta', k.capitalize()), "value": k} for k, v in sorted(guide_data.items())]; options.append({"label": "<-- முதன்மை பட்டி", "value": "back_to_start"}); next_state = 'AWAITING_CROP_CHOICE'
        elif current_state == 'AWAITING_TOPIC_CHOICE':
            if not selected_crop or selected_crop not in guide_data: response_text = "முதலில் ஒரு பயிரைத் தேர்ந்தெடுக்கவும்."; options = [{"label": v.get('name_ta', k.capitalize()), "value": k} for k, v in sorted(guide_data.items())]; options.append({"label": "<-- முதன்மை பட்டி", "value": "back_to_start"}); next_state = 'AWAITING_CROP_CHOICE'
            elif user_input in guide_data.get(selected_crop, {}):
                topic_key = user_input; info = guide_data[selected_crop].get(topic_key, "தகவல் இல்லை."); crop_name_display = guide_data[selected_crop].get('name_ta', selected_crop.capitalize()); topic_map = {"summary": "சுருக்கம்", "soil": "மண்", "water": "நீர் தேவை", "fertilizer": "உரம்", "pests_diseases": "பூச்சி/நோய்", "duration": "வளர்ச்சி காலம்", "planting_months": "நடவு மாதம்", "climate": "காலநிலை"}; topic_label = topic_map.get(topic_key, topic_key); response_text = f"**{crop_name_display} - {topic_label}:**\n{info}"
                options = [{"label": f"{crop_name_display} பற்றி மேலும்", "value": "ask_same_crop_again"}, {"label": "வேறு பயிர் பற்றி", "value": "ask_crop_info"}, {"label": "முதன்மை பட்டி", "value": "back_to_start"}, {"label": "நன்றி", "value": "end_chat"}]; next_state = 'SHOWING_INFO'
            elif user_input == 'ask_crop_info': response_text = "எந்த பயிர் பற்றி தெரிந்து கொள்ள விரும்புகிறீர்கள்?"; options = [{"label": v.get('name_ta', k.capitalize()), "value": k} for k, v in sorted(guide_data.items())]; options.append({"label": "<-- முதன்மை பட்டி", "value": "back_to_start"}); next_state = 'AWAITING_CROP_CHOICE'; session['selected_crop'] = None
            else:
                crop_name_display = guide_data[selected_crop].get('name_ta', selected_crop.capitalize()); response_text = f"'{crop_name_display}' பற்றி அறிய சரியான தலைப்பைத் தேர்ந்தெடுக்கவும்:"; available_topics = list(guide_data[selected_crop].keys()); options = []; topic_map = {"summary": "சுருக்கம்", "soil": "மண்", "water": "நீர் தேவை", "fertilizer": "உரம்", "pests_diseases": "பூச்சி/நோய்", "duration": "வளர்ச்சி காலம்", "planting_months": "நடவு மாதம்", "climate": "காலநிலை"}
                for topic_key, topic_label in topic_map.items():
                    if topic_key in available_topics and guide_data[selected_crop].get(topic_key): options.append({"label": topic_label, "value": topic_key})
                options.append({"label": "<-- வேறு பயிர்", "value": "ask_crop_info"}); next_state = 'AWAITING_TOPIC_CHOICE'
        elif current_state == 'AWAITING_SOIL_CHOICE':
            if user_input == 'back_to_start': response_text = "வேறு எப்படி உதவ முடியும்?"; options = [{"label": "பயிர் பற்றிய தகவல்", "value": "ask_crop_info"},{"label": "மண்ணுக்கேற்ற பயிர்", "value": "ask_soil_info"},{"label": "விவசாய குறிப்பு", "value": "get_tip"}]; next_state = 'AWAITING_MAIN_CHOICE'
            elif user_input in soil_crop_map:
                soil_type_key = user_input; crops = soil_crop_map.get(soil_type_key, []); response = f"**{soil_type_key.capitalize()} மண்:**\n" + (f"பரிந்துரைக்கப்படும் பயிர்கள்: {', '.join(crops)}." if crops else "இந்தப் மண்ணிற்கான பயிர் பரிந்துரைகள் இல்லை.")
                options = [{"label": "வேறு மண் வகை", "value": "ask_soil_info"}, {"label": "முதன்மை பட்டி", "value": "back_to_start"}, {"label": "நன்றி", "value": "end_chat"}]; next_state = 'SHOWING_INFO'
            else: response_text = "சரியான மண் வகையைத் தேர்ந்தெடுக்கவும்:"; options = [{"label": k.capitalize(), "value": k} for k in sorted(soil_crop_map.keys())]; options.append({"label": "<-- முதன்மை பட்டி", "value": "back_to_start"}); next_state = 'AWAITING_SOIL_CHOICE'
        elif current_state == 'SHOWING_INFO':
             if user_input == 'ask_same_crop_again':
                 if selected_crop and selected_crop in guide_data:
                      crop_name_display = guide_data[selected_crop].get('name_ta', selected_crop.capitalize()); response_text = f"{crop_name_display} பற்றி வேறு என்ன தகவல் வேண்டும்?"; available_topics = list(guide_data[selected_crop].keys()); options = []; topic_map = {"summary": "சுருக்கம்", "soil": "மண்", "water": "நீர் தேவை", "fertilizer": "உரம்", "pests_diseases": "பூச்சி/நோய்", "duration": "வளர்ச்சி காலம்", "planting_months": "நடவு மாதம்", "climate": "காலநிலை"}
                      for topic_key, topic_label in topic_map.items():
                           if topic_key in available_topics and guide_data[selected_crop].get(topic_key): options.append({"label": topic_label, "value": topic_key})
                      options.append({"label": "<-- வேறு பயிர்", "value": "ask_crop_info"}); next_state = 'AWAITING_TOPIC_CHOICE'
                 else: response_text = "முதலில் ஒரு பயிரைத் தேர்ந்தெடுக்கவும்."; options = [{"label": v.get('name_ta', k.capitalize()), "value": k} for k, v in sorted(guide_data.items())]; options.append({"label": "<-- முதன்மை பட்டி", "value": "back_to_start"}); next_state = 'AWAITING_CROP_CHOICE'
             elif user_input == 'ask_crop_info': response_text = "எந்த பயிர் பற்றி தெரிந்து கொள்ள விரும்புகிறீர்கள்?"; options = [{"label": v.get('name_ta', k.capitalize()), "value": k} for k, v in sorted(guide_data.items())]; options.append({"label": "<-- முதன்மை பட்டி", "value": "back_to_start"}); next_state = 'AWAITING_CROP_CHOICE'; session['selected_crop'] = None
             elif user_input == 'ask_soil_info': response_text = "எந்த மண் வகை பற்றி தெரிந்து கொள்ள விரும்புகிறீர்கள்?"; options = [{"label": k.capitalize(), "value": k} for k in sorted(soil_crop_map.keys())]; options.append({"label": "<-- முதன்மை பட்டி", "value": "back_to_start"}); next_state = 'AWAITING_SOIL_CHOICE'; session['selected_crop'] = None
             elif user_input == 'back_to_start': response_text = "வேறு எப்படி உதவ முடியும்?"; options = [{"label": "பயிர் பற்றிய தகவல்", "value": "ask_crop_info"},{"label": "மண்ணுக்கேற்ற பயிர்", "value": "ask_soil_info"},{"label": "விவசாய குறிப்பு", "value": "get_tip"}]; next_state = 'AWAITING_MAIN_CHOICE'; session['selected_crop'] = None
             elif user_input == 'end_chat': response_text = "நன்றி! வேறு உதவிக்கு மீண்டும் தொடர்பு கொள்ளவும்."; options = []; next_state = 'START'; session['selected_crop'] = None
             else:
                 response_text = "தயவுசெய்து கீழே உள்ள விருப்பங்களில் ஒன்றைத் தேர்ந்தெடுக்கவும்."
                 current_crop_name = guide_data[selected_crop].get('name_ta', selected_crop.capitalize()) if selected_crop and selected_crop in guide_data else "வேறு பயிர்"
                 options = [{"label": f"{current_crop_name} பற்றி மேலும்", "value": "ask_same_crop_again" if selected_crop else "ask_crop_info"}, {"label": "வேறு பயிர் பற்றி", "value": "ask_crop_info"}, {"label": "மண்ணுக்கேற்ற பயிர்", "value": "ask_soil_info"}, {"label": "முதன்மை பட்டி", "value": "back_to_start"}, {"label": "நன்றி", "value": "end_chat"}]
                 next_state = 'SHOWING_INFO'
        else: # Fallback for unknown state
            print(f"WARNING: Unhandled state '{current_state}'. Resetting.")
            response_text = "மன்னிக்கவும், உரையாடலில் ஒரு சிக்கல். மீண்டும் ஆரம்பிக்கலாம். எப்படி உதவ முடியும்?"; options = [{"label": "பயிர் பற்றிய தகவல்", "value": "ask_crop_info"},{"label": "மண்ணுக்கேற்ற பயிர்", "value": "ask_soil_info"},{"label": "விவசாய குறிப்பு", "value": "get_tip"}]; next_state = 'AWAITING_MAIN_CHOICE'; session['selected_crop'] = None

        response_payload = {"response": response_text, "next_state": next_state, "options": options }
        session['chat_state'] = next_state # Update session state
        print(f"INFO: Sending payload: {response_payload}")
        return jsonify(response_payload)

    except Exception as e:
        print(f"ERROR processing chatbot request: {e}"); traceback.print_exc()
        session['chat_state'] = 'START'; session['selected_crop'] = None
        return jsonify({"response": "மன்னிக்கவும், ஒரு எதிர்பாராத பிழை ஏற்பட்டது.", "next_state": "START", "options": []}), 500

# 9. Main Execution Guard
if __name__ == '__main__':
    print("INFO: Starting Flask development server...")
    host = os.environ.get('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_RUN_PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_default_secret_key_for_dev_123!@#') # Use env var or default
    if app.secret_key == 'a_default_secret_key_for_dev_123!@#' and debug_mode: print("\nWARNING: Using default Flask secret key!\n")
    print(f"--- Server Config ---"); print(f" * Host: {host}"); print(f" * Port: {port}"); print(f" * Debug Mode: {debug_mode}"); print(f"---------------------");
    if not os.path.exists(MARKET_DATA_CSV): print(f"\n *** FINAL WARNING: Market data CSV not found at '{MARKET_DATA_CSV}'. Ticker data will be unavailable. ***\n")
    if not crop_guide_data: print(f"\n *** FINAL WARNING: Crop guide data is empty or failed to load from '{CROP_GUIDE_JSON_PATH}'. Chatbot crop info will be unavailable. ***\n")

    app.run(host=host, port=port, debug=debug_mode)