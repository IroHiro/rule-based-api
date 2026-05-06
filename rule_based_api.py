"""
RULE-BASED FILTER API - WITH DEBUG LOGGING
Orchestrates suitability + yield predictions and adds business rules
"""

import numpy as np
import requests
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
import json
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ============================================
# CONFIGURATION
# ============================================

SUITABILITY_API_URL = os.environ.get('SUITABILITY_API_URL', 'https://suitability-api.onrender.com/predict')
YIELD_API_URL = os.environ.get('YIELD_API_URL', 'https://crop-yield-api-9c1l.onrender.com/predict')

# Enable debug mode
DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'

# Soil database
SOIL_DB = {
    'Ligao': {'n': 4, 'p': 2, 'k': 4, 'ph': 5.0, 'fertility': 3.33},
    'Malinao': {'n': 4, 'p': 2, 'k': 4, 'ph': 5.0, 'fertility': 3.33},
    'Oas': {'n': 2, 'p': 2, 'k': 2, 'ph': 5.0, 'fertility': 2.0},
    'Pio Duran': {'n': 1, 'p': 1, 'k': 1, 'ph': 5.0, 'fertility': 1.0},
    'Polangui': {'n': 2, 'p': 2, 'k': 2, 'ph': 5.0, 'fertility': 2.0}
}

# Climate norms (used for comparison)
CLIMATE_DB = {
    'Ligao': {'temp': 27.5, 'rainfall': 2100, 'humidity': 82, 'ndvi': 0.75, 'evi': 0.52},
    'Malinao': {'temp': 27.3, 'rainfall': 2000, 'humidity': 83, 'ndvi': 0.72, 'evi': 0.49},
    'Oas': {'temp': 27.2, 'rainfall': 1950, 'humidity': 81, 'ndvi': 0.70, 'evi': 0.48},
    'Pio Duran': {'temp': 27.0, 'rainfall': 1850, 'humidity': 79, 'ndvi': 0.68, 'evi': 0.46},
    'Polangui': {'temp': 27.4, 'rainfall': 1980, 'humidity': 81, 'ndvi': 0.71, 'evi': 0.49}
}

# Regional averages (kg/ha for rice/corn, nuts/ha for coconut)
REGIONAL_AVG = {'rice': 4500, 'corn': 3800, 'coconut': 12000}

# Risk levels
RISK_LEVELS = {
    'Ligao': ('Moderate', 'July–October'),
    'Malinao': ('High', 'June–November'),
    'Oas': ('Moderate', 'July–October'),
    'Polangui': ('Moderate', 'July–October'),
    'Pio Duran': ('Low', 'August–September')
}

# Crop parameters
CROP_PARAMS = {
    'rice': {'temp_optimal': 28, 'rain_optimal': 1500, 'hum_optimal': 80, 'ph_optimal': 6.2, 'days': 110, 'temp_min': 20, 'temp_max': 35, 'rain_min': 800, 'rain_max': 2500},
    'corn': {'temp_optimal': 27, 'rain_optimal': 800, 'hum_optimal': 70, 'ph_optimal': 6.5, 'days': 100, 'temp_min': 18, 'temp_max': 32, 'rain_min': 500, 'rain_max': 1800},
    'coconut': {'temp_optimal': 30, 'rain_optimal': 2000, 'hum_optimal': 80, 'ph_optimal': 5.5, 'days': 365, 'temp_min': 20, 'temp_max': 38, 'rain_min': 1000, 'rain_max': 3000}
}


# ============================================
# HELPER FUNCTIONS
# ============================================

def call_api_with_retry(url: str, payload: Dict, api_name: str, max_retries: int = 3, initial_delay: int = 2) -> Tuple[Optional[Dict], int]:
    """
    Call API with exponential backoff retry logic
    Returns: (response_json, status_code) or (None, error_status)
    """
    for attempt in range(max_retries):
        try:
            if DEBUG:
                print(f"   [DEBUG] Calling {api_name} (attempt {attempt + 1}/{max_retries})")
                print(f"   [DEBUG] URL: {url}")
                print(f"   [DEBUG] Payload keys: {list(payload.keys())}")
            
            response = requests.post(url, json=payload, timeout=30)
            
            if DEBUG:
                print(f"   [DEBUG] Response status: {response.status_code}")
                print(f"   [DEBUG] Response headers: {dict(response.headers)}")
            
            if response.status_code == 429:
                delay = initial_delay * (2 ** attempt)
                print(f"   ⚠️ Rate limited (429) for {api_name}, retrying in {delay}s...")
                
                # Log rate limit headers if available
                if 'X-RateLimit-Reset' in response.headers:
                    reset_time = response.headers['X-RateLimit-Reset']
                    print(f"   [DEBUG] Rate limit reset at: {reset_time}")
                
                time.sleep(delay)
                continue
                
            if response.status_code == 200:
                try:
                    return response.json(), 200
                except json.JSONDecodeError as e:
                    print(f"   ❌ JSON decode error for {api_name}: {e}")
                    print(f"   Response text: {response.text[:200]}")
                    return None, response.status_code
            else:
                print(f"   ❌ {api_name} error: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                return None, response.status_code
                
        except requests.exceptions.Timeout:
            print(f"   ⏰ Timeout for {api_name} (attempt {attempt + 1})")
            if attempt == max_retries - 1:
                return None, 504
            time.sleep(initial_delay * (2 ** attempt))
            
        except requests.exceptions.ConnectionError as e:
            print(f"   🔌 Connection error for {api_name}: {e}")
            if attempt == max_retries - 1:
                return None, 503
            time.sleep(initial_delay * (2 ** attempt))
            
        except Exception as e:
            print(f"   ❌ Unexpected error for {api_name}: {e}")
            if attempt == max_retries - 1:
                return None, 500
            time.sleep(initial_delay * (2 ** attempt))
    
    return None, 429


def calc_climate_match(crop: str, municipality: str, climate_data: Dict) -> float:
    """Calculate climate match percentage"""
    climate = CLIMATE_DB.get(municipality, CLIMATE_DB['Ligao'])
    p = CROP_PARAMS[crop]
    
    temp = climate_data.get('avg_temperature', climate['temp'])
    rainfall = climate_data.get('total_rainfall', climate['rainfall'])
    humidity = climate_data.get('humidity', climate['humidity'])
    
    if DEBUG:
        print(f"   [DEBUG] Climate match - temp: {temp}, rainfall: {rainfall}, humidity: {humidity}")
        print(f"   [DEBUG] Optimal - temp: {p['temp_optimal']}, rainfall: {p['rain_optimal']}, humidity: {p['hum_optimal']}")
    
    # Check if values are within viable ranges
    temp_in_range = p['temp_min'] <= temp <= p['temp_max']
    rain_in_range = p['rain_min'] <= rainfall <= p['rain_max']
    
    # Calculate scores with penalty for being outside range
    if temp_in_range:
        temp_score = 100 * np.exp(-((temp - p['temp_optimal'])**2) / 50)
    else:
        distance_to_range = min(abs(temp - p['temp_min']), abs(temp - p['temp_max']))
        temp_score = max(0, 100 - distance_to_range * 10)
    
    if rain_in_range:
        rain_score = 100 * np.exp(-((rainfall - p['rain_optimal'])**2) / 180000)
    else:
        distance_to_range = min(abs(rainfall - p['rain_min']), abs(rainfall - p['rain_max']))
        rain_score = max(0, 100 - distance_to_range * 0.05)
    
    hum_score = 100 * np.exp(-((humidity - p['hum_optimal'])**2) / 200)
    
    if DEBUG:
        print(f"   [DEBUG] Scores - temp: {temp_score:.1f}, rain: {rain_score:.1f}, hum: {hum_score:.1f}")
        print(f"   [DEBUG] In range - temp: {temp_in_range}, rain: {rain_in_range}")
    
    return (temp_score * 0.4) + (rain_score * 0.35) + (hum_score * 0.25)


def calc_soil_compat(crop: str, municipality: str) -> float:
    """Calculate soil compatibility percentage"""
    soil = SOIL_DB.get(municipality, SOIL_DB['Ligao'])
    p = CROP_PARAMS[crop]
    
    ph_score = 100 * np.exp(-((soil['ph'] - p['ph_optimal'])**2) / 0.18)
    
    # For coconut, NPK scores are more important (coconut needs high K)
    if crop == 'coconut':
        npk_score = (soil['fertility'] / 5) * 100
        # Bonus for good potassium (K score)
        k_bonus = min(20, soil.get('k', 2) * 5)
        npk_score = min(100, npk_score + k_bonus)
    else:
        npk_score = (soil['fertility'] / 5) * 100
    
    if DEBUG:
        print(f"   [DEBUG] Soil - ph: {soil['ph']}, fertility: {soil['fertility']}")
        print(f"   [DEBUG] Soil scores - ph: {ph_score:.1f}, npk: {npk_score:.1f}")
    
    return (ph_score * 0.6) + (npk_score * 0.4)


def get_soil_preparation(crop: str, municipality: str) -> List[str]:
    """Get soil preparation recommendations"""
    soil = SOIL_DB.get(municipality, SOIL_DB['Ligao'])
    soil_fertility = soil['fertility']
    
    soil_prep = []
    soil_prep.append("Plow and harrow 2-3 times until soil is well-pulverized")
    
    if soil_fertility < 2.5:
        soil_prep.append("Add organic compost (5-10 tons/ha) 2 weeks before planting")
    else:
        soil_prep.append("Add organic compost (3-5 tons/ha) 2 weeks before planting")
    
    if crop == 'rice':
        soil_prep.append("Level the field for uniform water distribution")
        soil_prep.append("Construct small dikes (20-30 cm high) around field borders")
    elif crop == 'coconut':
        soil_prep.append("Dig holes 60x60x60 cm at recommended spacing (8-9m between trees)")
        soil_prep.append("Backfill holes with topsoil mixed with compost and 500g complete fertilizer per hole")
        soil_prep.append("Ensure proper drainage - coconut doesn't tolerate waterlogging")
    else:  # corn
        soil_prep.append("Create furrows 75 cm apart for proper drainage")
    
    return soil_prep


def get_harvest_advice(crop: str, predicted_yield_kg: float) -> List[str]:
    """Get harvest recommendations"""
    days = CROP_PARAMS[crop]['days']
    
    if crop == 'coconut':
        harvest_advice = [
            f"First harvest typically begins 4-5 years after planting",
            f"Expected yield: {int(predicted_yield_kg):,} nuts/ha/year",
            "Harvest nuts every 45-60 days when nuts are mature",
            "Use climbing equipment or extendable pole harvesters for safety"
        ]
    else:
        harvest_advice = [
            f"Harvest at {days} days after planting when grains are mature",
            f"Expected yield: {int(predicted_yield_kg):,} kg/ha",
            "Dry immediately to 14% moisture to prevent mold"
        ]
    
    return harvest_advice


def get_typhoon_advice(municipality: str) -> Dict:
    """Get typhoon preparedness recommendations"""
    risk_level, risk_months = RISK_LEVELS.get(municipality, ('Moderate', 'July–October'))
    current_month = datetime.now().month
    typhoon_months = {
        'Ligao': [7, 8, 9, 10],
        'Malinao': [6, 7, 8, 9, 10, 11],
        'Oas': [7, 8, 9, 10],
        'Polangui': [7, 8, 9, 10],
        'Pio Duran': [8, 9]
    }
    
    is_typhoon_season = current_month in typhoon_months.get(municipality, [7, 8, 9, 10])
    
    if is_typhoon_season:
        if risk_level == 'High':
            advice = [
                "⚠️ ACTIVE TYPHOON SEASON - HIGH RISK",
                "Harvest mature crops immediately before typhoon",
                "Clear drainage canals, secure equipment",
                "For coconut: Check tree health, remove weak fronds"
            ]
        elif risk_level == 'Moderate':
            advice = [
                "⚠️ ACTIVE TYPHOON SEASON - MODERATE RISK",
                "Monitor weather forecasts daily",
                "Ensure drainage systems are clear"
            ]
        else:
            advice = [
                "ACTIVE TYPHOON SEASON - LOW RISK",
                "Monitor weather updates"
            ]
    else:
        advice = [
            "NO ACTIVE TYPHOON THREAT",
            "Current conditions are safe for farming"
        ]
    
    return {
        'risk_level': risk_level,
        'risk_months': risk_months,
        'is_typhoon_season': is_typhoon_season,
        'advice': advice
    }


def get_overall_recommendation(suitability: str, predicted_yield_kg: float, crop: str) -> Dict:
    """Get overall recommendation based on results"""
    is_suitable = suitability in ['High', 'Medium']
    is_good_yield = predicted_yield_kg > REGIONAL_AVG[crop]
    
    if is_suitable and is_good_yield:
        return {
            'status': 'Excellent',
            'color': '#2d6a4f',
            'message': f"✅ {crop.capitalize()} is highly suitable for this location with good yield potential. Consider expanding cultivation."
        }
    elif is_suitable and not is_good_yield:
        return {
            'status': 'Good but Needs Improvement',
            'color': '#f4a261',
            'message': f"⚠️ {crop.capitalize()} is suitable but yield is below average. Check farming practices and input management."
        }
    elif not is_suitable and is_good_yield:
        return {
            'status': 'Cautiously Optimistic',
            'color': '#f4a261',
            'message': f"⚠️ {crop.capitalize()} shows good yield but suitability is low. Monitor crop health closely."
        }
    else:
        return {
            'status': 'Not Recommended',
            'color': '#d62828',
            'message': f"❌ {crop.capitalize()} may not be optimal for this location. Consider alternative crops or soil improvement."
        }


def get_planting_advice(crop: str, municipality: str, climate_data: Dict) -> Dict:
    """Get planting advice based on current conditions"""
    temp = climate_data.get('avg_temperature', 27)
    rainfall = climate_data.get('total_rainfall', 1500)
    
    p = CROP_PARAMS[crop]
    
    temp_optimal = abs(temp - p['temp_optimal']) <= 3
    rain_optimal = p['rain_min'] <= rainfall <= p['rain_max']
    
    # For coconut, check if temperature is appropriate
    if crop == 'coconut':
        if temp < 20:
            temp_optimal = False
            message = f"❌ Temperature too low for coconut cultivation (needs >20°C)"
        elif temp > 38:
            temp_optimal = False
            message = f"❌ Temperature too high for coconut cultivation (needs <38°C)"
        elif temp_optimal and rain_optimal:
            message = f"✅ Current weather conditions are IDEAL for {crop} cultivation"
        elif temp_optimal or rain_optimal:
            message = f"⚠️ Current conditions are ACCEPTABLE but not optimal for {crop}"
        else:
            message = f"❌ Current conditions are NOT IDEAL for {crop} cultivation"
    elif temp_optimal and rain_optimal:
        message = f"✅ Current weather conditions are IDEAL for {crop} cultivation"
    elif temp_optimal or rain_optimal:
        message = f"⚠️ Current conditions are ACCEPTABLE but not optimal for {crop}"
    else:
        message = f"❌ Current conditions are NOT IDEAL for {crop} cultivation"
    
    return {
        'status': 'Excellent' if (temp_optimal and rain_optimal) else ('Moderate' if (temp_optimal or rain_optimal) else 'Poor'),
        'message': message,
        'temperature': round(temp, 1),
        'rainfall': round(rainfall, 0),
        'temp_optimal': temp_optimal,
        'rain_optimal': rain_optimal
    }


def calculate_fallback_yield(crop: str, temperature: float, rainfall: float, soil_fertility: float) -> float:
    """Calculate fallback yield when API is unavailable"""
    base_yield = REGIONAL_AVG[crop]
    p = CROP_PARAMS[crop]
    
    # Temperature factor (Gaussian, optimal at temp_optimal)
    temp_factor = np.exp(-((temperature - p['temp_optimal'])**2) / 50)
    
    # For coconut, penalty is higher outside viable range
    if crop == 'coconut':
        if temperature < p['temp_min'] or temperature > p['temp_max']:
            temp_factor *= 0.5
    temp_factor = max(0.3, min(1.0, temp_factor))
    
    # Rainfall factor (sigmoid-like, optimal at rain_optimal)
    if rainfall >= p['rain_optimal']:
        rain_factor = 1.0
    else:
        rain_factor = 0.5 + (rainfall / p['rain_optimal']) * 0.5
    
    # For coconut, steep penalty below minimum rainfall
    if crop == 'coconut' and rainfall < p['rain_min']:
        rain_factor *= 0.6
    
    rain_factor = max(0.3, min(1.0, rain_factor))
    
    # Soil fertility factor
    soil_factor = 0.6 + (soil_fertility / 5) * 0.4
    soil_factor = max(0.6, min(1.0, soil_factor))
    
    if DEBUG:
        print(f"   [DEBUG] Fallback factors - temp: {temp_factor:.2f}, rain: {rain_factor:.2f}, soil: {soil_factor:.2f}")
        print(f"   [DEBUG] Base yield: {base_yield} {'kg/ha' if crop != 'coconut' else 'nuts/ha'}")
    
    predicted_yield = base_yield * temp_factor * rain_factor * soil_factor
    
    return predicted_yield


# ============================================
# API ENDPOINTS
# ============================================

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        'service': 'Rule-Based Crop API',
        'status': 'running',
        'debug_mode': DEBUG,
        'endpoints': {
            '/predict': 'POST - Get crop predictions',
            '/health': 'GET - Health check',
            '/municipalities': 'GET - List supported municipalities',
            '/crop_params': 'GET - Crop parameters'
        },
        'supported_crops': ['rice', 'corn', 'coconut'],
        'version': '3.0.0'
    })


@app.route('/predict', methods=['POST'])
def predict_full():
    """Main prediction endpoint with debug logging"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data'}), 400
        
        crop = data.get('crop', '').lower()
        municipality = data.get('municipality', '')
        
        # Updated to include coconut
        if crop not in ['rice', 'corn', 'coconut']:
            return jsonify({'error': 'Crop must be "rice", "corn", or "coconut"'}), 400
        
        if not municipality or municipality not in SOIL_DB:
            return jsonify({'error': f'Invalid municipality. Must be one of: {list(SOIL_DB.keys())}'}), 400
        
        print(f"\n{'='*70}")
        print(f"📥 RULE-BASED API REQUEST")
        print(f"{'='*70}")
        print(f"   Crop: {crop}")
        print(f"   Municipality: {municipality}")
        print(f"   Timestamp: {datetime.now().isoformat()}")
        
        if DEBUG:
            print(f"\n   [DEBUG] Full request data:")
            for key, value in data.items():
                if key == 'raw_sequence' and value:
                    print(f"      {key}: {len(value)} weeks of data")
                elif key == 'raw_sequence':
                    print(f"      {key}: {value}")
                else:
                    print(f"      {key}: {value}")
        
        # Get climate defaults
        climate_defaults = CLIMATE_DB.get(municipality, CLIMATE_DB['Ligao'])
        
        # Get input values (use provided or defaults)
        weather_data = {
            'avg_temperature': data.get('avg_temperature', climate_defaults['temp']),
            'total_rainfall': data.get('total_rainfall', climate_defaults['rainfall']),
            'humidity': data.get('humidity', climate_defaults['humidity'])
        }
        
        ndvi = data.get('ndvi', climate_defaults['ndvi'])
        evi = data.get('evi', climate_defaults['evi'])
        soil_fertility = data.get('soil_fertility', SOIL_DB[municipality]['fertility'])
        n_score = data.get('n_score', 3)
        p_score = data.get('p_score', 3)
        k_score = data.get('k_score', 3)
        
        # Get raw_sequence if provided (from frontend weather data)
        raw_sequence = data.get('raw_sequence')
        season = data.get('season', 2)
        max_wind_kts = data.get('max_wind_kts', 0)
        min_pres_mb = data.get('min_pres_mb', 1013)
        duration_hrs = data.get('duration_hrs', 0)
        risk_score = data.get('risk_score', 1)
        
        # Get previous_yield for coconut (optional)
        previous_yield = data.get('previous_yield', 0)
        
        print(f"\n📊 INPUT VALUES SUMMARY:")
        print(f"   🌡️ Temperature: {weather_data['avg_temperature']:.1f}°C (optimal: {CROP_PARAMS[crop]['temp_optimal']}°C)")
        print(f"   🌧️ Rainfall: {weather_data['total_rainfall']:.0f}mm (optimal: {CROP_PARAMS[crop]['rain_optimal']}mm)")
        print(f"   💧 Humidity: {weather_data['humidity']:.0f}%")
        print(f"   🌿 NDVI: {ndvi} | 🍃 EVI: {evi}")
        print(f"   🪴 Soil Fertility: {soil_fertility}/5")
        print(f"   📊 N/P/K Scores: {n_score}/{p_score}/{k_score}")
        print(f"   📅 raw_sequence provided: {raw_sequence is not None}")
        if raw_sequence:
            print(f"      Weeks in sequence: {len(raw_sequence)}")
        if crop == 'coconut':
            print(f"   🥥 Previous yield: {previous_yield:,.0f} nuts")
        
        # ============================================
        # CALL SUITABILITY API WITH RETRY
        # ============================================
        suitability_payload = {
            'crop': crop,
            'ndvi': ndvi,
            'evi': evi,
            'temperature': weather_data['avg_temperature'],
            'rainfall': weather_data['total_rainfall'],
            'soil_fertility': soil_fertility,
            'n_score': n_score,
            'p_score': p_score,
            'k_score': k_score,
            'humidity': weather_data['humidity']
        }
        
        print(f"\n📤 Calling Suitability API...")
        suitability_result = None
        suitability_response_json, status_code = call_api_with_retry(
            SUITABILITY_API_URL, suitability_payload, "Suitability API"
        )
        
        if status_code == 200 and suitability_response_json:
            suitability_result = suitability_response_json
            print(f"✅ Suitability: {suitability_result.get('suitability')}")
            if DEBUG:
                print(f"   [DEBUG] Full response: {suitability_result}")
        else:
            print(f"❌ Suitability API failed (status {status_code}), using fallback")
            suitability_result = {
                'suitability': 'Medium',
                'confidence': 75,
                'probabilities': {'Low': 20, 'Medium': 75, 'High': 5}
            }
        
        # ============================================
        # CALL YIELD API WITH RETRY - NOW SUPPORTS COCONUT
        # ============================================
        print(f"\n📤 Calling Yield API...")
        
        # NOW INCLUDE COCONUT - use yield API for all crops
        use_yield_api = crop in ['rice', 'corn', 'coconut']  # FIXED: Added coconut
        
        if use_yield_api and raw_sequence and len(raw_sequence) > 0:
            # Build payload based on crop type
            if crop == 'coconut':
                yield_payload = {
                    'raw_sequence': raw_sequence,
                    'crop': 'coconut',  # Pass crop name
                    'municipality': municipality,
                    'previous_yield': previous_yield  # Optional lag feature
                }
                print(f"   Using coconut payload with {len(raw_sequence)} weeks")
            else:
                # Rice/Corn payload
                yield_payload = {
                    'raw_sequence': raw_sequence,
                    'crop': crop,  # ADDED: Pass crop name
                    'crop_encoded': 1 if crop == 'rice' else 0,
                    'municipality': municipality,
                    'season': season,
                    'max_wind_kts': max_wind_kts,
                    'min_pres_mb': min_pres_mb,
                    'duration_hrs': duration_hrs,
                    'risk_score': risk_score
                }
                print(f"   Using rice/corn payload with {len(raw_sequence)} weeks")
            
            if DEBUG:
                print(f"   [DEBUG] Yield payload keys: {list(yield_payload.keys())}")
                if crop == 'coconut':
                    print(f"   [DEBUG] Previous yield: {previous_yield}")
                else:
                    print(f"   [DEBUG] Season: {season}, risk_score: {risk_score}")
            
            yield_response_json, status_code = call_api_with_retry(
                YIELD_API_URL, yield_payload, "Yield API"
            )
            
            if status_code == 200 and yield_response_json:
                if crop == 'coconut':
                    # Handle coconut response - get nuts directly
                    predicted_yield_kg = yield_response_json.get('predicted_yield_nuts', REGIONAL_AVG[crop])
                    if predicted_yield_kg == 0 and 'predicted_yield_millions' in yield_response_json:
                        predicted_yield_kg = yield_response_json['predicted_yield_millions'] * 1000000
                    print(f"✅ Yield API returned: {predicted_yield_kg:.0f} nuts/ha")
                else:
                    # Rice/Corn response - get tons and convert to kg
                    predicted_yield_kg = yield_response_json.get('yield', REGIONAL_AVG[crop]) * 1000
                    print(f"✅ Yield API returned: {predicted_yield_kg:.0f} kg/ha")
                if DEBUG:
                    print(f"   [DEBUG] Full yield response: {yield_response_json}")
            else:
                print(f"❌ Yield API failed (status {status_code}), using fallback calculation")
                predicted_yield_kg = calculate_fallback_yield(
                    crop, weather_data['avg_temperature'], 
                    weather_data['total_rainfall'], soil_fertility
                )
                print(f"   Using fallback: {predicted_yield_kg:.0f} {'nuts/ha' if crop == 'coconut' else 'kg/ha'}")
        else:
            print(f"   No raw_sequence provided, using fallback calculation")
            predicted_yield_kg = calculate_fallback_yield(
                crop, weather_data['avg_temperature'], 
                weather_data['total_rainfall'], soil_fertility
            )
            print(f"   Fallback yield: {predicted_yield_kg:.0f} {'nuts/ha' if crop == 'coconut' else 'kg/ha'}")
        
        # ============================================
        # APPLY RULE-BASED FILTERS
        # ============================================
        
        climate_match = calc_climate_match(crop, municipality, weather_data)
        soil_compat = calc_soil_compat(crop, municipality)
        
        model_score = {'High': 85, 'Medium': 65, 'Low': 35}.get(suitability_result.get('suitability'), 65)
        overall_score = (model_score * 0.5) + (climate_match * 0.3) + (soil_compat * 0.2)
        
        if overall_score >= 80:
            overall_rating = "HIGHLY SUITABLE"
        elif overall_score >= 60:
            overall_rating = "MODERATELY SUITABLE"
        else:
            overall_rating = "MARGINALLY SUITABLE"
        
        # Get recommendations
        soil_preparation = get_soil_preparation(crop, municipality)
        harvest_advice = get_harvest_advice(crop, predicted_yield_kg)
        typhoon_info = get_typhoon_advice(municipality)
        planting_advice = get_planting_advice(crop, municipality, weather_data)
        overall_rec = get_overall_recommendation(suitability_result.get('suitability'), predicted_yield_kg, crop)
        
        regional_avg = REGIONAL_AVG[crop]
        vs_pct = ((predicted_yield_kg - regional_avg) / regional_avg) * 100
        vs_text = f"{'+' if vs_pct > 0 else ''}{vs_pct:.1f}% {'above' if vs_pct > 0 else 'below'} average"
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # ============================================
        # BUILD RESPONSE
        # ============================================
        
        response = {
            'status': 'success',
            'crop': crop.capitalize(),
            'municipality': municipality,
            'timestamp': datetime.now().isoformat(),
            'execution_time_ms': round(execution_time * 1000, 2),
            'model_results': {
                'suitability': suitability_result.get('suitability'),
                'suitability_confidence': suitability_result.get('confidence'),
                'suitability_probabilities': suitability_result.get('probabilities'),
                'predicted_yield': round(predicted_yield_kg, 0),
                'predicted_yield_unit': 'nuts/ha' if crop == 'coconut' else 'kg/ha'
            },
            'calculated_scores': {
                'climate_match_score': round(climate_match, 1),
                'soil_compatibility_score': round(soil_compat, 1),
                'overall_suitability_score': round(overall_score, 1),
                'overall_rating': overall_rating,
                'vs_regional_average': vs_text
            },
            'input_summary': {
                'temperature': round(weather_data['avg_temperature'], 1),
                'rainfall': round(weather_data['total_rainfall'], 0),
                'humidity': round(weather_data['humidity'], 0),
                'ndvi': ndvi,
                'evi': evi,
                'soil_fertility': soil_fertility,
                'n_score': n_score,
                'p_score': p_score,
                'k_score': k_score
            },
            'recommendations': {
                'overall': overall_rec,
                'soil_preparation': soil_preparation,
                'harvest_advice': harvest_advice,
                'planting_advice': planting_advice,
                'typhoon': typhoon_info
            }
        }
        
        print(f"\n✅ Overall Score: {overall_score:.1f}% - {overall_rating}")
        print(f"⏱️ Execution time: {execution_time*1000:.0f}ms")
        print(f"{'='*70}\n")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'rule-based-filter-api',
        'suitability_api': SUITABILITY_API_URL,
        'yield_api': YIELD_API_URL,
        'debug_mode': DEBUG,
        'supported_crops': ['rice', 'corn', 'coconut'],
        'timestamp': datetime.now().isoformat()
    })


@app.route('/municipalities', methods=['GET'])
def get_municipalities():
    """Get list of supported municipalities"""
    return jsonify({
        'municipalities': list(SOIL_DB.keys()),
        'soil_data': SOIL_DB,
        'climate_norms': CLIMATE_DB
    })


@app.route('/crop_params', methods=['GET'])
def get_crop_params():
    """Get crop parameters"""
    return jsonify({
        'crops': list(CROP_PARAMS.keys()),
        'parameters': CROP_PARAMS
    })


@app.route('/debug/test', methods=['GET'])
def debug_test():
    """Debug endpoint to test API connectivity"""
    results = {}
    
    # Test Suitability API
    print("\n🔍 Testing Suitability API...")
    test_crops = ['rice', 'corn', 'coconut']
    for crop in test_crops:
        try:
            test_payload = {
                'crop': crop,
                'ndvi': 0.7,
                'evi': 0.5,
                'temperature': 27 if crop != 'coconut' else 30,
                'rainfall': 1500 if crop != 'coconut' else 2000,
                'soil_fertility': 3,
                'n_score': 3,
                'p_score': 3,
                'k_score': 3,
                'humidity': 75 if crop != 'coconut' else 80
            }
            start = time.time()
            response = requests.post(SUITABILITY_API_URL, json=test_payload, timeout=10)
            results[f'suitability_api_{crop}'] = {
                'status': response.status_code,
                'response_time_ms': round((time.time() - start) * 1000, 2),
                'working': response.status_code == 200
            }
            if response.status_code == 200:
                results[f'suitability_api_{crop}']['data'] = response.json()
        except Exception as e:
            results[f'suitability_api_{crop}'] = {'error': str(e), 'working': False}
    
    # Test Yield API for all crops
    print("🔍 Testing Yield API...")
    
    # Test rice/corn
    try:
        test_payload = {
            'raw_sequence': [[0.7, 0.5, 30, 25, 27, 50, 80, 200, 10, 2024, 15, 2]],
            'crop': 'rice',
            'crop_encoded': 1,
            'municipality': 'Ligao',
            'season': 2,
            'max_wind_kts': 10,
            'min_pres_mb': 1010,
            'duration_hrs': 0,
            'risk_score': 1
        }
        start = time.time()
        response = requests.post(YIELD_API_URL, json=test_payload, timeout=10)
        results['yield_api_rice'] = {
            'status': response.status_code,
            'response_time_ms': round((time.time() - start) * 1000, 2),
            'working': response.status_code == 200
        }
        if response.status_code == 200:
            results['yield_api_rice']['data'] = response.json()
    except Exception as e:
        results['yield_api_rice'] = {'error': str(e), 'working': False}
    
    # Test coconut
    try:
        test_payload_coconut = {
            'raw_sequence': [[0.7, 0.5, 30, 25, 27, 50, 80, 200, 10, 2024, 15, 2]],
            'crop': 'coconut',
            'municipality': 'Ligao',
            'previous_yield': 0
        }
        start = time.time()
        response = requests.post(YIELD_API_URL, json=test_payload_coconut, timeout=10)
        results['yield_api_coconut'] = {
            'status': response.status_code,
            'response_time_ms': round((time.time() - start) * 1000, 2),
            'working': response.status_code == 200
        }
        if response.status_code == 200:
            results['yield_api_coconut']['data'] = response.json()
    except Exception as e:
        results['yield_api_coconut'] = {'error': str(e), 'working': False}
    
    return jsonify(results)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5003))
    print(f"\n{'='*70}")
    print(f"RULE-BASED FILTER API (DEBUG MODE: {DEBUG})")
    print(f"{'='*70}")
    print(f"Suitability API: {SUITABILITY_API_URL}")
    print(f"Yield API: {YIELD_API_URL}")
    print(f"Supported crops: rice, corn, coconut")
    print(f"Starting on port {port}")
    print(f"{'='*70}\n")
    app.run(host='0.0.0.0', port=port)
