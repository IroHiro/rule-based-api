"""
RULE-BASED FILTER API - CORRECTED VERSION
Properly calls Suitability API (simple) and Yield API (with sequence data)
"""

import numpy as np
import requests
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ============================================
# CONFIGURATION
# ============================================

SUITABILITY_API_URL = os.environ.get('SUITABILITY_API_URL', 'https://suitability-api.onrender.com/predict')
YIELD_API_URL = os.environ.get('YIELD_API_URL', 'https://crop-yield-api-9c1l.onrender.com/predict')

# Soil database
SOIL_DB = {
    'Ligao': {'n': 4, 'p': 2, 'k': 4, 'ph': 5.0, 'fertility': 3.33},
    'Malinao': {'n': 4, 'p': 2, 'k': 4, 'ph': 5.0, 'fertility': 3.33},
    'Oas': {'n': 2, 'p': 2, 'k': 2, 'ph': 5.0, 'fertility': 2.0},
    'Pio Duran': {'n': 1, 'p': 1, 'k': 1, 'ph': 5.0, 'fertility': 1.0},
    'Polangui': {'n': 2, 'p': 2, 'k': 2, 'ph': 5.0, 'fertility': 2.0}
}

# Climate norms
CLIMATE_DB = {
    'Ligao': {'temp': 27.5, 'rainfall': 2100, 'humidity': 82, 'ndvi': 0.75, 'evi': 0.52},
    'Malinao': {'temp': 27.3, 'rainfall': 2000, 'humidity': 83, 'ndvi': 0.72, 'evi': 0.49},
    'Oas': {'temp': 27.2, 'rainfall': 1950, 'humidity': 81, 'ndvi': 0.70, 'evi': 0.48},
    'Pio Duran': {'temp': 27.0, 'rainfall': 1850, 'humidity': 79, 'ndvi': 0.68, 'evi': 0.46},
    'Polangui': {'temp': 27.4, 'rainfall': 1980, 'humidity': 81, 'ndvi': 0.71, 'evi': 0.49}
}

# Regional averages
REGIONAL_AVG = {'rice': 4500, 'corn': 3800}

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
    'rice': {'temp_optimal': 28, 'rain_optimal': 1500, 'hum_optimal': 80, 'ph_optimal': 6.2, 'days': 110},
    'corn': {'temp_optimal': 27, 'rain_optimal': 800, 'hum_optimal': 70, 'ph_optimal': 6.5, 'days': 100}
}

# ============================================
# HELPER FUNCTIONS
# ============================================

def build_raw_sequence(weather_data, ndvi, evi):
    """
    Build the raw_sequence that Yield API expects (4 weeks of data)
    """
    all_days = weather_data.get('allDays', [])
    if not all_days:
        return None
    
    raw_sequence = []
    last_season = None
    
    for i in range(4):
        week_block = all_days[i * 7:(i * 7) + 7]
        if len(week_block) == 0:
            break
        
        tmax = max(d.get('tempmax', d.get('temp', 30)) for d in week_block)
        tmin = min(d.get('tempmin', d.get('temp', 24)) for d in week_block)
        tave = sum(d.get('temp', 27) for d in week_block) / len(week_block)
        rain = sum(d.get('precip', 0) for d in week_block)
        hum = sum(d.get('humidity', 75) for d in week_block) / len(week_block)
        solar = sum(d.get('solarradiation', 200) for d in week_block) / len(week_block)
        wind = sum(d.get('windspeed', 10) for d in week_block) / len(week_block)
        
        date_obj = datetime.strptime(week_block[0]['datetime'], '%Y-%m-%d')
        week_num = date_obj.isocalendar()[1]
        season = 2 if date_obj.month >= 6 else 1
        last_season = season
        
        raw_sequence.append([
            ndvi, evi,
            tmax, tmin, tave, rain, hum, solar, wind,
            date_obj.year, week_num, season
        ])
    
    return raw_sequence, last_season

def calc_climate_match(crop: str, municipality: str, climate_data: Dict) -> float:
    """Calculate climate match percentage"""
    climate = CLIMATE_DB.get(municipality, CLIMATE_DB['Ligao'])
    p = CROP_PARAMS[crop]
    
    temp = climate_data.get('avg_temperature', climate['temp'])
    rainfall = climate_data.get('total_rainfall', climate['rainfall'])
    humidity = climate_data.get('humidity', climate['humidity'])
    
    temp_score = 100 * np.exp(-((temp - p['temp_optimal'])**2) / 50)
    rain_score = 100 * np.exp(-((rainfall - p['rain_optimal'])**2) / 180000)
    hum_score = 100 * np.exp(-((humidity - p['hum_optimal'])**2) / 200)
    
    return (temp_score * 0.4) + (rain_score * 0.35) + (hum_score * 0.25)


def calc_soil_compat(crop: str, municipality: str) -> float:
    """Calculate soil compatibility percentage"""
    soil = SOIL_DB.get(municipality, SOIL_DB['Ligao'])
    p = CROP_PARAMS[crop]
    
    ph_score = 100 * np.exp(-((soil['ph'] - p['ph_optimal'])**2) / 0.18)
    npk_score = (soil['fertility'] / 5) * 100
    
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
    else:
        soil_prep.append("Create furrows 75 cm apart for proper drainage")
    
    return soil_prep


def get_harvest_advice(crop: str, predicted_yield_kg: float) -> List[str]:
    """Get harvest recommendations"""
    days = CROP_PARAMS[crop]['days']
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
                "Clear drainage canals, secure equipment"
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
    rain_optimal = 500 <= rainfall <= 2000
    
    if temp_optimal and rain_optimal:
        status = "Excellent"
        message = f"✅ Current weather conditions are IDEAL for {crop} cultivation"
    elif temp_optimal or rain_optimal:
        status = "Moderate"
        message = f"⚠️ Current conditions are ACCEPTABLE but not optimal for {crop}"
    else:
        status = "Poor"
        message = f"❌ Current conditions are NOT IDEAL for {crop} cultivation"
    
    return {
        'status': status,
        'message': message,
        'temperature': round(temp, 1),
        'rainfall': round(rainfall, 0),
        'temp_optimal': temp_optimal,
        'rain_optimal': rain_optimal
    }


# ============================================
# API ENDPOINTS
# ============================================

@app.route('/predict', methods=['POST'])
def predict_full():
    """
    Main prediction endpoint - CORRECTED to match both APIs
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data'}), 400
        
        crop = data.get('crop', '').lower()
        municipality = data.get('municipality', '')
        
        if crop not in ['rice', 'corn']:
            return jsonify({'error': 'Crop must be "rice" or "corn"'}), 400
        
        if not municipality or municipality not in SOIL_DB:
            return jsonify({'error': f'Invalid municipality. Must be one of: {list(SOIL_DB.keys())}'}), 400
        
        print(f"\n{'='*60}")
        print(f"📥 RULE-BASED API REQUEST")
        print(f"{'='*60}")
        print(f"   Crop: {crop}")
        print(f"   Municipality: {municipality}")
        
        # Get climate defaults
        climate_defaults = CLIMATE_DB.get(municipality, CLIMATE_DB['Ligao'])
        
        # Use provided weather data or defaults
        weather_data = {
            'avg_temperature': data.get('avg_temperature', climate_defaults['temp']),
            'total_rainfall': data.get('total_rainfall', climate_defaults['rainfall']),
            'humidity': data.get('humidity', climate_defaults['humidity']),
            'allDays': data.get('allDays', [])  # May be passed from frontend
        }
        
        ndvi = data.get('ndvi', climate_defaults['ndvi'])
        evi = data.get('evi', climate_defaults['evi'])
        soil_fertility = data.get('soil_fertility', SOIL_DB[municipality]['fertility'])
        
        print(f"\n📊 INPUT VALUES:")
        print(f"   Temperature: {weather_data['avg_temperature']}°C")
        print(f"   Rainfall: {weather_data['total_rainfall']}mm")
        print(f"   Humidity: {weather_data['humidity']}%")
        print(f"   NDVI: {ndvi}")
        print(f"   EVI: {evi}")
        print(f"   Soil Fertility: {soil_fertility}")
        
        # ============================================
        # CALL SUITABILITY API (Correct format)
        # ============================================
        suitability_payload = {
            'crop': crop,
            'ndvi': ndvi,
            'evi': evi,
            'temperature': weather_data['avg_temperature'],
            'rainfall': weather_data['total_rainfall'],
            'soil_fertility': soil_fertility,
            'n_score': data.get('n_score', 3),
            'p_score': data.get('p_score', 3),
            'k_score': data.get('k_score', 3),
            'humidity': weather_data['humidity']
        }
        
        print(f"\n📤 Calling Suitability API...")
        print(f"   URL: {SUITABILITY_API_URL}")
        
        try:
            suitability_response = requests.post(
                SUITABILITY_API_URL,
                json=suitability_payload,
                timeout=30
            )
            
            print(f"   Status: {suitability_response.status_code}")
            
            if suitability_response.status_code == 200:
                suitability_result = suitability_response.json()
                print(f"✅ Suitability: {suitability_result.get('suitability')}")
            else:
                print(f"❌ Suitability API error: {suitability_response.status_code}")
                suitability_result = {
                    'suitability': 'Medium',
                    'confidence': 75,
                    'probabilities': {'Low': 20, 'Medium': 75, 'High': 5}
                }
        except Exception as e:
            print(f"❌ Suitability API exception: {e}")
            suitability_result = {
                'suitability': 'Medium',
                'confidence': 75,
                'probabilities': {'Low': 20, 'Medium': 75, 'High': 5}
            }
        
        # ============================================
        # CALL YIELD API (CORRECT format with raw_sequence)
        # ============================================
        print(f"\n📤 Calling Yield API...")
        print(f"   URL: {YIELD_API_URL}")
        
        # Try to build raw_sequence if weather data available
        raw_sequence = data.get('raw_sequence')  # Could be passed from frontend
        
        if not raw_sequence and weather_data.get('allDays'):
            raw_sequence, season = build_raw_sequence(weather_data, ndvi, evi)
            print(f"   Built raw_sequence from weather data: {len(raw_sequence) if raw_sequence else 0} weeks")
        
        if raw_sequence and len(raw_sequence) > 0:
            # Proper format with sequence data
            # Calculate typhoon parameters from weather data
            all_days = weather_data.get('allDays', [])
            max_wind_kts = max([d.get('windspeed', 0) for d in all_days], default=0) * 0.539957
            min_pres_mb = min([d.get('pressure', 1013) for d in all_days], default=1013)
            duration_hrs = len([d for d in all_days if d.get('windspeed', 0) > 63]) * 24
            risk_score = 3 if max_wind_kts > 100 else (2 if max_wind_kts > 64 else 1)
            
            season = data.get('season', 2)  # Default to Wet season
            
            yield_payload = {
                'raw_sequence': raw_sequence,
                'crop_encoded': 1 if crop == 'rice' else 0,
                'municipality': municipality,
                'season': season,
                'max_wind_kts': max_wind_kts,
                'min_pres_mb': min_pres_mb,
                'duration_hrs': duration_hrs,
                'risk_score': risk_score
            }
        else:
            # Fallback: Use rule-based estimation
            print(f"   ⚠️ No raw_sequence available, using rule-based estimation")
            yield_payload = None
            base_yield = REGIONAL_AVG[crop]
            temp_factor = 1 - abs(weather_data['avg_temperature'] - CROP_PARAMS[crop]['temp_optimal']) / 20
            rain_factor = min(1, weather_data['total_rainfall'] / CROP_PARAMS[crop]['rain_optimal'])
            soil_factor = 0.7 + (soil_fertility / 5) * 0.5
            predicted_yield_kg = base_yield * max(0, temp_factor) * rain_factor * soil_factor
            print(f"   Rule-based yield: {predicted_yield_kg:.0f} kg/ha")
        
        if yield_payload:
            try:
                yield_response = requests.post(
                    YIELD_API_URL,
                    json=yield_payload,
                    timeout=30
                )
                
                print(f"   Status: {yield_response.status_code}")
                
                if yield_response.status_code == 200:
                    yield_result = yield_response.json()
                    predicted_yield_kg = yield_result.get('yield', REGIONAL_AVG[crop]) * 1000
                    print(f"✅ Yield API returned: {predicted_yield_kg:.0f} kg/ha")
                else:
                    print(f"❌ Yield API error: {yield_response.status_code}")
                    print(f"   Response: {yield_response.text[:200]}")
                    # Fallback to rule-based
                    base_yield = REGIONAL_AVG[crop]
                    temp_factor = 1 - abs(weather_data['avg_temperature'] - CROP_PARAMS[crop]['temp_optimal']) / 20
                    rain_factor = min(1, weather_data['total_rainfall'] / CROP_PARAMS[crop]['rain_optimal'])
                    soil_factor = 0.7 + (soil_fertility / 5) * 0.5
                    predicted_yield_kg = base_yield * max(0, temp_factor) * rain_factor * soil_factor
                    print(f"   Using fallback: {predicted_yield_kg:.0f} kg/ha")
                    
            except Exception as e:
                print(f"❌ Yield API exception: {e}")
                base_yield = REGIONAL_AVG[crop]
                temp_factor = 1 - abs(weather_data['avg_temperature'] - CROP_PARAMS[crop]['temp_optimal']) / 20
                rain_factor = min(1, weather_data['total_rainfall'] / CROP_PARAMS[crop]['rain_optimal'])
                soil_factor = 0.7 + (soil_fertility / 5) * 0.5
                predicted_yield_kg = base_yield * max(0, temp_factor) * rain_factor * soil_factor
                print(f"   Using fallback: {predicted_yield_kg:.0f} kg/ha")
        
        print(f"✅ Final Predicted Yield: {predicted_yield_kg:.0f} kg/ha")
        
        # ============================================
        # APPLY RULE-BASED FILTERS
        # ============================================
        
        climate_match = calc_climate_match(crop, municipality, weather_data)
        soil_compat = calc_soil_compat(crop, municipality)
        
        model_score = {
            'High': 85,
            'Medium': 65,
            'Low': 35
        }.get(suitability_result.get('suitability'), 65)
        
        overall_score = (model_score * 0.5) + (climate_match * 0.3) + (soil_compat * 0.2)
        
        if overall_score >= 80:
            overall_rating = "HIGHLY SUITABLE"
        elif overall_score >= 60:
            overall_rating = "MODERATELY SUITABLE"
        else:
            overall_rating = "MARGINALLY SUITABLE"
        
        soil_preparation = get_soil_preparation(crop, municipality)
        harvest_advice = get_harvest_advice(crop, predicted_yield_kg)
        typhoon_info = get_typhoon_advice(municipality)
        planting_advice = get_planting_advice(crop, municipality, weather_data)
        overall_rec = get_overall_recommendation(
            suitability_result.get('suitability'), 
            predicted_yield_kg, 
            crop
        )
        
        regional_avg = REGIONAL_AVG[crop]
        vs_pct = ((predicted_yield_kg - regional_avg) / regional_avg) * 100
        vs_text = f"{'+' if vs_pct > 0 else ''}{vs_pct:.1f}% {'above' if vs_pct > 0 else 'below'} average"
        
        # ============================================
        # BUILD RESPONSE
        # ============================================
        
        response = {
            'status': 'success',
            'crop': crop.capitalize(),
            'municipality': municipality,
            'timestamp': datetime.now().isoformat(),
            
            'model_results': {
                'suitability': suitability_result.get('suitability'),
                'suitability_confidence': suitability_result.get('confidence'),
                'suitability_probabilities': suitability_result.get('probabilities'),
                'predicted_yield_kg': round(predicted_yield_kg, 0),
                'predicted_yield_tons': round(predicted_yield_kg / 1000, 2)
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
                'soil_fertility': soil_fertility
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
        print(f"{'='*60}\n")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'rule-based-filter-api',
        'suitability_api': SUITABILITY_API_URL,
        'yield_api': YIELD_API_URL
    })


@app.route('/municipalities', methods=['GET'])
def get_municipalities():
    return jsonify({
        'municipalities': list(SOIL_DB.keys()),
        'soil_data': SOIL_DB,
        'climate_norms': CLIMATE_DB
    })


@app.route('/crop_params', methods=['GET'])
def get_crop_params():
    return jsonify({
        'crops': list(CROP_PARAMS.keys()),
        'parameters': CROP_PARAMS
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5003))
    print(f"\n{'='*60}")
    print(f"RULE-BASED FILTER API")
    print(f"{'='*60}")
    print(f"Suitability API: {SUITABILITY_API_URL}")
    print(f"Yield API: {YIELD_API_URL}")
    print(f"Starting on port {port}")
    print(f"{'='*60}\n")
    app.run(host='0.0.0.0', port=port)
