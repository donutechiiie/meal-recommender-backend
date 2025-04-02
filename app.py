from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import time
import requests

app = Flask(__name__)

# Supabase credentials
SUPABASE_URL = "https://jujxoskixfadyvrxlaru.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp1anhvc2tpeGZhZHl2cnhsYXJ1Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MjM4MjI5NSwiZXhwIjoyMDU3OTU4Mjk1fQ.Ka4RjOSpSr5ODKpklgvFJkx9iNPxgqwIMFLbQU5-NMo"
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}

# Nutritional targets (unchanged)
targets = {
    'breakfast': {
        'carbohydrate': 90, 'protein': 30, 'total_fat': 24, 'dietary_fibre_total': 10,
        'total_ascorbic_acid': 20, 'calcium_mg': 300, 'iron_mg': 4, 'vite': 5,
        'linoleic_c18_2n6': 5, 'alpha_linolenic_c18_3n3': 0.5, 'lys': 1.5, 'met': 0.5
    },
    'lunch': {
        'carbohydrate': 120, 'protein': 40, 'total_fat': 32, 'dietary_fibre_total': 15,
        'total_ascorbic_acid': 30, 'calcium_mg': 400, 'iron_mg': 6, 'vite': 7,
        'linoleic_c18_2n6': 7, 'alpha_linolenic_c18_3n3': 0.7, 'lys': 2.0, 'met': 0.7
    },
    'dinner': {
        'carbohydrate': 90, 'protein': 30, 'total_fat': 24, 'dietary_fibre_total': 10,
        'total_ascorbic_acid': 20, 'calcium_mg': 300, 'iron_mg': 4, 'vite': 5,
        'linoleic_c18_2n6': 5, 'alpha_linolenic_c18_3n3': 0.5, 'lys': 1.5, 'met': 0.5
    }
}

# Load and preprocess data once when the app starts
print("Fetching and merging data...")
start_time = time.time()

tables = {
    "nutritionaldata": "nutritional_df",
    "watersolublevitamins": "vitamins_df",
    "amino_acid_profile": "amino_df",
    "carotenoids": "carotenoids_df",
    "edibleoils": "oils_df",
    "fat_soluble_vitamins": "fat_vitamins_df",
    "fatty_acid_profile": "fatty_acids_df",
    "minerals_trace_elements": "minerals_df",
    "oligosaccharides_phytosterols": "oligo_df",
    "organic_acids": "organic_df",
    "polyphenols": "polyphenols_df",
    "starch_and_sugars": "starch_df"
}

dataframes = {}
for table, df_name in tables.items():
    response = requests.get(f"{SUPABASE_URL}/rest/v1/{table}", headers=HEADERS)
    if response.status_code == 200:
        dataframes[df_name] = pd.DataFrame(response.json())
    else:
        print(f"Failed to fetch {table}: {response.status_code} - {response.text}")
        dataframes[df_name] = pd.DataFrame()  # Empty DataFrame as fallback

combined_df = dataframes["nutritional_df"]
for df_name, df in dataframes.items():
    if df_name != "nutritional_df" and not df.empty:
        combined_df = combined_df.merge(df, on='food_code', how='left', suffixes=('', f'_{df_name}'))
combined_df.fillna(0, inplace=True)

print(f"Data fetched and merged in {time.time() - start_time:.2f} seconds")
print(f"Combined DF shape: {combined_df.shape}")

# Fetch all users using requests
def get_all_users():
    response = requests.get(f"{SUPABASE_URL}/rest/v1/UserTable?select=auth_uid,notes", headers=HEADERS)
    if response.status_code != 200:
        print(f"Failed to fetch users: {response.status_code} - {response.text}")
        return {}
    
    users = {}
    for row in response.json():
        user_id = row['auth_uid']
        notes = row.get('notes') or {}
        restrictions = notes.get('restrictions', []) if isinstance(notes, dict) else []
        users[user_id] = [r.lower() for r in restrictions if isinstance(r, str)]
    return users

# Unchanged functions
def train_decision_tree(df, meal_type):
    features = ['carbohydrate', 'protein', 'total_fat', 'dietary_fibre_total',
               'total_ascorbic_acid', 'calcium_mg', 'iron_mg', 'vite',
               'linoleic_c18_2n6', 'alpha_linolenic_c18_3n3', 'lys', 'met']

    X = df[features].values
    y = np.array([1 if (row['carbohydrate'] > 0 and row['carbohydrate'] <= targets[meal_type]['carbohydrate'] * 1.5 and
                        row['protein'] > 0 and row['protein'] <= targets[meal_type]['protein'] * 1.5 and
                        row['total_fat'] > 0 and row['total_fat'] <= targets[meal_type]['total_fat'] * 1.5)
                  else 0
                  for _, row in df.iterrows()])

    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, y)
    return clf

def generate_meal_name(meal, meal_type):
    if not meal:
        return f"{meal_type.capitalize()} Default Delight"
    key_words = []
    for food in meal[:3]:
        name_parts = food.split('(')[0].split(',')[0].strip().split()
        key_words.extend([word for word in name_parts if word.lower() not in ['whole', 'fresh', 'processed', 'tender']])
    if meal_type == 'breakfast':
        return f"Morning {key_words[0]} {key_words[1] if len(key_words) > 1 else 'Mix'}"
    elif meal_type == 'lunch':
        return f"{key_words[0]} {key_words[1] if len(key_words) > 1 else 'Feast'} Lunch"
    else:
        return f"Evening {key_words[0]} {key_words[1] if len(key_words) > 1 else 'Plate'}"

def recommend_meals_dt(meal_type, df, restrictions, num_meals=3):
    if restrictions:
        pattern = '|'.join([fr'\b{restr}\b' for restr in restrictions] +
                          [fr'{restr}.*' for restr in restrictions])
        filtered_df = df[~df['food_name_nutri'].str.lower().str.contains(pattern, na=False, case=False)]
    else:
        filtered_df = df.copy()

    clf = train_decision_tree(filtered_df, meal_type)
    features = ['carbohydrate', 'protein', 'total_fat', 'dietary_fibre_total',
               'total_ascorbic_acid', 'calcium_mg', 'iron_mg', 'vite',
               'linoleic_c18_2n6', 'alpha_linolenic_c18_3n3', 'lys', 'met']

    X = filtered_df[features].values
    predictions = clf.predict(X)
    suitable_foods = filtered_df[predictions == 1].sample(frac=1, random_state=42).reset_index(drop=True)

    meal_options = []
    used_meal_sets = set()
    min_targets = {k: v * 0.85 for k, v in targets[meal_type].items()}
    max_targets = {k: v * 1.2 for k, v in targets[meal_type].items()}

    for meal_num in range(num_meals):
        meal = []
        totals = {key: 0 for key in targets[meal_type].keys()}
        shuffled_foods = suitable_foods.sample(frac=1, random_state=meal_num * 100).reset_index(drop=True)
        idx = 0

        while idx < len(shuffled_foods):
            row = shuffled_foods.iloc[idx]
            food = row.get('food_name_nutri') or row.get('food_name')
            temp_totals = totals.copy()

            for nutrient in totals.keys():
                temp_totals[nutrient] += row.get(nutrient, 0)

            if all(temp_totals[n] <= max_targets[n] for n in temp_totals):
                meal.append(food)
                totals = temp_totals

            idx += 1

            if all(totals[n] >= min_targets[n] for n in totals):
                meal_tuple = tuple(sorted(meal))
                if meal_tuple not in used_meal_sets:
                    used_meal_sets.add(meal_tuple)
                    meal_name = generate_meal_name(meal, meal_type)
                    meal_options.append({'meal': meal, 'totals': totals, 'name': meal_name})
                    break

        if len(meal_options) <= meal_num and idx >= len(shuffled_foods):
            shuffled_foods = suitable_foods.sample(frac=1, random_state=meal_num * 200).reset_index(drop=True)
            idx = 0
            while idx < len(shuffled_foods):
                row = shuffled_foods.iloc[idx]
                food = row.get('food_name_nutri') or row.get('food_name')
                temp_totals = totals.copy()

                for nutrient in totals.keys():
                    temp_totals[nutrient] += row.get(nutrient, 0)

                if all(temp_totals[n] <= max_targets[n] for n in temp_totals):
                    meal.append(food)
                    totals = temp_totals

                idx += 1

                if all(totals[n] >= min_targets[n] for n in totals):
                    meal_tuple = tuple(sorted(meal))
                    if meal_tuple not in used_meal_sets:
                        used_meal_sets.add(meal_tuple)
                        meal_name = generate_meal_name(meal, meal_type)
                        meal_options.append({'meal': meal, 'totals': totals, 'name': meal_name})
                        break

        if len(meal_options) <= meal_num:
            meal_tuple = tuple(sorted(meal))
            if meal_tuple not in used_meal_sets:
                used_meal_sets.add(meal_tuple)
                meal_name = generate_meal_name(meal, meal_type)
                meal_options.append({'meal': meal, 'totals': totals, 'name': meal_name})

    while len(meal_options) < num_meals:
        base_meal = meal_options[0]['meal'].copy()
        base_totals = meal_options[0]['totals'].copy()
        shuffled_foods = suitable_foods.sample(frac=1, random_state=len(meal_options) * 300).reset_index(drop=True)
        extra_food = shuffled_foods.iloc[0].get('food_name_nutri') or shuffled_foods.iloc[0].get('food_name')
        new_meal = base_meal + [extra_food]
        for nutrient in base_totals.keys():
            base_totals[nutrient] += shuffled_foods.iloc[0].get(nutrient, 0)
        meal_tuple = tuple(sorted(new_meal))
        if meal_tuple not in used_meal_sets:
            used_meal_sets.add(meal_tuple)
            meal_name = generate_meal_name(new_meal, meal_type)
            meal_options.append({'meal': new_meal, 'totals': base_totals, 'name': meal_name})

    return meal_options[:num_meals]

# Flask API endpoint to get meal recommendations
@app.route('/api/meals', methods=['GET'])
def get_meal_recommendations():
    user_id = request.args.get('user_id')
    meal_type = request.args.get('meal_type', 'all')  # Default to 'all' if not specified

    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400

    all_users = get_all_users()
    if user_id not in all_users:
        return jsonify({'error': 'User not found'}), 404

    restrictions = all_users[user_id]
    result = {}

    if meal_type == 'all':
        for mt in ['breakfast', 'lunch', 'dinner']:
            options = recommend_meals_dt(mt, combined_df, restrictions)
            result[mt] = options
    elif meal_type in ['breakfast', 'lunch', 'dinner']:
        options = recommend_meals_dt(meal_type, combined_df, restrictions)
        result[meal_type] = options
    else:
        return jsonify({'error': 'Invalid meal_type. Use breakfast, lunch, dinner, or all'}), 400

    return jsonify({
        'user_id': user_id,
        'restrictions': restrictions,
        'meals': result
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
