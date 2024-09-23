import pandas as pd
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np


def calculate_bmi(weight, height):  # Formula 1
    bmi = weight / (height ** 2)
    return bmi


def w_over(bmi, height):  # Formula 2
    weight_over = bmi * (height ** 2)
    return weight_over


def w_upper(height):  # Formula 3
    weight_upper = 24.9 * (height ** 2)
    return weight_upper


def w_over_minus_min(weight, weight_upper):  # Formula 4
    min_over = weight - weight_upper
    return min_over


def w_lower(height):  # Formula 5
    val = 18.5 * (height ** 2)
    return val


def w_over_minus_max(weight, w_lower):  # Formula 6
    val = weight - w_lower
    return val


def harris_benedict(weight, height_in_cm, age, gender):
    bmr = 0
    if gender == "man":
        bmr = 88.362 + (13.397 * weight) + (4.799 * height_in_cm) - (5.677 * age)  # Formula 7
    elif gender == "woman":
        bmr = 447.593 + (9.247 * weight) + (3.098 * height_in_cm) - (4.33 * age)  # Formula 8
    else:
        print("Invalid gender")
    return bmr


def DNC_categories(bmr, active):
    if active == 1:
        return bmr * 1.2
    elif active == 2:
        return bmr * 1.375
    elif active == 3:
        return bmr * 1.55
    elif active == 4:
        return bmr * 1.725
    elif active == 5:
        return bmr * 1.9
    else:
        return "Invalid activity level"


def ndn_min(w_over_min_min):  # Formula 10
    val = w_over_min_min / 0.06485
    return val


def ndn_max(w_over_min_max):  # Formula 11
    val = w_over_min_max / 0.06485
    return val


def prepare_data_for_clustering(food_data):
    features = food_data[['Calories']].values
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features


def perform_knn_clustering(normalized_features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(normalized_features)
    return cluster_labels


def create_meal_plan_with_clustering(dnc, dietary_preference, selected_constraints, dnc_sat):
    food_data = pd.read_csv('Updated_Food_and_Calories.csv')
    food_data['Calories'] = food_data['Calories'].str.replace('cal', '').astype(float)

    if dietary_preference == "vegetarian":
        food_data = food_data[food_data['User type'].isin(['VG/NV'])]
    elif dietary_preference == "non-vegetarian":
        food_data = food_data[food_data['User type'].isin(['NV', 'VG/NV'])]
    food_data = food_data[food_data['Constraint variable'].isin(selected_constraints)]

    normalized_features = prepare_data_for_clustering(food_data)
    cluster_labels = perform_knn_clustering(normalized_features)
    food_data['Cluster'] = cluster_labels

    def select_foods_for_meal(target_calories):
        selected_foods = []
        total_calories = 0
        available_clusters = set(food_data['Cluster'])

        while total_calories < target_calories and available_clusters:
            cluster = random.choice(list(available_clusters))
            cluster_foods = food_data[food_data['Cluster'] == cluster]

            if not cluster_foods.empty:
                food = cluster_foods.sample(1).iloc[0]
                if total_calories + food['Calories'] <= target_calories:
                    selected_foods.append(
                        [food['Food'], food['Calories'], food['Serving'], food['Constraint variable']])
                    total_calories += food['Calories']
                else:
                    available_clusters.remove(cluster)
            else:
                available_clusters.remove(cluster)

        return selected_foods, total_calories

    breakfast_calories = dnc * 0.3
    lunch_calories = dnc * 0.4
    dinner_calories = dnc * 0.3

    breakfast, breakfast_total = select_foods_for_meal(breakfast_calories)
    lunch, lunch_total = select_foods_for_meal(lunch_calories)
    dinner, dinner_total = select_foods_for_meal(dinner_calories)
    total_calories = breakfast_total + lunch_total + dinner_total

    def format_meal_plan(meal, total_calories):
        table = f"{'Food':<30} {'Calories':<10} {'Serving':<25} {'Constraints':<20}\n"
        table += "-" * 80 + "\n"
        for item in meal:
            table += f"{item[0]:<30} {item[1]:<10} {item[2]:<30} {item[3]:<20}\n"
        table += "-" * 80 + "\n"
        table += f"{'Total Calories':<30} {total_calories:<10}\n"
        return table

    print("Meal Plan for the Day:")
    print("Breakfast:")
    print(format_meal_plan(breakfast, breakfast_total))
    print("Lunch:")
    print(format_meal_plan(lunch, lunch_total))
    print("Dinner:")
    print(format_meal_plan(dinner, dinner_total))
    print(f"Total Meal Day Calories: {total_calories}")

    fn = 1
    if total_calories > dnc_sat:
        fn = 1 / (total_calories - dnc_sat)

    return fn


def get_bmi_category(bmi, height):
    if bmi < 15:
        return "Very severely underweight"
    elif 15 <= bmi < 16:
        return "Severely underweight"
    elif 16 <= bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    elif 30 <= bmi < 35:
        return "Class I obesity"
    elif 35 <= bmi < 40:
        return "Class II obesity"
    else:
        return "Class III obesity"


def main_program():
    selected_constraints = []
    print("Welcome to the BMI Calculator!")
    weight = float(input("Please enter your weight in kilograms: "))
    height = float(input("Please enter your height in meters: "))
    height_cm = height * 100
    age = float(input("Please enter your age: "))
    gender = input("Please enter your gender(man / woman): ")
    how_active = int(input(
        "How active are you? (1 = Sedentary, 2 = Lightly active, 3 = Moderately active, 4 = Very active, 5 = Extra active ): "))
    dietary_preference = input("Are you vegetarian or non-vegetarian? ").strip().lower()
    if dietary_preference == "vegetarian":
        choices = input(
            "Select the types of food you would like: M(Milk Products), F(Fruits), O(Onions), H(Mushrooms), U(Tofu/Soy Products), V(Vegetables), R(Carbs), Z(Indian Food), I(Italian Food), W(Snacks), N(Nuts): ").split()
        for choice in choices:
            selected_constraints.append(choice)
    elif dietary_preference == "non-vegetarian":
        choices = input(
            "Select the types of food you would like: M(Milk Products), F(Fruits), O(Onions), H(Mushrooms), U(Tofu/Soy Products), V(Vegetables), R(Carbs), Z(Indian Food), I(Italian Food), W(Snacks), N(Nuts), C(Chicken), B(Beef), P(Pork), S(Seafood), G(Egg based), L(Lamb): ").split()
        for choice in choices:
            selected_constraints.append(choice)

    if weight <= 0 or height <= 0:
        print("Weight and height must be positive numbers.")
        return

    bmi = calculate_bmi(weight, height)
    w_over_val = w_over(bmi, height)
    w_upper_val = w_upper(height)
    category = get_bmi_category(bmi, height)
    w_over_minus_min_val = w_over_minus_min(weight, w_upper_val)
    w_lower_val = w_lower(height)
    w_over_minus_max_val = w_over_minus_max(weight, w_lower_val)
    bmr = harris_benedict(weight, height_cm, age, gender)
    dnc = DNC_categories(bmr, how_active)
    dnc_sat = dnc - 500
    ndn_min_val = ndn_min(w_over_minus_min_val)
    ndn_max_val = ndn_max(w_over_minus_max_val)

    print(f"Your BMI is: {bmi:.2f}")
    print(f"Your weight over: {w_over_val:.2f}")
    print(f"You are in the {category} category.")
    print(f"Your weight over the upper limit is: {w_over_minus_min_val:.2f}")
    print(f"Your weight over the lower limit is: {w_over_minus_max_val:.2f}")
    print(f"Your BMR is: {bmr:.2f}")
    print(f"Your daily caloric needs are: {dnc:.2f}")
    print(f"Your daily caloric needs to lose weight are: {dnc_sat:.2f}")
    print(f"Your NDN min is: {ndn_min_val:.2f}")
    print(f"Your NDN max is: {ndn_max_val:.2f}")
    fn = create_meal_plan_with_clustering(dnc, dietary_preference, selected_constraints, dnc_sat)


main_program()