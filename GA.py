import random
import pandas as pd

# Constants for the genetic algorithm
POPULATION_SIZE = 50  # Number of different meal plans created each generation
GENERATIONS = 100  # Number of times the meal plans evolve (iterations)
MUTATION_RATE = 0.1  # Chance of small random changes in a meal plan
CROSSOVER_RATE = 0.8  # Chance of combining parts of two meal plans

# Load the food data from a CSV file
food_data = pd.read_csv('Updated_Food_and_Calories.csv')

# Remove 'cal' from the 'Calories' column and convert to a numerical format
food_data['Calories'] = food_data['Calories'].str.replace('cal', '').astype(float)


# Function to calculate how good a meal plan is (its fitness)
def calculate_fitness(meal_plan, target_calories):
    # Calculate total calories for the meal plan
    total_calories = sum(food_data.iloc[food_index]['Calories'] for food_index in meal_plan)
    # Fitness is better if the total calories are closer to the target
    return 1 / (abs(total_calories - target_calories) + 1)


# Create a random population of meal plans
def create_random_population(population_size, meals_per_day):
    # Each meal plan consists of randomly chosen food items
    return [random.sample(range(len(food_data)), meals_per_day) for _ in range(population_size)]


# Function to select the best meal plans from the population using tournament selection
def select_meal_plan(population, fitnesses, tournament_size):
    # Randomly select a few meal plans and pick the best one
    tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
    return max(tournament, key=lambda x: x[1])[0]  # Return the best meal plan


# Combine two meal plans (parents) to create two new meal plans (children)
def combine_meal_plans(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        # Randomly choose a point to split the parents and combine them
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + [x for x in parent2[crossover_point:] if
                                              x not in parent1[:crossover_point]]
        child2 = parent2[:crossover_point] + [x for x in parent1[crossover_point:] if
                                              x not in parent2[:crossover_point]]
        return child1, child2
    return parent1, parent2  # If no crossover, return the original parents


# Mutate a meal plan by randomly changing one or more food items
def mutate_meal_plan(meal_plan):
    for i in range(len(meal_plan)):
        if random.random() < MUTATION_RATE:
            new_food = random.randint(0, len(food_data) - 1)
            # Ensure that the new food isn't already in the meal plan
            while new_food in meal_plan:
                new_food = random.randint(0, len(food_data) - 1)
            meal_plan[i] = new_food


# Main function that runs the genetic algorithm to find the best meal plan
def find_best_meal_plan(target_calories, meals_per_day):
    # Create an initial population of random meal plans
    population = create_random_population(POPULATION_SIZE, meals_per_day)

    # Evolve the population over many generations
    for generation in range(GENERATIONS):
        # Calculate the fitness score for each meal plan in the population
        fitnesses = [calculate_fitness(meal_plan, target_calories) for meal_plan in population]

        # Sort the population based on fitness (best meal plans first)
        population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]
        fitnesses.sort(reverse=True)

        # Print the best fitness score for the current generation
        print(f"Generation {generation + 1}: Best fitness = {fitnesses[0]}")

        # Create a new generation of meal plans
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            # Select two parents using tournament selection
            parent1 = select_meal_plan(population, fitnesses, 3)
            parent2 = select_meal_plan(population, fitnesses, 3)
            # Combine the parents to create children, then mutate the children
            child1, child2 = combine_meal_plans(parent1, parent2)
            mutate_meal_plan(child1)
            mutate_meal_plan(child2)
            new_population.extend([child1, child2])  # Add children to the new population

        # Replace the old population with the new one
        population = new_population[:POPULATION_SIZE]

    # After all generations, return the best meal plan
    best_meal_plan = max(population, key=lambda x: calculate_fitness(x, target_calories))
    return best_meal_plan


# Convert the meal plan indices into actual food items for display
def get_meal_plan_details(meal_plan):
    selected_foods = [food_data.iloc[food_index] for food_index in meal_plan]
    total_calories = sum(food['Calories'] for food in selected_foods)

    # Create a detailed list of food items
    meal_plan_details = []
    for food in selected_foods:
        meal_plan_details.append([food['Food'], food['Calories'], food['Serving'], food['Constraint variable']])

    return meal_plan_details, total_calories


# Main function to create a meal plan with dietary preferences and constraints
def create_meal_plan_with_ga(daily_calorie_target, dietary_preference, selected_constraints):
    global food_data

    # Filter the food data based on the user's dietary preference
    if dietary_preference == "vegetarian":
        food_data = food_data[food_data['User type'].isin(['VG/NV'])]
    elif dietary_preference == "non-vegetarian":
        food_data = food_data[food_data['User type'].isin(['NV', 'VG/NV'])]

    # Further filter the food data based on selected constraints (e.g., allergies)
    food_data = food_data[food_data['Constraint variable'].isin(selected_constraints)]

    # Find the best meal plan using the genetic algorithm
    best_meal_plan = find_best_meal_plan(target_calories=daily_calorie_target, meals_per_day=7)

    # Get the details of the selected meal plan
    meal_plan_details, total_calories = get_meal_plan_details(best_meal_plan)

    # Format and display the meal plan
    print("Meal Plan for the Day:")
    print(format_meal_plan(meal_plan_details, total_calories))

    # Calculate the fitness score based on the total calories
    fitness_score = 1
    if total_calories > daily_calorie_target:
        fitness_score = 1 / (total_calories - daily_calorie_target)

    return fitness_score


# Format the meal plan for display
def format_meal_plan(meal, total_calories):
    table = f"{'Food':<30} {'Calories':<10} {'Serving':<25} {'Constraints':<20}\n"
    table += "-" * 80 + "\n"
    for item in meal:
        table += f"{item[0]:<30} {item[1]:<10} {item[2]:<30} {item[3]:<20}\n"
    table += "-" * 80 + "\n"
    table += f"{'Total Calories':<30} {total_calories:<10}\n"
    return table
