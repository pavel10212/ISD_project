# BMI Calculator and Meal Planner

This project combines a BMI (Body Mass Index) calculator with a meal planner that uses either a linear approach or a genetic algorithm to create personalized meal plans based on user input and dietary needs.

## Features

- BMI calculation and categorization
- Daily caloric needs estimation
- Personalized meal planning
- Support for vegetarian and non-vegetarian diets
- Consideration of dietary constraints and preferences
- Option to use either a linear approach or a genetic algorithm for meal planning

## Requirements

- Python 3.x
- pandas
- scikit-learn
- numpy (implied by scikit-learn dependency)

## Installation

1. Clone this repository
2. Install the required packages:

    `pip install pandas scikit-learn`

3. Ensure you have the `Updated_Food_and_Calories.csv` file in the same directory as the script.

## Usage

Run the `main_program()` function to start the interactive BMI calculator and meal planner. The program will prompt you for the following information:

- Weight (in kilograms)(Example: 70)
- Height (in meters)(Example: 1.75)
- Age (Example: 25)
- Gender (man)
- Activity level (1: Sedentary, 2: Lightly active, 3: Moderately active, 4: Very active, 5: Super active)
- Dietary preference (vegetarian or non-vegetarian) (Example: vegetarian)
- Food type preferences (M F O)
- Preferred algorithm (clustering or genetic) (genetic)

Based on your inputs, the program will:

1. Calculate your BMI and categorize it
2. Estimate your daily caloric needs
3. Generate a personalized meal plan
4. Display the meal plan along with calorie information

## Meal Planning Algorithms

### Linear Approach

The linear approach uses K-means clustering to group foods based on their calorie content and then selects foods from different clusters to create a balanced meal plan.

### Genetic Algorithm

The genetic algorithm evolves a population of meal plans over multiple generations to find an optimal solution that meets the calorie target and dietary preferences.

## Files

- `main_program.py`: Contains the main program logic, BMI calculations, and linear meal planning approach
- `GA.py`: Implements the genetic algorithm for meal planning
- `Updated_Food_and_Calories.csv`: Database of foods with their calorie content and other attributes

## Note

This program is for educational purposes only and should not be used as a substitute for professional medical or nutritional advice.