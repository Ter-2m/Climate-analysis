import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- CONFIGURATION ---
RANDOM_SEED = 42

def create_synthetic_data():
    """
    Creates a simple synthetic dataset simulating factors influencing CO2 emissions.
    Real-world data would be loaded from a CSV file (e.g., Kaggle CO2 Emissions datasets).
    """
    np.random.seed(RANDOM_SEED)
    
    # Independent variables (features)
    years = np.arange(2000, 2025)
    population = np.linspace(10, 15, len(years)) + np.random.normal(0, 0.5, len(years)) # in millions
    gdp_per_capita = np.linspace(5000, 12000, len(years)) + np.random.normal(0, 1000, len(years)) # in USD
    fossil_fuel_consumption = np.linspace(50, 80, len(years)) + np.random.normal(0, 3, len(years)) # in units
    
    # Dependent variable (target: CO2 Emissions in megatons)
    # The formula models a positive correlation with all features, plus random noise
    co2_emissions = (
        0.5 * population + 
        0.003 * gdp_per_capita + 
        1.5 * fossil_fuel_consumption
    ) * 10 + np.random.normal(0, 20, len(years))
    
    data = pd.DataFrame({
        'Year': years,
        'Population_M': population,
        'GDP_per_Capita': gdp_per_capita,
        'Fossil_Fuel_Consumption': fossil_fuel_consumption,
        'CO2_Emissions_Mt': co2_emissions
    })
    
    return data

def preprocess_and_split(df):
    """
    Prepares the data for the regression model.
    """
    print("--- Data Preprocessing ---")
    
    # 1. Define Features (X) and Target (y)
    # The target is what we want to predict: CO2_Emissions_Mt
    X = df[['Population_M', 'GDP_per_Capita', 'Fossil_Fuel_Consumption']]
    y = df['CO2_Emissions_Mt']
    
    # 2. Split the data into training and testing sets (80/20 split is common)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    print(f"Training set size (samples): {len(X_train)}")
    print(f"Testing set size (samples): {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def train_regression_model(X_train, y_train):
    """
    Trains a Multiple Linear Regression model.
    """
    print("\n--- Model Training (Multiple Linear Regression) ---")
    
    # Initialize the Linear Regression model
    model = LinearRegression()
    
    # Train the model using the training data
    model.fit(X_train, y_train)
    
    print("Training complete.")
    
    # Output the learned coefficients (the importance of each feature)
    print("\nModel Coefficients:")
    for feature, coef in zip(X_train.columns, model.coef_):
        print(f"  {feature}: {coef:.4f}")
    print(f"  Intercept (Baseline): {model.intercept_:.4f}")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model's performance on the test set.
    """
    print("\n--- Model Evaluation ---")
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Mt CO2")
    print(f"R-squared (R2 Score): {r2:.4f} (Closer to 1 is better)")
    
    return y_pred

def visualize_results(df, model):
    """
    Visualizes the relationship between a key feature and the target, 
    including the model's prediction line.
    """
    # Use Fossil Fuel Consumption as the primary visualization feature
    feature_to_plot = 'Fossil_Fuel_Consumption'
    
    # 1. Create a plot to show the relationship
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[feature_to_plot], y=df['CO2_Emissions_Mt'], label='Actual Data Points', color='blue')
    
    # 2. Get the model's prediction for this feature across its range
    # Note: For a true multiple regression plot, this is a simplified view.
    # We fix other variables at their mean to isolate the effect of the plotted feature.
    
    # Create an array of values for the plotted feature
    x_range = np.linspace(df[feature_to_plot].min(), df[feature_to_plot].max(), 100)
    
    # Create prediction data by setting other features to their mean
    mean_population = df['Population_M'].mean()
    mean_gdp = df['GDP_per_Capita'].mean()
    
    prediction_df = pd.DataFrame({
        'Population_M': mean_population,
        'GDP_per_Capita': mean_gdp,
        'Fossil_Fuel_Consumption': x_range
    })
    
    # Make predictions
    predicted_line = model.predict(prediction_df)
    
    # 3. Plot the regression line
    plt.plot(x_range, predicted_line, color='red', linewidth=2, label='Regression Prediction Line')
    
    # 4. Finalize plot
    plt.title(f'CO2 Emissions vs. {feature_to_plot} (Climate Action Analysis)')
    plt.xlabel(f'{feature_to_plot} (Standardized Units)')
    plt.ylabel('CO2 Emissions (Megatons)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    
    # 1. Data Collection/Creation
    data_df = create_synthetic_data()
    print("Raw Data Sample:")
    print(data_df.head())
    
    # 2. Data Preprocessing and Splitting
    X_train, X_test, y_train, y_test = preprocess_and_split(data_df)
    
    # 3. Model Training
    co2_model = train_regression_model(X_train, y_train)
    
    # 4. Model Evaluation
    y_predictions = evaluate_model(co2_model, X_test, y_test)

    # 5. Visualization (optional but recommended for assignments)
    visualize_results(data_df, co2_model)
    
    # 6. Forecasting Example (for the year 2025)
    print("\n--- New Scenario Forecasting ---")
    new_scenario = pd.DataFrame({
        'Population_M': [16.0],
        'GDP_per_Capita': [15000.0],
        'Fossil_Fuel_Consumption': [90.0]
    })
    
    # Use the trained model to predict CO2 emissions for this future scenario
    forecasted_emission = co2_model.predict(new_scenario)
    
    print(f"Scenario: Pop=16M, GDP=15k, Fuel Consump=90")
    print(f"Forecasted CO2 Emission for 2025: {forecasted_emission[0]:.2f} Megatons")