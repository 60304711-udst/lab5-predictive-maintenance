import pandas as pd
import numpy as np
import time
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from deap import base, creator, tools, algorithms
import random
import mlflow

# Auto-log metrics to Azure
mlflow.autolog()

# --- 1. DATA INGESTION & PREP ---
def load_and_prep_data(filepath):
    print("Loading data...")
    columns = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]
    df = pd.read_csv(filepath, sep='\s+', header=None, names=columns)
    
    max_cycles = df.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycles.rename(columns={'time_cycles': 'max_cycle'}, inplace=True)
    df = df.merge(max_cycles, on='unit_nr')
    df['RUL'] = df['max_cycle'] - df['time_cycles']
    df.drop(columns=['max_cycle'], inplace=True)
    return df

# --- 2. FAST FEATURE EXTRACTION ---
def extract_ts_features(df):
    print("Extracting features with tsfresh...")
    start_time = time.time()
    
    sensor_cols = [col for col in df.columns if 's_' in col]
    df_features = df[['unit_nr', 'time_cycles'] + sensor_cols]
    
    # Efficient parameters for the "Fastest Pipeline" bonus
    extracted_features = extract_features(df_features, 
                                          column_id='unit_nr', 
                                          column_sort='time_cycles',
                                          default_fc_parameters=EfficientFCParameters(),
                                          n_jobs=2) 
    
    extracted_features.dropna(axis=1, inplace=True)
    print(f"Extraction Time: {time.time() - start_time:.2f} seconds")
    return extracted_features

# --- 3. FILTER-BASED SELECTION ---
def filter_features(X, y):
    print(f"Original feature count: {X.shape[1]}")
    
    # Variance Threshold
    vt = VarianceThreshold(threshold=0.01)
    X_vt = vt.fit_transform(X)
    X = pd.DataFrame(X_vt, columns=X.columns[vt.get_support()], index=X.index)
    
    # Correlation Filter
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    X.drop(columns=to_drop, inplace=True)
    
    # Mutual Information
    k_best = min(50, X.shape[1])
    mi = SelectKBest(score_func=mutual_info_regression, k=k_best)
    X_mi = mi.fit_transform(X, y)
    X_filtered = pd.DataFrame(X_mi, columns=X.columns[mi.get_support()], index=X.index)
    
    print(f"Feature count after filtering: {X_filtered.shape[1]}")
    return X_filtered

# --- 4. GENETIC ALGORITHM (DEAP) ---
def optimize_features_ga(X_train, y_train):
    print("Starting Genetic Algorithm Feature Selection...")
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -0.1))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X_train.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    eval_model = xgb.XGBRegressor(n_estimators=10, random_state=42, n_jobs=-1)
    
    def evaluate(individual):
        selected_indices = [i for i, bit in enumerate(individual) if bit == 1]
        if len(selected_indices) == 0: return 9999, 9999 
        
        X_subset = X_train.iloc[:, selected_indices]
        Xt, Xv, yt, yv = train_test_split(X_subset, y_train, test_size=0.2, random_state=42)
        eval_model.fit(Xt, yt)
        preds = eval_model.predict(Xv)
        return np.sqrt(mean_squared_error(yv, preds)), len(selected_indices)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=20) 
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=5, stats=stats, halloffame=hof, verbose=False)
    
    best_features = X_train.columns[[i for i, bit in enumerate(hof[0]) if bit == 1]].tolist()
    return best_features

# --- 5. EXECUTION ---
if __name__ == "__main__":
    total_start = time.time()
    
    df = load_and_prep_data("data/train_FD001.txt")
    y_target = df.groupby('unit_nr')['RUL'].max()
    
    X_extracted = extract_ts_features(df)
    X_extracted = X_extracted.reindex(y_target.index)
    X_filtered = filter_features(X_extracted, y_target)
    
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_target, test_size=0.2, random_state=42)
    
    best_features = optimize_features_ga(X_train, y_train)
    
    print("Training Final XGBoost Model...")
    final_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    final_model.fit(X_train[best_features], y_train)
    
    final_rmse = np.sqrt(mean_squared_error(y_test, final_model.predict(X_test[best_features])))
    total_time = time.time() - total_start
    
    # Log final custom metrics to Azure
    mlflow.log_metric("Final_RMSE", final_rmse)
    mlflow.log_metric("Feature_Count", len(best_features))
    mlflow.log_metric("Total_Runtime", total_time)
    
    print(f" DONE! Final RMSE: {final_rmse:.2f} | Features: {len(best_features)} | Time: {total_time:.2f}s")