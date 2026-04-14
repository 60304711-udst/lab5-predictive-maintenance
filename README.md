# Cloud-Native Predictive Maintenance Pipeline
**Automated Remaining Useful Life (RUL) Prediction via Evolutionary Feature Selection**

## Executive Summary
This repository contains a fully containerized, automated Machine Learning pipeline designed to predict the Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS dataset. Deployed via the Azure ML SDK v2, this architecture leverages highly parallelized feature extraction and a multi-stage selection process—culminating in a Genetic Algorithm—to optimize an XGBoost regressor.

## System Architecture
The pipeline is designed for scalability and minimal local footprint, pushing all compute and environment provisioning to Azure.

1. **Automated Cloud Orchestration (`submit_to_azure.py`)**: Uses Azure ML SDK to provision compute clusters, build Dockerized Anaconda environments (`env.yml`), and submit training jobs without manual portal interaction.
2. **Massive Feature Extraction (`tsfresh`)**: Extracts comprehensive time-series characteristics across all sensor telemetry, utilizing multiprocessing to slash extraction times.
3. **Multi-Stage Feature Reduction**:
    * **Variance Thresholding**: Eliminates static/zero-variance sensor readings.
    * **Collinearity Filtering**: Removes highly correlated features (Pearson > 0.90) to reduce redundancy.
    * **Mutual Information**: Selects the top 50 features with the highest non-linear dependency to the RUL target.
4. **Evolutionary Optimization (DEAP)**: A Genetic Algorithm navigates the feature space to find the optimal subset that minimizes model RMSE while aggressively penalizing feature bloat.
5. **Model Training & Telemetry (MLflow)**: An XGBoost Regressor is trained on the genetically optimized feature set. All hyperparameter trials, metrics, and artifacts are auto-logged directly to the Azure Studio dashboard.

## Cloud Execution Results
The pipeline successfully executed on an Azure `Standard_DS3_v2` compute cluster with exceptional efficiency. The Genetic Algorithm successfully eliminated noise, aggressively down-selecting the feature space while maintaining high predictive accuracy.

* **Final Optimized Feature Count:** 23 features
* **Evaluation Metric (RMSE):** 8.087
* **Total Cloud Compute Runtime:** 82.03 seconds

### Execution Telemetry
*(See the Azure ML Studio tracking dashboard for the completed run metrics)*
![alt text](image-1.png)
![alt text](image.png)

## Reproduction Steps
To replicate this cloud execution in your own Azure environment:

1. **Prerequisites**: Ensure the Azure CLI and `azure-ai-ml` SDK are installed locally.
2. **Authentication**: Run `az login` to authenticate your terminal.
3. **Configuration**: Update the `subscription_id`, `resource_group`, and `workspace_name` in `submit_to_azure.py` to match your Azure target.
4. **Data Placement**: Ensure the C-MAPSS `train_FD001.txt` file is located in the `/data/` directory.
5. **Launch**:
   ```bash
   python submit_to_azure.py