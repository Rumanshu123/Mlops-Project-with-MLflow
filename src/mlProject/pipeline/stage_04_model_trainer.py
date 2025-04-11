# In src/mlProject/pipeline/stage_04_model_trainer.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
import optuna  # Import Optuna
from mlProject.config.configuration import ConfigurationManager
from mlProject import logger

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        self.train_and_evaluate_with_optuna(model_trainer_config)

    def train_and_evaluate_with_optuna(self, config):
        train_df = pd.read_csv(config.train_data_path)
        test_df = pd.read_csv(config.test_data_path)

        train_x = train_df.drop([config.target_column], axis=1)
        test_x = test_df.drop([config.target_column], axis=1)
        train_y = train_df[[config.target_column]]
        test_y = test_df[[config.target_column]]

        def objective(trial):
            alpha = trial.suggest_float("alpha", config.alpha_min, config.alpha_max)
            l1_ratio = trial.suggest_float("l1_ratio", config.l1_ratio_min, config.l1_ratio_max)

            with mlflow.start_run():
                lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
                lr.fit(train_x, train_y)

                predicted_qualities = lr.predict(test_x)

                (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

                mlflow.log_param("alpha", alpha)
                mlflow.log_param("l1_ratio", l1_ratio)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                mlflow.sklearn.log_model(lr, "model")

                return rmse  # Optimize for RMSE (you can change this)

        study = optuna.create_study(direction="minimize")  # Minimize RMSE
        study.optimize(objective, n_trials=10)  # Adjust the number of trials

        best_trial = study.best_trial
        logger.info(f"Best trial: score {best_trial.value}, params {best_trial.params}")

        # Train the final model with the best hyperparameters
        best_alpha = best_trial.params["alpha"]
        best_l1_ratio = best_trial.params["l1_ratio"]

        final_model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, random_state=42)
        final_model.fit(train_x, train_y)

        model_path = os.path.join(config.root_dir, config.model_name)
        joblib.dump(final_model, model_path)

        logger.info(f"Best model saved at: {model_path}")

# The if __name__ == '__main__': block should be in main.py