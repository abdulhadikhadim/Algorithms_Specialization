from typing import Optional
from TwoStagePredictor import TwoStagePredictor
from DataDrivenRiskPredictor import DataDrivenRiskPredictor
from itertools import product
import pandas as pd
import numpy as np
import os

class CombinedRiskPredictor:
    def __init__(self, two_stage_predictor: TwoStagePredictor, data_driven_risk_predictor: DataDrivenRiskPredictor):
        """
        Aggregates predictions from TwoStagePredictor and DataDrivenRiskPredictor.

        Args:
            two_stage_predictor (TwoStagePredictor): An instance of the TwoStagePredictor class.
            data_driven_risk_predictor (DataDrivenRiskPredictor): An instance of the DataDrivenRiskPredictor class.
        """
        self.two_stage_predictor = two_stage_predictor
        self.data_driven_risk_predictor = data_driven_risk_predictor
        self._prediction_model_predictions: Optional[pd.DataFrame] = None
        self._risk_predictor_predictions: Optional[pd.DataFrame] = None

    def get_combined_predictions(self, specific_week: int, selected_year: int) -> None:
        """
        Generates predictions from both models and combines them.

        Args:
            specific_week (int): The specific week for predictions.
            selected_year (int): The year for predictions.
        """
        # Train or load models if not already available
        if not hasattr(self.two_stage_predictor, 'direction_model') or \
           not hasattr(self.two_stage_predictor, 'regression_model'):
            self.two_stage_predictor.direction_model, self.two_stage_predictor.regression_model = \
                self.two_stage_predictor.train_or_load_models(
                    current_year=selected_year,
                    specific_week=specific_week,
                    direction_model_path='direction_model.pkl',
                    regression_model_path='regression_model.pkl'
                )

        # Run predictions from TwoStagePredictor
        self._prediction_model_predictions = self.two_stage_predictor.run_predictions_for_combinations(
            predictor=self.two_stage_predictor,
            current_year=selected_year,
            specific_week=specific_week,
            direction_model=self.two_stage_predictor.direction_model,
            regression_model=self.two_stage_predictor.regression_model
        )

        print(f"TwoStagePredictor Predictions: {self._prediction_model_predictions.shape}")

        # Run predictions from DataDrivenRiskPredictor
        self._risk_predictor_predictions = self.data_driven_risk_predictor.run_risk_predictions_for_combinations(
            predictor=self.data_driven_risk_predictor,
            specific_week=specific_week,
            selected_year=selected_year
        )

        print(f"DataDrivenRiskPredictor Predictions: {self._risk_predictor_predictions.shape}")

        # Standardize casing and remove whitespaces for consistent merging
        for df in [self._prediction_model_predictions, self._risk_predictor_predictions]:
            df['State'] = df['State'].astype(str).str.upper().str.strip()
            df['ICD-10'] = df['ICD-10'].astype(str).str.upper().str.strip()

    def get_final_risk_score(
        self,
        classifier_weightage: float = 0.5,
        data_driven_weightage: float = 0.5,
        threshold: float = 0.5,
        patients_threshold: int = 30
    ) -> pd.DataFrame:
        """
        Combines predictions from both models into a final risk score.

        Args:
            classifier_weightage (float): Weightage for the classifier's score.
            data_driven_weightage (float): Weightage for the data-driven model's score.
            threshold (float): Threshold for determining a risky situation.
            patients_threshold (int): Minimum number of expected patients to flag as risky.

        Returns:
            pd.DataFrame: DataFrame with combined predictions and final risk assessment.
        """
        # Merge the prediction DataFrames on 'State' and 'ICD-10'
        combined_df = pd.merge(
            self._prediction_model_predictions,
            self._risk_predictor_predictions,
            on=['State', 'ICD-10'],
            how='inner'  # Ensures only overlapping combinations are merged
        )

        # Check if the merged DataFrame is empty
        if combined_df.empty:
            print("Merged DataFrame is empty. Check if there are overlapping 'State' and 'ICD-10' values.")
            # Optionally, return the empty DataFrame or handle as needed
            return combined_df

        # Ensure 'Risk Level Score' is numeric
        combined_df['Risk Level Score'] = pd.to_numeric(
            combined_df['Risk Level Score'], errors='coerce'
        )

        # Calculate the final combined score using the specified weightages
        combined_df['Final_score'] = (
            combined_df['Confidence Score'] * classifier_weightage +
            combined_df['Risk Level Score'] * data_driven_weightage
        )

        # Define conditions for flagging a risky situation
        combined_df['risky_situation'] = np.where(
            (combined_df['Final_score'] >= threshold) &
            (combined_df['Direction Prediction'] == 1) &  # Binary classification: 1 for risky
            (combined_df['Expected Num of Patients'] > patients_threshold),
            1,
            0
        )

        return combined_df  # Ensure the DataFrame is returned

if __name__ == '__main__':
    # Initialize individual predictors
    data_driver = DataDrivenRiskPredictor(
        "combined_dataset_directional_classification.csv",
        train_models=False
    )
    two_stage_predictor = TwoStagePredictor(
        "combined_dataset_directional_classification.csv",
        train_models=False
    )

    # Create CombinedRiskPredictor
    combined_predictor = CombinedRiskPredictor(two_stage_predictor, data_driver)

    # Get predictions from both models
    combined_predictor.get_combined_predictions(specific_week=41, selected_year=2023)

    # Calculate final risk scores
    final_predictions = combined_predictor.get_final_risk_score()

    # Check if final_predictions is empty
    if final_predictions.empty:
        print("Final predictions DataFrame is empty.")

    # Save results to CSV if not empty
    if not final_predictions.empty:
        final_predictions.to_csv('final_risk_predictions.csv', index=False)
        print("Final risk predictions saved to 'final_risk_predictions.csv'.")
    else:
        print("No predictions to save.")