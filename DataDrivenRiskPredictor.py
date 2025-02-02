from itertools import product
import pandas as pd
import numpy as np
import os
from typing import Callable, List, Optional, Tuple


class DataDrivenRiskPredictor:
    def __init__(
        self,
        combined_data: str,
        year_start: int = 2010,
        year_end: int = 2023,
        train_models: bool = False
    ) -> None:
        """
        Initialize the DataDrivenRiskPredictor with the combined dataset and year range.

        Args:
            combined_data (str): Path to the combined dataset CSV file.
            year_start (int, optional): Start year for analysis. Defaults to 2010.
            year_end (int, optional): End year for analysis. Defaults to 2023.
            train_models (bool, optional): Flag to indicate if models should be trained
                or pre-trained models should be used. Defaults to False.
        """
        # Load the combined dataset from the specified CSV file
        self._combined_data: pd.DataFrame = pd.read_csv(combined_data)
        
        # Define the range of years for analysis
        self.year_start: int = year_start
        self.year_end: int = year_end
        
        # Flag to determine whether to train models or load pre-trained models
        self.train_models: bool = train_models

    def train_or_load_risk_model(self, model_path: str) -> Callable:
        """
        Train a new risk model or load an existing pre-trained model based on the
        `train_models` flag.

        Args:
            model_path (str): Path to save (if training) or load (if not training)
                the risk model.

        Returns:
            Callable: A function representing the trained risk model.
        """
        if self.train_models:
            # If training is required, train the risk model
            print("Training risk model...")
            risk_model = self._train_risk_model()
            self._save_model(risk_model, model_path)
            print(f"Risk model trained and saved to {model_path}.")
        else:
            # If not training, attempt to load the pre-trained model
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Pre-trained model not found at {model_path}. "
                    "Please train the model first."
                )
            print("Loading pre-trained risk model...")
            risk_model = self._load_model(model_path)
            print(f"Risk model loaded from {model_path}.")
        
        return risk_model

    def _global_risk_model(*args, **kwargs) -> float:
        """
        Placeholder for the global risk model logic.

        Returns:
            float: A default risk score.
        """
        # Placeholder implementation; replace with actual model logic
        return 0.5

    def _train_risk_model(self) -> Callable:
        """
        Placeholder method for training the risk model.

        Returns:
            Callable: A trained risk model function.
        """
        # Placeholder implementation; replace with actual training logic
        return self._global_risk_model

    @staticmethod
    def _save_model(model: Callable, model_path: str) -> None:
        """
        Save the trained model to the specified file path using pickle.

        Args:
            model (Callable): The trained risk model to save.
            model_path (str): Path where the model will be saved.
        """
        # Serialize and save the model using pandas' pickle functionality
        pd.to_pickle(model, model_path)

    @staticmethod
    def _load_model(model_path: str) -> Callable:
        """
        Load a trained model from the specified file path using pickle.

        Args:
            model_path (str): Path from where the model will be loaded.

        Returns:
            Callable: The loaded risk model function.
        """
        # Deserialize and load the model using pandas' pickle functionality
        return pd.read_pickle(model_path)

    @staticmethod
    def run_risk_predictions_for_combinations(
        predictor: 'DataDrivenRiskPredictor',
        specific_week: int,
        selected_year: int,
        alpha: float = 0.7,
        beta: float = 0.3
    ) -> pd.DataFrame:
        """
        Generate risk predictions for all combinations of states and ICD-10 codes
        for a specific week and year. Assign confidence levels based on quartile
        distribution.

        Args:
            predictor (DataDrivenRiskPredictor): An instance of the
                DataDrivenRiskPredictor class.
            specific_week (int): The specific week number to filter data.
            selected_year (int): The current year for which predictions are being made.
            alpha (float, optional): Weight for encounter changes. Defaults to 0.7.
            beta (float, optional): Weight for temporal decay. Defaults to 0.3.

        Returns:
            pd.DataFrame: DataFrame containing predictions for each combination with
                columns:
                ['State', 'ICD-10', 'Normalized Risk Score', 'Risk Level',
                'Risk Level Score'].
        """
        # Extract unique states and ICD-10 codes from the combined dataset
        states: np.ndarray = predictor._combined_data['STATE'].unique()
        icd_10_codes: np.ndarray = predictor._combined_data['Valid_ICD_10'].unique()

        # Create all possible combinations of states and ICD-10 codes
        combinations: List[Tuple[str, str]] = list(product(states, icd_10_codes))

        results: List[dict] = []

        for state, icd_10_code in combinations:
            try:
                print(f"Processing State: {state}, ICD-10: {icd_10_code}")

                # Calculate the normalized risk score for the current combination
                normalized_score: float = (
                    predictor._calculate_risk_of_disease_outbreak_for_week_no(
                        state_name=state,
                        ICD10_code=icd_10_code,
                        week_no=specific_week,
                        selected_year=selected_year,
                        alpha=alpha,
                        beta=beta
                    )
                )

                # Append the result without assigning risk levels yet
                results.append({
                    'State': state,
                    'ICD-10': icd_10_code,
                    'Normalized Risk Score': normalized_score
                })

            except Exception as e:
                # Handle any exceptions during processing of a combination
                print(
                    f"Error processing State: {state}, ICD-10: {icd_10_code}. "
                    f"Error: {e}"
                )

        # Convert the list of results to a pandas DataFrame
        results_df: pd.DataFrame = pd.DataFrame(results)

        if results_df.empty:
            print("No results to process.")
            return results_df

        # Calculate quartile thresholds for assigning risk levels
        q1: float = results_df['Normalized Risk Score'].quantile(0.25)
        q3: float = results_df['Normalized Risk Score'].quantile(0.75)

        # Assign risk levels based on quartile thresholds
        results_df['Risk Level'] = pd.cut(
            results_df['Normalized Risk Score'],
            bins=[-float('inf'), q1, q3, float('inf')],
            labels=['Low', 'Medium', 'High']
        )

        # Map risk levels to numerical scores
        risk_level_mapping: dict = {'Low': 0.3, 'Medium': 0.6, 'High': 1.0}
        results_df['Risk Level Score'] = results_df['Risk Level'].map(
            risk_level_mapping
        )

        return results_df

    def _calculate_risk_of_disease_outbreak_for_week_no(
        self,
        state_name: str,
        ICD10_code: str,
        week_no: int,
        selected_year: int,
        alpha: float = 0.7,
        beta: float = 0.3
    ) -> float:
        """
        Calculate the risk of disease outbreak for a specific week and normalize
        the total score by the number of years.

        Args:
            state_name (str): State name to filter by.
            ICD10_code (str): ICD10 code to filter by.
            week_no (int): Week number for analysis.
            selected_year (int): The current year for which predictions are being made.
            alpha (float, optional): Weight for encounter changes. Defaults to 0.7.
            beta (float, optional): Weight for temporal decay. Defaults to 0.3.

        Returns:
            float: Total normalized cumulative risk score for the specified week
                across all prior years, normalized by the number of years.
        """
        # Filter the dataset for the specified state and ICD10 code
        filtered_data: pd.DataFrame = self._combined_data[
            (self._combined_data['Valid_ICD_10'] == ICD10_code) &
            (self._combined_data['STATE'] == state_name)
        ]

        # Exclude data from the selected year and years beyond
        filtered_data = filtered_data[filtered_data['Year'] < selected_year]

        # Create additional columns for next and previous week's encounters
        filtered_data['Next_Week_Encounters'] = (
            filtered_data['Num_Encounters'].shift(-1)
        )
        filtered_data['Prev_Week_Encounters'] = (
            filtered_data['Num_Encounters'].shift(1)
        )
        filtered_data['Change'] = (
            filtered_data['Num_Encounters'] - filtered_data['Prev_Week_Encounters']
        )

        # Generate all year-week combinations up to the selected year
        year_week_combinations: pd.DataFrame = pd.DataFrame(
            [
                (year, week)
                for year in range(self.year_start, selected_year)
                for week in range(1, 53)
            ],
            columns=['Year', 'Week']
        )

        # Merge the generated combinations with the filtered data
        complete_data: pd.DataFrame = pd.merge(
            year_week_combinations,
            filtered_data,
            on=['Year', 'Week'],
            how='left'
        )

        # Fill missing values for critical columns
        complete_data['STATE'] = complete_data['STATE'].fillna(state_name)
        complete_data['Valid_ICD_10'] = complete_data['Valid_ICD_10'].fillna(ICD10_code)
        complete_data['Prev_Week_Encounters'] = (
            complete_data['Prev_Week_Encounters'].fillna(0.0)
        )
        complete_data['Next_Week_Encounters'] = (
            complete_data['Next_Week_Encounters'].fillna(0.0)
        )
        complete_data['Change'] = complete_data['Change'].fillna(0.0)

        # Compute the average number of encounters
        average_encounters: float = complete_data['Num_Encounters'].mean()

        # Determine the maximum year in the data for temporal weighting
        max_year: int = complete_data['Year'].max()

        # Calculate encounter change and temporal weight
        complete_data['Encounter_Change'] = (
            complete_data['Num_Encounters'] - complete_data['Prev_Week_Encounters']
        )
        complete_data['Temporal_Weight'] = np.exp(
            -0.1 * (max_year - complete_data['Year'])
        )

        # Calculate the risk score based on encounter changes and temporal decay
        complete_data['Risk_Score'] = (
            alpha * (complete_data['Encounter_Change'] / average_encounters).fillna(0) +
            beta * complete_data['Temporal_Weight']
        )

        # Filter the data for the specified week number
        week_data: pd.DataFrame = complete_data[complete_data['Week'] == week_no]

        # Group by year and sum the risk scores to get cumulative scores per year
        cumulative_scores: pd.Series = week_data.groupby('Year')['Risk_Score'].sum()

        # Normalize the cumulative scores to a range of 0-1
        max_score: float = cumulative_scores.max()
        if max_score > 0:
            normalized_scores: pd.Series = cumulative_scores / max_score
        else:
            normalized_scores = cumulative_scores

        # Calculate the total normalized score across all years
        total_normalized_score: float = normalized_scores.sum()
        num_years: int = len(normalized_scores)
        if num_years > 0:
            total_normalized_score /= num_years

        return abs(total_normalized_score)


if __name__ == '__main__':
    # Initialize the DataDrivenRiskPredictor with the combined dataset
    data_driver: DataDrivenRiskPredictor = DataDrivenRiskPredictor(
        "combined_dataset_directional_classification.csv",
        train_models=True
    )

    # Train or load the risk model based on the train_models flag
    risk_model: Callable = data_driver.train_or_load_risk_model("risk_model.pkl")

    # Run risk predictions for all combinations of states and ICD-10 codes
    risk_predictions: pd.DataFrame = DataDrivenRiskPredictor.run_risk_predictions_for_combinations(
        predictor=data_driver,
        specific_week=41,
        selected_year=2023
    )
    # Save the risk predictions to a CSV file
    risk_predictions.to_csv('data_driven_risk_predictions.csv')
    print("Risk predictions saved to 'data_driven_risk_predictions.csv'.")
