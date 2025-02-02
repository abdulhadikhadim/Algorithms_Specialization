from itertools import product
import pandas as pd
import numpy as np

class DataDrivenRiskPredictor:
    def __init__(self, combined_data: str, year_start: int = 2010, year_end: int = 2023):
        """
        Initializes the predictor with the combined dataset and year range.

        Args:
            combined_data (str): Path to the combined dataset CSV file.
            year_start (int): Start year for analysis (default: 2010).
            year_end (int): End year for analysis (default: 2023).
        """
        self.combined_data = pd.read_csv(combined_data, index_col=0)
        self.year_start = year_start
        self.year_end = year_end

    @staticmethod
    def run_risk_predictions_for_combinations(predictor, specific_week, selected_year, alpha=0.7, beta=0.3):
        """
        Creates combinations of all ICD-10 codes and states present in the data, calculates the risk score 
        for the specified week and year, and assigns confidence levels based on the quartile distribution.

        Args:
            predictor (DataDrivenRiskPredictor): An instance of the DataDrivenRiskPredictor class.
            specific_week (int): The specific week to filter data.
            selected_year (int): The current year for which predictions are being made.
            alpha (float): Weight for encounter changes (default: 0.7).
            beta (float): Weight for temporal decay (default: 0.3).

        Returns:
            pd.DataFrame: DataFrame containing predictions for each combination with columns:
                        ['State', 'ICD-10', 'Normalized Risk Score', 'Risk Level', 'Risk Level Score'].
        """
        # Get unique states and ICD-10 codes
        states = predictor.combined_data['STATE'].unique()
        icd_10_codes = predictor.combined_data['Valid_ICD_10'].unique()

        # Create combinations of states and ICD-10 codes
        combinations = list(product(states, icd_10_codes))

        results = []

        for state, icd_10_code in combinations:
            try:
                print(f"Processing State: {state}, ICD-10: {icd_10_code}")

                # Calculate the normalized risk score for the current combination
                normalized_score = predictor.calculate_risk_of_disease_outbreak_for_week_no(
                    state_name=state,
                    ICD10_code=icd_10_code,
                    week_no=specific_week,
                    selected_year=selected_year,
                    alpha=alpha,
                    beta=beta
                )

                # Append results without assigning risk levels yet
                results.append({
                    'State': state,
                    'ICD-10': icd_10_code,
                    'Normalized Risk Score': normalized_score
                })

            except Exception as e:
                print(f"Error processing State: {state}, ICD-10: {icd_10_code}. Error: {e}")

        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)

        # Calculate quartile thresholds for risk level assignment
        q1 = results_df['Normalized Risk Score'].quantile(0.25)
        q3 = results_df['Normalized Risk Score'].quantile(0.75)

        # Assign risk levels based on quartile thresholds
        results_df['Risk Level'] = pd.cut(
            results_df['Normalized Risk Score'],
            bins=[-float('inf'), q1, q3, float('inf')],
            labels=['Low', 'Medium', 'High']
        )

        # Assign numerical scores based on risk levels
        risk_level_mapping = {'Low': 0.3, 'Medium': 0.6, 'High': 1.0}
        results_df['Risk Level Score'] = results_df['Risk Level'].map(risk_level_mapping)

        return results_df

    def calculate_risk_of_disease_outbreak_for_week_no(self, state_name: str, ICD10_code: str, week_no: int, selected_year: int, alpha: float = 0.7, beta: float = 0.3) -> float:
        """
        Calculates the risk of disease outbreak for a specific week and normalizes the total score by the number of years.

        Args:
            state_name (str): State name to filter by.
            ICD10_code (str): ICD10 code to filter by.
            week_no (int): Week number for analysis.
            selected_year (int): The current year for which predictions are being made.
            alpha (float): Weight for encounter changes (default: 0.7).
            beta (float): Weight for temporal decay (default: 0.3).

        Returns:
            float: Total normalized cumulative risk score for the specified week across all prior years, normalized by the number of years.
        """
        # Filter the dataset for the specified state and ICD10 code
        filtered_data = self.combined_data[
            (self.combined_data['Valid_ICD_10'] == ICD10_code) &
            (self.combined_data['STATE'] == state_name)
        ]

        # Exclude data from the selected year and years beyond
        filtered_data = filtered_data[filtered_data['Year'] < selected_year]

        # Create additional columns for next and previous week's encounters
        filtered_data['Next_Week_Encounters'] = filtered_data['Num_Encounters'].shift(-1)
        filtered_data['Prev_Week_Encounters'] = filtered_data['Num_Encounters'].shift(1)
        filtered_data['Change'] = filtered_data['Num_Encounters'] - filtered_data['Prev_Week_Encounters']

        # Generate all year-week combinations up to the selected year (exclusive)
        year_week_combinations = pd.DataFrame(
            [(year, week) for year in range(self.year_start, selected_year) for week in range(1, 53)],
            columns=['Year', 'Week']
        )
        complete_data = pd.merge(year_week_combinations, filtered_data, on=['Year', 'Week'], how='left')

        # Fill missing values for critical columns
        complete_data['STATE'] = complete_data['STATE'].fillna(state_name)
        complete_data['Valid_ICD_10'] = complete_data['Valid_ICD_10'].fillna(ICD10_code)
        complete_data['Prev_Week_Encounters'] = complete_data['Prev_Week_Encounters'].fillna(0.0)
        complete_data['Next_Week_Encounters'] = complete_data['Next_Week_Encounters'].fillna(0.0)
        complete_data['Change'] = complete_data['Change'].fillna(0.0)

        # Compute the average number of encounters
        average_encounters = complete_data['Num_Encounters'].mean()

        # Add encounter change and temporal weight columns
        max_year = complete_data['Year'].max()
        complete_data['Encounter_Change'] = complete_data['Num_Encounters'] - complete_data['Prev_Week_Encounters']
        complete_data['Temporal_Weight'] = np.exp(-0.1 * (max_year - complete_data['Year']))

        # Calculate risk scores
        complete_data['Risk_Score'] = (
            alpha * (complete_data['Encounter_Change'] / average_encounters).fillna(0) +
            beta * complete_data['Temporal_Weight']
        )

        # Filter for the specified week number and calculate cumulative risk scores
        week_data = complete_data[complete_data['Week'] == week_no]
        cumulative_scores = week_data.groupby('Year')['Risk_Score'].sum()

        # Normalize the cumulative scores to a range of 0-1
        max_score = cumulative_scores.max()
        if max_score > 0:
            normalized_scores = cumulative_scores / max_score
        else:
            normalized_scores = cumulative_scores

        # Calculate total normalized score across all years and normalize by the number of years
        total_normalized_score = normalized_scores.sum()
        num_years = len(normalized_scores)
        if num_years > 0:
            total_normalized_score /= num_years

        return abs(total_normalized_score)


if __name__ == '__main__':
    data_driver = DataDrivenRiskPredictor("combined_dataset_directional_classification.csv")
    risk_predictions = data_driver.run_risk_predictions_for_combinations(
        predictor=data_driver,
        specific_week=40,
        selected_year=2023
    )
    print(risk_predictions)
    risk_predictions.to_csv('risk_predictions.csv')

