import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Optional
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


class DiseaseAnomalyDetector:
    """A class to detect anomalies in disease encounter data using various statistical methods."""

    def __init__(
        self,
        input_data_directory: str,
        add_isolation_prediction: bool = True
    ) -> None:
        """
        Initialize the DiseaseAnomalyDetector.

        Args:
            input_data_directory (str): Path to the input CSV data file.
            add_isolation_prediction (bool, optional): Whether to add Isolation
                Forest predictions. Defaults to True.
        """
        self.input_df = pd.read_csv(input_data_directory)
        self.add_isolation_prediction = add_isolation_prediction
        
        # Derive Year from year_week_int if not present
        if 'Year' not in self.input_df.columns:
            self.input_df['Year'] = (self.input_df['year_week_int'] // 100).astype(int)
        
        self.input_df = self.input_df[self.input_df['year_week_int'] >= 201000].copy()

    def _filter_disease_and_state(
        self,
        df: pd.DataFrame,
        diseases: Optional[List[str]] = None,
        states: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Filter the dataset based on specified diseases and states."""
        if diseases:
            df = df[df['Valid_ICD_10'].isin(diseases)]
        if states:
            df = df[df['STATE'].isin(states)]
        print(f"Filter applied:\nFiltered size: {len(df)}")
        return df

    def _filter_by_week(self, df: pd.DataFrame, week: int) -> pd.DataFrame:
        """Filter the DataFrame based on the 'Week' column."""
        df['Week'] = pd.to_numeric(df['Week'], errors='coerce')
        if isinstance(week, int):
            week = [week]
        return df[df['Week'].isin(week)]

    def _filter_by_year_week(
        self,
        df: pd.DataFrame,
        start: Optional[int] = None,
        end: Optional[int] = None
    ) -> pd.DataFrame:
        """Filter the DataFrame based on the 'year_week_int' column."""
        df['year_week_int'] = pd.to_numeric(df['year_week_int'])
        if start is not None:
            df = df[df['year_week_int'] >= start]
        if end is not None:
            df = df[df['year_week_int'] <= end]
        return df

    def _add_lagged_features(self) -> None:
        """Add lagged features and moving averages to the input DataFrame."""
        group = self.input_df.groupby(['STATE', 'Valid_ICD_10'])['Num_Encounters']
        self.input_df['Lag_1_Week'] = group.shift(1)
        self.input_df['MA_2_Weeks'] = group.shift(1).rolling(2, min_periods=1).mean()
        self.input_df['MA_3_Weeks'] = group.shift(1).rolling(3, min_periods=1).mean()
        self.input_df['MA_4_Weeks'] = group.shift(1).rolling(4, min_periods=1).mean()
        self.input_df['MA_12_Weeks'] = group.shift(1).rolling(12, min_periods=1).mean()

    # Updated _calculate_risk_score method
    def _calculate_risk_score(self, alpha: float = 0.7, beta: float = 0.3) -> None:
        """Calculate the risk score using encounter change (positive only) and temporal decay."""
        df = self.input_df.copy()
        df['Prev_Week_Encounters'] = df.groupby(['STATE', 'Valid_ICD_10'])['Num_Encounters'].shift(1)
        df['Encounter_Change'] = (df['Num_Encounters'] - df['Prev_Week_Encounters']).fillna(0)
        
        # Consider only positive changes
        df['Encounter_Change_Pos'] = np.where(df['Encounter_Change'] > 0, df['Encounter_Change'], 0)
        
        avg_encounters = df.groupby(['STATE', 'Valid_ICD_10'])['Num_Encounters'].transform('mean')
        df['Encounter_Change_Norm'] = (df['Encounter_Change_Pos'] / avg_encounters.replace(0, np.nan)).fillna(0)
        
        max_year = df['Year'].max()
        df['Temporal_Weight'] = np.exp(-0.1 * (max_year - df['Year']))
        
        df['Risk_Score'] = (alpha * df['Encounter_Change_Norm']) + (beta * df['Temporal_Weight'])
        max_risk_score = df['Risk_Score'].max()
        if max_risk_score > 0:
            df['Risk_Score'] /= max_risk_score  # Normalize to 0-1 range
        
        # Categorize risk levels based on quantiles
        q1, q3 = df['Risk_Score'].quantile([0.25, 0.75])
        df['Risk_Level'] = pd.cut(df['Risk_Score'], 
                                bins=[-np.inf, q1, q3, np.inf],
                                labels=['Low', 'Medium', 'High'])
        risk_level_mapping = {'Low': 0.3, 'Medium': 0.6, 'High': 1.0}
        df['Risk_Level_transition'] = df['Risk_Level'].map(risk_level_mapping).astype(float)
        self.input_df = df

    def _add_flags(
        self,
        flag_1_pct: float = 0.30,
        flag_ma_2_pct: float = 0.25,
        flag_ma3_pct: float = 0.15,
        flag_ma_4_pct: float = 0.15,
        flag_ma_12_pct: float = 0.10 
    ) -> None:
        """Add flag columns based on percentage changes."""
        df = self.input_df.copy()
        epsilon = 1e-6  # Prevent division by zero

        df['Flag_Lag_1_Week'] = (
            (df['Num_Encounters'] - df['Lag_1_Week']) / 
            (df['Lag_1_Week'] + epsilon)
        ) > flag_1_pct

        df['Flag_MA_2_Weeks'] = (
            (df['Num_Encounters'] - df['MA_2_Weeks']) / 
            (df['MA_2_Weeks'] + epsilon)
        ) > flag_ma_2_pct

        df['Flag_MA_3_Weeks'] = (
            (df['Num_Encounters'] - df['MA_3_Weeks']) / 
            (df['MA_3_Weeks'] + epsilon)
        ) > flag_ma3_pct

        df['Flag_MA_4_Weeks'] = (
            (df['Num_Encounters'] - df['MA_4_Weeks']) / 
            (df['MA_4_Weeks'] + epsilon)
        ) > flag_ma_4_pct

        df['Flag_MA_12_Weeks'] = (
            (df['Num_Encounters'] - df['MA_12_Weeks']) / 
            (df['MA_12_Weeks'] + epsilon)
        ) > flag_ma_12_pct

        if self.add_isolation_prediction:
            df['Flag_Iso_Forest'] = df["isolation_forest_prediction"] == -1

        self.input_df = df

    def _isolation_forest_anomaly_adder(self, contamination: float = 0.1) -> None:
        """Apply Isolation Forest to add anomaly predictions."""
        df = self.input_df.copy()
        features = df[['Lag_1_Week', 'MA_2_Weeks', 'MA_3_Weeks', 'MA_4_Weeks', 'MA_12_Weeks']].dropna()
        
        iso_forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
        iso_forest.fit(features)
        
        # Align predictions with original index
        df.loc[features.index, 'isolation_forest_prediction'] = iso_forest.predict(features)
        self.input_df = df

    # Updated _add_anomalous_score_with_regression method
    def _add_anomalous_score_with_regression(
        self,
        score_flag1: float = 0.10,
        score_flag2: float = 0.10,
        score_flag3: float = 0.10,
        score_flag4: float = 0.10,
        score_flag12: float = 0.10,
        risk_level_transition: float = 0.11,
        score_flag_isolation: float = 0.10,
        score_residual: float = 0.29
    ) -> None:
        """Combine scores into a single anomaly score with dynamic residuals (positive only)."""
        df = self.input_df.copy()
        scaler = MinMaxScaler()
        df['Residual'] = 0

        # Calculate residuals for each disease-state group (positive only)
        for (disease, state), group in df.groupby(['Valid_ICD_10', 'STATE']):
            if len(group) < 2:
                continue
            X = group['year_week_int'].values.reshape(-1, 1)
            y = group['Num_Encounters'].values
            reg = LinearRegression().fit(X, y)
            residuals = y - reg.predict(X)
            
            # Set negative residuals to zero (ignore drops)
            positive_residuals = np.where(residuals > 0, residuals, 0)
            
            # Scale positive residuals to 0-1 range
            if positive_residuals.max() > 0:  # Avoid scaling if all zeros
                scaled_residuals = scaler.fit_transform(positive_residuals.reshape(-1, 1)).flatten()
            else:
                scaled_residuals = positive_residuals
            df.loc[group.index, 'Residual'] = scaled_residuals

        # Calculate composite anomaly score
        df['Anomaly_Score'] = (
            df['Flag_Lag_1_Week'] * score_flag1 +
            df['Flag_MA_2_Weeks'] * score_flag2 +
            df['Flag_MA_3_Weeks'] * score_flag3 +
            df['Flag_MA_4_Weeks'] * score_flag4 +
            df['Flag_MA_12_Weeks'] * score_flag12 +
            df['Risk_Level_transition'] * risk_level_transition +
            df['Residual'] * score_residual
        )

        if self.add_isolation_prediction:
            # Only consider Isolation Forest flags if current encounters > previous week
            df['Flag_Iso_Forest'] = (df["isolation_forest_prediction"] == -1) & (df['Num_Encounters'] > df['Lag_1_Week'].fillna(0))
            df['Anomaly_Score'] += df['Flag_Iso_Forest'] * score_flag_isolation

        # Dynamic thresholding based on 90th percentile
        dynamic_thresholds = df.groupby(['Valid_ICD_10', 'STATE'])['Anomaly_Score'].transform(lambda x: x.quantile(0.90))
        df['Anomalous'] = df['Anomaly_Score'] > dynamic_thresholds
        self.input_df = df
    
    
    
    def save_combined_data_to_csv(self, file_name: str) -> None:
        """Save the dataset with anomaly scores to a CSV file."""
        try:
            self.input_df.to_csv(f'{file_name}.csv', index=False)
            print(f"Data saved to {file_name}.csv")
        except Exception as e:
            raise IOError(f"Error saving file: {e}")

    def get_anomalous_scores_for_all(self) -> None:
        """Execute all steps to compute anomalous scores."""
        self._add_lagged_features()
        self._isolation_forest_anomaly_adder()
        self._add_flags()  # Now uncommented to generate necessary flags
        self._calculate_risk_score()
        self._add_anomalous_score_with_regression()


if __name__ == '__main__':
    try:
        anomaly_detector = DiseaseAnomalyDetector("combined_dataset_data_driven.csv")
    except FileNotFoundError:
        raise FileNotFoundError("Missing input dataset. Generate 'combined_dataset_data_driven.csv' first.")
    
    anomaly_detector.get_anomalous_scores_for_all()
    anomaly_detector.save_combined_data_to_csv(r'WebApp/FlaskApp/static/data/anomaly_scores2')