import pandas as pd
import numpy as np
import config
from typing import Optional ,List


class DatasetCreator:
    """
    Supports dataset creation for the following three different tasks:
        1. data_driven
        2. directional_classification
        3. outbreak_classification
    """

    def __init__(
        self,
        diagnosis_records_directory: str,
        disease_trait_flag: bool = False,
        lagged_feature_flag: bool = False,
        task: str = 'DataDriven',
        census_flag: bool = False,
        weather_flag: bool = False,
        census_data_directory: Optional[str] = None,
        weather_data_directory: Optional[str] = None
    ) -> None:
        """
        Initialize the DatasetCreator.

        Args:
            diagnosis_records_directory (str): Path to the diagnosis records CSV file.
            disease_trait_flag (bool, optional): Flag to include disease traits. Defaults to False.
            lagged_feature_flag (bool, optional): Flag to include lagged features. Defaults to False.
            task (str, optional): Task type ('DataDriven', 'directional_classification', 
                'outbreak_classification'). Defaults to 'DataDriven'.
            census_flag (bool, optional): Flag to include census data. Defaults to False.
            weather_flag (bool, optional): Flag to include weather data. Defaults to False.
            census_data_directory (str, optional): Path to the census data CSV file.
            weather_data_directory (str, optional): Path to the weather data CSV file.
        """
        self._task = task
        self._disease_trait_flag  = disease_trait_flag
        self._lagged_feature_flag  = lagged_feature_flag

        # Load diagnosis records
        try:
            self.df_state_wise_data = pd.read_csv(
                diagnosis_records_directory, index_col=0
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Diagnosis records file not found: {e}")

        # Load weather data if flagged
        if weather_flag:
            if not weather_data_directory:
                raise ValueError('Please specify the weather data directory.')
            try:
                self.df_weather_data = pd.read_csv(
                    weather_data_directory, index_col=0
                )
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Weather data file not found: {e}")

        # Load census data if flagged
        if census_flag:
            if not census_data_directory:
                raise ValueError('Please specify the census data directory.')
            try:
                self.df_census_data = pd.read_csv(
                    census_data_directory, index_col=0
                )
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Census data file not found: {e}")

    def _preprocess_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert 'Date_Time' to datetime and remove invalid dates (e.g., February 29).

        Args:
            df (pd.DataFrame): DataFrame containing the 'Date_Time' column.

        Returns:
            pd.DataFrame: Preprocessed DataFrame with valid dates.
        """
        df['Date_Time'] = pd.to_datetime(df['Date_Time'], errors='coerce')
        # Remove February 29 to avoid leap year issues
        df = df[~((df['Date_Time'].dt.month == 2) & (df['Date_Time'].dt.day == 29))]
        return df

    def _filter_valid_icd_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter and map ICD-10 codes to valid prefixes or make them disease-agnostic for 'data_driven'.

        Args:
            df (pd.DataFrame): DataFrame containing the 'ICD_10' column.

        Returns:
            pd.DataFrame: DataFrame with a new 'Valid_ICD_10' column and filtered rows.
        """
        def filter_icd_codes(icd_code: str) -> Optional[str]:
            icd_code_str = str(icd_code)

            # Handle the 'data_driven' task
            if self._task.lower() == 'data_driven':
                # Use only the part before the decimal for 'data_driven'
                return icd_code_str.split('.')[0]

            # Handle the 'prediction' task
            elif self._task.lower() == 'prediction':
                # Filter based on mappings in config for prediction
                for prefix in config.disease_mapping_prediction.keys():
                    if icd_code_str.startswith(prefix):
                        return prefix
            return None  # Exclude if no match found

        # Apply the filtering logic
        df['Valid_ICD_10'] = df['ICD_10'].apply(filter_icd_codes)

        # Drop rows without valid ICD codes (invalid or unmapped)
        df = df.dropna(subset=['Valid_ICD_10'])
        return df


    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features such as week, year, and cyclical encodings for week and month.

        Args:
            df (pd.DataFrame): DataFrame containing the 'Date_Time' column.

        Returns:
            pd.DataFrame: DataFrame with added temporal features.
        """
        df['Week'] = df['Date_Time'].dt.dayofyear // 7 + 1
        df['Year'] = df['Date_Time'].dt.year
        df['Month'] = df['Date_Time'].dt.month
        # Cyclical encoding for week
        df['week_sin'] = np.sin(2 * np.pi * df['Week'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['Week'] / 52)
        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        return df

    def _group_weekly_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Group data by Year, Week, Valid_ICD_10, and STATE, and count encounters.

        Args:
            df (pd.DataFrame): DataFrame with temporal and ICD-10 information.

        Returns:
            pd.DataFrame: Grouped DataFrame with 'Num_Encounters'.
        """
        return df.groupby(['Year', 'Week', 'Valid_ICD_10', 'STATE']).size().reset_index(name='Num_Encounters')

    def _create_full_combinations(
        self,
        df: pd.DataFrame,
        disease_weekly_patients: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create all possible combinations of Year, Week, Valid_ICD_10, and STATE.

        Args:
            df (pd.DataFrame): Original DataFrame with temporal and ICD-10 data.
            disease_weekly_patients (pd.DataFrame): Grouped DataFrame with encounter counts.

        Returns:
            pd.DataFrame: DataFrame with all possible combinations and encounter counts.
        """
        all_weeks = pd.DataFrame(
            [(year, week) for year in df['Year'].unique() for week in range(1, 53)],
            columns=['Year', 'Week']
        )
        icd_codes = pd.DataFrame(df['Valid_ICD_10'].unique(), columns=['Valid_ICD_10'])
        states = pd.DataFrame(df['STATE'].unique(), columns=['STATE'])

        # Create cross join of all weeks, ICD codes, and states
        full_combination = pd.merge(all_weeks, icd_codes, how='cross')
        full_combination = pd.merge(full_combination, states, how='cross')

        # Merge with actual encounter counts, filling missing with 0
        full_data = pd.merge(
            full_combination,
            disease_weekly_patients,
            on=['Year', 'Week', 'Valid_ICD_10', 'STATE'],
            how='left'
        ).fillna({'Num_Encounters': 0})

        return full_data

    def _add_combined_features(self, full_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add combined year-week features for sorting and analysis.

        Args:
            full_data (pd.DataFrame): DataFrame with all combinations and encounter counts.

        Returns:
            pd.DataFrame: DataFrame with added 'year_week_int' and sorted.
        """
        full_data['year_week_int'] = full_data['Year'] * 100 + full_data['Week']
        full_data = full_data.sort_values(by=['STATE', 'Valid_ICD_10', 'year_week_int'])
        return full_data

    def _calculate_disease_specific_weekly_patients(self, df1: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate disease-specific weekly patient data.

        Args:
            df1 (pd.DataFrame): Raw diagnosis records DataFrame.

        Returns:
            pd.DataFrame: Processed DataFrame with weekly patient counts.
        """
        df = df1.copy()
        df = self._preprocess_datetime(df)
        df = self._filter_valid_icd_codes(df)
        df = self._add_temporal_features(df)
        disease_weekly_patients = self._group_weekly_counts(df)
        full_data = self._create_full_combinations(df, disease_weekly_patients)
        full_data = self._add_combined_features(full_data)
        return full_data

 

    def _add_lagged_and_future_features(
            self,
            full_data: pd.DataFrame,
            num_of_lags: List[int] = [1, 2, 3]
        ) -> pd.DataFrame:
            """
            Add lagged, rolling, and seasonal features to the dataset.

            Args:
                full_data (pd.DataFrame): DataFrame containing weekly patient encounter data.
                num_of_lags (List[int], optional): List of lag periods to add as features. Defaults to [1, 2, 3].

            Returns:
                pd.DataFrame: DataFrame with added lagged and rolling features.
            """
            # Calculate lagged features
            for lag in num_of_lags:
                full_data[f'Lag_{lag}'] = full_data.groupby(['STATE', 'Valid_ICD_10'])['Num_Encounters'].shift(lag)

            # Calculate rolling features based on the maximum lag
            max_lag = max(num_of_lags)
            rolling_group = full_data.groupby(['STATE', 'Valid_ICD_10'])['Num_Encounters']

            full_data[f'Rolling_Mean_{max_lag}'] = rolling_group.rolling(max_lag, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
            full_data[f'Rolling_Sum_{max_lag}'] = rolling_group.rolling(max_lag, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
            full_data[f'Rolling_Std_{max_lag}'] = rolling_group.rolling(max_lag, min_periods=1).std().reset_index(level=[0, 1], drop=True)

            # Add last year's encounters (shifted by 52 weeks)
            full_data['Last_Year_Encounters'] = rolling_group.shift(52)

            # Add last year's rolling mean (3-week rolling mean from the same week last year)
            full_data['Last_Year_Rolling_Mean'] = full_data.groupby(['STATE', 'Valid_ICD_10'])['Num_Encounters'].transform(
                    lambda x: x.shift(52).rolling(3, min_periods=1).mean())


            # Smoothing to avoid zero division errors
            full_data['Num_Encounters'] = full_data['Num_Encounters'] + 1

            # Add first and second order derivatives for 'Num_Encounters'
            full_data['temp_diff'] = full_data['Num_Encounters'].diff(1)
            full_data['first_derivative_encounters'] = full_data['temp_diff'].shift(1)
            full_data['temp_diff2'] = full_data['temp_diff'].diff(1)
            full_data['second_derivative_encounters'] = full_data['temp_diff2'].shift(1)
            full_data.drop(columns=['temp_diff', 'temp_diff2'], inplace=True)  # Drop intermediate columns

            # Identify whether encounters are above the seasonal mean
            rolling_mean_col = f'Rolling_Mean_{max_lag}'
            full_data['Above_Seasonal_Mean'] = (full_data['Num_Encounters'] > full_data[rolling_mean_col]).astype(int)

            # Fill NaN values in new columns
            fill_values = {
                col: 0 for col in full_data.columns if col.startswith('Lag_') or col.startswith('Rolling_')
            }
            fill_values.update({
                f'Rolling_Std_{max_lag}': 1,  # Avoid division by zero in z-score
                'Last_Year_Encounters': 0,
                'Last_Year_Rolling_Mean': 0,
                'Above_Seasonal_Mean': 0,
            })
            full_data.fillna(fill_values, inplace=True)

            return full_data


    def add_disease_traits(self, df: pd.DataFrame, transmission_mode: bool = False) -> pd.DataFrame:
        """
        Add disease-specific traits from the configuration file.

        Args:
            df (pd.DataFrame): DataFrame to which traits will be added.
            transmission_mode (bool, optional): Flag to include transmission mode. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with added disease traits.
        """
        df1 = df.copy()
        df1['R_0'] = df['Valid_ICD_10'].map(lambda x: config.disease_traits_mapping.get(x, {}).get('R_0'))
        df1['Recovery Period'] = df['Valid_ICD_10'].map(lambda x: config.disease_traits_mapping.get(x, {}).get('Recovery Period'))
        if transmission_mode:
            df1['Transmission Mode'] = df['Valid_ICD_10'].map(lambda x: config.disease_traits_mapping.get(x, {}).get('Transmission Mode'))
        return df1

    def add_census_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add state-wise census data including population, population density, and average income.

        Args:
            df (pd.DataFrame): DataFrame to which census data will be added.

        Returns:
            pd.DataFrame: DataFrame with added census data.

        Raises:
            AttributeError: If census data is not loaded.
        """
        if hasattr(self, 'df_census_data'):
            return pd.merge(df, self.df_census_data, on='STATE', how='left')
        else:
            raise AttributeError("Census data not loaded. Enable `census_flag` during initialization.")

    def add_weekly_weather_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add weather statistics to the dataset.

        Args:
            df (pd.DataFrame): DataFrame to which weather data will be added.

        Returns:
            pd.DataFrame: DataFrame with added weather statistics.
        """
        weather_df = self.df_weather_data.copy()

        # Drop unnecessary columns
        weather_df.drop(
            columns=['Unnamed: 0', 'location_id', 'utc_offset_seconds', 'timezone', 'timezone_abbreviation'],
            inplace=True
        )

        # Convert 'time' to datetime
        weather_df['time'] = pd.to_datetime(weather_df['time'], errors='coerce')

        # Extract Year and Week
        weather_df['Year'] = weather_df['time'].dt.year
        weather_df['Week'] = weather_df['time'].dt.isocalendar().week

        # Map state names to abbreviations
        weather_df['STATE'] = weather_df['state_name'].map(config.state_abbreviation_mapping)

        # Group by Year, Week, STATE and calculate aggregated weather statistics
        weekly_stats = weather_df.groupby(['Year', 'Week', 'STATE']).agg(
            mean_temperature=('temperature_2m_mean (°C)', 'mean'),
            mean_precipitation=('precipitation_sum (mm)', 'mean'),
            mean_evaporation=('et0_fao_evapotranspiration (mm)', 'mean'),
            latitude=('latitude', 'first'),
            longitude=('longitude', 'first'),
            elevation=('elevation', 'first')
        ).reset_index()

        # Sort to ensure proper lag calculation
        weekly_stats = weekly_stats.sort_values(by=['STATE', 'Year', 'Week'])

        # Add future weather as placeholders for next week's prediction
        weekly_stats['future_temperature'] = weekly_stats.groupby('STATE')['mean_temperature'].shift(-1)
        weekly_stats['future_precipitation'] = weekly_stats.groupby('STATE')['mean_precipitation'].shift(-1)

        # Merge with the main DataFrame
        df = pd.merge(df, weekly_stats, on=['Year', 'Week', 'STATE'], how='inner')

        return df

    def save_combined_data_to_csv(self, df: pd.DataFrame, file_name: str) -> None:
        """
        Save the dataset to a CSV file.

        Args:
            df (pd.DataFrame): DataFrame to save.
            file_name (str): Base name of the output CSV file.
        """
        try:
            df.to_csv(f'{file_name}_{self._task}.csv', index=False)
            print(f"Data saved to {file_name}_{self._task}.csv")
        except Exception as e:
            raise IOError(f"Error saving file: {e}")

    def _add_classification_target_column_directional(
        self,
        df: pd.DataFrame,
        increase_threshold: float = 0.025,
        decrease_threshold: float = -0.025
    ) -> pd.DataFrame:
        """
        Prepare the dataset for multi-label logistic regression by generating target labels.

        Labels:
            0: Decreasing
            1: Steady (within ±threshold of change)
            2: Increasing

        Args:
            df (pd.DataFrame): DataFrame to which target labels will be added.
            increase_threshold (float, optional): Threshold for increasing trend. Defaults to 0.025.
            decrease_threshold (float, optional): Threshold for decreasing trend. Defaults to -0.025.

        Returns:
            pd.DataFrame: DataFrame with added 'Target' column.
        """
        df = df.copy()

        # Calculate next week's encounters
        df['Next_Encounter'] = df.groupby(['STATE', 'Valid_ICD_10'])['Num_Encounters'].shift(-1)

        # Calculate percentage change between current and next week's encounters
        df['Week_Over_Week_Change'] = (df['Next_Encounter'] - df['Num_Encounters']) / df['Num_Encounters']

        # Initialize the target column as steady
        df['Target'] = 1

        # Apply thresholds to classify changes
        df.loc[df['Week_Over_Week_Change'] > increase_threshold, 'Target'] = 2  # Increasing
        df.loc[df['Week_Over_Week_Change'] < decrease_threshold, 'Target'] = 0  # Decreasing

        # Drop helper columns and rows with NaN values
        df.drop(columns=['Next_Encounter', 'Week_Over_Week_Change'], inplace=True)
        df.dropna(inplace=True)

        return df

    def _add_classification_target_column_find_peaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a classification column indicating outbreaks.

        Args:
            df (pd.DataFrame): DataFrame to which the outbreak classification will be added.

        Returns:
            pd.DataFrame: DataFrame with added 'Outbreak' column.
        """
        df['Outbreak'] = 0  # Placeholder for outbreak classification
        # Implementation for peak finding can be added here
        return df

    def combine_all(self) -> pd.DataFrame:
        """
        Combine all preprocessing steps to create the final dataset.

        Returns:
            pd.DataFrame: The fully processed dataset ready for analysis or modeling.
        """
        # Calculate disease-specific weekly patient data
        df = self._calculate_disease_specific_weekly_patients(self.df_state_wise_data)

        if self._task.lower() == 'data_driven':
            return df.dropna()

        # Add lagged and rolling features if flagged
        if self._lagged_feature_flag:
            df = self._add_lagged_and_future_features(df)

        # Add disease traits if flagged
        if self._disease_trait_flag:
            df = self.add_disease_traits(df)

        # Add census data if available
        if hasattr(self, 'df_census_data'):
            df = self.add_census_data(df)

        # Add weather statistics if available
        if hasattr(self, 'df_weather_data'):
            df = self.add_weekly_weather_stats(df)

        # Add classification target columns based on the task
        if self._task.lower() == 'directional_classification':
            df = self._add_classification_target_column_directional(df)
        elif self._task.lower() == 'outbreak_classification':
            df = self._add_classification_target_column_find_peaks(df)

        return df.dropna()

    def _drop_highly_correlated(
        self,
        df: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None,
        threshold: float = 0.8
    ) -> pd.DataFrame :
        """
        Drop columns from the DataFrame that have correlations higher than the specified threshold.

        Args:
            df (pd.DataFrame): Input DataFrame.
            exclude_columns (List[str], optional): Columns to exclude from correlation calculation.
                Defaults to ['Valid_ICD_10', 'STATE', 'year_week_int'].
            threshold (float, optional): Correlation threshold above which columns will be dropped.
                Defaults to 0.8.

        Returns:
            Tuple[pd.DataFrame, List[str]]: Reduced DataFrame with highly correlated columns removed,
                and a list of columns that were dropped.
        """
        if exclude_columns is None:
            exclude_columns = ['Valid_ICD_10', 'STATE', 'year_week_int']

        # Drop specified columns from the correlation calculation
        df_corr = df.drop(columns=exclude_columns, errors='ignore')

        # Calculate the absolute correlation matrix using Kendall's method
        corr_matrix = df_corr.corr(method='kendall').abs()

        # Select the upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Identify columns to drop based on the threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        # Drop the identified columns from the original DataFrame
        df_reduced = df.drop(columns=to_drop, errors='ignore')

        return df_reduced, to_drop


if __name__ == '__main__':
    # Initialize the DatasetCreator with appropriate parameters
    creator = DatasetCreator(
        diagnosis_records_directory=r'C:\Users\hadi.khadim\Documents\CureMD_Job\Capstone\eda_pipeline\cleaned_data.csv',
        task='data_driven',  # Options: 'data_driven', 'directional_classification', 'outbreak_classification'
        disease_trait_flag=True,
        lagged_feature_flag=True,
        census_flag=True,
        weather_flag=True,
        census_data_directory=r'capstone_2.0\DISEASE OUTBREAK PREDICTION\census\census_data.csv',
        weather_data_directory=r'capstone_2.0\DISEASE OUTBREAK PREDICTION\temperature\temp_merged.csv'
    )

    # Combine all preprocessing steps to create the dataset
    df = creator.combine_all()

    # Drop highly correlated columns to reduce multicollinearity
    df_reduced, dropped_columns = creator._drop_highly_correlated(df)
    print(f"Dropped Columns: {dropped_columns}")

    # Display the final DataFrame
    print(df_reduced.head())

    # Save the combined dataset to a CSV file
    creator.save_combined_data_to_csv(df_reduced, "combined_dataset")
