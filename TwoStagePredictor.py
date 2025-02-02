from itertools import product
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
import joblib
import os
import numpy as np
from typing import Callable, Tuple, Optional

# Suppress specific pandas warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


class TwoStagePredictor:
    def __init__(
        self,
        csv_path: str,
        train_models: bool = True,
        date_col: str = 'year_week_int'
    ) -> None:
        """
        Initializes the TwoStagePredictor by loading the dataset and storing parameters.

        Args:
            csv_path (str): Path to the CSV file containing data.
            train_models (bool): Flag to indicate if models should be trained or pre-trained models should be used.
            date_col (str): Column representing the year-week integer.
        """
        # Load the dataset from the specified CSV file
        self.df: pd.DataFrame = pd.read_csv(csv_path)
        self.date_col: str = date_col

        # Columns to be dropped during feature engineering
        self.columns_to_drop: list = ['Year', 'Week', 'year_week_int',
                                      'Valid_ICD_10', 'STATE']

        # Placeholder for dummy columns generated from 'Confidence_Level'
        self.dummy_columns: Optional[list] = None

        # Flag to determine whether to train models or load existing ones
        self.train_models: bool = train_models

    def _train_classifier(
        self,
        current_year: int,
        specific_week: int,
        target_column: str = 'Target',
        save_path: Optional[str] = None
    ) -> Tuple[RandomForestClassifier, pd.DataFrame]:
        """
        Trains a RandomForestClassifier on the filtered dataset.

        Args:
            current_year (int): The current year for filtering data.
            specific_week (int): The specific week for filtering data.
            target_column (str, optional): The target column for classification. Defaults to 'Target'.
            save_path (str, optional): Path to save the trained classifier. Defaults to None.

        Returns:
            Tuple[RandomForestClassifier, pd.DataFrame]: The trained classifier and the filtered DataFrame with predictions.
        """
        # Initialize the RandomForestClassifier with specified hyperparameters
        random_forest_model: RandomForestClassifier = RandomForestClassifier(
            n_estimators=100,
            class_weight={0: 1, 1: 1, 2: 1.5},
            max_depth=5,
            random_state=42,
            max_features='sqrt',
            bootstrap=True,
            min_samples_leaf=5,
            min_samples_split=3
        )

        # Define the cutoff for training data based on current year and week
        valid_time: int = current_year * 100 + specific_week

        # Create a copy of the dataset to avoid modifying the original
        df: pd.DataFrame = self.df.copy()

        # Filter the dataset to include only records before the valid_time
        filtered_df: pd.DataFrame = df[df['year_week_int'] < valid_time]

        # Drop unnecessary columns for training
        filtered_df_without_columns: pd.DataFrame = filtered_df.drop(
            columns=self.columns_to_drop
        )

        # Define training features and target
        X_train: pd.DataFrame = filtered_df_without_columns.drop(
            columns=target_column
        )
        y_train: pd.Series = filtered_df_without_columns[target_column]

        # Train the classifier
        random_forest_model.fit(X_train, y_train)

        # Predict probabilities on the training data
        probs_train: np.ndarray = random_forest_model.predict_proba(X_train)

        # Add prediction results to the filtered DataFrame
        filtered_df['Max_Prob'] = probs_train.max(axis=1)
        filtered_df['Predicted_Class'] = probs_train.argmax(axis=1)
        filtered_df['Confidence_Level'] = filtered_df['Max_Prob'].apply(
            self._classify_risk
        )

        # Save the trained classifier if a save path is provided
        if save_path:
            joblib.dump(random_forest_model, save_path)
            print(f"Classifier saved to {save_path}.")

        return random_forest_model, filtered_df

    def _train_regressor(
        self,
        df: pd.DataFrame,
        target_column: str = 'Target_Week_1',
        save_path: Optional[str] = None
    ) -> RandomForestRegressor:
        """
        Trains a RandomForestRegressor on the provided DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to train the regressor on.
            target_column (str, optional): The target column for regression. Defaults to 'Target_Week_1'.
            save_path (str, optional): Path to save the trained regressor. Defaults to None.

        Returns:
            RandomForestRegressor: The trained regressor model.
        """
        # Initialize the RandomForestRegressor with specified hyperparameters
        regressor_model: RandomForestRegressor = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_features='sqrt',
            min_samples_split=5,
            min_samples_leaf=3
        )

        # Create a copy of the DataFrame to avoid modifying the original
        filtered_df: pd.DataFrame = df.copy()

        # Add regression target columns (e.g., Target_Week_1)
        filtered_df = self._add_regression_target_column(filtered_df, num_weeks=[1])

        # Generate dummy variables from 'Confidence_Level'
        dummy_columns: pd.DataFrame = pd.get_dummies(
            filtered_df['Confidence_Level'],
            prefix='Confidence_Level'
        )
        self.dummy_columns = dummy_columns.columns.tolist()  # Save column names for alignment

        # Concatenate dummy variables with the DataFrame and drop the original 'Confidence_Level' column
        filtered_df: pd.DataFrame = pd.concat(
            [filtered_df, dummy_columns],
            axis=1
        ).drop(columns="Confidence_Level")

        # Define training features and target for regression
        X_train: pd.DataFrame = filtered_df.drop(
            columns=self.columns_to_drop + [target_column]
        )
        y_train: pd.Series = filtered_df[target_column]

        # Train the regressor
        regressor_model.fit(X_train, y_train)

        # Save the trained regressor and dummy_columns if a save path is provided
        if save_path:
            joblib.dump(
                {'model': regressor_model, 'dummy_columns': self.dummy_columns},
                save_path
            )
            print(f"Regressor and dummy columns saved to {save_path}.")

        return regressor_model

    def _get_direction_prediction(
        self,
        model: RandomForestClassifier,
        current_year: int,
        specific_week: int,
        disease: str,
        state: str,
        target_column: str = 'Target'
    ) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
        """
        Predicts the direction of disease outbreaks for a specific combination.

        Args:
            model (RandomForestClassifier): The trained classification model.
            current_year (int): The current year for filtering data.
            specific_week (int): The specific week for filtering data.
            disease (str): The ICD-10 code to filter by.
            state (str): The state to filter by.
            target_column (str, optional): The target column for classification. Defaults to 'Target'.

        Returns:
            Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]: The predicted direction and the DataFrame with predictions.
        """
        # Filter the dataset for the specified disease, state, week, and year
        data_to_predict: pd.DataFrame = self.df[
            (self.df['Valid_ICD_10'] == disease) &
            (self.df['STATE'] == state) &
            (self.df['Week'] == specific_week) &
            (self.df['Year'] == current_year)
        ].copy()

        # Check if there is data to predict
        if data_to_predict.empty:
            print(
                f"No data to predict for State: {state}, ICD-10: {disease}, "
                f"Week: {specific_week}, Year: {current_year}"
            )
            return None, None

        # Drop unnecessary columns, including the target
        X_test: pd.DataFrame = data_to_predict.drop(
            columns=self.columns_to_drop + [target_column],
            errors='ignore'
        )

        # Ensure feature alignment by adding missing features with default value 0
        expected_features: np.ndarray = model.feature_names_in_
        for col in expected_features:
            if col not in X_test.columns:
                X_test[col] = 0  # Add missing features with default value

        # Reorder columns to match the training data
        X_test = X_test[expected_features]

        # Predict the direction classes
        direction_prediction: np.ndarray = model.predict(X_test)

        # Predict probabilities for each class
        probs_test: np.ndarray = model.predict_proba(X_test)

        # Add prediction results to the DataFrame
        data_to_predict['Max_Prob'] = probs_test.max(axis=1)
        data_to_predict['Predicted_Class'] = probs_test.argmax(axis=1)
        data_to_predict['Confidence_Level'] = data_to_predict['Max_Prob'].apply(
            self._classify_risk
        )

        return direction_prediction, data_to_predict

    @staticmethod
    def _classify_risk(max_prob: float) -> str:
        """
        Classifies the confidence level based on maximum probability.

        Args:
            max_prob (float): The maximum probability from model predictions.

        Returns:
            str: The confidence level ('High Confidence', 'Medium Confidence', 'Low Confidence').
        """
        if max_prob > 0.75:
            return 'High Confidence'
        elif 0.45 <= max_prob <= 0.75:
            return 'Medium Confidence'
        else:
            return 'Low Confidence'

    def _add_regression_target_column(
        self,
        full_data: pd.DataFrame,
        num_weeks: list = [1]
    ) -> pd.DataFrame:
        """
        Adds target columns for regression by shifting the 'Num_Encounters' column.

        Args:
            full_data (pd.DataFrame): The complete dataset.
            num_weeks (list, optional): List of weeks to shift for target columns. Defaults to [1].

        Returns:
            pd.DataFrame: The DataFrame with added target columns.

        Raises:
            ValueError: If no valid data remains after adding target columns.
        """
        # Create target columns by shifting 'Num_Encounters' for specified weeks
        for week in num_weeks:
            full_data[f'Target_Week_{week}'] = full_data.groupby(
                ['STATE', 'Valid_ICD_10']
            )['Num_Encounters'].shift(-week)

        # Drop rows with any NaN values resulting from the shift
        full_data = full_data.dropna()

        # Raise an error if no data is left after dropping NaNs
        if full_data.empty:
            raise ValueError(
                "No valid data for regression after adding target column."
            )

        return full_data

    def _get_regression_prediction(
        self,
        model: RandomForestRegressor,
        data_to_predict: pd.DataFrame
    ) -> np.ndarray:
        """
        Predicts patient counts using the trained regression model.

        Args:
            model (RandomForestRegressor): The trained regression model.
            data_to_predict (pd.DataFrame): Data to make predictions on.

        Returns:
            np.ndarray: Predicted patient counts.
        """
        # Generate dummy columns for 'Confidence_Level'
        dummy_columns: pd.DataFrame = pd.get_dummies(
            data_to_predict['Confidence_Level'],
            prefix='Confidence_Level'
        )
        data_to_predict = pd.concat([data_to_predict, dummy_columns], axis=1)

        # Align dummy columns to match those used during training
        data_to_predict = self._align_dummy_columns(
            data_to_predict, self.dummy_columns
        )

        # Drop unnecessary columns
        X_test: pd.DataFrame = data_to_predict.drop(
            columns=self.columns_to_drop, errors='ignore'
        )
        X_test = X_test.drop(columns=['Confidence_Level'], errors='ignore')

        # Reorder columns in X_test to match the order in the training data
        X_test = X_test[[col for col in model.feature_names_in_]]

        # Predict patient counts
        return model.predict(X_test)

    @staticmethod
    def _align_dummy_columns(
        df: pd.DataFrame,
        reference_columns: list
    ) -> pd.DataFrame:
        """
        Aligns the dummy columns in the DataFrame to match the reference columns.

        Args:
            df (pd.DataFrame): The DataFrame to align.
            reference_columns (list): The list of reference dummy column names.

        Returns:
            pd.DataFrame: The aligned DataFrame with all reference dummy columns.
        """
        for col in reference_columns:
            if col not in df.columns:
                df[col] = 0  # Add missing dummy columns with default value 0
        return df

    @staticmethod
    def _load_model(
        model_path: str,
        is_regressor: bool = False
    ) -> Tuple[Callable, Optional[list]]:
        """
        Loads a pre-trained model from a file. If the model is a regressor, also loads the dummy columns.

        Args:
            model_path (str): Path to the model file.
            is_regressor (bool, optional): Flag indicating if the model is a regressor. Defaults to False.

        Returns:
            Tuple[Callable, Optional[list]]: The loaded model and dummy columns (if regressor).
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        print(f"Loading model from {model_path}")

        loaded_data = joblib.load(model_path)

        if is_regressor:
            # Expecting a dictionary with 'model' and 'dummy_columns'
            if not isinstance(loaded_data, dict):
                raise ValueError(
                    f"Regressor model file {model_path} does not contain a dictionary with 'model' and 'dummy_columns'."
                )
            return loaded_data['model'], loaded_data['dummy_columns']
        else:
            # For classifiers, the loaded_data is the model itself
            return loaded_data, None

    @staticmethod
    def _read_filtering_dictionary(filter_data: dict) -> Tuple[int, int, str, str]:
        """
        Extracts values for filtering from a dictionary.

        Args:
            filter_data (dict): Dictionary containing filtering parameters.

        Returns:
            Tuple[int, int, str, str]: Extracted week, year, state, and disease.
        """
        week: int = filter_data["Week"]
        year: int = filter_data["Year"]
        state: str = filter_data["STATE"]
        disease: str = filter_data["Valid_ICD_10"]
        return week, year, state, disease

    def train_or_load_models(
        self,
        current_year: int,
        specific_week: int,
        direction_model_path: str = 'direction_model.pkl',
        regression_model_path: str = 'regression_model.pkl'
    ) -> Tuple[RandomForestClassifier, RandomForestRegressor]:
        """
        Trains or loads the classifier and regressor models based on the train_models flag.

        Args:
            current_year (int): The current year for filtering data.
            specific_week (int): The specific week for filtering data.
            direction_model_path (str, optional): Path to save or load the direction model. Defaults to 'direction_model.pkl'.
            regression_model_path (str, optional): Path to save or load the regression model. Defaults to 'regression_model.pkl'.

        Returns:
            Tuple[RandomForestClassifier, RandomForestRegressor]: The direction classifier and regression model.
        """
        if self.train_models:
            print("Training models...")
            # Train the classifier
            direction_model, filtered_data_for_regression_training = self._train_classifier(
                current_year=current_year,
                specific_week=specific_week,
                save_path=direction_model_path
            )
            # Train the regressor
            regression_model: RandomForestRegressor = self._train_regressor(
                filtered_data_for_regression_training,
                save_path=regression_model_path
            )
        else:
            print("Loading pre-trained models...")
            # Load the classifier
            direction_model, _ = self._load_model(
                direction_model_path, is_regressor=False
            )
            # Load the regressor and its dummy columns
            regression_model, loaded_dummy_columns = self._load_model(
                regression_model_path, is_regressor=True
            )
            self.dummy_columns = loaded_dummy_columns  # Set the dummy_columns for alignment

        return direction_model, regression_model

    @staticmethod
    def run_predictions_for_combinations(
        predictor: 'TwoStagePredictor',
        current_year: int,
        specific_week: int,
        direction_model: RandomForestClassifier,
        regression_model: RandomForestRegressor
    ) -> pd.DataFrame:
        """
        Creates combinations of all ICD-10 codes and states present in the data, filters the data for the specified
        week and year, and runs classification and regression predictions for each combination using pre-trained models.

        Args:
            predictor (TwoStagePredictor): An instance of the TwoStagePredictor class.
            current_year (int): The current year to filter data.
            specific_week (int): The specific week to filter data.
            direction_model (RandomForestClassifier): Pre-trained classification model.
            regression_model (RandomForestRegressor): Pre-trained regression model.

        Returns:
            pd.DataFrame: DataFrame containing predictions for each combination with columns:
                        ['State', 'ICD-10', 'Direction Prediction', 'Predicted Probability',
                        'Confidence Level', 'Confidence Score', 'Expected Num of Patients'].
        """
        # Define the cutoff for prediction data based on current year and week
        valid_time: int = current_year * 100 + specific_week

        # Filter the dataset to include only records before the valid_time
        filtered_data: pd.DataFrame = predictor.df[predictor.df['year_week_int'] < valid_time]

        # Extract unique states and ICD-10 codes from the filtered dataset
        states: np.ndarray = filtered_data['STATE'].unique()
        icd_10_codes: np.ndarray = filtered_data['Valid_ICD_10'].unique()

        # Create all possible combinations of states and ICD-10 codes
        combinations: list = list(product(states, icd_10_codes))

        # Initialize a list to store prediction results
        results: list = []

        for state, icd_10_code in combinations:
            try:
                print(f"Processing State: {state}, ICD-10: {icd_10_code}")

                # Get direction prediction and the corresponding data
                direction_prediction, regression_data = predictor._get_direction_prediction(
                    model=direction_model,
                    current_year=current_year,
                    specific_week=specific_week,
                    disease=icd_10_code,
                    state=state
                )

                # Check if predictions were made
                if direction_prediction is None or regression_data is None:
                    print(
                        f"No prediction data for State: {state}, ICD-10: {icd_10_code}."
                    )
                    continue

                # Extract predicted probabilities and confidence levels
                predicted_probability: float = regression_data['Max_Prob'].iloc[0]
                confidence_level: str = regression_data['Confidence_Level'].iloc[0]

                # Map confidence levels to numerical scores
                confidence_score: float
                if confidence_level == 'Low Confidence':
                    confidence_score = 0.3
                elif confidence_level == 'Medium Confidence':
                    confidence_score = 0.6
                else:
                    confidence_score = 1.0

                # Run regression prediction to estimate expected number of patients
                expected_num_of_patients: np.ndarray = predictor._get_regression_prediction(
                    model=regression_model,
                    data_to_predict=regression_data
                )

                # Append the prediction results to the list
                results.append({
                    'State': state,
                    'ICD-10': icd_10_code,
                    'Direction Prediction': direction_prediction[0],
                    'Predicted Probability': predicted_probability,
                    'Confidence Level': confidence_level,
                    'Confidence Score': confidence_score,
                    'Expected Num of Patients': expected_num_of_patients[0]
                })

            except Exception as e:
                # Handle any exceptions during prediction
                print(
                    f"Error processing State: {state}, ICD-10: {icd_10_code}. "
                    f"Error: {e}"
                )

        # Convert the list of results to a pandas DataFrame
        return pd.DataFrame(results)


if __name__ == '__main__':
    # Initialize the TwoStagePredictor with the dataset and training flag
    predictor: TwoStagePredictor = TwoStagePredictor(
        csv_path='combined_dataset_directional_classification.csv',
        train_models=True
    )

    # Define paths for saving/loading models
    direction_model_path: str = 'direction_model.pkl'
    regression_model_path: str = 'regression_model.pkl'

    # Train or load the models based on the train_models flag
    direction_model, regression_model = predictor.train_or_load_models(
        current_year=2023,
        specific_week=40,
        direction_model_path=direction_model_path,
        regression_model_path=regression_model_path
    )

    # Run predictions for all state and ICD-10 code combinations
    all_predictions: pd.DataFrame = TwoStagePredictor.run_predictions_for_combinations(
        predictor=predictor,
        current_year=2023,
        specific_week=41,
        direction_model=direction_model,
        regression_model=regression_model
    )

    # Save the prediction results to a CSV file
    all_predictions.to_csv('all_predictions.csv', index=False)
    print("All predictions saved to 'all_predictions.csv'.")
