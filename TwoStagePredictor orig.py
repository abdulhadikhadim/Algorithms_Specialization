from itertools import product
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

class TwoStagePredictor:
    def __init__(self, csv_path, date_col='year_week_int'):
        """
        Initializes the TwoStagePipeline by loading the dataset and storing parameters.

        Args:
            csv_path (str): Path to the CSV file containing data.
            date_col (str): Column representing the year-week integer.

        """
        self.df = pd.read_csv(csv_path, index_col=0)
        self.date_col = date_col
        self.columns_to_drop = ['Year', 'Week', 'year_week_int', 'Valid_ICD_10', 'STATE']
        self.dummy_columns = None

    def train_classifier(self, current_year, specific_week, target_column = 'Target'):
        columns_to_drop = ['Year','Week','year_week_int','Valid_ICD_10','STATE']
        # Random Forest
        random_forest_model = RandomForestClassifier(n_estimators=100,
                                                    class_weight={0:1,1:1,2:2},
                                                    max_depth=5, random_state=42,max_features='sqrt',
                                                    bootstrap=True,
                                                    min_samples_leaf= 5,
                                                    min_samples_split=3)
        valid_time = current_year * 100 + specific_week
        df = self.df.copy()
        filtered_df = df[df['year_week_int'] < valid_time]

        filtered_df_without_columns = filtered_df.drop(columns=columns_to_drop)

        X_train = filtered_df_without_columns.drop(columns = target_column)
        y_train = filtered_df_without_columns[target_column]

        random_forest_model.fit(X_train, y_train)

        probs_train = random_forest_model.predict_proba(X_train)
        filtered_df['Max_Prob'] = probs_train.max(axis=1)
        filtered_df['Predicted_Class'] = probs_train.argmax(axis=1)
        filtered_df['Confidence_Level'] = filtered_df['Max_Prob'].apply(self.classify_risk)


        return random_forest_model , filtered_df


    def get_direction_prediction(self, model, current_year, specific_week,disease, state, target_column = 'Target'):

        data_to_predict = self.df[
            (self.df['Valid_ICD_10'] == disease) &
            (self.df['STATE'] == state) &
            (self.df['Week'] == specific_week) &
            (self.df['Year'] == current_year)
        ].copy()

        X_test = data_to_predict.drop(columns=self.columns_to_drop + [target_column])
        direction_prediction = model.predict(X_test)
        probs_test = model.predict_proba(X_test)
        data_to_predict['Max_Prob'] = probs_test.max(axis=1)
        data_to_predict['Predicted_Class'] = probs_test.argmax(axis=1)
        data_to_predict['Confidence_Level'] = data_to_predict['Max_Prob'].apply(self.classify_risk)

        return direction_prediction, data_to_predict

    @staticmethod
    def classify_risk(max_prob):
        if max_prob > 0.75:
            return 'High Confidence'
        elif 0.45 <= max_prob <= 0.75:
            return 'Medium Confidence'
        else:
            return 'Low Confidence'

    def add_regression_target_column(self, full_data, num_weeks=[1]):
        for week in num_weeks:
            full_data[f'Target_Week_{week}'] = full_data.groupby(['STATE', 'Valid_ICD_10'])['Num_Encounters'].shift(-week)
        full_data = full_data.dropna()
        if full_data.empty:
            raise ValueError("No valid data for regression after adding target column.")
        return full_data

    def train_regressor(self, df, target_column='Target_Week_1'):
        regressor_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_features='sqrt',
            min_samples_split=5,
            min_samples_leaf=3
        )
        filtered_df = df.copy()
        filtered_df = self.add_regression_target_column(filtered_df)
        dummy_columns = pd.get_dummies(filtered_df['Confidence_Level'], prefix='Confidence_Level')
        self.dummy_columns = dummy_columns.columns.tolist()  # Save column names for alignment

        filtered_df = pd.concat([filtered_df, dummy_columns], axis=1).drop(columns="Confidence_Level")
        X_train = filtered_df.drop(columns=self.columns_to_drop + [target_column])
        y_train = filtered_df[target_column]

        regressor_model.fit(X_train, y_train)

        return regressor_model

    def get_regression_prediction(self, model, data_to_predict: pd.DataFrame):
        """
        Predicts patient counts using the trained regression model.

        Args:
            model (RandomForestRegressor): The trained regression model.
            data_to_predict (pd.DataFrame): Data to make predictions on.

        Returns:
            np.ndarray: Predicted patient counts.
        """
        # Generate dummy columns for Confidence_Level
        dummy_columns = pd.get_dummies(data_to_predict['Confidence_Level'], prefix='Confidence_Level')
        data_to_predict = pd.concat([data_to_predict, dummy_columns], axis=1)

        # Align dummy columns to match those used during training
        data_to_predict = self._align_dummy_columns(data_to_predict, self.dummy_columns)

        # Drop unnecessary columns
        X_test = data_to_predict.drop(columns=self.columns_to_drop, errors='ignore')
        X_test = X_test.drop(columns=['Confidence_Level'], errors='ignore')

        # Reorder columns in X_test to match the order in the training data
        X_test = X_test[[col for col in model.feature_names_in_]]

        # Predict patient counts
        return model.predict(X_test)

    @staticmethod
    def _align_dummy_columns(df, reference_columns):
        for col in reference_columns:
            if col not in df.columns:
                df[col] = 0
        return df

    @staticmethod
    def read_filtering_dictionary(filter_data):
        """
        Extracts values for filtering from a dictionary.

        Args:
            filter_data (dict): Dictionary containing filtering parameters.

        Returns:
            tuple: Extracted week, year, state, and disease.
        """

        week = filter_data["Week"]
        year = filter_data["Year"]
        state = filter_data["STATE"]
        disease = filter_data["Valid_ICD_10"]
        return week, year, state, disease

    @staticmethod
    def run_predictions_for_combinations(predictor, current_year, specific_week, direction_model, regression_model):
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
                        'Confidence Level', 'Expected Num of Patients'].
        """
        # Filter the data by the specified year and week
        valid_time = current_year * 100 + specific_week
        filtered_data = predictor.df[predictor.df['year_week_int'] < valid_time]

        # Get unique states and ICD-10 codes
        states = filtered_data['STATE'].unique()
        icd_10_codes = filtered_data['Valid_ICD_10'].unique()

        # Create combinations of states and ICD-10 codes
        combinations = list(product(states, icd_10_codes))

        results = []

        for state, icd_10_code in combinations:
            try:
                print(f"Processing State: {state}, ICD-10: {icd_10_code}")

                # Filter data for the current combination
                data_to_predict = filtered_data[
                    (filtered_data['Valid_ICD_10'] == icd_10_code) &
                    (filtered_data['STATE'] == state)
                ]

                if data_to_predict.empty:
                    print(f"No data for State: {state}, ICD-10: {icd_10_code}")
                    continue

                # Run classification prediction
                direction_prediction, regression_data = predictor.get_direction_prediction(
                    model=direction_model,
                    current_year=current_year,
                    specific_week=specific_week,
                    disease=icd_10_code,
                    state=state
                )

                # Extract predicted probabilities and confidence levels
                predicted_probability = regression_data['Max_Prob'].iloc[0] if not regression_data.empty else None
                confidence_level = regression_data['Confidence_Level'].iloc[0] if not regression_data.empty else None
                if confidence_level == 'Low Confidence':
                    confidence_score = 0.3
                elif confidence_level == 'Medium Confidence':
                    confidence_score = 0.6
                else:
                    confidence_score = 1.0
                # Run regression prediction
                expected_num_of_patients = predictor.get_regression_prediction(
                    model=regression_model,
                    data_to_predict=regression_data
                )

                # Append results
                results.append({
                    'State': state,
                    'ICD-10': icd_10_code,
                    'Direction Prediction': direction_prediction[0] if len(direction_prediction) > 0 else None,
                    'Predicted Probability': predicted_probability,
                    'Confidence Level': confidence_level,
                    'Confidence Score' : confidence_score,
                    'Expected Num of Patients': expected_num_of_patients[0] if len(expected_num_of_patients) > 0 else None
                })

            except Exception as e:
                print(f"Error processing State: {state}, ICD-10: {icd_10_code}. Error: {e}")

        return pd.DataFrame(results)



if __name__ == '__main__':
    predictor = TwoStagePredictor(csv_path='combined_dataset_directional_classification.csv')
    current_year = 2023
    specific_week = 40
    # Train the models once
    direction_model, filtered_data_for_regression_training = predictor.train_classifier(current_year=current_year, specific_week=specific_week)
    regression_model = predictor.train_regressor(filtered_data_for_regression_training)

    # Run predictions for all combinations
    all_predictions = predictor.run_predictions_for_combinations(
        predictor=predictor,
        current_year=current_year,
        specific_week=specific_week,
        direction_model=direction_model,
        regression_model=regression_model
    )
    all_predictions.to_csv('all_predicitons.csv')

    