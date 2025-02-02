import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

class DataFiltererAndPlotter:
    def __init__(self, inputdf_directory: str) -> None:
        """
        Initializes the DataFiltererAndPlotter by loading the dataset.

        Args:
            inputdf_directory (str): Path to the CSV file containing anomaly scores.
        """
        # Load the dataset from the specified CSV file
        self._input_df: pd.DataFrame = pd.read_csv(inputdf_directory, index_col=0)
        

    def plot_recent_weeks(
        self,
        disease: str,
        state: str,
        current_year: int,
        current_week: int,
        num_weeks: int = 3,
        pct_change_threshold: float = 10.0
    ) -> Optional[plt.Figure]:
        """
        Plots the number of encounters for the specified number of weeks prior to the given
        year-week combination for a specific disease and state. Adds a red dotted line indicating
        the percentage change threshold.

        Args:
            disease (str): Disease code to filter (Valid_ICD_10 column).
            state (str): State to filter (STATE column).
            current_year (int): Current year.
            current_week (int): Current week.
            num_weeks (int, optional): Number of weeks to include in the plot prior to the given
                year-week. Defaults to 3.
            pct_change_threshold (float, optional): Percentage change threshold for the red line.
                Defaults to 10.0.

        Returns:
            Optional[plt.Figure]: The matplotlib figure object if plotting is successful; otherwise, None.
        """
        # Filter the data for the specified disease and state
        
        df_filtered = self._input_df[
            (self._input_df['Valid_ICD_10'] == disease) &
            (self._input_df['STATE'] == state)
        ]
      


        # Add a helper column for sorting across years and weeks
        df_filtered['year_week'] = df_filtered['Year'] * 100 + df_filtered['Week']
      
        # Calculate the current year-week integer
        current_year_week: int = current_year * 100 + current_week

        # Select rows up to the specified year-week
        df_filtered = df_filtered[df_filtered['year_week'] <= current_year_week]

        # Check for sufficient data
        if len(df_filtered) < num_weeks + 1:
            print(
                f"Insufficient data to plot {num_weeks} weeks for disease {disease}, "
                f"state {state}, and week {current_week}."
            )
            return None

        # Select the last `num_weeks` and the current week
        df_recent: pd.DataFrame = df_filtered.tail(num_weeks + 1)

        # Ensure the recent data is not empty
        if df_recent.empty:
            print("No recent data available for plotting.")
            return None

        # Prepare x-axis labels to handle week wrapping
        df_recent['x_label'] = (
            df_recent['Year'].astype(str) + "-W" + df_recent['Week'].astype(str)
        )

        # Calculate the percentage change
        current_value: float = df_recent.iloc[-1]['Num_Encounters']
        # previous_value: float = df_recent.iloc[-2]['Num_Encounters']
        previous_value: float = df_recent.iloc[:-1]['Num_Encounters'].mean()
        
        pct_change: Optional[float] = None
        if previous_value != 0:
            pct_change = ((current_value - previous_value) / previous_value) * 100

        # Plot the number of encounters
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            df_recent['x_label'],
            df_recent['Num_Encounters'],
            marker='o',
            label='Num Encounters'
        )

        # Add a red dotted line for the percentage change threshold
        if previous_value != 0:
            threshold_line: float = previous_value * (1 + pct_change_threshold / 100)
            ax.axhline(
                y=threshold_line,
                color='red',
                linestyle='--',
                label=f'{pct_change_threshold}% of Avergae Threshold'
            )

        # Annotate the percentage change
        if pct_change is not None:
            ax.text(
                len(df_recent['x_label']) - 1,
                current_value,
                f'{pct_change:.2f}%',
                color='black',
                fontsize=10,
                verticalalignment='bottom'
            )
        else:
            ax.text(
                len(df_recent['x_label']) - 1,
                current_value,
                'Undefined % Change',
                color='black',
                fontsize=10,
                verticalalignment='bottom'
            )

        # Adjust the x-axis for better readability
        ax.set_xticks(range(len(df_recent['x_label'])))
        ax.set_xticklabels(df_recent['x_label'], rotation=45, ha='right')

        # Plot title and labels
        ax.set_title(
            f"Recent {num_weeks} Weeks Num Encounters for {disease} in {state}"
        )
        ax.set_xlabel("Week")
        ax.set_ylabel("Num Encounters")
        ax.grid()
        ax.legend()

        # Use tight layout for proper spacing
        fig.tight_layout()

        return fig

    def get_past_years_data(
        self,
        current_year: int,
        specific_week: int,
        state: str,
        disease: str
    ) -> pd.DataFrame:
        """
        Retrieves data for a specific week across all past years up to the current year
        for a given disease and state.

        Args:
            current_year (int): The current year.
            specific_week (int): The specific week to filter.
            state (str): The state to filter.
            disease (str): The disease code to filter.

        Returns:
            pd.DataFrame: Filtered DataFrame containing relevant columns.
        """
        df_filtered: pd.DataFrame = self._input_df[
            (self._input_df['Valid_ICD_10'] == disease) &
            (self._input_df['STATE'] == state) &
            (self._input_df['Week'] == specific_week) &
            (self._input_df['Year'] <= current_year)
        ]

        return df_filtered[
            ['Year', 'Week', 'STATE', 'Valid_ICD_10', 'Num_Encounters', 'Anomaly_Score']
        ]

    def get_anomaly_score(
        self,
        current_year: int,
        specific_week: int,
        state: str,
        disease: str
    ) -> Optional[float]:
        """
        Retrieves the Anomaly Score for a specific combination of year, week, state, and disease.

        Args:
            current_year (int): The year to filter.
            specific_week (int): The week to filter.
            state (str): The state to filter.
            disease (str): The disease code to filter.

        Returns:
            Optional[float]: The anomaly score if data exists; otherwise, None.
        """
        # Filter the DataFrame
        df_filtered: pd.DataFrame = self._input_df[
            (self._input_df['Valid_ICD_10'] == disease) &
            (self._input_df['STATE'] == state) &
            (self._input_df['Week'] == specific_week) &
            (self._input_df['Year'] == current_year)
        ]

        # Check if any rows match the criteria
        if df_filtered.empty:
            print(
                f"No data found for Year: {current_year}, Week: {specific_week}, "
                f"State: {state}, Disease: {disease}."
            )
            return None

        # If a single row matches, return the scalar value; otherwise, return all matching scores
        if len(df_filtered) == 1:
            return df_filtered['Anomaly_Score'].iloc[0]
        else:
            return df_filtered['Anomaly_Score'].mean()




    def plot_past_years_weekly_encounters(
        self,
        disease: str,
        state: str,
        specific_week: int,
        current_year: int,
        pct_change_threshold: float = 10.0
    ) -> Optional[plt.Figure]:
        """
        Plots the number of encounters for a specific week across all past years up to the
        current year with proper year handling and percentage changes.
        """
        # Filter and prepare data
        df_filtered = self._input_df[
            (self._input_df['Valid_ICD_10'] == disease) &
            (self._input_df['STATE'] == state) &
            (self._input_df['Week'] == specific_week) &
            (self._input_df['Year'] <= current_year)
        ].copy()

        if df_filtered.empty:
            print(f"No data found for disease {disease}, state {state}, and week {specific_week}.")
            return None

        # Create complete year range
        min_year = int(self._input_df['Year'].min())
        print(min_year)
        current_year = int(current_year)
        all_years = pd.DataFrame({'Year': range(min_year, current_year + 1)})
        print(all_years)
        # Merge with actual data
        df_complete = all_years.merge(
            df_filtered[['Year', 'Num_Encounters']],
            on='Year',
            how='left'
        ).fillna({'Num_Encounters': 0})

        # Calculate percentage changes relative to previous year
        df_complete['Prev_Year_Encounters'] = df_complete['Num_Encounters'].shift(1)
        df_complete['Pct_Change'] = ((df_complete['Num_Encounters'] - df_complete['Prev_Year_Encounters']) /
                                    df_complete['Prev_Year_Encounters'].replace(0, np.nan)) * 100

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 7))
        years = df_complete['Year'].astype(int)
        encounters = df_complete['Num_Encounters']

        # Plot main data line
        ax.plot(years, encounters, marker='o', linestyle='-', 
                markersize=8, linewidth=2, label='Weekly Encounters')

        # Add threshold line (based on previous year's value)
        if current_year > min_year:
            prev_year = current_year - 1
            prev_value = df_complete[df_complete['Year'] == prev_year]['Num_Encounters'].values
            if len(prev_value) > 0 and prev_value[0] > 0:
                threshold = prev_value[0] * (1 + pct_change_threshold/100)
                ax.axhline(threshold, color='r', linestyle='--', 
                          label=f'{pct_change_threshold}% Threshold ({prev_year} Baseline)')

        # Add annotations for percentage changes
        for idx, row in df_complete.iterrows():
            if idx == 0 or pd.isna(row['Pct_Change']):
                continue
            ax.annotate(f"{row['Pct_Change']:.1f}%", 
                       (row['Year'], row['Num_Encounters']),
                       textcoords="offset points",
                       xytext=(0,10),
                       ha='center',
                       fontsize=9,
                       color='darkgreen')

        # Formatting
        ax.set_title(f"Weekly Encounters for Week {specific_week}\n{disease} in {state}", pad=20)
        ax.set_xlabel("Year", labelpad=10)
        ax.set_ylabel("Number of Encounters", labelpad=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax.set_xticks(years)
        ax.set_xticklabels(years, rotation=45, ha='right')
        ax.set_xlim(min_year - 0.5, current_year + 0.5)
        
        plt.tight_layout()
        return fig



    @staticmethod
    def _read_filtering_dictionary(filter_data: Dict[str, Any]) -> Tuple[int, int, str, str]:
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

    def save_figure(self, fig: plt.Figure, filename: str) -> None:
        """
        Saves a matplotlib figure to the specified filename.

        Args:
            fig (plt.Figure): The figure to save.
            filename (str): The name of the file (without extension).
        """
        # Ensure the directory exists
        directory: str = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Save the figure with high resolution
        fig.savefig(f"{filename}.png", dpi=300)
        plt.close(fig)  # Close the figure to free memory
        print(f"Figure saved as '{filename}.png'.")


if __name__ == '__main__':
    # Initialize the DataFiltererAndPlotter with the anomaly scores CSV
    plotter: DataFiltererAndPlotter = DataFiltererAndPlotter(
        inputdf_directory='WebApp\\FlaskApp\\static\\data\\anomaly_scores_filt1.csv'
    )

    # Define filter parameters
    filter_params: Dict[str, Any] = {
        "STATE": "TX",
        "Valid_ICD_10": "K29",
        "Week": 5,
        "Year": 2021
    }

    # Extract filter values
    week, year, state, disease = plotter._read_filtering_dictionary(filter_data=filter_params)


    # Plot recent weeks
    graph_recent_weeks: Optional[plt.Figure] = plotter.plot_recent_weeks(
        disease=disease,
        state=state,
        current_year=year,
        current_week=week,
        num_weeks=6,
        pct_change_threshold=20.0
    )
    if graph_recent_weeks:
        plotter.save_figure(
            graph_recent_weeks,
            f"recent_weeks_{disease}_{state}_{year}_{week}"
        )

    # Plot past years' weekly encounters
    graph_past_years: Optional[plt.Figure] = plotter.plot_past_years_weekly_encounters(
        disease=disease,
        state=state,
        specific_week=week,
        current_year=year
    )
    if graph_past_years:
        plotter.save_figure(
            graph_past_years,
            f"all_years_{disease}_{state}_{week}"
        )

    # Get filtered data for past years
    filtered_df: pd.DataFrame = plotter.get_past_years_data(
        disease=disease,
        state=state,
        current_year=year,
        specific_week=week
    )


    # Get anomaly score
    anomaly_score: Optional[float] = plotter.get_anomaly_score(
        disease=disease,
        state=state,
        current_year=year,
        specific_week=week
    )
    print(f"Anomaly Score: {anomaly_score}")
