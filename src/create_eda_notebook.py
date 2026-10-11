"""
Script to generate notebooks/02_eda.ipynb.
"""
import json

notebook = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# RoadSense AI: Multi-Task Machine Learning for Accident Severity and Casualties\n',
                '## Notebook 02: Exploratory Data Analysis & Statistical Risk Insights\n',
                '**Course Module**: Artificial Intelligence (CO5: Classification, Regression and Prediction Problems)\n',
                '**Target Team Member**: Member 2 (Exploratory Data Analysis & Visualizations Lead)\n',
                '\n',
                '---\n',
                '\n',
                '### Objectives:\n',
                '1. Explore class distributions for Task A (Binary) and Task B (3-class) to quantify severe class imbalance.\n',
                '2. Analyze bivariate relationship between posted speed limit and danger rate (Fatal + Serious injury probability).\n',
                '3. Examine environmental risk factors: weather conditions, road geometry, and ambient lighting.\n',
                '4. Analyze physiological vulnerability of road users (pedal cycles, motorcycles vs passenger cars).\n',
                '5. Identify temporal risk spikes across 24-hour diurnal cycle.\n',
                '6. Inspect casualty count distribution and heavy-tailed extreme outliers (70 casualties bus collision).\n',
                '7. Save publication-quality figures to `../outputs/charts/`.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'import os\n',
                'import sys\n',
                'import pandas as pd\n',
                'import numpy as np\n',
                'import matplotlib.pyplot as plt\n',
                'import seaborn as sns\n',
                '\n',
                'sys.path.append(os.path.abspath(".."))\n',
                'from src.eda import (\n',
                '    plot_class_distributions,\n',
                '    plot_danger_by_speed_limit,\n',
                '    plot_danger_by_weather,\n',
                '    plot_danger_by_road_type,\n',
                '    plot_danger_by_vehicle_type,\n',
                '    plot_danger_by_hour,\n',
                '    plot_danger_by_light_condition,\n',
                '    plot_casualties_distribution\n',
                ')\n',
                'print("Imports loaded successfully.")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 1. Load Clean Master Dataset\n',
                'Loading the multi-year dataset (2021-2023) created in Stage 1.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'df = pd.read_parquet("../data/clean_roadsense.parquet")\n',
                'print(f"Loaded master clean dataset: {len(df):,} records across {len(df.columns)} columns.")\n',
                'df.head(3)'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 2. Target Class Distributions & Severe Imbalance\n',
                '- **Task A**: Minor (~76.7%) vs Dangerous (~23.3%)\n',
                '- **Task B**: Slight (~76.7%) vs Serious (~21.9%) vs Fatal (~1.48%)\n',
                'The severe scarcity of Fatal accidents demands specialized reweighting, SMOTE, and PR-AUC tracking in subsequent stages.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'plot_class_distributions(df, save_path="../outputs/charts/01_class_distribution.png")\n',
                'print("Task A Counts:\\n", df["target_severity_binary"].value_counts(normalize=True)*100)\n',
                'print("Task B Counts:\\n", df["target_severity_3class"].value_counts(normalize=True)*100)'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 3. Danger Rate vs. Posted Speed Limit\n',
                'Because kinetic energy scales quadratically ($E_k = \\frac{1}{2}mv^2$), collisions on high-speed roads (60-70 mph) yield significantly higher danger rates (>35%) and fatal rates compared to urban 20-30 mph zones (<20%).'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'plot_danger_by_speed_limit(df, save_path="../outputs/charts/02_danger_rate_by_speed_limit.png")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 4. Environmental Risk: Weather & Road Surface\n',
                'Analyzing risk rates across adverse weather conditions (Fog, High Winds, Heavy Rain).'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'plot_danger_by_weather(df, save_path="../outputs/charts/03_danger_rate_by_weather.png")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 5. Road Geometry: Junctions & Carriageway Types\n',
                'Roundabouts show the lowest danger rate despite frequent minor collisions, due to forced low-speed angle mergers.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'plot_danger_by_road_type(df, save_path="../outputs/charts/04_danger_rate_by_road_type.png")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 6. Vehicle Vulnerability Hierarchy\n',
                'Vulnerable road users (cyclists and motorcyclists) lack protective metal cages, crumple zones, and airbags. Motorcyclists face an extreme danger rate exceeding 40%.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'plot_danger_by_vehicle_type(df, save_path="../outputs/charts/05_danger_rate_by_vehicle_type.png")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 7. Diurnal Rhythm: Volume vs Lethality\n',
                'Peak accident volume occurs during evening rush hour (16:00 - 18:00), but **peak danger rate occurs in early morning hours (01:00 - 04:00)** due to higher speeds, reduced visibility, and driver fatigue.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'plot_danger_by_hour(df, save_path="../outputs/charts/06_danger_rate_by_hour.png")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 8. Environmental Lighting\n',
                'Darkness without street lighting presents the highest risk of serious and fatal injuries.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'plot_danger_by_light_condition(df, save_path="../outputs/charts/07_danger_rate_by_light_condition.png")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 9. Casualty Count Distribution & Extreme Outliers\n',
                '- Over 80.6% of collisions result in exactly 1 casualty.\n',
                '- The 2023 dataset includes an extreme 70-casualty bus crash.\n',
                '- As instructed, this outlier is preserved in the test set to evaluate Poisson loss and robust metrics (Median AE alongside MAE and RMSE).'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'plot_casualties_distribution(df, save_path="../outputs/charts/08_casualties_distribution_and_outlier.png")\n',
                'print("Casualties summary statistics:")\n',
                'print(df["target_casualties"].describe(percentiles=[0.5, 0.75, 0.9, 0.99]))'
            ]
        }
    ],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3 (ipykernel)',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {'name': 'ipython', 'version': 3},
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.11.0'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

with open('notebooks/02_eda.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)
print('Saved notebooks/02_eda.ipynb successfully.')
