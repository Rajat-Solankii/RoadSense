"""
Utility script to generate notebooks/01_data_cleaning.ipynb with clean cells.
"""
import json

notebook = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# RoadSense AI: Multi-Task Machine Learning for Accident Severity and Casualties\n',
                '## Notebook 01: Data Ingestion, Verification, Cleaning & Feature Engineering\n',
                '**Course Module**: Artificial Intelligence (CO5: Classification, Regression and Prediction Problems)\n',
                '**Target Team Member**: Member 1 (Data Engineering & Preprocessing Lead)\n',
                '\n',
                '---\n',
                '\n',
                '### Objectives:\n',
                '1. Ingest official UK STATS19 road safety data (Collisions and Vehicles for 2021, 2022, 2023).\n',
                '2. Verify schema consistency and column alignment across all years.\n',
                '3. Enforce **strict anti-leakage rules** (exclude crash dynamics, post-hoc injury adjustments, police response).\n',
                '4. Apply **per-column unknown-code mapping** based on the official DfT Data Guide.\n',
                '5. Engineer collision-level vehicle interaction features without arbitrary single-vehicle bias.\n',
                '6. Define ground truth targets for Tasks A, B, and C.\n',
                '7. Export the clean master dataset and data-quality audit report.'
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
                '\n',
                '# Ensure project root is on sys.path\n',
                'sys.path.append(os.path.abspath(".."))\n',
                'from src.data import (\n',
                '    UNKNOWN_CODE_MAP, \n',
                '    EXCLUDED_COLUMNS_RATIONALE, \n',
                '    verify_schema_consistency,\n',
                '    build_master_dataset\n',
                ')\n',
                'print("Imports and configuration loaded successfully.")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 1. Schema Consistency Verification\n',
                'Before concatenating datasets across multiple years, we verify that all 44 collision columns and 32 vehicle columns exist with identical naming and data types across 2021, 2022, and 2023.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'schema_check = verify_schema_consistency(data_dir="../data", years=[2021, 2022, 2023])\n',
                'print("All years schema-consistent:", schema_check["consistent"])\n',
                'for yr, info in schema_check["collision_files"].items():\n',
                '    print(f"Year {yr}: Collision cols = {info[\'col_count\']}, Vehicle cols = {schema_check[\'vehicle_files\'][yr][\'col_count\']}")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 2. Per-Column Unknown-Code Mapping\n',
                'DfT STATS19 does not use a blanket missing value code. While `-1` generally indicates unrecorded data, several features use specialized codes (e.g. `99` for self-reported unknown speed limit or vehicle type, `9` for unknown weather/road surface, `7` for unknown lighting, `3` for unallocated urban/rural). We map each feature individually according to the DfT Data Guide.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'for col, codes in UNKNOWN_CODE_MAP.items():\n',
                '    print(f"Feature \'{col:<25}\' -> Unknown codes: {codes}")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 3. Strict Data Leakage Prevention Log\n',
                'To prevent target leakage and unrealistic pre-crash model dependencies, we strictly exclude post-hoc outcomes and crash-dynamics fields.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'leakage_df = pd.DataFrame([\n',
                '    {"Dropped Column": col, "Leakage / Exclusion Rationale": reason}\n',
                '    for col, reason in EXCLUDED_COLUMNS_RATIONALE.items()\n',
                '])\n',
                'print(f"Total explicitly excluded columns: {len(leakage_df)}")\n',
                'leakage_df.head(10)'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 4. End-to-End Pipeline Execution\n',
                'We execute the complete `build_master_dataset` pipeline, which:\n',
                '- Aggregates vehicle records into collision-level interaction features (`any_motorcycle`, `any_pedal_cycle`, `any_hgv_or_bus`, `most_vulnerable_vehicle_type`, `age_of_vehicle_max`, `driver_age_band_youngest`).\n',
                '- Cleans unknown codes per map.\n',
                '- Extracts temporal markers (`hour`, `month`).\n',
                '- Encodes targets for Tasks A (binary), B (3-class), and C (regression).\n',
                '- Exports `data/clean_roadsense.parquet`.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'master_df, dq_df = build_master_dataset(data_dir="../data", output_dir="../data", tables_dir="../outputs/tables")\n',
                'print("Master clean shape:", master_df.shape)\n',
                'master_df.head()'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 5. Data Quality Report & Missing Rates Audit\n',
                'Examining missing rates and imputation strategies planned for `sklearn.compose.ColumnTransformer`.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'dq_report = pd.read_csv("../outputs/tables/data_quality_report.csv")\n',
                'dq_report.sort_values(by="missing_percent", ascending=False)'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 6. Target Distributions & COVID Distribution Shift Note\n',
                '### Target Balances:\n',
                '- **Task A (Binary)**: Dangerous (~23.3%) vs Minor (~76.7%)\n',
                '- **Task B (3-Class)**: Fatal (~1.48%), Serious (~21.8%), Slight (~76.7%)\n',
                '- **Task C (Regression)**: Number of casualties (Mean 1.28, heavily right-skewed)\n',
                '\n',
                '### 2021 COVID Distribution Shift:\n',
                '> **Important**: The UK was subject to national COVID-19 lockdowns in early 2021, which resulted in reduced overall vehicular traffic but higher average speeds and a higher proportion of vulnerable road users (cyclists/pedestrians). Models trained on 2021–2022 and tested on 2023 capture this real-world macroeconomic and behavioral distribution shift.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'for yr in [2021, 2022, 2023]:\n',
                '    sub = master_df[master_df["collision_year"] == yr]\n',
                '    print(f"Year {yr} ({len(sub):,} accidents):")\n',
                '    print(f"  Dangerous rate: {sub[\'target_severity_binary\'].mean()*100:.2f}%")\n',
                '    print(f"  Fatal rate: {(sub[\'target_severity_3class\'] == 0).mean()*100:.2f}%")\n',
                '    print(f"  Avg casualties: {sub[\'target_casualties\'].mean():.2f}")'
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

with open('notebooks/01_data_cleaning.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)
print('Saved notebooks/01_data_cleaning.ipynb successfully.')
