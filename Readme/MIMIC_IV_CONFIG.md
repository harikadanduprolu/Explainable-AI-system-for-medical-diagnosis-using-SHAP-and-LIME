# MIMIC-IV v3.1 Configuration Guide

## Overview
This project has been updated to support **MIMIC-IV v3.1**, the latest version of the Medical Information Mart for Intensive Care dataset published by MIT.

### Key Changes from MIMIC-III
- **ICD Coding**: Changed from ICD-9 to ICD-10 codes
- **Schema**: Modular approach with `hosp` (hospital) and `icu` (ICU) modules
- **Data Organization**: Better data provenance and module separation
- **Dates**: Deidentified to 2100-2200 range with anchor year groups

## Dataset Access

### Option 1: Local Files
Download MIMIC-IV v3.1 from [PhysioNet](https://physionet.org/content/mimiciv/3.1/):

1. Create a PhysioNet account and request credentialed access
2. Download the dataset to a local directory with structure:
```
/path/to/mimic-iv-v3.1/
├── hosp/
│   ├── admissions.csv
│   ├── patients.csv
│   ├── diagnoses_icd.csv
│   ├── labevents.csv
│   ├── procedures_icd.csv
│   └── ... (other hosp tables)
├── icu/
│   ├── icustays.csv
│   ├── chartevents.csv
│   ├── inputevents.csv
│   └── ... (other icu tables)
```

### Option 2: BigQuery Access
MIMIC-IV v3.1 is available on Google BigQuery:
- Database: `mimiciv_v3_1_hosp` and `mimiciv_v3_1_icu`
- Requires: GCP project with BigQuery API enabled

## Data Loading Scripts

### Load Data for Training
```bash
# Local files
python load_mimic_for_training.py --mimic-path /path/to/mimic-iv-v3.1 --output mimic_training_data.csv

# BigQuery
python load_mimic_for_training.py --bigquery --project-id YOUR_PROJECT --output mimic_training_data.csv
```

### Multi-Disease Prediction System
```python
from multi_disease_explainable_system import MultiDiseaseExplainableSystem

# Load from local files
system = MultiDiseaseExplainableSystem(mimic_data_path="/path/to/mimic-iv-v3.1")

# Or use BigQuery
system = MultiDiseaseExplainableSystem(use_bigquery=True)

# Load data
data = system.load_mimic_data()
```

## ICD-10 Disease Mappings

The project uses the following ICD-10 codes for disease prediction:

```python
DISEASE_ICD10_CODES = {
    'sepsis': ['A40', 'A41', 'R65.2'],
    'kidney_failure': ['N17', 'N18', 'N19'],
    'heart_disease': ['I21', 'I22', 'I23', 'I24', 'I25'],
    'diabetes': ['E10', 'E11', 'E13', 'E14'],
    'anemia': ['D50', 'D51', 'D52', 'D53', 'D55', 'D56', 'D57', 'D58', 'D59'],
    'thalassemia': ['D56'],
    'thrombocytopenia': ['D69'],
    'hypertension': ['I10', 'I11', 'I12', 'I13', 'I14', 'I15'],
}
```

## MIMIC-IV Key Metrics
- **Patients**: 364,627 unique individuals
- **Admissions**: 546,028 hospitalizations (2008-2022)
- **ICU Stays**: 94,458 unique ICU stays
- **Data Modules**: 2 (hosp, icu)
- **Version**: 3.1 (Oct 11, 2024)

## MIMIC-IV Tables

### HOSP Module
- `patients.csv` - Patient demographics with anchor year information
- `admissions.csv` - Hospital admissions with expire flags
- `transfers.csv` - Hospital transfers and ward assignments
- `diagnoses_icd.csv` - Diagnosis codes (ICD-10)
- `procedures_icd.csv` - Procedure codes (ICD-10)
- `labevents.csv` - Laboratory measurements
- `labevents_detail.csv` - Lab measurement details
- `emar.csv` - Electronic medication administration records
- `prescriptions.csv` - Medication prescriptions
- `pharmacy.csv` - Pharmacy records
- `chartevents.csv` (when in hosp) - Charted vital signs and complications
- `omr.csv` - Online medical record data (height, weight, BP at baseline)
- `services.csv` - Hospital service information

### ICU Module  
- `icustays.csv` - ICU patient stays
- `chartevents.csv` - Vital signs, mental status, complications (from MetaVision)
- `datetimeevents.csv` - Date/time events from MetaVision
- `inputevents.csv` - Intravenous and oral inputs
- `outputevents.csv` - Measured outputs (urine, feces, drains)
- `procedureevents.csv` - Procedures performed during ICU stay
- `d_items.csv` - Dictionary of chart event concepts/item IDs

## Date Deidentification

MIMIC-IV uses date shifting for privacy:
- All dates shifted to 2100-2200 range
- Each patient has a single anchor year (same shift applied to all their dates)
- Time intervals between events remain accurate
- Patients from same year NOT temporally comparable (by design)
- `anchor_year_group` indicates original time period:
  - '2008 - 2010'
  - '2011 - 2013'
  - '2014 - 2016'
  - '2017 - 2019'
  - '2020 - 2022'

## Citation

When using MIMIC-IV v3.1, please cite:

```bibtex
@data{johnson2024mimic,
    author = {Johnson, A.E.W. and Bulgarelli, L. and Pollard, T. and 
              Gow, B. and Moody, B. and Horng, S. and Celi, L.A. and Mark, R.},
    title = {MIMIC-IV (version 3.1)},
    year = {2024},
    doi = {10.13026/kpb9-mt58},
    url = {https://physionet.org/content/mimiciv/3.1/}
}
```

Also cite the original paper:
```bibtex
@article{johnson2023mimic,
    author = {Johnson, A.E.W. and Bulgarelli, L. and Shen, L. and others},
    title = {MIMIC-IV, a freely accessible electronic health record dataset},
    journal = {Scientific Data},
    volume = {10},
    pages = {1},
    year = {2023},
    doi = {10.1038/s41597-022-01899-x}
}
```

## Installation Requirements

```bash
# Core dependencies
pip install pandas numpy scikit-learn xgboost lightgbm

# Explainability
pip install shap lime

# Visualization
pip install matplotlib seaborn plotly

# BigQuery support (optional)
pip install google-cloud-bigquery
```

## Troubleshooting

### Issue: "File not found" errors
- Ensure directory structure matches: `/hosp/` and `/icu/` subdirectories
- MIMIC-IV uses lowercase filenames: `admissions.csv` not `ADMISSIONS.csv`
- Check that you're using MIMIC-IV v3.1 (might be v2.2 or v3.0)

### Issue: Memory errors loading large files
- Use `nrows` parameter to sample data: `pd.read_csv(file, nrows=50000)`
- Split processing into chunks for CHARTEVENTS and LABEVENTS
- Consider using BigQuery for larger-scale analysis

### Issue: ICD code matching fails
- MIMIC-IV uses ICD-10 with periods: `A40.0` vs `A40`
- Use `.startswith()` with normalized codes (remove periods)
- Verify disease code mappings in `DISEASE_ICD10_CODES`

## Resources

- [MIMIC-IV PhysioNet Page](https://physionet.org/content/mimiciv/3.1/)
- [MIMIC Code Repository](https://github.com/MIT-LCP/mimic-code)
- [MIMIC Online Documentation](https://mimic.mit.edu)
- [BigQuery Quick Start](https://cloud.google.com/bigquery/docs/quickstart)

---
**Last Updated**: March 18, 2026
**MIMIC-IV Version**: 3.1
