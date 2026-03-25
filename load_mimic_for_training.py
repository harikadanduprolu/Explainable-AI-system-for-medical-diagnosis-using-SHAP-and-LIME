"""
MIMIC-IV Data Loader for Multi-Disease Training
================================================
Extracts clinical features from MIMIC-IV v3.1 dataset for training 8 disease models.
Supports both local files and BigQuery access.

MIMIC-IV v3.1 is available on BigQuery under schemas:
- mimiciv_v3_1_hosp (hospital module)
- mimiciv_v3_1_icu (ICU module)

Usage:
    # Local file access:
    python load_mimic_for_training.py --mimic-path /path/to/mimic-iv-v3.1 --output mimic_training_data.csv

    # BigQuery access:
    python load_mimic_for_training.py --bigquery --project-id YOUR_PROJECT --output mimic_training_data.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ICD-10 code mappings for 8 diseases (MIMIC-IV uses ICD-10)
DISEASE_ICD10_CODES = {
    'sepsis': ['A40', 'A41', 'R65.2'],  # Streptococcal sepsis, other sepsis, SIRS
    'kidney_failure': ['N17', 'N18', 'N19'],  # Acute/chronic/unspecified kidney failure
    'heart_disease': ['I21', 'I22', 'I23', 'I24', 'I25'],  # MI and coronary disease
    'diabetes': ['E10', 'E11', 'E13', 'E14'],  # All types of diabetes
    'anemia': ['D50', 'D51', 'D52', 'D53', 'D55', 'D56', 'D57', 'D58', 'D59', 'D60', 'D61', 'D64'],  # All anemia types
    'thalassemia': ['D56'],  # Thalassemia (alpha/beta)
    'thrombocytopenia': ['D69'],  # Other specified hemorrhagic conditions
    'hypertension': ['I10', 'I11', 'I12', 'I13', 'I14', 'I15'],  # All hypertension types
}

# Core event item mappings aligned to MIMIC-IV structure.
VITAL_ITEMIDS = {
    'heart_rate': [220045],
    'systolic_bp': [220179],
    'diastolic_bp': [220180],
    'temperature': [223761, 223762],
    'respiratory_rate': [220210],
}

LAB_ITEMIDS = {
    'wbc_count': [51301],
    'hemoglobin': [50811],
    'platelet_count': [51265],
    'creatinine': [50912],
    'bun': [51006],
    'glucose': [50809, 50931],
    'lactate': [50813],
}


class MIMICDataLoader:
    """Load and preprocess MIMIC-IV v3.1 data for disease prediction."""
    
    def __init__(
        self,
        mimic_path: str = None,
        use_bigquery: bool = False,
        project_id: str = None,
        credentials_json: str = None,
    ):
        """
        Initialize MIMIC data loader.
        
        Args:
            mimic_path: Path to local MIMIC-IV files (hosp and icu subdirectories)
            use_bigquery: Use BigQuery instead of local files
            project_id: GCP project ID for BigQuery access
        """
        self.mimic_path = Path(mimic_path) if mimic_path else None
        self.use_bigquery = use_bigquery
        self.project_id = project_id
        self.credentials_json = credentials_json
        self.patients_df = None
        self.admissions_df = None
        self.icustays_df = None
        
        if use_bigquery:
            print(f"📊 Using BigQuery (project: {project_id})")
            print("   Schemas: mimiciv_v3_1_hosp, mimiciv_v3_1_icu")
            self._init_bigquery()
        else:
            print(f"📂 Loading MIMIC-IV v3.1 from: {self.mimic_path}")
            self._load_base_tables()
    
    def _init_bigquery(self):
        """Initialize BigQuery client (requires google-cloud-bigquery)."""
        try:
            from google.cloud import bigquery

            if self.credentials_json:
                service_account = __import__(
                    "google.oauth2.service_account", fromlist=["Credentials"]
                )
                creds = service_account.Credentials.from_service_account_file(
                    self.credentials_json
                )
                self.bq_client = bigquery.Client(
                    project=self.project_id, credentials=creds
                )
            else:
                self.bq_client = bigquery.Client(project=self.project_id)

            self.use_bigquery = True
            print("✅ BigQuery client initialized")
            self._load_base_tables_from_bigquery()
        except ImportError:
            print("⚠️  google-cloud-bigquery not installed. Install with:")
            print("   pip install google-cloud-bigquery")
            self.use_bigquery = False
        except Exception as e:
            print("⚠️  BigQuery initialization failed:")
            print(f"   {e}")
            print("   Authentication options:")
            print("   1) Set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON")
            print("   2) Pass --credentials-json C:/path/to/service-account.json")
            print("   3) Install gcloud and run: gcloud auth application-default login")
            self.use_bigquery = False

    def _load_base_tables_from_bigquery(self):
        """Load core MIMIC-IV tables from BigQuery schemas."""
        print("⏳ Loading core MIMIC-IV tables from BigQuery...")
        try:
            base = "physionet-data"
            patients_query = f"SELECT * FROM `{base}.mimiciv_v3_1_hosp.patients`"
            admissions_query = f"SELECT * FROM `{base}.mimiciv_v3_1_hosp.admissions`"
            icustays_query = f"SELECT * FROM `{base}.mimiciv_v3_1_icu.icustays`"

            self.patients_df = self.bq_client.query(patients_query).result().to_dataframe()
            self.admissions_df = self.bq_client.query(admissions_query).result().to_dataframe()
            self.icustays_df = self.bq_client.query(icustays_query).result().to_dataframe()

            self.patients_df.columns = self.patients_df.columns.str.upper()
            self.admissions_df.columns = self.admissions_df.columns.str.upper()
            self.icustays_df.columns = self.icustays_df.columns.str.upper()

            print(f"✅ patients: {len(self.patients_df):,} records")
            print(f"✅ admissions: {len(self.admissions_df):,} records")
            print(f"✅ icustays: {len(self.icustays_df):,} records")
        except Exception as e:
            print(f"❌ Failed to load base tables from BigQuery: {e}")
            self.patients_df = None
            self.admissions_df = None
            self.icustays_df = None
    
    def _load_base_tables(self):
        """Load core MIMIC-IV tables from local files."""
        print("⏳ Loading core MIMIC-IV tables (hosp module)...")
        
        hosp_path = self.mimic_path / "hosp"
        
        # MIMIC-IV v3.1 uses lowercase filenames in the hosp module
        files_to_load = {
            'patients_df': 'patients.csv',
            'admissions_df': 'admissions.csv',
        }
        
        for attr_name, filename in files_to_load.items():
            file_path = hosp_path / filename
            if not file_path.exists():
                # Try uppercase version
                file_path = hosp_path / filename.upper()
            
            if file_path.exists():
                df = pd.read_csv(file_path)
                # Normalize to uppercase columns for consistency
                df.columns = df.columns.str.upper()
                setattr(self, attr_name, df)
                print(f"✅ {filename}: {len(df):,} records")
            else:
                print(f"⚠️  {filename} not found at {file_path}")
        
        # Load ICU stays from icu module
        icu_path = self.mimic_path / "icu"
        icustays_file = icu_path / "icustays.csv"
        if not icustays_file.exists():
            icustays_file = icu_path / "ICUSTAYS.csv"
        
        if icustays_file.exists():
            self.icustays_df = pd.read_csv(icustays_file)
            self.icustays_df.columns = self.icustays_df.columns.str.upper()
            print(f"✅ ICU Stays: {len(self.icustays_df):,} records")
        else:
            print("⚠️  ICU stays file not found")

    @staticmethod
    def _normalize_icd(code: str) -> str:
        return str(code).replace('.', '').strip().upper()

    @staticmethod
    def _aggregate_feature_stats(df: pd.DataFrame, key_col: str, feature_col: str) -> pd.DataFrame:
        """Aggregate event rows into mean/min/max per key and feature."""
        if df.empty:
            return pd.DataFrame()
        grouped = (
            df.groupby([key_col, feature_col])['VALUENUM']
            .agg(['mean', 'min', 'max'])
            .reset_index()
        )
        wide = grouped.pivot(index=key_col, columns=feature_col)
        wide.columns = [f"{feat}_{stat}" for stat, feat in wide.columns]
        return wide.reset_index()

    @staticmethod
    def _map_itemid_to_feature(itemid: int, mapping: Dict[str, list]) -> Optional[str]:
        for feature, ids in mapping.items():
            if int(itemid) in ids:
                return feature
        return None
    
    def extract_diagnoses(self) -> pd.DataFrame:
        """Extract diagnosis codes for disease labeling (ICD-10 in MIMIC-IV)."""
        print("⏳ Loading diagnoses (MIMIC-IV uses ICD-10)...")

        if self.use_bigquery:
            try:
                diagnoses_query = "SELECT * FROM `physionet-data.mimiciv_v3_1_hosp.diagnoses_icd`"
                diagnoses_df = self.bq_client.query(diagnoses_query).result().to_dataframe()
                diagnoses_df.columns = diagnoses_df.columns.str.upper()
                print(f"✅ Diagnoses: {len(diagnoses_df):,} records (ICD-10)")
                return diagnoses_df
            except Exception as e:
                print(f"⚠️  Failed to load diagnoses from BigQuery: {e}")
                return pd.DataFrame()
        
        hosp_path = self.mimic_path / "hosp"
        diagnoses_file = hosp_path / "diagnoses_icd.csv"
        
        if not diagnoses_file.exists():
            diagnoses_file = hosp_path / "DIAGNOSES_ICD.csv"
        
        if diagnoses_file.exists():
            diagnoses_df = pd.read_csv(diagnoses_file)
            diagnoses_df.columns = diagnoses_df.columns.str.upper()
            print(f"✅ Diagnoses: {len(diagnoses_df):,} records (ICD-10)")
            return diagnoses_df
        else:
            print(f"⚠️  Diagnoses file not found")
            return pd.DataFrame()
    
    def extract_vitals_and_labs(self, icustays_df: pd.DataFrame, max_items: int = 200000) -> Dict[str, pd.DataFrame]:
        """Extract and aggregate event data into model-ready features.

        Returns:
            Dict with keys:
            - vitals_by_stay: aggregated chart events by ICUSTAY_ID
            - labs_by_hadm: aggregated lab events by HADM_ID
        """
        print("⏳ Extracting chart events and lab events from MIMIC-IV...")
        stay_ids = icustays_df['ICUSTAY_ID'].dropna().astype(int).unique().tolist()
        hadm_ids = icustays_df['HADM_ID'].dropna().astype(int).unique().tolist()

        if not stay_ids or not hadm_ids:
            return {'vitals_by_stay': pd.DataFrame(), 'labs_by_hadm': pd.DataFrame()}

        vital_itemids = [i for v in VITAL_ITEMIDS.values() for i in v]
        lab_itemids = [i for v in LAB_ITEMIDS.values() for i in v]

        vitals_events = pd.DataFrame()
        labs_events = pd.DataFrame()

        if self.use_bigquery:
            try:
                stay_ids_str = ",".join(str(s) for s in stay_ids[:2000])
                hadm_ids_str = ",".join(str(h) for h in hadm_ids[:2000])
                vital_itemids_str = ",".join(str(i) for i in vital_itemids)
                lab_itemids_str = ",".join(str(i) for i in lab_itemids)

                vitals_query = f"""
                    SELECT stay_id AS ICUSTAY_ID, itemid, charttime, valuenum
                    FROM `physionet-data.mimiciv_v3_1_icu.chartevents`
                    WHERE stay_id IN ({stay_ids_str})
                      AND itemid IN ({vital_itemids_str})
                      AND valuenum IS NOT NULL
                    LIMIT {max_items}
                """
                labs_query = f"""
                    SELECT hadm_id AS HADM_ID, itemid, charttime, valuenum
                    FROM `physionet-data.mimiciv_v3_1_hosp.labevents`
                    WHERE hadm_id IN ({hadm_ids_str})
                      AND itemid IN ({lab_itemids_str})
                      AND valuenum IS NOT NULL
                    LIMIT {max_items}
                """
                vitals_events = self.bq_client.query(vitals_query).result().to_dataframe()
                labs_events = self.bq_client.query(labs_query).result().to_dataframe()
                print(f"✅ Chart events: {len(vitals_events):,} rows, Lab events: {len(labs_events):,} rows")
            except Exception as e:
                print(f"⚠️  Failed to extract events from BigQuery: {e}")
                return {'vitals_by_stay': pd.DataFrame(), 'labs_by_hadm': pd.DataFrame()}
        else:
            try:
                icu_path = self.mimic_path / "icu"
                hosp_path = self.mimic_path / "hosp"
                chartevents_file = icu_path / "chartevents.csv"
                if not chartevents_file.exists():
                    chartevents_file = icu_path / "CHARTEVENTS.csv"

                labevents_file = hosp_path / "labevents.csv"
                if not labevents_file.exists():
                    labevents_file = hosp_path / "LABEVENTS.csv"

                if chartevents_file.exists():
                    chunks = []
                    for chunk in pd.read_csv(
                        chartevents_file,
                        chunksize=500000,
                        usecols=['stay_id', 'itemid', 'charttime', 'valuenum']
                    ):
                        chunk.columns = [c.upper() for c in chunk.columns]
                        sub = chunk[
                            chunk['STAY_ID'].isin(stay_ids) &
                            chunk['ITEMID'].isin(vital_itemids) &
                            chunk['VALUENUM'].notna()
                        ]
                        if not sub.empty:
                            chunks.append(sub)
                        if sum(len(x) for x in chunks) >= max_items:
                            break
                    if chunks:
                        vitals_events = pd.concat(chunks, ignore_index=True)
                        vitals_events = vitals_events.rename(columns={'STAY_ID': 'ICUSTAY_ID'})

                if labevents_file.exists():
                    chunks = []
                    for chunk in pd.read_csv(
                        labevents_file,
                        chunksize=500000,
                        usecols=['hadm_id', 'itemid', 'charttime', 'valuenum']
                    ):
                        chunk.columns = [c.upper() for c in chunk.columns]
                        sub = chunk[
                            chunk['HADM_ID'].isin(hadm_ids) &
                            chunk['ITEMID'].isin(lab_itemids) &
                            chunk['VALUENUM'].notna()
                        ]
                        if not sub.empty:
                            chunks.append(sub)
                        if sum(len(x) for x in chunks) >= max_items:
                            break
                    if chunks:
                        labs_events = pd.concat(chunks, ignore_index=True)

                print(f"✅ Chart events: {len(vitals_events):,} rows, Lab events: {len(labs_events):,} rows")
            except Exception as e:
                print(f"⚠️  Failed to load local events: {e}")
                return {'vitals_by_stay': pd.DataFrame(), 'labs_by_hadm': pd.DataFrame()}

        if not vitals_events.empty:
            vitals_events['FEATURE'] = vitals_events['ITEMID'].apply(lambda x: self._map_itemid_to_feature(x, VITAL_ITEMIDS))
            vitals_events = vitals_events[vitals_events['FEATURE'].notna()]
            vitals_by_stay = self._aggregate_feature_stats(vitals_events, 'ICUSTAY_ID', 'FEATURE')
        else:
            vitals_by_stay = pd.DataFrame()

        if not labs_events.empty:
            labs_events['FEATURE'] = labs_events['ITEMID'].apply(lambda x: self._map_itemid_to_feature(x, LAB_ITEMIDS))
            labs_events = labs_events[labs_events['FEATURE'].notna()]
            labs_by_hadm = self._aggregate_feature_stats(labs_events, 'HADM_ID', 'FEATURE')
        else:
            labs_by_hadm = pd.DataFrame()

        return {
            'vitals_by_stay': vitals_by_stay,
            'labs_by_hadm': labs_by_hadm,
        }
    
    def create_disease_labels(self, diagnoses_df: pd.DataFrame, hadm_ids: list) -> pd.DataFrame:
        """Create binary labels for 8 diseases based on ICD-10 codes (MIMIC-IV)."""
        print("⏳ Creating disease labels from ICD-10 codes...")
        
        # Filter diagnoses for our admissions
        diagnoses_df = diagnoses_df[diagnoses_df['HADM_ID'].isin(hadm_ids)].copy()
        
        # Initialize disease columns
        for disease in DISEASE_ICD10_CODES.keys():
            diagnoses_df[disease] = 0
        
        # Mark diseases based on ICD-10 codes (prefix matching)
        for disease, icd_codes in DISEASE_ICD10_CODES.items():
            for icd_code in icd_codes:
                # ICD-10 codes in MIMIC-IV are stored with periods (e.g., A40.0).
                # Use normalized prefix matching.
                mask = diagnoses_df['ICD_CODE'].astype(str).map(self._normalize_icd).str.startswith(self._normalize_icd(icd_code))
                diagnoses_df.loc[mask, disease] = 1
        
        # Aggregate by admission (max value if multiple diagnoses)
        disease_labels = diagnoses_df.groupby('HADM_ID')[list(DISEASE_ICD10_CODES.keys())].max().reset_index()
        
        # Add mortality label (from ADMISSIONS.hospital_expire_flag in MIMIC-IV)
        if 'HOSPITAL_EXPIRE_FLAG' in self.admissions_df.columns:
            mortality_map = self.admissions_df.set_index('HADM_ID')['HOSPITAL_EXPIRE_FLAG'].to_dict()
            disease_labels['mortality'] = disease_labels['HADM_ID'].map(mortality_map).fillna(0).astype(int)
        elif 'HOSPITA_EXPIRE_FLAG' in self.admissions_df.columns:  # Handle typo if present
            mortality_map = self.admissions_df.set_index('HADM_ID')['HOSPITA_EXPIRE_FLAG'].to_dict()
            disease_labels['mortality'] = disease_labels['HADM_ID'].map(mortality_map).fillna(0).astype(int)
        else:
            # Estimate from DOD if available
            disease_labels['mortality'] = 0
        
        # Print prevalences
        print("\n📊 Disease Prevalences (ICD-10):")
        for disease in list(DISEASE_ICD10_CODES.keys()) + ['mortality']:
            count = disease_labels[disease].sum()
            pct = 100 * count / len(disease_labels) if len(disease_labels) > 0 else 0
            print(f"   {disease:20s}: {count:5d} ({pct:5.1f}%)")
        
        return disease_labels
    
    def build_feature_matrix(
        self,
        icustays_df: pd.DataFrame,
        event_data: Dict[str, pd.DataFrame],
        max_patients: int = 5000
    ) -> pd.DataFrame:
        """Build feature matrix using MIMIC hierarchy: subject_id -> hadm_id -> stay_id."""
        print(f"\n⏳ Building feature matrix for {len(icustays_df)} ICU stays...")
        
        # Sample if too many
        if len(icustays_df) > max_patients:
            print(f"   Sampling {max_patients} of {len(icustays_df)} stays...")
            icustays_df = icustays_df.sample(n=max_patients, random_state=42)
        
        # Merge with patient demographics
        data = icustays_df.merge(
            self.patients_df[['SUBJECT_ID', 'GENDER'] + ([c for c in ['ANCHOR_AGE'] if c in self.patients_df.columns])],
            on='SUBJECT_ID',
            how='left'
        )

        if 'ANCHOR_AGE' in data.columns:
            data['age'] = data['ANCHOR_AGE'].fillna(data['ANCHOR_AGE'].median()).clip(18, 91)
        else:
            np.random.seed(42)
            n = len(data)
            data['age'] = np.random.normal(65, 15, n).clip(18, 89)
        
        # Gender encoding
        data['gender'] = (data['GENDER'] == 'M').astype(int)
        
        vitals_by_stay = event_data.get('vitals_by_stay', pd.DataFrame())
        labs_by_hadm = event_data.get('labs_by_hadm', pd.DataFrame())

        if not vitals_by_stay.empty:
            data = data.merge(vitals_by_stay, on='ICUSTAY_ID', how='left')
        if not labs_by_hadm.empty:
            data = data.merge(labs_by_hadm, on='HADM_ID', how='left')

        # Backward-compatible canonical features expected by downstream pipeline.
        canonical_map = {
            'heart_rate': ['heart_rate_mean', 'heart_rate_max', 'heart_rate_min'],
            'systolic_bp': ['systolic_bp_mean', 'systolic_bp_max', 'systolic_bp_min'],
            'diastolic_bp': ['diastolic_bp_mean', 'diastolic_bp_max', 'diastolic_bp_min'],
            'temperature': ['temperature_mean', 'temperature_max', 'temperature_min'],
            'respiratory_rate': ['respiratory_rate_mean', 'respiratory_rate_max', 'respiratory_rate_min'],
            'wbc_count': ['wbc_count_mean', 'wbc_count_max', 'wbc_count_min'],
            'hemoglobin': ['hemoglobin_mean', 'hemoglobin_max', 'hemoglobin_min'],
            'platelet_count': ['platelet_count_mean', 'platelet_count_max', 'platelet_count_min'],
            'creatinine': ['creatinine_mean', 'creatinine_max', 'creatinine_min'],
            'bun': ['bun_mean', 'bun_max', 'bun_min'],
            'glucose': ['glucose_mean', 'glucose_max', 'glucose_min'],
            'lactate': ['lactate_mean', 'lactate_max', 'lactate_min'],
        }

        for feature, candidates in canonical_map.items():
            existing = [c for c in candidates if c in data.columns]
            if existing:
                data[feature] = data[existing].mean(axis=1)
            else:
                data[feature] = np.nan

        # Missingness handling for sparse events.
        for feature in canonical_map.keys():
            data[feature] = data[feature].fillna(data[feature].median())

        # If a feature remains all-null (rare), fill safe clinical defaults.
        defaults = {
            'heart_rate': 80,
            'systolic_bp': 120,
            'diastolic_bp': 70,
            'temperature': 98.6,
            'respiratory_rate': 18,
            'wbc_count': 10,
            'hemoglobin': 12,
            'platelet_count': 220,
            'creatinine': 1.1,
            'bun': 20,
            'glucose': 120,
            'lactate': 1.8,
        }
        for feature, default_val in defaults.items():
            if data[feature].isna().all():
                data[feature] = default_val
        
        print(f"✅ Feature matrix: {data.shape}")
        
        return data[['HADM_ID', 'ICUSTAY_ID', 'age', 'gender', 
                     'heart_rate', 'systolic_bp', 'diastolic_bp', 'temperature',
                     'respiratory_rate', 'wbc_count', 'hemoglobin', 'platelet_count',
                     'creatinine', 'bun', 'glucose', 'lactate']]
    
    def create_training_dataset(self, max_patients: int = 5000, output_file: str = None) -> pd.DataFrame:
        """Create complete training dataset with features and labels."""
        print("\n" + "="*60)
        print("Creating MIMIC-IV v3.1 Training Dataset")
        print("="*60)

        if self.icustays_df is None or self.admissions_df is None or self.patients_df is None:
            raise RuntimeError(
                "Required base tables are not loaded. "
                "For local mode, verify --mimic-path points to a folder containing 'hosp' and 'icu'. "
                "For BigQuery mode, install google-cloud-bigquery and ensure project access to physionet-data."
            )

        if self.icustays_df.empty:
            raise RuntimeError("ICU stays table is empty. Cannot build training dataset.")
        
        # Get ICU stays
        icustays = self.icustays_df.sample(
            n=min(max_patients, len(self.icustays_df)),
            random_state=42
        ).copy()
        
        # Extract diagnoses
        diagnoses_df = self.extract_diagnoses()
        
        # Create disease labels
        disease_labels = self.create_disease_labels(
            diagnoses_df,
            icustays['HADM_ID'].unique().tolist()
        )
        
        # Extract event data using MIMIC hierarchy and aggregate to model features.
        event_data = self.extract_vitals_and_labs(icustays, max_items=200000)
        
        # Build feature matrix
        features = self.build_feature_matrix(icustays, event_data, max_patients)
        
        # Merge features with labels
        dataset = features.merge(disease_labels, on='HADM_ID', how='inner')
        
        # Drop ID columns
        dataset = dataset.drop(['HADM_ID', 'ICUSTAY_ID'], axis=1)
        
        print(f"\n✅ Final dataset: {dataset.shape}")
        print(f"   Features: {dataset.shape[1] - 8} clinical variables")
        print(f"   Labels: 8 diseases")
        print(f"   Samples: {len(dataset)} patients")
        
        # Save if output file specified
        if output_file:
            dataset.to_csv(output_file, index=False)
            print(f"\n💾 Saved to: {output_file}")
        
        return dataset


def main():
    parser = argparse.ArgumentParser(description="Load MIMIC-IV v3.1 data for disease model training")
    parser.add_argument(
        '--mimic-path',
        type=str,
        help="Path to MIMIC-IV v3.1 dataset (should contain 'hosp' and 'icu' subdirectories)"
    )
    parser.add_argument(
        '--bigquery',
        action='store_true',
        help="Use BigQuery instead of local files (requires google-cloud-bigquery)"
    )
    parser.add_argument(
        '--project-id',
        type=str,
        help="GCP project ID for BigQuery access"
    )
    parser.add_argument(
        '--credentials-json',
        type=str,
        help="Path to GCP service-account JSON for BigQuery auth (no gcloud required)"
    )
    parser.add_argument(
        '--output',
        type=str,
        default="mimic_training_data.csv",
        help="Output CSV file"
    )
    parser.add_argument(
        '--max-patients',
        type=int,
        default=5000,
        help="Maximum number of patients to extract"
    )
    
    args = parser.parse_args()
    
    # Handle data source selection
    if args.bigquery:
        if not args.project_id:
            print("❌ --project-id required for BigQuery access")
            return
        if args.credentials_json and not Path(args.credentials_json).exists():
            print(f"❌ credentials file not found: {args.credentials_json}")
            return

        loader = MIMICDataLoader(
            use_bigquery=True,
            project_id=args.project_id,
            credentials_json=args.credentials_json,
        )
        if not loader.use_bigquery:
            print("❌ BigQuery mode unavailable due to initialization/authentication failure.")
            return
    else:
        # Check if path exists
        if not args.mimic_path:
            print("❌ MIMIC path not provided")
            print("\n💡 Options:")
            print("   1. Local files: python load_mimic_for_training.py --mimic-path /path/to/mimic-iv-v3.1")
            print("   2. BigQuery: python load_mimic_for_training.py --bigquery --project-id YOUR_PROJECT")
            print("\n📥 Download MIMIC-IV v3.1 from PhysioNet:")
            print("   https://physionet.org/content/mimiciv/3.1/")
            print("   Files location: /hosp and /icu subdirectories with CSV files")
            return
        
        mimic_path = Path(args.mimic_path)
        if not mimic_path.exists():
            print(f"❌ MIMIC path not found: {mimic_path}")
            return
        
        loader = MIMICDataLoader(args.mimic_path)

    if loader.icustays_df is None:
        print("❌ Could not load icustays table. Check your data source and credentials.")
        return
    
    # Load and process data
    dataset = loader.create_training_dataset(
        max_patients=args.max_patients,
        output_file=args.output
    )
    
    print("\n" + "="*60)
    print("🎉 MIMIC-IV Dataset Ready for Training!")
    print("="*60)
    print(f"\n📝 Next steps:")
    print(f"   1. Train models: python training_pipeline.py --data-source csv --csv-path {args.output}")
    print(f"   2. Verify output: head {args.output}")
    print(f"   3. Check disease distributions in the data")
    print(f"\n📊 Dataset: {dataset.shape[0]} samples, {dataset.shape[1]-9} features")
    
    return dataset


if __name__ == "__main__":
    main()
