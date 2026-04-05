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
from typing import Dict, Tuple, Optional, List
import warnings

from sklearn.model_selection import StratifiedShuffleSplit

from feature_engineering import BASE_FEATURES, add_derived_features, get_all_feature_columns
from dataset.path_config import (
    detect_mimic_iv_path,
    detect_mimic_cxr_metadata,
    detect_mimic_cxr_labels,
)

warnings.filterwarnings('ignore')

DEFAULT_MIMIC_IV_PATH = detect_mimic_iv_path()
DEFAULT_CXR_METADATA_PATH = detect_mimic_cxr_metadata()
DEFAULT_CXR_LABELS_PATH = detect_mimic_cxr_labels()


def _read_local_csv(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def _resolve_table_file(folder: Path, table_name: str) -> Optional[Path]:
    """Resolve local MIMIC table file supporting csv/csv.gz and case variants."""
    candidates = [
        folder / f"{table_name}.csv",
        folder / f"{table_name}.csv.gz",
        folder / f"{table_name.upper()}.csv",
        folder / f"{table_name.upper()}.csv.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_local_csv(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    file_path = Path(path)
    if not file_path.exists():
        print(f"⚠️  File not found: {file_path}")
        return pd.DataFrame()
    return _read_local_csv(file_path)


def prepare_chexpert_labels(
    labels_df: pd.DataFrame,
    label_column: str,
    uncertain_policy: str = "ones",
) -> Dict[str, int]:
    if labels_df.empty:
        return {}
    df = labels_df.copy()
    df.columns = [c.lower() for c in df.columns]
    if label_column.lower() not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in CheXpert file. "
            f"Available columns: {list(df.columns)}"
        )
    series = df[label_column.lower()]
    if uncertain_policy == "ones":
        series = series.replace(-1, 1)
    elif uncertain_policy == "zeros":
        series = series.replace(-1, 0)
    else:
        series = series.replace(-1, np.nan)
    df = df.assign(_label=series).dropna(subset=["dicom_id", "_label"])
    df = df[df["_label"].isin([0, 1])]
    return {str(dicom): int(label) for dicom, label in zip(df["dicom_id"], df["_label"])}


def load_cxr_metadata_df(path: Optional[str]) -> pd.DataFrame:
    metadata = load_local_csv(path)
    if metadata.empty:
        return metadata
    metadata.columns = [c.upper() for c in metadata.columns]
    required = {"HADM_ID", "DICOM_ID", "STUDY_ID"}
    if not required.issubset(metadata.columns):
        print(
            f"⚠️  CXR metadata missing required columns {required}. "
            f"Found: {metadata.columns.tolist()}"
        )
        return pd.DataFrame()
    metadata = metadata.dropna(subset=["HADM_ID", "DICOM_ID"]).copy()
    metadata["HADM_ID"] = metadata["HADM_ID"].astype(int)
    return metadata


def attach_cxr_data(
    dataset: pd.DataFrame,
    metadata_df: pd.DataFrame,
    chexpert_map: Dict[str, int],
) -> pd.DataFrame:
    if metadata_df.empty:
        enriched = dataset.copy()
        enriched["DICOM_ID"] = ""
        enriched["CXR_STUDY_ID"] = np.nan
        enriched["CXR_IMAGE_PATH"] = ""
        enriched["CXR_LABEL"] = np.nan
        print("⚠️  CXR metadata not provided; continuing with clinical-only dataset.")
        return enriched
    dedup = metadata_df.drop_duplicates("HADM_ID")
    columns = ["HADM_ID", "DICOM_ID", "STUDY_ID"]
    image_col = None
    for candidate in ("PATH", "JPEG_PATH", "FILEPATH", "IMG_PATH"):
        if candidate in dedup.columns:
            columns.append(candidate)
            image_col = candidate
            break
    merged = dataset.merge(dedup[columns], on="HADM_ID", how="inner")
    merged = merged.rename(columns={"STUDY_ID": "CXR_STUDY_ID"})
    if image_col:
        merged = merged.rename(columns={image_col: "CXR_IMAGE_PATH"})
    else:
        merged["CXR_IMAGE_PATH"] = ""
    if chexpert_map:
        merged["CXR_LABEL"] = merged["DICOM_ID"].map(chexpert_map)
        merged = merged.dropna(subset=["CXR_LABEL"])
    else:
        merged["CXR_LABEL"] = np.nan
    if merged.empty:
        enriched = dataset.copy()
        enriched["DICOM_ID"] = ""
        enriched["CXR_STUDY_ID"] = np.nan
        enriched["CXR_IMAGE_PATH"] = ""
        enriched["CXR_LABEL"] = np.nan
        print("⚠️  No overlap with CXR metadata/labels; continuing with clinical-only dataset.")
        return enriched
    return merged


def apply_stratified_splits(
    dataset: pd.DataFrame,
    disease_cols: List[str],
    seed: int = 42,
) -> pd.DataFrame:
    dataset = dataset.copy()
    dataset["split"] = "train"
    if dataset.empty:
        return dataset
    if not disease_cols:
        n = len(dataset)
        train_end = int(0.7 * n)
        val_end = train_end + int(0.15 * n)
        dataset.iloc[train_end:val_end, dataset.columns.get_loc("split")] = "val"
        dataset.iloc[val_end:, dataset.columns.get_loc("split")] = "test"
        return dataset

    signature = dataset[disease_cols].astype(str).agg("".join, axis=1)
    if signature.nunique() <= 1:
        n = len(dataset)
        train_end = int(0.7 * n)
        val_end = train_end + int(0.15 * n)
        dataset.iloc[train_end:val_end, dataset.columns.get_loc("split")] = "val"
        dataset.iloc[val_end:, dataset.columns.get_loc("split")] = "test"
        return dataset

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=seed,
    )
    try:
        train_idx, temp_idx = next(splitter.split(dataset, signature))
        temp_signature = signature.iloc[temp_idx]
        temp_indices = dataset.index[temp_idx]
        temp_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.50,
            random_state=seed,
        )
        val_idx, test_idx = next(temp_splitter.split(temp_signature, temp_signature))
        dataset.loc[temp_indices[val_idx], "split"] = "val"
        dataset.loc[temp_indices[test_idx], "split"] = "test"
    except ValueError:
        # Fallback for tiny cohorts where one or more label signatures are too rare.
        shuffled_idx = dataset.sample(frac=1.0, random_state=seed).index
        n = len(dataset)
        train_end = int(0.7 * n)
        val_end = train_end + int(0.15 * n)
        dataset.loc[shuffled_idx[train_end:val_end], "split"] = "val"
        dataset.loc[shuffled_idx[val_end:], "split"] = "test"
    return dataset


# ICD-10 code mappings for clinically defensible tabular targets (MIMIC-IV uses ICD-10)
DISEASE_ICD10_CODES = {
    'sepsis': ['A40', 'A41', 'R65.2'],  # Streptococcal sepsis, other sepsis, SIRS
    'kidney_failure': ['N17', 'N18', 'N19'],  # Acute/chronic/unspecified kidney failure
    'diabetes': ['E10', 'E11', 'E13', 'E14'],  # All types of diabetes
    'anemia': ['D50', 'D51', 'D52', 'D53', 'D55', 'D56', 'D57', 'D58', 'D59', 'D60', 'D61', 'D64'],  # All anemia types
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
            if 'STAY_ID' in self.icustays_df.columns and 'ICUSTAY_ID' not in self.icustays_df.columns:
                self.icustays_df = self.icustays_df.rename(columns={'STAY_ID': 'ICUSTAY_ID'})

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
            'patients_df': 'patients',
            'admissions_df': 'admissions',
        }
        
        for attr_name, table_name in files_to_load.items():
            file_path = _resolve_table_file(hosp_path, table_name)
            
            if file_path is not None:
                df = _read_local_csv(file_path)
                # Normalize to uppercase columns for consistency
                df.columns = df.columns.str.upper()
                setattr(self, attr_name, df)
                print(f"✅ {file_path.name}: {len(df):,} records")
            else:
                print(f"⚠️  {table_name}.csv(.gz) not found under {hosp_path}")
        
        # Load ICU stays from icu module
        icu_path = self.mimic_path / "icu"
        icustays_file = _resolve_table_file(icu_path, "icustays")
        
        if icustays_file is not None:
            self.icustays_df = _read_local_csv(icustays_file)
            self.icustays_df.columns = self.icustays_df.columns.str.upper()
            if 'STAY_ID' in self.icustays_df.columns and 'ICUSTAY_ID' not in self.icustays_df.columns:
                self.icustays_df = self.icustays_df.rename(columns={'STAY_ID': 'ICUSTAY_ID'})
            print(f"✅ {icustays_file.name}: {len(self.icustays_df):,} records")
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
        diagnoses_file = _resolve_table_file(hosp_path, "diagnoses_icd")

        if diagnoses_file is not None:
            diagnoses_df = _read_local_csv(diagnoses_file)
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
                chartevents_file = _resolve_table_file(icu_path, "chartevents")
                labevents_file = _resolve_table_file(hosp_path, "labevents")

                if chartevents_file is not None:
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

                if labevents_file is not None:
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
        """Create binary labels for configured diseases based on ICD-10 codes (MIMIC-IV)."""
        print("⏳ Creating disease labels from ICD-10 codes...")
        
        # Filter diagnoses for our admissions
        diagnoses_df = diagnoses_df[diagnoses_df['HADM_ID'].isin(hadm_ids)].copy()

        # Start from all admissions so missing diagnoses still receive zero labels.
        base_labels = pd.DataFrame({'HADM_ID': pd.Series(hadm_ids, dtype='int64')}).drop_duplicates()
        
        # Normalize ICD code once (vectorized) to keep memory usage stable on full MIMIC-IV.
        icd_norm = (
            diagnoses_df['ICD_CODE']
            .astype('string')
            .str.replace('.', '', regex=False)
            .str.strip()
            .str.upper()
            .fillna('')
        )

        # Initialize disease columns.
        for disease in DISEASE_ICD10_CODES.keys():
            diagnoses_df[disease] = 0

        # Mark diseases by checking any normalized ICD prefix in one vectorized operation.
        for disease, icd_codes in DISEASE_ICD10_CODES.items():
            prefixes = tuple(self._normalize_icd(code) for code in icd_codes)
            mask = icd_norm.str.startswith(prefixes)
            diagnoses_df.loc[mask, disease] = 1
        
        # Aggregate by admission (max value if multiple diagnoses)
        if diagnoses_df.empty:
            disease_labels = base_labels.copy()
            for disease in DISEASE_ICD10_CODES.keys():
                disease_labels[disease] = 0
        else:
            grouped = diagnoses_df.groupby('HADM_ID')[list(DISEASE_ICD10_CODES.keys())].max().reset_index()
            disease_labels = base_labels.merge(grouped, on='HADM_ID', how='left').fillna(0)
            for disease in DISEASE_ICD10_CODES.keys():
                disease_labels[disease] = disease_labels[disease].astype(int)
        
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
        max_patients: Optional[int] = None
    ) -> pd.DataFrame:
        """Build feature matrix using MIMIC hierarchy: subject_id -> hadm_id -> stay_id."""
        print(f"\n⏳ Building feature matrix for {len(icustays_df)} ICU stays...")
        
        # Sample only when an explicit cap is provided.
        if max_patients is not None and len(icustays_df) > max_patients:
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
    
    def create_training_dataset(
        self,
        max_patients: Optional[int] = None,
        max_event_rows: int = 200000,
        output_file: str = None,
        cxr_metadata: Optional[str] = None,
        cxr_labels: Optional[str] = None,
        cxr_label_column: str = "Pneumonia",
        cxr_uncertain_policy: str = "ones",
    ) -> pd.DataFrame:
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
        
        # Get ICU stays. By default, use all available MIMIC-IV mini stays.
        if max_patients is None:
            icustays = self.icustays_df.copy()
            print(f"📌 Using all {len(icustays):,} ICU stays from the dataset")
        else:
            icustays = self.icustays_df.sample(
                n=min(max_patients, len(self.icustays_df)),
                random_state=42
            ).copy()
            print(f"📌 Using a capped sample of {len(icustays):,} ICU stays")
        
        # Extract diagnoses
        diagnoses_df = self.extract_diagnoses()
        
        # Create disease labels
        disease_labels = self.create_disease_labels(
            diagnoses_df,
            icustays['HADM_ID'].unique().tolist()
        )
        
        # Extract event data using MIMIC hierarchy and aggregate to model features.
        event_data = self.extract_vitals_and_labs(icustays, max_items=max_event_rows)
        
        # Build feature matrix
        features = self.build_feature_matrix(icustays, event_data, max_patients)
        
        # Merge features with labels
        dataset = features.merge(disease_labels, on='HADM_ID', how='inner')

        # Derived clinical features
        engineered = add_derived_features(dataset[BASE_FEATURES])
        dataset = dataset.drop(columns=BASE_FEATURES, errors='ignore')
        dataset = pd.concat([dataset, engineered], axis=1)

        print(f"\n📷 Linking to MIMIC-CXR metadata...")
        cxr_metadata_df = load_cxr_metadata_df(cxr_metadata)
        chexpert_df = load_local_csv(cxr_labels)
        chexpert_map = prepare_chexpert_labels(
            chexpert_df,
            cxr_label_column,
            cxr_uncertain_policy,
        )
        dataset = attach_cxr_data(dataset, cxr_metadata_df, chexpert_map)
        print(f"   ✅ Patients with paired clinical + imaging data: {len(dataset)}")

        disease_cols = [col for col in list(DISEASE_ICD10_CODES.keys()) + ['mortality'] if col in dataset.columns]
        dataset = apply_stratified_splits(dataset, disease_cols)
        print("   ✅ Stratified splits assigned (70/15/15)")
        
        print(f"\n✅ Final dataset: {dataset.shape}")
        print(f"   Features: {len(get_all_feature_columns(dataset))} engineered variables")
        print(f"   Labels: {len(disease_cols)} diseases")
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
        default=str(DEFAULT_MIMIC_IV_PATH) if DEFAULT_MIMIC_IV_PATH else None,
        help=(
            "Path to MIMIC-IV v3.1 dataset (should contain 'hosp' and 'icu' subdirectories). "
            "Defaults to the detected dataset/mimic4 directory (or env:MIMIC_IV_PATH) when available."
        )
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
        '--cxr-metadata',
        type=str,
        default=str(DEFAULT_CXR_METADATA_PATH) if DEFAULT_CXR_METADATA_PATH else None,
        help="Path to mimic-cxr metadata CSV (must contain HADM_ID, STUDY_ID, DICOM_ID)"
    )
    parser.add_argument(
        '--cxr-labels',
        type=str,
        default=str(DEFAULT_CXR_LABELS_PATH) if DEFAULT_CXR_LABELS_PATH else None,
        help="Path to CheXpert label CSV (mimic-cxr-jpg-2.1.0-chexpert.csv.gz)"
    )
    parser.add_argument(
        '--cxr-label-column',
        type=str,
        default="Pneumonia",
        help="CheXpert label column to use when filtering imaging data"
    )
    parser.add_argument(
        '--cxr-uncertain-policy',
        choices=['ones', 'zeros', 'drop'],
        default='ones',
        help="How to treat uncertain (-1) CheXpert labels when linking imaging data"
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
            print("❌ Could not auto-detect a local MIMIC-IV directory (checked env:MIMIC_IV_PATH, dataset/mimic4, and mimic_dataset_path.txt).")
            print("\n💡 Options:")
            print("   1. Place the PhysioNet download under dataset/mimic4 (with hosp/ and icu/ folders).")
            print("   2. Provide an explicit path: python load_mimic_for_training.py --mimic-path /path/to/mimic-iv-v3.1")
            print("   3. Use BigQuery: python load_mimic_for_training.py --bigquery --project-id YOUR_PROJECT")
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
        output_file=args.output,
        cxr_metadata=args.cxr_metadata,
        cxr_labels=args.cxr_labels,
        cxr_label_column=args.cxr_label_column,
        cxr_uncertain_policy=args.cxr_uncertain_policy,
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
