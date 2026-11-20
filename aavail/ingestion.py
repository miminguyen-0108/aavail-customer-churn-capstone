"""Automated, auditable ingestion of AAVAIL customer and stream records.

Identifying names are deliberately excluded from the published customer source.
Missing stream IDs are unknown categorical identifiers, not missing numeric measures.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

CUSTOMER_COLUMNS = {'customer_id', 'country_name', 'age', 'is_subscriber', 'subscriber_type', 'num_streams'}
STREAM_COLUMNS = {'customer_id', 'stream_id', 'date', 'invoice_item_id', 'subscription_stopped'}
MODEL_FEATURES = ['country_name', 'subscriber_type', 'age', 'num_streams']
TARGET = 'churned'


def _require_fields(df: pd.DataFrame, fields: set[str], name: str) -> None:
    missing = fields - set(df.columns)
    if missing:
        raise ValueError(f'{name}: missing required columns {sorted(missing)}')


def _boolean_series(series: pd.Series) -> pd.Series:
    """Strictly normalize booleans; reject labels that would silently flip."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    truth = {'true': True, 'false': False, '1': True, '0': False}
    norm = series.astype('string').str.strip().str.lower().map(truth)
    if norm.isna().any():
        raise ValueError('is_subscriber must contain only True/False (or 1/0).')
    return norm.astype(bool)


def ingest_data(customers_csv: str | Path, streams_csv: str | Path, output_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    """Audit sources, preserve events, flag/impute unknown stream IDs and join data.

    Writes audit.json, streams_imputed.csv, and customer_features.csv to output_dir.
    The returned customer frame contains the untouched subscription outcome.
    No input file is modified.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    customers = pd.read_csv(customers_csv)
    streams = pd.read_csv(streams_csv)
    _require_fields(customers, CUSTOMER_COLUMNS, 'customer source')
    _require_fields(streams, STREAM_COLUMNS, 'stream source')
    null_customer = {k:int(v) for k,v in customers.isna().sum().items()}
    null_stream = {k:int(v) for k,v in streams.isna().sum().items()}
    if customers['customer_id'].isna().any() or streams['customer_id'].isna().any():
        raise ValueError('Cannot ingest rows with null customer IDs')
    if customers['customer_id'].duplicated().any():
        raise ValueError('Duplicate customer IDs found in customer source')
    if customers[list(CUSTOMER_COLUMNS)].isna().any().any():
        raise ValueError('Missing required customer values must be resolved before modeling')
    if streams[['date','invoice_item_id','subscription_stopped']].isna().any().any():
        raise ValueError('Unsupported missing stream values: date/invoice_item_id/subscription_stopped')
    customers = customers.copy()
    customers['is_subscriber'] = _boolean_series(customers['is_subscriber'])
    customers['age'] = pd.to_numeric(customers['age'],errors='raise')
    customers['num_streams'] = pd.to_numeric(customers['num_streams'],errors='raise')
    if (customers['age'] < 0).any() or (customers['num_streams'] < 0).any():
        raise ValueError('age and num_streams must be nonnegative')
    parsed_dates = pd.to_datetime(streams['date'], errors='coerce')
    if parsed_dates.isna().any():
        raise ValueError('Invalid date in stream source')
    if not streams['customer_id'].isin(customers['customer_id']).all():
        raise ValueError('Some streams refer to unknown customers')
    # A numeric sentinel is safe ONLY for categorical stream IDs, not an inferred title.
    streams = streams.copy()
    streams['stream_id_was_missing'] = streams['stream_id'].isna()
    streams['stream_id'] = streams['stream_id'].fillna(-1).astype('int64')
    grouped = streams.groupby('customer_id')['stream_id_was_missing'].agg(
        stream_events='size', unknown_stream_id_events='sum'
    ).reset_index()
    grouped['unknown_stream_id_rate'] = grouped['unknown_stream_id_events'] / grouped['stream_events']
    customer_features = customers.merge(grouped, on='customer_id', how='left', validate='one_to_one')
    for col in ['stream_events', 'unknown_stream_id_events']:
        customer_features[col] = customer_features[col].fillna(0).astype(int)
    customer_features['unknown_stream_id_rate'] = customer_features['unknown_stream_id_rate'].fillna(0.0)
    customer_features[TARGET] = (~customer_features['is_subscriber']).astype(int)
    if len(customer_features) != len(customers):
        raise AssertionError('Unexpected customer count after join')
    if not customer_features['customer_id'].is_unique:
        raise AssertionError('Customer key uniqueness lost')
    audit = {
        'source_customer_rows': int(len(customers)),
        'source_stream_rows': int(len(streams)),
        'source_customer_columns': int(len(null_customer)),
        'source_stream_columns': int(len(null_stream)),
        'source_customer_nulls': null_customer,
        'source_stream_nulls': null_stream,
        'missing_stream_id_imputed': int(streams['stream_id_was_missing'].sum()),
        'stream_id_imputation': 'sentinel -1 + stream_id_was_missing flag (unknown identifier)',
        'stream_events_dropped': 0,
        'source_columns_dropped_in_ingestion': [],
        'privacy_columns_excluded_before_publication': ['customer_name'],
        'model_columns_excluded_to_prevent_leakage': ['customer_id','is_subscriber','churned','stream_id','subscription_stopped'],
        'final_customer_rows': int(len(customer_features)),
        'final_stream_rows': int(len(streams)),
        'remaining_nulls_customer_features': {k:int(v) for k,v in customer_features.isna().sum().items()},
        'remaining_nulls_imputed_streams': {k:int(v) for k,v in streams.isna().sum().items()},
        'countries': {str(k):int(v) for k,v in customers['country_name'].value_counts().items()}
    }
    (output_dir/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    streams.to_csv(output_dir/'streams_imputed.csv',index=False)
    customer_features.to_csv(output_dir/'customer_features.csv',index=False)
    return customer_features,audit
