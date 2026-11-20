import json
import pandas as pd
import pytest
from aavail.ingestion import ingest_data


def test_missing_identified_and_imputed_without_dropping_rows(fixture_sources,tmp_path):
    cs,ss,customers,streams=fixture_sources
    features,audit=ingest_data(cs,ss,tmp_path/'isolated')
    events=pd.read_csv(tmp_path/'isolated'/'streams_imputed.csv')
    assert audit['source_stream_nulls']['stream_id']==streams.stream_id.isna().sum()
    assert audit['missing_stream_id_imputed']==events['stream_id_was_missing'].sum()
    assert len(events)==len(streams) and len(features)==len(customers)
    assert events['stream_id'].isna().sum()==0
    assert (events.loc[events.stream_id_was_missing,'stream_id']==-1).all()
    assert len(audit['source_columns_dropped_in_ingestion'])==0
    assert features['churned'].tolist()==(~customers['is_subscriber']).astype(int).tolist()


def test_missing_required_column_fails(fixture_sources,tmp_path):
    cs,ss,customers,_=fixture_sources
    customers.drop(columns='is_subscriber').to_csv(cs,index=False)
    with pytest.raises(ValueError,match='missing required columns'):
        ingest_data(cs,ss,tmp_path/'isolated')


def test_duplicate_customer_key_fails(fixture_sources,tmp_path):
    cs,ss,customers,_=fixture_sources
    pd.concat([customers,customers.iloc[[0]]]).to_csv(cs,index=False)
    with pytest.raises(ValueError,match='Duplicate customer IDs'):
        ingest_data(cs,ss,tmp_path/'isolated')


def test_unrecognized_subscriber_label_rejected(fixture_sources,tmp_path):
    cs,ss,customers,_=fixture_sources
    customers['is_subscriber']=customers['is_subscriber'].astype(str)
    customers.loc[0,'is_subscriber']='not_valid'
    customers.to_csv(cs,index=False)
    with pytest.raises(ValueError,match='must contain only'):
        ingest_data(cs,ss,tmp_path/'isolated')
