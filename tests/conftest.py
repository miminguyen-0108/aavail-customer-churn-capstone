"""Tiny isolated fixtures: no test reads/writes production model or log paths."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from aavail.ingestion import ingest_data
from aavail.training import fit_and_compare

@pytest.fixture
def fixture_sources(tmp_path:Path):
    n=60
    rng=np.random.default_rng(19)
    customers=pd.DataFrame({
        'customer_id':np.arange(1,n+1),
        'country_name':['singapore' if i%2 else 'united_states' for i in range(n)],
        'age':rng.integers(20,60,n),
        'is_subscriber':[i%3!=0 for i in range(n)],
        'subscriber_type':[['aavail_basic','aavail_premium','aavail_unlimited'][i%3] for i in range(n)],
        'num_streams':rng.integers(2,40,n),
    })
    streams=pd.DataFrame({
        'customer_id':np.repeat(customers.customer_id.to_numpy(),2),
        'stream_id':np.where(np.arange(n*2)%7==0,np.nan,np.arange(1,n*2+1)),
        'date':['2019-01-01']*(n*2),
        'invoice_item_id':[1]*(n*2),
        'subscription_stopped':[0]*(n*2)
    })
    src=tmp_path/'source';src.mkdir()
    cs=src/'customers.csv';ss=src/'streams.csv'
    customers.to_csv(cs,index=False)
    streams.to_csv(ss,index=False)
    return cs,ss,customers,streams

@pytest.fixture
def isolated_pipeline(fixture_sources,tmp_path):
    cs,ss,_,_=fixture_sources
    out=tmp_path/'artifacts'
    features,audit=ingest_data(cs,ss,out)
    bundle,ranking,metrics=fit_and_compare(features,out,folds=3)
    return out,features,audit,bundle,ranking,metrics
