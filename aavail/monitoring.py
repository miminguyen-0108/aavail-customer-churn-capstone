"""Online request counters and optional labeled-batch performance checks."""
from __future__ import annotations
from collections import Counter
from threading import Lock
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from .ingestion import MODEL_FEATURES, _boolean_series

class RequestMonitor:
    def __init__(self):
        self.lock=Lock()
        self.requests=0
        self.errors=0
        self.total_latency_ms=0.0
        self.routes=Counter()

    def observe(self,path:str,status_code:int,latency_ms:float):
        with self.lock:
            self.requests+=1
            self.errors+=int(status_code>=400)
            self.total_latency_ms+=float(latency_ms)
            self.routes[path]+=1

    def snapshot(self):
        with self.lock:
            return {'request_count':self.requests,'error_count':self.errors,
                    'mean_latency_ms':round(self.total_latency_ms/self.requests,3) if self.requests else 0.,
                    'requests_by_path':dict(self.routes)}


def evaluate_labeled_batch(bundle:dict, batch:pd.DataFrame, min_count:int=20) -> dict:
    """Score a NEW labeled cohort; don't report training data as live monitoring.

    Returned metrics are descriptive if the supplied cohort is not independent.
    """
    missing=set(MODEL_FEATURES+['is_subscriber'])-set(batch.columns)
    if missing:
        raise ValueError(f'Labeled batch missing fields: {sorted(missing)}')
    if len(batch)<min_count:
        raise ValueError(f'At least {min_count} labeled records required')
    true=(~_boolean_series(batch['is_subscriber'])).astype(int)
    prob=bundle['model'].predict_proba(batch[MODEL_FEATURES])[:,1]
    pred=(prob>=bundle['threshold']).astype(int)
    result={'n':len(batch),'observed_churn_rate':float(true.mean()),
            'predicted_churn_rate':float(pred.mean()),
            'f1':float(f1_score(true,pred,zero_division=0)),
            'roc_auc':float(roc_auc_score(true,prob)) if true.nunique()==2 else None,
            'note':'Metrics require NEW, independently labeled records to represent post-deployment performance.'}
    ref=bundle['test_metrics']['selected_test_metrics']
    if result['roc_auc'] is not None:
        result['roc_auc_drop_vs_holdout']=float(ref['roc_auc']-result['roc_auc'])
        result['roc_auc_alert']=result['roc_auc_drop_vs_holdout']>0.10
    return result
