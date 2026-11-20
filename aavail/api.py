"""FastAPI churn prediction service.

GET /predict?country=singapore predicts that country's customers.
GET /predict?country=all predicts all customers (paginated responses).
POST /predict scores an individual anonymous feature vector.
"""
from __future__ import annotations
from pathlib import Path
from time import perf_counter
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field
from .config import FEATURES_PATH, LOG_DIR, MODEL_PATH
from .ingestion import MODEL_FEATURES
from .logging_utils import configure_event_logger,log_event
from .monitoring import RequestMonitor

class PredictionInput(BaseModel):
    country_name:Literal['singapore','united_states']
    subscriber_type:Literal['aavail_basic','aavail_premium','aavail_unlimited']
    age:int=Field(ge=0,le=110)
    num_streams:int=Field(ge=0)


def create_app(model_path: str | Path | None=None,
               customers_path: str | Path | None=None,
               log_path: str | Path | None=None) -> FastAPI:
    """Dependency injection allows tests to use temporary models/logs only."""
    app=FastAPI(title='AAVAIL Customer Churn API',version='1.0.0')
    app.state.model_path=Path(model_path) if model_path is not None else MODEL_PATH
    app.state.customers_path=Path(customers_path) if customers_path is not None else FEATURES_PATH
    app.state.log_path=Path(log_path) if log_path is not None else LOG_DIR/'api_events.jsonl'
    app.state.monitor=RequestMonitor()
    app.state.bundle=None
    app.state.customers=None
    app.state.logger=None

    def load_artifacts():
        if app.state.bundle is None:
            if not app.state.model_path.is_file():
                raise HTTPException(503,'Model artifact missing. Run python scripts/run_pipeline.py')
            app.state.bundle=joblib.load(app.state.model_path)
        if app.state.customers is None:
            if not app.state.customers_path.is_file():
                raise HTTPException(503,'Customer features missing. Run python scripts/run_pipeline.py')
            app.state.customers=pd.read_csv(app.state.customers_path)
        return app.state.bundle,app.state.customers

    @app.middleware('http')
    async def audit_requests(request:Request,call_next):
        start=perf_counter()
        response=None
        try:
            response=await call_next(request)
            return response
        finally:
            status=response.status_code if response is not None else 500
            ms=round((perf_counter()-start)*1000,3)
            app.state.monitor.observe(request.url.path,status,ms)
            if app.state.logger is None:
                app.state.logger=configure_event_logger(app.state.log_path)
            log_event(app.state.logger,'http_request',route=request.url.path,status=status,latency_ms=ms)

    @app.get('/health')
    def health():
        return {'status':'ok' if app.state.model_path.is_file() and app.state.customers_path.is_file() else 'not_ready',
                'model_ready':app.state.model_path.is_file(), 'data_ready':app.state.customers_path.is_file()}

    @app.get('/metrics')
    def metrics():
        return app.state.monitor.snapshot()

    @app.get('/predict')
    def predict_cohort(country:Literal['all','singapore','united_states']='all',
                       limit:int=Query(20,ge=1,le=1000),offset:int=Query(0,ge=0)):
        bundle,customers=load_artifacts()
        subset=customers if country=='all' else customers[customers['country_name']==country]
        if subset.empty:
            raise HTTPException(404,'No customers for this country')
        if offset>=len(subset):
            raise HTTPException(400,'offset out of range')
        probs=bundle['model'].predict_proba(subset[MODEL_FEATURES])[:,1]
        pred=probs>=bundle['threshold']
        page=subset.iloc[offset:offset+limit]
        entries=[]
        for i,(_,row) in enumerate(page.iterrows(),start=offset):
            entries.append({'customer_id':int(row['customer_id']),
                            'country_name':str(row['country_name']),
                            'predicted_churn':bool(pred[i]),
                            'churn_probability':round(float(probs[i]),5)})
        country_summary={}
        for name in sorted(subset['country_name'].unique()):
            mask=(subset['country_name'].to_numpy()==name)
            country_summary[name]={'count':int(mask.sum()),'predicted_churn_rate':round(float(pred[mask].mean()),5),
                                   'mean_churn_probability':round(float(probs[mask].mean()),5)}
        return {'requested_country':country,'total_customers':len(subset),
                'model_name':bundle['model_name'], 'offset':offset,'limit':limit,
                'country_summary':country_summary, 'predictions':entries}

    @app.post('/predict')
    def predict_one(features:PredictionInput):
        bundle,_=load_artifacts()
        frame=pd.DataFrame([features.model_dump()])
        p=float(bundle['model'].predict_proba(frame[MODEL_FEATURES])[:,1][0])
        return {'predicted_churn':p>=bundle['threshold'],'churn_probability':round(p,5),
                'model_name':bundle['model_name']}

    return app

app=create_app()
