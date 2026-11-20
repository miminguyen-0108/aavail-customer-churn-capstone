"""Fit and compare models on a stratified train/test split.

Model selection is based only on 5-fold cross-validation inside the training
partition. The test partition is untouched until after final model selection.
"""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .ingestion import MODEL_FEATURES, TARGET


def create_pipeline(estimator):
    preprocessor = ColumnTransformer(transformers=[
        ('cats', OneHotEncoder(handle_unknown='ignore'), ['country_name','subscriber_type']),
        ('nums', StandardScaler(), ['age','num_streams']),
    ])
    return Pipeline([('preprocessor',preprocessor),('model',estimator)])


def candidate_models(seed=42):
    return {
        'baseline_majority': create_pipeline(DummyClassifier(strategy='most_frequent')),
        'logistic_regression': create_pipeline(LogisticRegression(max_iter=1500,random_state=seed)),
        'random_forest': create_pipeline(RandomForestClassifier(n_estimators=140,min_samples_leaf=3,random_state=seed,n_jobs=1)),
    }


def _metrics(model, x, y):
    pred = model.predict(x)
    prob = model.predict_proba(x)[:,1]
    return {
        'accuracy':float(accuracy_score(y,pred)),
        'precision':float(precision_score(y,pred,zero_division=0)),
        'recall':float(recall_score(y,pred,zero_division=0)),
        'f1':float(f1_score(y,pred,zero_division=0)),
        'roc_auc':float(roc_auc_score(y,prob)) if y.nunique()==2 else None,
    }


def fit_and_compare(customer_features:pd.DataFrame, output_dir:str|Path,seed:int=42,folds:int=5):
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    if not set(MODEL_FEATURES+[TARGET]).issubset(customer_features.columns):
        raise ValueError('Missing required model features or target')
    x=customer_features[MODEL_FEATURES]
    y=customer_features[TARGET].astype(int)
    if y.nunique()!=2 or y.value_counts().min()<folds*2:
        raise ValueError('Insufficient representation of both churn classes for stratification')
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,stratify=y,random_state=seed)
    cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    candidates=candidate_models(seed)
    rows=[]
    for name,model in candidates.items():
        scores=cross_val_score(model,x_train,y_train,cv=cv,scoring='roc_auc',n_jobs=1)
        rows.append({'model':name,'cv_roc_auc_mean':float(scores.mean()),'cv_roc_auc_std':float(scores.std())})
    leaderboard=pd.DataFrame(rows).sort_values('cv_roc_auc_mean',ascending=False).reset_index(drop=True)
    best_name=str(leaderboard.loc[0,'model'])
    selected=candidates[best_name].fit(x_train,y_train)
    baseline=candidates['baseline_majority'].fit(x_train,y_train)
    test_results={'selected_model':best_name,'train_size':int(len(x_train)),
                  'test_size':int(len(x_test)),'random_state':seed,
                  'selection_metric':'mean train-fold ROC AUC (5-fold CV)',
                  'selected_test_metrics':_metrics(selected,x_test,y_test),
                  'baseline_test_metrics':_metrics(baseline,x_test,y_test),
                  'test_country_breakdown':{}}
    for country in sorted(x_test['country_name'].unique()):
        mask=x_test['country_name']==country
        test_results['test_country_breakdown'][country]={
            'count':int(mask.sum()),'selected':_metrics(selected,x_test.loc[mask],y_test.loc[mask])
        }
    bundle={'model':selected,'model_name':best_name,'features':MODEL_FEATURES,
            'threshold':0.5,'test_metrics':test_results,'reference_data_size':len(customer_features)}
    joblib.dump(bundle,output_dir/'churn_model.joblib')
    leaderboard.to_csv(output_dir/'model_comparison.csv',index=False)
    (output_dir/'model_metrics.json').write_text(json.dumps(test_results,indent=2),encoding='utf-8')
    return bundle,leaderboard,test_results
