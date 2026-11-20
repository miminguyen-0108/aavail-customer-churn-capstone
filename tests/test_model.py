import joblib
import numpy as np
from aavail.ingestion import MODEL_FEATURES
from aavail.training import candidate_models


def test_model_comparison_baseline_and_saved_artifact(isolated_pipeline):
    path,features,audit,bundle,ranking,metrics=isolated_pipeline
    assert set(ranking['model'])=={'baseline_majority','logistic_regression','random_forest'}
    assert metrics['train_size']+metrics['test_size']==len(features)
    assert metrics['selected_model']==ranking.iloc[0]['model']
    assert (path/'churn_model.joblib').is_file()
    assert joblib.load(path/'churn_model.joblib')['model_name']==metrics['selected_model']


def test_probability_bounds_and_leakage_prevention(isolated_pipeline):
    _,features,_,bundle,_,_=isolated_pipeline
    assert 'churned' not in MODEL_FEATURES
    assert 'is_subscriber' not in MODEL_FEATURES
    assert 'customer_id' not in MODEL_FEATURES
    probs=bundle['model'].predict_proba(features[MODEL_FEATURES])
    assert probs.shape==(len(features),2)
    assert np.all((probs>=0)&(probs<=1))
    assert np.allclose(probs.sum(axis=1),1)


def test_baseline_is_deterministic(isolated_pipeline):
    _,features,_,bundle,_,_=isolated_pipeline
    x=features[MODEL_FEATURES]
    assert bundle['model'].predict(x).shape==(len(features),)
