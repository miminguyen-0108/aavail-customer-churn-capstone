import pytest
from aavail.monitoring import RequestMonitor,evaluate_labeled_batch


def test_request_monitor():
    m=RequestMonitor();m.observe('/predict',200,12);m.observe('/predict',503,8)
    assert m.snapshot()['request_count']==2
    assert m.snapshot()['error_count']==1
    assert m.snapshot()['mean_latency_ms']==10


def test_batch_monitoring_uses_injected_model(isolated_pipeline):
    _,features,_,bundle,_,_=isolated_pipeline
    report=evaluate_labeled_batch(bundle,features,min_count=20)
    assert report['n']==len(features)
    assert 0<=report['f1']<=1
    assert 0<=report['roc_auc']<=1
    assert 'independently labeled' in report['note']
    with pytest.raises(ValueError,match='At least'):
        evaluate_labeled_batch(bundle,features.head(2),min_count=20)
