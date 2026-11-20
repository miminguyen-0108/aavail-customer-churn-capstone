from fastapi.testclient import TestClient
from aavail.api import create_app
from aavail.logging_utils import read_jsonl


def test_health_countries_all_and_single_prediction(isolated_pipeline,tmp_path):
    source,features,audit,bundle,ranking,metrics=isolated_pipeline
    path=tmp_path/'logs'/'api.jsonl'
    api=create_app(model_path=source/'churn_model.joblib',
                   customers_path=source/'customer_features.csv', log_path=path)
    with TestClient(api) as client:
        assert client.get('/health').json()['model_ready']
        all_result=client.get('/predict',params={'country':'all','limit':10})
        assert all_result.status_code==200
        a=all_result.json()
        assert a['total_customers']==len(features)
        assert len(a['predictions'])==10
        assert {'singapore','united_states'}==set(a['country_summary'])
        sg=client.get('/predict',params={'country':'singapore','limit':100})
        assert sg.status_code==200
        s=sg.json()
        assert s['total_customers']==(features.country_name=='singapore').sum()
        assert all(x['country_name']=='singapore' for x in s['predictions'])
        us=client.get('/predict',params={'country':'united_states','limit':5})
        assert us.status_code==200
        assert all(x['country_name']=='united_states' for x in us.json()['predictions'])
        post=client.post('/predict',json={'country_name':'singapore','subscriber_type':'aavail_premium','age':30,'num_streams':10})
        assert post.status_code==200
        assert 0<=post.json()['churn_probability']<=1
        assert client.get('/metrics').json()['request_count']>=4
    assert len(read_jsonl(path))>=5


def test_api_validation_and_missing_model_isolated(tmp_path):
    api=create_app(model_path=tmp_path/'nonexistent_model.joblib',
                   customers_path=tmp_path/'nonexistent_data.csv',
                   log_path=tmp_path/'test.jsonl')
    with TestClient(api) as client:
        assert client.get('/health').json()['status']=='not_ready'
        assert client.get('/predict').status_code==503
        assert client.get('/predict',params={'country':'mars'}).status_code==422
        assert client.post('/predict',json={'country_name':'mars','subscriber_type':'unknown','age':-1,'num_streams':5}).status_code==422
