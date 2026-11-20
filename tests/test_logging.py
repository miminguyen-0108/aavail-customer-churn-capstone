import json
from aavail.logging_utils import configure_event_logger,log_event,read_jsonl


def test_jsonl_logger_isolated_in_tmp_path(tmp_path):
    log=tmp_path/'test_logs'/'requests.jsonl'
    logger=configure_event_logger(log)
    log_event(logger,'test_event',country='singapore',status=200,latency_ms=12.5)
    assert log.is_file()
    items=read_jsonl(log)
    assert len(items)==1
    assert items[0]['country']=='singapore'
    assert items[0]['event']=='test_event'
    assert 'ts' in items[0]
    assert read_jsonl(tmp_path/'missing.jsonl')==[]


def test_logging_same_path_does_not_duplicate_handlers(tmp_path):
    p=tmp_path/'events.jsonl'
    one=configure_event_logger(p)
    two=configure_event_logger(p)
    assert one is two and len(one.handlers)==1
    log_event(two,'once')
    assert len(read_jsonl(p))==1
