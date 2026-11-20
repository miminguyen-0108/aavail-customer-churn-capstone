#!/usr/bin/env python3
"""Monitor a labeled batch supplied after deployment; writes only to requested path."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib,pandas as pd
from aavail.config import MODEL_PATH
from aavail.monitoring import evaluate_labeled_batch

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True,help='NEW labeled customer CSV; must contain is_subscriber and model features')
    parser.add_argument('--model',default=str(MODEL_PATH))
    parser.add_argument('--output',default='artifacts/latest_monitoring.json')
    args=parser.parse_args()
    result=evaluate_labeled_batch(joblib.load(args.model),pd.read_csv(args.input))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if result.get('roc_auc_alert'):
        print('ALERT: ROC AUC is > 0.10 below the original holdout estimate')
if __name__=='__main__':main()
