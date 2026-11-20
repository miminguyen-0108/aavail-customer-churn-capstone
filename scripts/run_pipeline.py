#!/usr/bin/env python3
"""One-command ingestion, model comparison, and EDA generation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aavail.config import DATA_DIR,ARTIFACT_DIR
from aavail.ingestion import ingest_data
from aavail.training import fit_and_compare
from aavail.visualizations import generate_figures

def main():
    features,audit=ingest_data(DATA_DIR/'aavail-customers-anonymized.csv',DATA_DIR/'aavail-streams.csv',ARTIFACT_DIR)
    _,leaderboard,metrics=fit_and_compare(features,ARTIFACT_DIR)
    generate_figures(features,audit,leaderboard,metrics,ARTIFACT_DIR/'figures')
    print('Ingestion:',audit['source_customer_rows'],'customers,',audit['source_stream_rows'],'stream events')
    print('Missing stream IDs handled:',audit['missing_stream_id_imputed'])
    print('Best model:',metrics['selected_model'])
    print('Holdout metrics:',metrics['selected_test_metrics'])
    print('Artifacts saved to:',ARTIFACT_DIR)
if __name__=='__main__':main()
