"""Source-grounded EDA and holdout model-vs-baseline visualizations."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAVY='#1D3551'; TEAL='#138A89'; ORANGE='#E78B45'; BLUE='#4388BE'

def _style(ax,title,ylabel=None):
    ax.set_title(title,loc='left',fontsize=13,fontweight='bold',color=NAVY)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2)
    ax.set_axisbelow(True)
    if ylabel: ax.set_ylabel(ylabel)

def _save(fig,path):
    fig.savefig(path,dpi=160,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def generate_figures(customers:pd.DataFrame,audit:dict,leaderboard:pd.DataFrame,metrics:dict,figure_dir:str|Path):
    figure_dir=Path(figure_dir);figure_dir.mkdir(parents=True,exist_ok=True)
    # The source null counts predate imputation, so they document the genuine defect.
    stream_missing=audit['source_stream_nulls']
    fig,ax=plt.subplots(figsize=(9,4.7))
    keys=list(stream_missing.keys()); vals=list(stream_missing.values())
    bars=ax.barh(keys,vals,color=[ORANGE if v else BLUE for v in vals]);ax.invert_yaxis()
    _style(ax,'Source audit: missing values by stream column','')
    ax.set_xlabel('Missing entries (before imputation)');ax.set_xlim(0,max(vals)*1.3)
    for bar,val in zip(bars,vals):ax.text(val+14,bar.get_y()+bar.get_height()/2,str(val),va='center',fontsize=10)
    fig.subplots_adjust(bottom=.22,left=.22)
    fig.text(.22,.04,f"{audit['missing_stream_id_imputed']:,} / {audit['source_stream_rows']:,} stream IDs missing; 0 event rows/columns dropped; unknown IDs set to -1.",fontsize=9,color=NAVY)
    _save(fig,figure_dir/'01_missing_values.png')

    country=customers.groupby('country_name')['churned'].agg(['sum','count','mean']).loc[['united_states','singapore']]
    fig,ax=plt.subplots(figsize=(9,5));names=['United States','Singapore'];bars=ax.bar(names,country['mean']*100,color=[BLUE,TEAL],width=.55)
    _style(ax,'Singapore churn is substantially higher','Churn rate (%)');ax.set_ylim(0,80)
    for b,(num,n) in zip(bars,zip(country['sum'],country['count'])):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+2,f'{b.get_height():.1f}% ({num}/{n})',ha='center',fontweight='bold')
    fig.subplots_adjust(bottom=.19)
    fig.text(.12,.04,'Churn = is_subscriber == False. Rates are descriptive; differences do not establish causation.',color=NAVY,fontsize=9)
    _save(fig,figure_dir/'02_churn_country.png')

    tiers=['aavail_basic','aavail_premium','aavail_unlimited']
    rates=customers.groupby(['country_name','subscriber_type'])['churned'].agg(['mean','count','sum'])
    fig,ax=plt.subplots(figsize=(10,5.2));xs=np.arange(3)
    for offset,country_name,label,color in [(-.18,'united_states','United States',BLUE),(.18,'singapore','Singapore',TEAL)]:
        vals=[100*rates.loc[(country_name,t),'mean'] for t in tiers]
        bars=ax.bar(xs+offset,vals,.36,label=label,color=color)
        for b,t in zip(bars,tiers):
            r=rates.loc[(country_name,t)]
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+1.4,f"{r['mean']:.1%}\n({int(r['sum'])}/{int(r['count'])})",ha='center',fontsize=9)
    ax.set_xticks(xs,['Basic','Premium','Unlimited']);ax.legend(frameon=False)
    ax.set_ylim(0,100);_style(ax,'Market × subscription tier: largest gaps in higher tiers','Churn rate (%)')
    fig.subplots_adjust(bottom=.18)
    fig.text(.11,.045,'Singapore Premium and Unlimited show the highest observed churn rates; possible confounders are not measured.',fontsize=9,color=NAVY)
    _save(fig,figure_dir/'03_churn_market_tier.png')

    tmp=customers.copy();tmp['age_band']=pd.cut(tmp.age,bins=[-1,24,34,44,150],labels=['≤24','25–34','35–44','45+'])
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,group,order,title in [
        (axes[0],'age_band',['≤24','25–34','35–44','45+'],'Age bands'),
        (axes[1],'num_streams',None,'Streaming activity bands')]:
        if group=='num_streams':
            tmp['activity_band']=pd.qcut(tmp['num_streams'],q=4,duplicates='drop').astype(str)
            group='activity_band'
            order=sorted(tmp[group].unique(),key=lambda x:float(x.split(',')[0].lstrip('([')))
            display_labels=[f'Q{j+1}' for j in range(len(order))]
        for i,(market,label,color) in enumerate([('united_states','US',BLUE),('singapore','SG',TEAL)]):
            rate=tmp[tmp.country_name==market].groupby(group,observed=False)['churned'].mean()
            x=np.arange(len(order));ax.bar(x+(.33*(i-.5)),[100*rate.get(t,0) for t in order],.33,color=color,label=label)
        ax.set_xticks(np.arange(len(order)),display_labels if group=='activity_band' else order,rotation=0)
        ax.set_ylim(0,100);_style(ax,title,'Churn rate (%)');ax.legend(frameon=False)
    fig.suptitle('Singapore churn remains elevated across observed segments',fontsize=15,fontweight='bold',color=NAVY)
    fig.subplots_adjust(bottom=.20,wspace=.28,top=.84)
    fig.text(.10,.035,'Activity = quartiles (Q1–Q4) of stream counts. Age/activity correlations are not established causes.',fontsize=9,color=NAVY)
    _save(fig,figure_dir/'04_segment_robustness.png')

    fig,ax=plt.subplots(figsize=(9,5))
    pairs=[('Baseline (majority)',metrics['baseline_test_metrics']),('Selected model',metrics['selected_test_metrics'])]
    xs=np.arange(3);keys=['accuracy','f1','roc_auc']
    for i,(name,result) in enumerate(pairs):
        bars=ax.bar(xs+(.35*(i-.5)),[result[k] for k in keys],.35,label=name,color=BLUE if i==0 else TEAL)
        for b in bars:ax.text(b.get_x()+b.get_width()/2,b.get_height()+.02,f'{b.get_height():.2f}',ha='center',fontsize=9)
    ax.set_xticks(xs,['Accuracy','F1 (churn)','ROC AUC']);ax.set_ylim(0,1.16);ax.legend(frameon=False)
    _style(ax,f"Untouched holdout: {metrics['selected_model']} vs majority baseline",'Metric (higher is better)')
    fig.subplots_adjust(bottom=.18)
    fig.text(.11,.045,f"Train n={metrics['train_size']}; test n={metrics['test_size']}. Model chosen via training-fold ROC AUC, not test data.",fontsize=9,color=NAVY)
    _save(fig,figure_dir/'05_model_vs_baseline.png')
