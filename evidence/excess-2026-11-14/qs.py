import sys, pandas as pd
pd.set_option('display.width',250)
for f in sys.argv[1:]:
    s=pd.read_csv(f)
    provs=[l for l in s.location.unique() if l!='CD']
    if 'CD' not in s.location.unique():
        nat=s.groupby(['target_end_date','sample']).value.sum().reset_index().assign(location='CD'); s=pd.concat([s,nat])
    x=s[s.location.isin(['CD','CD-IT','CD-NK','CD-HU','CD-TO'])]
    q=x.groupby(['location','target_end_date']).value.quantile([.05,.5,.95]).unstack().round(0)
    q['s']=q[0.5].astype(int).astype(str)+' ['+q[0.05].astype(int).astype(str)+','+q[0.95].astype(int).astype(str)+']'
    print(f.split('/')[-1]); print(q['s'].unstack(0).tail(6).to_string())
