import pandas as pd
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

def comp_eer_sklearn(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)

    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    print(f'EER is {eer*100} and threshold is {thresh}')
    return eer*100, thresh

df1 = pd.read_csv('scores_densenet.tsv', sep='\t')
df2 = pd.read_csv('scores_resnet.tsv', sep='\t')
comp_eer_sklearn(df1['score']+df2['score'], df1['label'])