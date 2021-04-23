from pyannote.core.utils.hierarchy import linkage, fcluster_auto
import numpy as np
from pyannote.core.utils.distance import l2_normalize
from scipy.cluster.hierarchy import fcluster
from pyannote.core import Annotation
import os
from pyannote.database.util import load_rttm, load_uem
from segment_wav import timeline2embed
from utils import my_load_rttm


# this class only do inner loop
def comp_linkage_mat(embeds, tls_segmented, wav_path, utt_id, linkage_method='ward'):
    Z = HAC(method=linkage_method)(embeds)
    if Z.ndim < 2:
        print('something is wrong')
        breakpoint()
    return {
        "Z": Z,
        "tls": tls_segmented,
        "wav_path": wav_path,
        'utt_id': utt_id,
    }


def get_pred(data, threshold):
    labels = fcluster(data["Z"], threshold, criterion="distance")
    pred = Annotation(uri=data["utt_id"])
    for label, seg in zip(labels, data["tls"]):
        if seg.duration <= 0.001:
            continue
        else:
            pred[seg, '_'] = int(label)
    return pred


def eval_diar(data, threshold, metric):
    pred = get_pred(data, threshold)
    eval_der(pred, data["utt_id"], data["wav_path"], metric)
    return metric


def get_hypothesized_spk_embed(data, spk_id, utt_id, model, trans, threshold, return_mfcc=False):
    pred = get_pred(data, threshold)
    groundtruth = my_load_rttm(data["wav_path"].replace(".wav", ".rttm").replace("/wav/", "/rttm/"))[utt_id]
    data['att_hypo'] = label_pred(pred, groundtruth)
    # sr, wav = wavfile.read(data["wav_path"])
    tl = data['att_hypo'].label_timeline(spk_id)
    # Todo handble spk_id is not in tl case
    if len(tl) == 0:
        print(f'spk_id {spk_id} not found in sentence {utt_id}')
        return None
    else:
        return timeline2embed(model=model,
                               wav_path=data["wav_path"],
                               tl=data['att_hypo'].label_timeline(spk_id),
                               trans=trans,
                               return_mfcc=return_mfcc)
        # return embed


def label_pred(pred, groundtruth):
    # todo groupby pred first
    ann_final = Annotation()
    for label in pred.labels():
        cluster_results = Annotation()
        for seg in pred.label_timeline(label):
            cluster_results.update(groundtruth.crop(seg))
        final_label = cluster_results.argmax()
        # cluster_results = cluster_results.support()
        for seg in cluster_results.get_timeline():
            ann = Annotation()
            ann[seg, '_'] = final_label
            ann_final.update(ann)
    return ann_final


def eval_der(
    pred,
    utt_id,
    path,
    metric,
):
    groundtruth = my_load_rttm(path.replace(".wav", ".rttm").replace("/wav/", "/rttm/"))[utt_id]
    if os.path.exists(path.replace(".wav", ".uem")):
        uem = load_uem(path.replace(".wav", ".uem"))[utt_id]
    else:
        uem = None
    metric(groundtruth, pred.support(), uem=uem)
    # breakpoint()
    return metric


class HAC:
    def __init__(self,
                 method,
                 # threshold,
                 metric='cosine',
                 normalize=False):
        self.method = method
        self.normalize = normalize
        self.metric = metric
        # self.threshold = threshold

    def __call__(self, X):
        n_samples, _ = X.shape

        if n_samples < 1:
            msg = 'There should be at least one sample in `X`.'
            raise ValueError(msg)

        elif n_samples == 1:
            # clustering of just one element
            return np.array([1], dtype=int)

        if self.normalize:
            X = l2_normalize(X)

        # compute agglomerative clustering all the way up to one cluster
        Z = linkage(X, method=self.method, metric=self.metric)

        # obtain flat clusters
        return Z

