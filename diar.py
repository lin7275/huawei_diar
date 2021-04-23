import numpy as np
from pyannote.metrics.diarization import DiarizationErrorRate
import glob
from pathlib import Path
from cluster import  comp_linkage_mat, get_pred, eval_diar
from utils import natural_sort, load_rttm_single, get_embed_from_mfcc
from resnet import resnet101
import torch


class Reader:
    def __init__(
        self,
        model,
        skip_overlap=True,
        collar=0.25,
        linkage_method='ward',
        threshold=None,
        n_subset=None,
        blind_trial=False,
    ):
        self.gpu_id = None
        self.n_subset = n_subset
        self.blind_trial = blind_trial
        self.linkage_method = linkage_method
        self.collar = collar
        self.skip_overlap = skip_overlap
        self.model = model
        self.model.eval()
        if threshold:
            self.threshold = [threshold]
            self.metric = {threshold: DiarizationErrorRate(skip_overlap=skip_overlap, collar=collar)}
        else:
            self.threshold = np.linspace(0, 9, 40) if linkage_method == 'ward' else np.linspace(0, 2, 40)
            self.metric = {threshold: DiarizationErrorRate(skip_overlap=skip_overlap, collar=collar)
                           for threshold in self.threshold}

    def read(self, read_dir):
        for tl_file in glob.glob(f"{read_dir}/segment_timeline/*.rttm"):
            tl = load_rttm_single(tl_file).get_timeline()
            mfcc_path = tl_file.replace("segment_timeline", "mfcc").replace(".rttm", "")
            wav_path = tl_file.replace("segment_timeline", "wav").replace(".rttm", ".wav")
            embeds = get_embed_from_mfcc(mfcc_path, self.model)
            data = comp_linkage_mat(embeds, tl, wav_path, Path(mfcc_path).stem)

            for threshold in self.threshold:
                eval_diar(data, threshold, self.metric[threshold])

        best_der = 1000
        best_der_report = []
        best_threshod = 0
        for threshold, metric in self.metric.items():
            der = metric.report().loc["TOTAL", "diarization error rate"].squeeze()
            if der < best_der:
                best_der = der
                best_der_report = metric.report()
                best_threshod = threshold
        print(f"best_threshod is {best_threshod}")
        print(best_der_report)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_file', required=True)
    parser.add_argument('-c', '--corpus_dir', required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.model_file, map_location="cpu")
    model = resnet101(n_classes=7323)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Todo load checkpoint
    model = model.cuda()
    reader = Reader(
        model=model,
    )
    reader.read(args.corpus_dir)