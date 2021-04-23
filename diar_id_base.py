import numpy as np
from scipy.io import wavfile
from sklearn.metrics.pairwise import cosine_similarity
from pyannote.core import Annotation, Segment, Timeline
import pandas as pd
from pyannote.metrics.identification import (
    IdentificationErrorRate,
)
from pathlib import Path
from utils import comp_eer_sklearn
from cluster import get_hypothesized_spk_embed, comp_linkage_mat
from segment_wav import segment_wav_and_get_embed, wav2embed


class DiarWithID:
    def __init__(
        self,
        step,
        duration,
        model,
        trans,
        verf_threhold,
        tsv_file,
        vad_config,
        threshold,
        auto_enroll,
        eer_collar=None,
        cohort_tsv=None,
        spk_turn_threshold=None,
        collar=0.25,
        score_norm=None,
        metric_choice="acc",
        enroll_seg_len=None,
        subset=999999999,
    ):
        self.threshold = threshold
        # self.cluster = Cluster(linkage_method='ward')
        self.spk_turn_threshold = spk_turn_threshold
        self.collar = collar
        self.vad_config = vad_config
        self.model = model
        self.model.eval()
        self.trans = trans
        self.win_config = {"step": step, "duration": duration}
        self.metric_choice = metric_choice
        self.metric = IdentificationErrorRate(collar=self.collar)
        self.verf_threhold = verf_threhold
        self.df = pd.read_csv(
            tsv_file, sep="\t", dtype={"start": float, "end": float, "enroll_spk_id": str}
        ).iloc[:subset]


        # breakpoint()
        self.scores = []
        self.labels = []
        self.enroll_seg_len = enroll_seg_len
        self.score_norm = score_norm
        self.min = 100
        self.max = -100
        if cohort_tsv:
            self.df_cohort = pd.read_csv(cohort_tsv, sep='\t')
        else:
            self.df_cohort = None
        self.eer_collar = eer_collar
        self.auto_enroll = auto_enroll

    # def score_pair(self):
    #     for i in range(0, len(self.df), 2):
    #         se_enroll, se_test = self.df.iloc[i], self.df.iloc[i+1]
    #         data_test = self.prepare_test_sentence(se_test, se_enroll)
    #         embed_enroll = self.prepare_enroll_data(se_enroll)
    #         embed_cohort = self.prepare_cohort()
    #         if embed_enroll is not None:
    #             self.score_eer(embed_enroll, data_test, embed_cohort)
    #     comp_eer_sklearn(self.scores, self.labels)


    def score_pair(self):
        for _, se in self.df.iterrows():
            data_test = self.prepare_test_sentence(se["test_wav_path"],
                                                   se["test_rttm_path"],
                                                   se["enroll_spk_id"])
            if self.auto_enroll:
                embed_enroll = self.prepare_enroll_data(se["enroll_wav_path"],
                                                        se["enroll_sentence_id"],
                                                        se["enroll_spk_id"])
            else:
                embed_enroll = wav2embed(wavfile.read(se["enroll_wav_path"])[1],
                                         model=self.model, trans=self.trans)
            embed_cohort = self.prepare_cohort()
            self.score_eer(embed_enroll, data_test, embed_cohort)
        comp_eer_sklearn(self.scores, self.labels)

    # def score_eer(self, embed_enroll, data_test, embed_cohort=None):
    #     dists = self._score(embed_enroll, data_test["embeds"], embed_cohort)
    #     ann_pred = Annotation(uri=data_test["sentence_file_dir"])
    #     ann_groundtruth = Annotation(uri=data_test["sentence_file_dir"])
    #     for tl, dist in zip(data_test["tls_seg"], dists):
    #         for seg in tl:
    #             if seg.duration <= 0.001:
    #                 continue
    #             else:
    #                 for seg_inter, _, label in (
    #                     data_test["groundtruth"]
    #                     .crop(seg)
    #                     .itertracks(yield_label=True)
    #                 ):
    #                     if self.eer_collar:
    #                         if seg_inter.duration < self.eer_collar:
    #                             continue
    #                     ann_pred[seg_inter] = dist
    #                     ann_groundtruth[seg_inter] = label
    #                     self.scores.append(dist)
    #                     self.labels.append(label)


    def score_eer(self, embed_enroll, data_test, embed_cohort=None):
        dists = self._score(embed_enroll, data_test["embeds"], embed_cohort)
        # ann_pred = Annotation(uri=data_test["sentence_file_dir"])
        # ann_groundtruth = Annotation(uri=data_test["sentence_file_dir"])
        ann_pred = Annotation()
        ann_groundtruth = Annotation()
        for seg, dist in zip(data_test["tls_seg"], dists):
            if seg.duration <= 0.001:
                continue
            else:
                for seg_inter, _, label in (
                    data_test["groundtruth"]
                    .crop(seg)
                    .itertracks(yield_label=True)
                ):
                    if self.eer_collar:
                        if seg_inter.duration < self.eer_collar:
                            continue
                    ann_pred[seg_inter] = dist
                    ann_groundtruth[seg_inter] = label
                    self.scores.append(dist)
                    self.labels.append(label)

    @staticmethod
    def _score(embed_enroll, embed_sentence, embed_cohort=None):
        dists = cosine_similarity(embed_enroll, embed_sentence).squeeze()
        if embed_cohort is not None:
            dists_tnorm = (dists - dists.mean()) / dists.std()
            #
            cohort_dist_matrix = cosine_similarity(embed_sentence, embed_cohort)
            dists_znorm = (
                dists - cohort_dist_matrix.mean(-1)
            ) / cohort_dist_matrix.std(-1)

            dists = (dists_tnorm + dists_znorm) / 2
            # dists = dists_tnorm
        return dists

    def prepare_cohort(self):
        embed_cohort = []
        for file in self.df_cohort.sample(30).files:
            embed_cohort.append(
                wav2embed(wavfile.read(file)[1], model=self.model, trans=self.trans)
            )
        embed_cohort = np.concatenate(embed_cohort)
        return embed_cohort


    def prepare_test_sentence(
        self, wav_path, rttm_path, enroll_spk_id, return_mfcc=False,
    ):
        tls, embeds = segment_wav_and_get_embed(
            wav_file=wav_path,
            model=self.model,
            trans=self.trans,
            vad_config=self.vad_config,
            win_config=self.win_config,
            return_mfcc=return_mfcc
        )
        df_rttm = pd.read_csv(
            rttm_path, sep=" ", usecols=[3, 4, 7], names=["start", "duration", "spk_id"]
        )
        df_rttm["end"] = df_rttm["start"] + df_rttm["duration"]
        url = Path(enroll_spk_id).stem + "-" + Path(wav_path).stem
        if self.metric_choice == "eer":
            groundtruth = df2rttm(
                df_rttm, url, target_spkid=int(enroll_spk_id), encode_scheme="binary"
            )
            # breakpoint()
        else:
            groundtruth = df2rttm(
                df_rttm, url, target_spkid=enroll_spk_id, encode_scheme="only_target"
            )
        # breakpoint()
        return {
            # "groundtruth": df2rttm_with_target(sub_df, url, enroll_spkid),
            "groundtruth": groundtruth,
            "tls_seg": tls,
            # "sentence_file_dir": wav_path,
            "embeds": embeds,
        }

    def prepare_enroll_data(self, wav_path, sentence_id, spk_id):
        tls, embeds = segment_wav_and_get_embed(
            wav_file=wav_path,
            model=self.model,
            trans=self.trans,
            vad_config=self.vad_config,
            win_config=self.win_config
        )
        data = comp_linkage_mat(embeds,
                                tls,
                                wav_path=wav_path,
                                utt_id=sentence_id)
        embed = get_hypothesized_spk_embed(data,
                                           spk_id=spk_id,
                                           utt_id=sentence_id,
                                           model=self.model,
                                           trans=self.trans,
                                           threshold=self.threshold)
        return embed


def df2rttm(df, url, target_spkid, encode_scheme):
    ann = Annotation(url)
    # breakpoint()
    df["spk_id"] = df["spk_id"].astype(np.int)
    target_spkid = int(target_spkid)
    for _, row in df.iterrows():
        if encode_scheme == "only_target":
            if row["spk_id"] != target_spkid:
                continue
            else:
                ann[Segment(row.start, row.end)] = row["spk_id"]
        elif encode_scheme == "binary":
            if row["spk_id"] != target_spkid:
                ann[Segment(row.start, row.end)] = 0
            else:
                ann[Segment(row.start, row.end)] = 1
        elif encode_scheme == "multi_class":
            ann[Segment(row.start, row.end)] = row["spk_id"]
        else:
            raise ValueError(f"not support encode_scheme {encode_scheme}")
    if encode_scheme == "binary":
        for seg in ann.get_timeline().gaps():
            ann[seg] = 0
    return ann


def df2sad(df):
    return Timeline([Segment(row["start"], row["end"]) for _, row in df.iterrows()])
