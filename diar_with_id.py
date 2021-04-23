from diar_id_base import DiarWithID
import torch
from utils import my_load_rttm, load_rttm_single
from utils import get_embed_from_mfcc, comp_eer_sklearn


class Reader(DiarWithID):
    def score_pair_from_disk(self):
        for _, se in self.df.iterrows():
            if not self.auto_enroll:
                embed_enroll = self.read_enroll_data_from_disk(se)
            else:
                embed_enroll = self.read_enroll_data_from_disk(se)
            # if embed_enroll is None:
            #     print('skip')
            #     continue
            data_test = self.read_test_data_from_disk(se)
            embed_cohort = self.prepare_cohort()
            # embed_cohort = None
            self.score_eer(embed_enroll, data_test, embed_cohort)

        self.labels = [int(label) for label in self.labels]
        comp_eer_sklearn(self.scores, self.labels)

    def read_enroll_data_from_disk(self, se):
        return get_embed_from_mfcc(se["enroll_mfcc_path"], self.model)

    def read_test_data_from_disk(self, se):
        embeds = get_embed_from_mfcc(se["test_mfcc_path"], self.model)
        groundtruth = load_rttm_single(se["test_groundtruth_path"])
        tl = load_rttm_single(se["test_segment_timeline_path"]).get_timeline()
        return {
            "groundtruth": groundtruth,
            "tls_seg": tl,
            "embeds": embeds,
        }


if __name__ == '__main__':
    import argparse
    from trans import KaldiFbank
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_file', required=True)
    parser.add_argument('-t', '--tsv_file', required=True)
    args = parser.parse_args()

    from resnet import resnet101
    checkpoint = torch.load(args.model_file, map_location="cpu")
    model = resnet101(n_classes=7323)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    reader = Reader(
        # tsv_file="/home8a/wwlin/corpus/respk_clean/all_new_clean_double_cluster_libsph.tsv",
        # tsv_file="/home8a/wwlin/corpus/respk_clean/new_correct_enroll_setence_trial.tsv",
        # cohort_tsv="/home8a/wwlin/corpus/respk_music/libsph_cohort.tsv",
        cohort_tsv='cohort_list.tsv',
        # tsv_file="/home8a/wwlin/corpus/respk_clean/manual_enroll_list_for_disk_diar.tsv",
        tsv_file=args.tsv_file,
        # auto_enroll=True,
        trans=KaldiFbank(sample_frequency=16000),
        auto_enroll=True,
        verf_threhold=-0.1,
        model=model,
        step=1.5,
        duration=1.5,
        # step=1,
        # duration=1,
        vad_config={
            "type": "energy",
            "paras": {
                "energy_threshold": 5.5,
                "vad_energy_mean_scale": 0.5,
                "vad_proportion_threshold": 0.12,
                "vad_frames_context": 2,
            },
        },
        # subset=200,
        metric_choice="eer",
        score_norm=True,
        threshold=1.38,
        eer_collar=0.3,
        # enroll_seg_len=4,
    )
    reader.score_pair_from_disk()
