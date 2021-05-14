import pandas as pd
import numpy as np
from pathlib import Path


def get_same_spk_in_multi_utt(family_corpus_wav_dir, save2, write_dir, sample_per_spk=400, max_entry=None):
    '''
    :param family_corpus_wav_dir: family corpus directory containing wav and some docs
    :param save2: save trial to
    :param write_dir: dir save features, timeline and rttm
    :param sample_per_spk: how many sample per spk
    :param max_entry: ignore
    :return:
    '''
    family_corpus_wav_dir = family_corpus_wav_dir + '/'
    segments_spkid_file = f"{Path(family_corpus_wav_dir).parent}/docs/segments_spkid"
    df = pd.read_csv(segments_spkid_file, sep=' ', usecols=[0, 1, 4],
                     names=["spk_ids_string", "sentence_id", "start", "end", "spk_id"],
                     dtype=str)
    df['wav_path'] = family_corpus_wav_dir + df['sentence_id'] + '.wav'
    df['rttm_path'] = family_corpus_wav_dir.replace('/wav', '/rttm') + df['sentence_id'] + '.rttm'
    # only need libsph data
    df_enroll, df_test = [], []
    df = df[df.spk_ids_string.str.match(r"\d+-\d+-\d+")].drop('spk_ids_string', axis=1)
    for spk_id, spk_grp in df.groupby("spk_id"):
        x = spk_grp.sample(sample_per_spk, replace=True)
        df_enroll.append(x.iloc[::2])
        df_test.append(x.iloc[1::2])
    df_enroll = pd.concat(df_enroll)
    df_enroll.columns = 'enroll_' + df_enroll.columns
    df_test = pd.concat(df_test)
    df_test.columns = 'test_' + df_test.columns
    df = pd.concat([df_enroll.reset_index(drop=True), df_test.reset_index(drop=True)], axis=1)
    df = df.sample(frac=1)

    # writing path
    df["enroll_mfcc_path"] = df["enroll_wav_path"].apply(
        lambda x: f"{write_dir}mfcc/enroll/{Path(x).stem}")

    df["enroll_segment_timeline_path"] = df["enroll_wav_path"].apply(
        lambda x: f"{write_dir}segment_timeline/enroll/{Path(x).stem}.rttm")

    df["test_mfcc_path"] = df["test_wav_path"].apply(
        lambda x: f"{write_dir}mfcc/test/{Path(x).stem}")
    df["test_groundtruth_path"] = df["test_wav_path"].apply(
        lambda x: f"{write_dir}groundtruth/test/{Path(x).stem}.rttm")
    df["test_segment_timeline_path"] = df["test_wav_path"].apply(
        lambda x: f"{write_dir}segment_timeline/test/{Path(x).stem}.rttm")
    col = ["enroll_mfcc_path",  "test_mfcc_path", "test_groundtruth_path",
            "test_rttm_path",
            "test_segment_timeline_path", "enroll_wav_path", "test_wav_path", "enroll_spk_id"]
    col.extend(["enroll_sentence_id", "enroll_segment_timeline_path"])
    # breakpoint()
    # df.sample(frac=1).iloc[:1000].to_csv(save2, sep='\t', columns=col, index=False)
    df.drop_duplicates(['enroll_mfcc_path']).drop_duplicates(['test_mfcc_path']).sample(frac=1).iloc[:max_entry].to_csv(save2, sep='\t', columns=col, index=False)


if __name__ == '__main__':
    # df = get_same_spk_in_multi_utt("/home8a/wwlin/corpus/respk_clean/wav",
    #                                "/home8a/wwlin/corpus/respk_clean/final_auto_enroll_libsph.tsv",
    #                                write_dir='/home8a/wwlin/corpus/corpus_for_huawei/diar_auto_enroll_id')
    get_same_spk_in_multi_utt("/home10b/wwlin/corpus/respk_music/wav",
                              "/home10b/wwlin/corpus/respk_music/auto_enroll.tsv",
                              write_dir='')
