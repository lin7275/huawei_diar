import pandas as pd
import glob
from pathlib import Path
import numpy as np

def get_sentence_with_libsph_spk(sentence_file_dir, sentence_spkid_info_file):
    return (
        pd.read_csv(
            sentence_spkid_info_file,
            sep=" ",
            names=["utt_id", "sentence_id"],
            usecols=[0, 1],
        )
        .loc[lambda x: x.utt_id.str.contains(r"^\d+-\d+-\d+$")]
        .assign(enroll_spk_id=lambda x: x.utt_id.str.split("-").str[0])
        .assign(
            test_wav_path=lambda x: sentence_file_dir + "/" + x.sentence_id + ".wav"
        )
    )


def read_libsph_spk_info(spk_file_dir):
    return (
        pd.DataFrame(
            {"enroll_wav_path": glob.glob(f"{spk_file_dir}/**/*.wav", recursive=True)}
        )
        .assign(
            spk_id=lambda x: x.enroll_wav_path.apply(lambda x: Path(x).parents[1].stem)
        )
        .assign(utt_id=lambda x: x.enroll_wav_path.apply(lambda x: Path(x).stem))
    )


def sample_true_enroll_file(df_grp_sentence, df_enroll_spk):
    # Todo make sure the sample utt is not the same as test utt
    # for each test utt sample a libsph file that has the same spk in the test file
    # For example test file 2_2_00000.wav have libsph spk 1355 and 332, then we will sample a utt from 1355 or 332
    mask = df_enroll_spk.spk_id.isin(df_grp_sentence.enroll_spk_id)
    # breakpoint()
    spkid2path = sample_enroll(df_enroll_spk[mask], df_grp_sentence)
    # spkid2path = (
    #     df_enroll_spk[mask]
    #     .groupby("spk_id")
    #     .sample()
    #     .set_index("spk_id")
    #     .enroll_wav_path
    # )
    df_grp_sentence["enroll_wav_path"] = df_grp_sentence.enroll_spk_id.map(spkid2path)
    ##junk
        # if df_smp.utt_id.isin(df_grp_sentence.utt_id).squeeze():

        # print(sub_df.sample())
    return df_grp_sentence


def sample_enroll(df_enroll_spk_mased, df_grp_sentence):
    spkid2path = []
    df = df_enroll_spk_mased
    for _, sub_df in df.groupby("spk_id"):
        # make sure sub_df has more than two utt per spk
        # breakpoint()
        assert (sub_df.groupby('spk_id').utt_id.nunique() >= 2).any()
        df_smp = df_grp_sentence.iloc[[0]]
        while df_smp.utt_id.isin(df_grp_sentence.utt_id).squeeze():
            df_smp = sub_df.sample()
        spkid2path.append(df_smp)
    return pd.concat(spkid2path, ignore_index=True).set_index("spk_id").enroll_wav_path


def read_libsph_spkid_mapper(segments_spkid_file):
    id_mapper = pd.read_csv(
        segments_spkid_file, sep=" ", usecols=[0, 4], names=["libsph_id", "diar_id"]
    )
    id_mapper = id_mapper[id_mapper.libsph_id.str.match(r"\d+-\d+-\d+")]
    id_mapper["libsph_id"] = id_mapper["libsph_id"].str.split("-").str[0]
    id_mapper = id_mapper.drop_duplicates()
    return id_mapper.set_index("libsph_id").diar_id


def generate_trial(
    enroll_wav_dir,
    test_wav_dir,
    test_spkid_info_file,
    max_entry=None,
    save2=None,
):
    df_enroll_spk = read_libsph_spk_info(enroll_wav_dir)
    df_libsph = get_sentence_with_libsph_spk(
        test_wav_dir, test_spkid_info_file
    )
    # df_libsph include all libsph segments in the family corpus data
    # the family corpus has around 9k utt, but sone utt may contain several segments from libsph
    # so df_libsph may be 3-4 times larger than the total amount of family corpus

    # for each libsph segment in the family corpus sample the same spk in the libsph data
    df_trial_pair = (
        df_libsph.groupby("sentence_id")
        .apply(sample_true_enroll_file, df_enroll_spk)
        .sample(frac=1)
    )

    # breakpoint()
    # drop enroll and test utt that appear twice
    df_trial_pair = (
        df_trial_pair.drop_duplicates("enroll_wav_path")
        .drop_duplicates("test_wav_path")
        .iloc[:max_entry]
    )
    df_trial_pair["test_rttm_path"] = df_trial_pair["test_wav_path"].str.replace("wav", "rttm")
    enroll_spkid_mapper = read_libsph_spkid_mapper(test_spkid_info_file)
    # breakpoint()
    df_trial_pair["enroll_spk_id"] = df_trial_pair["enroll_spk_id"].map(enroll_spkid_mapper)
    # assert not df_trial_pair.isnull().values.any()
    if df_trial_pair.isnull().values.any():
        breakpoint()
    if save2:
        df_trial_pair.to_csv(save2, sep="\t", index=False)
    return df_trial_pair


def generate_list_with_mfcc(
    enroll_wav_dir,
    test_wav_dir,
    test_spkid_info_file,
    write_dir,
    max_entry=None,
    save2=None,
    auto_enroll=False,
):
    '''
    :param enroll_wav_dir: enroll speech will be selected from here
    :param test_wav_dir: test speech will be seleted from here.
    :param test_spkid_info_file: a file that encode test speech spkid info
    :param write_dir: the directory where features, segment timeline and rttm will be written.
    if you want to use relative path enter a empty string here
    :param max_entry: max trial number
    :param save2: save trial file to
    :param auto_enroll: todo
    :return: None
    '''
    df = generate_trial(
        enroll_wav_dir, test_wav_dir, test_spkid_info_file, max_entry
    )
    df["enroll_mfcc_path"] = df["enroll_wav_path"].apply(
        lambda x: f"{write_dir}mfcc/enroll/{Path(x).stem}"
    )
    if auto_enroll:
        df["enroll_segment_timeline_path"] = df["enroll_wav_path"].apply(
            lambda x: f"{write_dir}segment_timeline/enroll/{Path(x).stem}.rttm"
        )

    df["test_mfcc_path"] = df["test_wav_path"].apply(
        lambda x: f"{write_dir}mfcc/test/{Path(x).stem}"
    )
    df["test_groundtruth_path"] = df["test_wav_path"].apply(
        lambda x: f"{write_dir}groundtruth/test/{Path(x).stem}.rttm"
    )
    df["test_segment_timeline_path"] = df["test_wav_path"].apply(
        lambda x: f"{write_dir}segment_timeline/test/{Path(x).stem}.rttm"
    )
    col = [
        "enroll_mfcc_path",
        "test_mfcc_path",
        "test_groundtruth_path",
        "test_rttm_path",
        "test_segment_timeline_path",
        "enroll_wav_path",
        "test_wav_path",
        "enroll_spk_id",
    ]
    # if auto_enroll:
    #     col.extend(["enroll_sentence_id", "enroll_segment_timeline_path"])
    if save2:
        df.to_csv(save2, sep="\t", columns=col, index=False)
    return df


if __name__ == "__main__":
    df = generate_list_with_mfcc(
        enroll_wav_dir="/home10b/wwlin/corpus/LibriSpeech_wav/train-clean-100",
        test_wav_dir="/home10b/wwlin/corpus/respk_clean/wav",
        write_dir="",
        test_spkid_info_file="/home10b/wwlin/corpus/respk_clean//docs/segments_spkid",
        save2=None,
    )
