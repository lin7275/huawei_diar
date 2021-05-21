import pandas as pd
from pathlib import Path
import glob
import numpy as np
from itertools import cycle
import random


def generate_trial(n_trials, save2):
    df_enroll = pd.DataFrame({'mfcc_dir': glob.glob('mfcc/enroll/*/*')})
    df_enroll['spk_id'] = df_enroll['mfcc_dir'].apply(lambda x: Path(x).parent.stem).astype(int)
    enroll_spk_ids = df_enroll['spk_id'].unique()
    df_enroll = df_enroll.set_index('spk_id')

    trials = {
        'enroll_mfcc_path': [],
        'test_mfcc_path': [],
        'test_groundtruth_path': [],
        'test_rttm_path': [],
        'test_segment_timeline_path': [],
        'enroll_spk_id': [],
    }
    test_mfcc_dirs = list(Path('mfcc/test').glob('*'))
    random.shuffle(test_mfcc_dirs)
    for _, test_mfcc_dir in zip(range(n_trials), cycle(test_mfcc_dirs)):
        spk_ids = pd.read_csv(f'groundtruth/test/{test_mfcc_dir.stem}.rttm', sep=' ', usecols=[7]).squeeze()
        # break
        spk_ids_libsph = spk_ids[spk_ids.isin(enroll_spk_ids)]
        if len(spk_ids_libsph) > 0:
            spk_ids_sampled = np.random.choice(spk_ids_libsph)
        else:
            continue
        # breakpoint()
        enroll_mfcc_dir = df_enroll.loc[spk_ids_sampled].sample().mfcc_dir
        if type(enroll_mfcc_dir) is not str:
            enroll_mfcc_dir = enroll_mfcc_dir.squeeze()
        trials['enroll_mfcc_path'].append(enroll_mfcc_dir)
        trials['test_mfcc_path'].append(str(test_mfcc_dir))
        trials['enroll_spk_id'].append(spk_ids_sampled)
        trials['test_groundtruth_path'].append(f"groundtruth/test/{test_mfcc_dir.stem}.rttm")
        trials['test_rttm_path'].append(f"groundtruth/test/{test_mfcc_dir.stem}.rttm")
        trials['test_segment_timeline_path'].append(f"segment_timeline/test/{test_mfcc_dir.stem}.rttm")
    pd.DataFrame(trials).to_csv(save2, sep='\t', index=False)


if __name__ == '__main__':
  # run it in the corpus dir (with folders such as mfcc, segment_timeline)
    generate_trial(n_trials=1000, save2='trials.tsv')
