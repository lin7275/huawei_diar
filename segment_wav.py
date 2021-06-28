from utils import read_sad, MySlidingWindow
from trans import kaldi_vad
from scipy.io import wavfile
import numpy as np
import math
from pyannote.core import Annotation, Segment, Timeline, SlidingWindow
import torch


def timeline2embed(model, trans, tl, wav_path, return_mfcc=False):
    sr, wav = wavfile.read(wav_path)
    if wav.ndim > 1:
        wav = wav[:, 0]
    wav_cat = []
    for seg in tl:
        wav_cat.append(wav[math.floor(seg.start * sr):math.floor(seg.end * sr)])
    wav_cat = np.concatenate(wav_cat)
    return wav2embed(wav_cat, model, trans, return_mfcc=return_mfcc)
    # mfcc = trans(wav_cat)
    # mfcc = torch.tensor(mfcc[None, :]).cuda()
    # with torch.no_grad():
    #     embed = model.extract(mfcc).to("cpu").numpy()
    # return embed


def segment_wav_and_get_embed(wav_file, model, trans, vad_config, win_config,
                              spk_turn_threshold=None,
                              return_mfcc=False):
    tl, wavs = get_wav_segments(wav_file=wav_file,
                                      sad=wav_file.replace("/wav/", "/sad/").replace(r".wav", ".lab"),
                                      vad_config=vad_config,
                                      win_config=win_config,
                                      trans=trans,
                                      spk_turn_threshold=spk_turn_threshold)
    if return_mfcc:
        return tl, [trans(wav) for wav in wavs]
    else:
        embeds = np.concatenate([wav2embed(wav, model, trans) for wav in wavs])
        return tl, embeds


def wav2embed(wav, model, trans, return_mfcc=False):
    mfcc = trans(wav)
    with torch.no_grad():
        # mfcc = torch.tensor(mfcc[None, :]).cuda()
        mfcc = mfcc[None, :].cuda()
        embed = model.extract(mfcc).to("cpu").numpy()
    if return_mfcc:
        return mfcc
    else:
        return embed


# def get_wav_segments(wav_file, sad, vad_config, win_config, trans=None, model=None, spk_turn_threshold=None, min_dur=0.1):
#     sr, wav = wavfile.read(wav_file)
#     if wav.ndim == 2:
#         wav = wav[:, 0]
#     wav = wav.astype(np.float32)
#     tls = _get_vad_wav_segments(wav, sad, vad_config, win_config, sr)
#     if spk_turn_threshold:
#         assert model is not None, 'a model has to be provided'
#         tls = speaker_turn(tls, wav, sr, spk_turn_threshold, trans, model)
#     else:
#         tls = greedy_merge(tls, min_dur=min_dur)
#     wavs = []
#     for tl in tls:
#         wav_cat = []
#         for seg in tl:
#             wav_cat.append(wav[math.floor(seg.start * sr):math.floor(seg.end * sr)])
#         wavs.append(np.concatenate(wav_cat))
#
#     return tls, wavs


def get_wav_segments(wav_file, sad, vad_config, win_config, trans=None, model=None, spk_turn_threshold=None, min_dur=0.1):
    sr, wav = wavfile.read(wav_file)
    if wav.ndim == 2:
        wav = wav[:, 0]
    wav = wav.astype(np.float32)
    tl = _get_vad_wav_segments(wav, sad, vad_config, win_config, sr)
    # if spk_turn_threshold:
    #     assert model is not None, 'a model has to be provided'
    #     tls = speaker_turn(tls, wav, sr, spk_turn_threshold, trans, model)
    # else:
    #     tls = greedy_merge(tls, min_dur=min_dur)
    wavs = []
    for seg in tl:
        wavs.append(wav[math.floor(seg.start * sr):math.floor(seg.end * sr)])

    return tl, wavs



def _get_vad_wav_segments(wav, sad, vad_config, win_config, sr):
    if vad_config['type'] == "oracle":
        if type(sad) == str:
            tl_sad = read_sad(sad)
        elif type(sad) == Timeline:
            tl_sad = sad
        else:
            raise NotImplementedError("unsupport sad")
        tl_vad_seg = split_time_line(tl_sad,
                                     max_len=win_config["duration"],
                                     min_len=0.1,
                                     duration=win_config["duration"],
                                     step=win_config["step"])
    elif vad_config['type'] == "energy":
        tl_sad = kaldi_vad(wav,
                           sample_frequency=sr,
                           **vad_config["paras"]
                           )
        tl_vad_seg = split_time_line(tl_sad.support(),
                                     max_len=win_config["duration"],
                                     min_len=0.1,
                                     duration=win_config["duration"],
                                     step=win_config["step"])
    else:
        tl_vad_seg = MySlidingWindow(duration=win_config["duration"],
                                     step=win_config["step"],
                                     end=len(wav) / sr)
        # tl_vad_seg = SlidingWindow(duration=win_config["duration"],
        #                              step=win_config["step"],
        #                            end=len(wav) / sr)
        # breakpoint()
    return tl_vad_seg


def split_time_line(tl, max_len, min_len, duration, step):
    new_tl = []
    for seg in tl:
        if seg.duration > max_len:
            for seg in MySlidingWindow(start=seg.start, end=seg.end, duration=duration, step=step):
                if seg.duration < min_len:
                    continue
                new_tl.append(seg)
            # new_tl.append([seg for seg in MySlidingWindow(start=seg.start, end=seg.end, duration=duration, step=step)])
        elif seg.duration > min_len:
            new_tl.append(seg)
    return Timeline(new_tl)