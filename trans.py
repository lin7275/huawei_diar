from torchaudio.compliance.kaldi import (
    POVEY,
    _get_waveform_and_window_properties,
    _get_window,
    get_mel_banks,
    _get_epsilon,
    _subtract_column_mean,
)
import torch
import numpy as np
from pyannote.core import Timeline
from utils import MySlidingWindow


class KaldiFbank:
    def __init__(
        self,
        sample_frequency,
        high_freq=None,
        vad_config={},
        apply_vad=False,
        energy_floor=0,
        subtract_mean=True,
        frame_length=25,
        low_freq=20,
        num_mel_bins=40,
        snip_edges=False,
        use_energy=False,
        dither=0,
    ):
        if high_freq is None:
            if sample_frequency == 16000:
                self.high_freq = 7600
            elif sample_frequency == 8000:
                self.high_freq = 3700
            else:
                raise ValueError(f"unsupport sample frequency {sample_frequency}")
        else:
            high_freq = high_freq
        self.sample_frequency = sample_frequency
        self.energy_floor = energy_floor
        self.frame_length = frame_length
        self.low_freq = low_freq
        # self.high_freq = high_freq
        self.num_mel_bins = num_mel_bins
        self.snip_edges = snip_edges
        self.dither = dither
        self.subtract_mean = subtract_mean
        self.apply_vad = apply_vad
        self.use_energy = use_energy
        self.vad_config = vad_config
        if apply_vad:
            print("use VAD")

    def __call__(self, x, y=None):
        # import torchaudio
        return fbank_with_vad(
            **self.vad_config,
            use_energy=self.use_energy,
            waveform=torch.tensor(x[None, :].astype(np.float32)),
            apply_vad=self.apply_vad,
            sample_frequency=self.sample_frequency,
            energy_floor=self.energy_floor,
            frame_length=self.frame_length,
            low_freq=self.low_freq,
            subtract_mean=self.subtract_mean,
            high_freq=self.high_freq,
            num_mel_bins=self.num_mel_bins,
            snip_edges=self.snip_edges,
            dither=self.dither,
        ).transpose(1, 0)


def fbank_with_vad(
    waveform,
    energy_threshold=5.5,
    vad_energy_mean_scale=0.5,
    vad_proportion_threshold=0.12,
    vad_frames_context=2,
    apply_vad=False,
    blackman_coeff: float = 0.42,
    channel: int = -1,
    dither: float = 0.0,
    energy_floor: float = 1.0,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    high_freq: float = 0.0,
    htk_compat: bool = False,
    low_freq: float = 20.0,
    min_duration: float = 0.0,
    num_mel_bins: int = 23,
    preemphasis_coefficient: float = 0.97,
    raw_energy: bool = True,
    remove_dc_offset: bool = True,
    round_to_power_of_two: bool = True,
    sample_frequency: float = 16000.0,
    snip_edges: bool = True,
    subtract_mean: bool = False,
    use_energy: bool = False,
    use_log_fbank: bool = True,
    use_power: bool = True,
    vtln_high: float = -500.0,
    vtln_low: float = 100.0,
    vtln_warp: float = 1.0,
    window_type: str = POVEY,
):
    device, dtype = waveform.device, waveform.dtype

    (
        waveform,
        window_shift,
        window_size,
        padded_window_size,
    ) = _get_waveform_and_window_properties(
        waveform,
        channel,
        sample_frequency,
        frame_shift,
        frame_length,
        round_to_power_of_two,
        preemphasis_coefficient,
    )

    if len(waveform) < min_duration * sample_frequency:
        # signal is too short
        return torch.empty(0, device=device, dtype=dtype)

    # strided_input, size (m, padded_window_size) and signal_log_energy, size (m)
    strided_input, signal_log_energy = _get_window(
        waveform,
        padded_window_size,
        window_size,
        window_shift,
        window_type,
        blackman_coeff,
        snip_edges,
        raw_energy,
        energy_floor,
        dither,
        remove_dc_offset,
        preemphasis_coefficient,
    )

    # size (m, padded_window_size // 2 + 1, 2)
    fft = torch.rfft(strided_input, 1, normalized=False, onesided=True)

    power_spectrum = fft.pow(2).sum(2)  # size (m, padded_window_size // 2 + 1)
    if not use_power:
        power_spectrum = power_spectrum.pow(0.5)

    # size (num_mel_bins, padded_window_size // 2)
    mel_energies, _ = get_mel_banks(
        num_mel_bins,
        padded_window_size,
        sample_frequency,
        low_freq,
        high_freq,
        vtln_low,
        vtln_high,
        vtln_warp,
    )
    mel_energies = mel_energies.to(device=device, dtype=dtype)

    # pad right column with zeros and add dimension, size (num_mel_bins, padded_window_size // 2 + 1)
    mel_energies = torch.nn.functional.pad(
        mel_energies, (0, 1), mode="constant", value=0
    )

    # sum with mel fiterbanks over the power spectrum, size (m, num_mel_bins)
    mel_energies = torch.mm(power_spectrum, mel_energies.T)
    if use_log_fbank:
        # avoid log of zero (which should be prevented anyway by dithering)
        mel_energies = torch.max(mel_energies, _get_epsilon(device, dtype)).log()

    # if use_energy then add it as the last column for htk_compat == true else first column
    if use_energy:
        # signal_log_energy = signal_log_energy.unsqueeze(1)  # size (m, 1)
        # returns size (m, num_mel_bins + 1)
        if htk_compat:
            mel_energies = torch.cat((mel_energies, signal_log_energy.unsqueeze(1)), dim=1)
        else:
            mel_energies = torch.cat((signal_log_energy.unsqueeze(1), mel_energies), dim=1)

    if apply_vad:
        # breakpoint()
        # vad_result = vad_torch(signal_log_energy, **vad_config)
        vad_result = vad_torch(signal_log_energy,
                               energy_threshold=energy_threshold,
                               vad_energy_mean_scale=vad_energy_mean_scale,
                               vad_proportion_threshold=vad_proportion_threshold,
                               vad_frames_context=vad_frames_context,
                               )
        mask = vad_result.gt(0)
        # breakpoint()
        mel_energies = mel_energies[mask]

    mel_energies = _subtract_column_mean(mel_energies, subtract_mean)
    return mel_energies


def vad_torch(
    log_energies,
    energy_threshold=5.5,
    vad_energy_mean_scale=0.5,
    vad_proportion_threshold=0.12,
    vad_frames_context=2,
):
    # log_energies = comp_log_energy(wav)
    energy_threshold += log_energies.mean() * vad_energy_mean_scale
    left = []
    for i in range(vad_frames_context):
        if (
            log_energies[: i + 2] > energy_threshold
        ).sum() > vad_proportion_threshold * len(log_energies[: i + 2]):
            left.append(1)
        else:
            left.append(0)

    right = []
    for i in range(len(log_energies) - vad_frames_context, len(log_energies)):
        if (
            log_energies[i - 2 :] > energy_threshold
        ).sum() > vad_proportion_threshold * len(log_energies[i - 2 :]):
            right.append(1)
        else:
            right.append(0)

    overcount = (
        log_energies.unfold(0, 2 * vad_frames_context + 1, 1) > energy_threshold
    ).sum(1)
    middle = torch.where(
        overcount > vad_proportion_threshold * (2 * vad_frames_context + 1),
        torch.tensor(1),
        torch.tensor(0),
    )
    return torch.cat([torch.tensor(left), middle, torch.tensor(right)])


def kaldi_vad(
    waveform,
    return_mask=False,
    energy_threshold=5.5,
    vad_energy_mean_scale=0.5,
    vad_proportion_threshold=0.12,
    vad_frames_context=2,
    blackman_coeff: float = 0.42,
    channel: int = -1,
    dither: float = 0.0,
    energy_floor: float = 1.0,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    min_duration: float = 0.0,
    preemphasis_coefficient: float = 0.97,
    raw_energy: bool = True,
    remove_dc_offset: bool = True,
    round_to_power_of_two: bool = True,
    sample_frequency: float = 16000.0,
    # snip_edges: bool = True,
    snip_edges: bool = False,
    window_type: str = POVEY,
):
    waveform = torch.tensor(waveform[None, :].astype(np.float32))
    # frame_shift = frame_length
    (
        waveform,
        window_shift,
        window_size,
        padded_window_size,
    ) = _get_waveform_and_window_properties(
        waveform,
        channel,
        sample_frequency,
        frame_shift,
        frame_length,
        round_to_power_of_two,
        preemphasis_coefficient,
    )

    if len(waveform) < min_duration * sample_frequency:
        # signal is too short
        raise ValueError

    _, signal_log_energy = _get_window(
        waveform,
        padded_window_size,
        window_size,
        window_shift,
        window_type,
        blackman_coeff,
        snip_edges,
        raw_energy,
        energy_floor,
        dither,
        remove_dc_offset,
        preemphasis_coefficient,
    )

    vad_results = vad_torch(signal_log_energy,
    energy_threshold=energy_threshold,
    vad_energy_mean_scale=vad_energy_mean_scale,
    vad_proportion_threshold=vad_proportion_threshold,
    vad_frames_context=vad_frames_context,).gt(0)
    if return_mask:
        return vad_results
    # sw = MySlidingWindow(step=frame_length/1000, duration=frame_length/1000, end=len(waveform)/sample_frequency, start=0)
    sw = MySlidingWindow(step=frame_shift / 1000, duration=frame_length / 1000, end=len(waveform) / sample_frequency,
                         start=0)
    # print(len(waveform)/sample_frequency)
    # print(len(vad_results))
    # print(len(sw))
    # breakpoint()
    assert (len(vad_results) == len(sw) - 1) or (len(vad_results) == len(sw))
    tl = Timeline([seg for seg, vad_result in zip(sw, vad_results) if vad_result])
    return tl.support()
