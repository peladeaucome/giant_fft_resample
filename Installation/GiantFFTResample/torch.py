import torch
import numpy as np
from .resample import get_P_and_Q, Resampler


def get_taper(length_sp: int, device: torch.device) -> torch.Tensor:
    n = torch.arange(length_sp, device=device)
    return (n * (np.pi * 0.5 / length_sp)).cos().square()


class TorchResampler(Resampler):
    def resample(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        original_shape = x.shape
        Nin = original_shape[-1]

        x_2d = x.reshape(-1, Nin)
        num_rows = x_2d.shape[0]

        M = int(2 * np.ceil((Nin + self.in_samplerate * 0.5) / (self.Q * 2)))

        Nx = M * self.Q
        Ny = M * self.P

        nbins_in = Nx // 2 + 1
        nbins_out = Ny // 2 + 1

        x_fft = torch.fft.rfft(x_2d, n=Nx, dim=-1)

        if self.out_samplerate > self.in_samplerate:
            Ntaper = int(self.lowpass_ratio * nbins_in)
            taper = get_taper(Ntaper, device=device).reshape(1, Ntaper)
            x_fft[:, -Ntaper:] *= taper
            y_fft = torch.zeros(num_rows, nbins_out, dtype=torch.complex64, device=device)
            y_fft[:, :nbins_in] = x_fft
        else:
            y_fft = torch.zeros(num_rows, nbins_out, dtype=torch.complex64, device=device)
            y_fft[:, :nbins_out] = x_fft[:, :nbins_out]
            y_fft[:, -1] = y_fft[:, -1].real.clone()

            Ntaper = int(self.lowpass_ratio * nbins_out)
            taper = get_taper(Ntaper, device=device).reshape(1, Ntaper)
            y_fft[:, -Ntaper:] *= taper

        y = torch.fft.irfft(y_fft, n=Ny, dim=-1)

        y = y * self.P / self.Q

        Nout = int(Nin * self.P / self.Q)
        y = y[:, :Nout]
        return y.reshape(original_shape[:-1] + (Nout,))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.resample(x)


def resample(
    x: torch.Tensor, in_samplerate: int, out_samplerate: int, lowpass_ratio: float = 0.05
) -> torch.Tensor:
    """
    Resample a tensor along its last axis.

    args:
    -----
     - `x` torch.Tensor, shape (..., Nsamples): signal to resample
     - `in_samplerate` input samplerate
     - `out_samplerate` output samplerate
     - `lowpass_ratio` ratio of frequencies lowpassed. The lower the value, the higher the lowpass frequency
    """
    RS = TorchResampler(in_samplerate, out_samplerate, lowpass_ratio)
    return RS(x)
