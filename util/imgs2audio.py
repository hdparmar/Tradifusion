import io
import typing as T
import numpy as np
from PIL import Image
import pydub
from scipy.io import wavfile
from scipy.signal.windows import tukey
import torch
import torchaudio
import argparse
import matplotlib.pyplot as plt


def spectrogram_from_image(
        image: Image.Image,
        max_volume: float = 50,
        power_for_image: float = 0.25
) -> np.ndarray:
    """
    Compute a spectrogram magnitude array from a spectrogram image.

    TODO(hayk): Add image_from_spectrogram and call this out as the reverse.
    """
    # Convert to a numpy array of floats
    data = np.array(image).astype(np.float32)
    # Flip Y take a single channel
    if len(data.shape) < 3:
        data = data[::-1]
    else:
        data = data[::-1, :, 0]
    # Invert
    data = 255 - data
    # Rescale to max volume
    data = data * max_volume / 255
    # Reverse the power curve
    data = np.power(data, 1 / power_for_image)
    return data

def waveform_from_spectrogram(
    Sxx: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    num_samples: int,
    sample_rate: int,
    mel_scale: bool = True,
    n_mels: int = 512,
    max_mel_iters: int = 200,
    num_griffin_lim_iters: int = 32,
    device: str = "cuda:0",
) -> np.ndarray:
    """
    Reconstruct a waveform from a spectrogram.

    This is an approximate inverse of spectrogram_from_waveform, using the Griffin-Lim algorithm
    to approximate the phase.
    """
    Sxx_torch = torch.from_numpy(Sxx).to(device)

    # TODO(hayk): Make this a class that caches the two things

    if mel_scale:
        mel_inv_scaler = torchaudio.transforms.InverseMelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=60,
            f_max=8500,
            n_stft=n_fft // 2 + 1,
            norm=None,
            mel_scale="htk",
            max_iter=max_mel_iters,
        ).to(device)

        Sxx_torch = mel_inv_scaler(Sxx_torch)

    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=1.0,
        n_iter=num_griffin_lim_iters,
    ).to(device)

    waveform = griffin_lim(Sxx_torch).cpu().numpy()

    return waveform

def display_results(original_image, waveform, windowed_waveform, sample_rate):
    """
    Displays the original image, waveform, and windowed waveform in a subplot.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # Display original image
    axs[0].imshow(np.array(original_image))
    axs[0].set_title("Original Image")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Frequency: Mel-scale")

    # Display waveform
    #axs[1].plot(waveform)
    axs[1].set_title("Waveform")
    time_axis = np.arange(waveform.size) / sample_rate
    axs[1].plot(time_axis, waveform)
    axs[1].set_xlabel("Time (s)")

    # Display windowed waveform
    #axs[2].plot(windowed_waveform)
    axs[2].set_title("Windowed Waveform")
    time_axis = np.arange(windowed_waveform.size) / sample_rate
    axs[2].plot(time_axis, windowed_waveform)
    axs[2].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


# Function to calculate alpha for the Tukey window
def calculate_alpha(sample_rate, taper_duration_ms, total_length_samples):
    taper_length_samples = int(taper_duration_ms * sample_rate / 1000)
    alpha = (taper_length_samples * 2) / total_length_samples
    print(f"Calculated alpha for Tukey window: {alpha:.2f}")
    return alpha


def wav_bytes_from_spectrogram_image(image: Image.Image, duration: int, nmels: int, maxvol: int, power_for_image: float, device: str ="cuda:0", display=False) -> T.Tuple[io.BytesIO, float]:
    """
    Reconstruct a WAV audio clip from a spectrogram image. Also returns the duration in seconds.
    """

    max_volume = maxvol
    # power_for_image = 0.25
    Sxx = spectrogram_from_image(image, max_volume=max_volume, power_for_image=power_for_image)

    sample_rate = 44100  # [Hz]
    clip_duration_ms = duration  # [ms]

    bins_per_image = 512
    n_mels = nmels

    # FFT parameters
    window_duration_ms = 100  # [ms]
    padded_duration_ms = 400  # [ms]
    step_size_ms = 10  # [ms]

    # Derived parameters
    num_samples = int(image.width / float(bins_per_image) * clip_duration_ms) * sample_rate
    n_fft = int(padded_duration_ms / 1000.0 * sample_rate)
    hop_length = int(step_size_ms / 1000.0 * sample_rate)
    win_length = int(window_duration_ms / 1000.0 * sample_rate)

    samples = waveform_from_spectrogram(
        Sxx=Sxx,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        num_samples=num_samples,
        sample_rate=sample_rate,
        mel_scale=True,
        n_mels=n_mels,
        max_mel_iters=200,
        num_griffin_lim_iters=32,
        device=device,
    )

    print(f"Waveform length (in samples): {len(samples)}")
    print(f"Sample rate: {sample_rate}")

     # Calculate alpha for the Tukey window
    taper_duration_ms = 100  # or 200?
    alpha = calculate_alpha(sample_rate, taper_duration_ms, len(samples))

    # Apply the Tukey window to the waveform
    print("Applying the Tukey window to the waveform...")
    window = tukey(len(samples), alpha=alpha)
    windowed_samples = samples * window

    if display:
        display_results(image, samples, windowed_samples, sample_rate)
    wav_bytes = io.BytesIO()
    wavfile.write(wav_bytes, sample_rate, windowed_samples.astype(np.int16))
    wav_bytes.seek(0)

    duration_s = float(len(windowed_samples)) / sample_rate
    print(f"Audio duration (in seconds): {duration_s:.2f}")

    return wav_bytes, duration_s

def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet.
    """
    if not filename.endswith('.wav'):
        filename += '.wav'

    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input file to process, anything that FFMPEG supports, but wav and mp3 are recommended")
parser.add_argument("-o", "--output", help="Output Image")
parser.add_argument("-d", "--duration", default=5119, help="Image duration")
parser.add_argument("-m", "--maxvol", default=100, help="Max Volume, 255 for identical results")
parser.add_argument("-p", "--powerforimage", default=0.25, help="Power for Image")
parser.add_argument("-n", "--nmels", default=512, help="n_mels to use for Image, basically width. Higher = more fidelity")
parser.add_argument("--display", action="store_true", help="Display original image and waveforms")
args = parser.parse_args()

# Main execution
if __name__ == "__main__":
    filename = args.input
    image = Image.open(filename)
    wav_bytes, duration_s = wav_bytes_from_spectrogram_image(image, duration=int(args.duration), nmels=int(args.nmels), maxvol=int(args.maxvol), power_for_image=float(args.powerforimage), device="cpu", display=args.display)
    write_bytesio_to_file(args.output, wav_bytes)