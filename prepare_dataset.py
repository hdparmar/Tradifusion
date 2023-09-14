import tensorflow as tf
import numpy as np
from scipy.ndimage import zoom
import librosa
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer

# Constants for text processing
PADDING_TOKEN = 49407
MAX_PROMPT_LENGTH = 77

# Load the Clip tokenizer from Keras-cv
tokenizer = SimpleTokenizer()
text_encoder = TextEncoder(MAX_PROMPT_LENGTH)

def process_text(caption):
    tokens = tokenizer.encode(caption)
    tokens = tokens + [PADDING_TOKEN] * (MAX_PROMPT_LENGTH - len(tokens))
    return np.array(tokens)



# Constants and preprocessing functions
RESOLUTION = 256
AUTO = tf.data.AUTOTUNE
POS_IDS = tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)

def time_stretch(image, rate=1):
    stretched = zoom(image, (rate, 1, 1), order=1)
    if rate < 1:
        # If we've compressed it, let's pad it to the original size
        pad_shape = [RESOLUTION - stretched.shape[0], stretched.shape[1], stretched.shape[2]]
        stretched = np.pad(stretched, [(0, pad_shape[0]), (0, pad_shape[1]), (0, pad_shape[2])], 'constant')
    else:
        # If we've stretched it, let's crop it to the original size
        stretched = stretched[:RESOLUTION, :, :]
    return stretched

def pitch_shift(image, rate=1):
    shifted = zoom(image, (1, rate, 1), order=1)
    if rate < 1:
        pad_shape = [shifted.shape[0], RESOLUTION - shifted.shape[1], shifted.shape[2]]
        shifted = np.pad(shifted, [(0, pad_shape[0]), (0, pad_shape[1]), (0, pad_shape[2])], 'constant')
    else:
        shifted = shifted[:, :RESOLUTION, :]
    return shifted


def custom_augmentation(image_batch, token):
    print("shape of image batch: ", tf.shape(image_batch))

    # Time Stretch
    def time_stretch_tf(image):
        return tf.py_function(func=time_stretch, inp=[image], Tout=tf.float32)

    def pitch_shift_tf(image):
        return tf.py_function(func=pitch_shift, inp=[image], Tout=tf.float32)

    image_batch = tf.map_fn(time_stretch_tf, image_batch, dtype=tf.float32)
    image_batch = tf.map_fn(pitch_shift_tf, image_batch, dtype=tf.float32)

    # Rescaling
    image_batch = (image_batch / 127.5) - 1

    return image_batch, token

@tf.function
def tf_apply_augmentation(image_batch, token_batch):
    augmented_images = tf.py_function(
        func=custom_augmentation,
        inp=image_batch,
        Tout=[tf.float32]
    )[0]
    return augmented_images, token_batch

def calculate_deltas(mel_spectrogram):
    delta = librosa.feature.delta(mel_spectrogram)
    delta_delta = librosa.feature.delta(mel_spectrogram, order=2)
    return delta, delta_delta



def process_image(image_path, tokenized_text):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_png(image, 1)
    image = tf.image.resize(image, (RESOLUTION, RESOLUTION))
    # Replicate the single channel to make it 3-channel
    image = tf.repeat(image, repeats=3, axis=-1)

    def numpy_ops(image_np, tokenized_text_np):
        delta, delta_delta = calculate_deltas(image_np[:,:,0])  # Assuming you want to calculate deltas for the first channel
        three_channel_image = np.stack([image_np[:,:,0], delta, delta_delta], axis=-1)
        return three_channel_image, tokenized_text_np

    image_out, text_out = tf.py_function(
        numpy_ops, [image, tokenized_text], [tf.float32, tf.float64]
    )

    # Explicitly set the shape of the output tensors
    image_out.set_shape([RESOLUTION, RESOLUTION, 3])
    text_out.set_shape([MAX_PROMPT_LENGTH])

    return image_out, text_out

def run_text_encoder(image_batch, token_batch):
    return (
        image_batch,
        token_batch,
        text_encoder([token_batch, POS_IDS], training=False),
    )

def prepare_dict(image_batch, token_batch, encoded_text_batch):
    return {
        "images": image_batch,
        "tokens": token_batch,
        "encoded_text": encoded_text_batch,
    }

def prepare_dataset(image_paths, tokenized_texts, batch_size=1):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, tokenized_texts))
    dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.map(process_image, num_parallel_calls=AUTO).batch(batch_size)
    dataset = dataset.map(custom_augmentation, num_parallel_calls=AUTO)
    dataset = dataset.map(run_text_encoder, num_parallel_calls=AUTO)
    dataset = dataset.map(prepare_dict, num_parallel_calls=AUTO)
    return dataset.prefetch(AUTO)
