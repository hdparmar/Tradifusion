from textwrap import wrap
import os
import csv
import keras_cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from sklearn import train_test_split
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder
from stable_diffusion_tf.stable_diffusion import StableDiffusion as StableDiffusionPy
from tensorflow.keras.callbacks import TensorBoard

from tensorflow import keras
from trainerClass import Trainer
    
# Load and process the dataset
data_path = "dataset"
data_frame = pd.read_csv(os.path.join(data_path, "data_1.csv"))
data_frame["image"] = data_frame["image"].apply(lambda x: os.path.join(data_path, x))
print(data_frame.head())

# Constants for text processing
PADDING_TOKEN = 49407
MAX_PROMPT_LENGTH = 77

# Load the Clip tokenizer from Keras-cv
tokenizer = SimpleTokenizer()

def process_text(caption):
    tokens = tokenizer.encode(caption)
    tokens = tokens + [PADDING_TOKEN] * (MAX_PROMPT_LENGTH - len(tokens))
    return np.array(tokens)


# Collate the tokenized captions into an array
tokenized_texts = np.empty((len(data_frame), MAX_PROMPT_LENGTH))
all_captions = list(data_frame["caption"].values)
for i, caption in enumerate(all_captions):
    tokenized_texts[i] = process_text(caption)

# Constants and preprocessing functions
RESOLUTION = 256
AUTO = tf.data.AUTOTUNE
POS_IDS = tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)
augmenter = keras.Sequential(
    layers=[
        keras_cv.layers.CenterCrop(RESOLUTION, RESOLUTION),
        keras_cv.layers.RandomFlip(),
        tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
    ]
)

text_encoder = TextEncoder(MAX_PROMPT_LENGTH)

def process_image(image_path, tokenized_text):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_png(image, 3)
    image = tf.image.resize(image, (RESOLUTION, RESOLUTION))
    return image, tokenized_text

def apply_augmentation(image_batch, token_batch):
    return augmenter(image_batch), token_batch

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
    dataset = dataset.map(apply_augmentation, num_parallel_calls=AUTO)
    dataset = dataset.map(run_text_encoder, num_parallel_calls=AUTO)
    dataset = dataset.map(prepare_dict, num_parallel_calls=AUTO)
    return dataset.prefetch(AUTO)

# Dataset preparation
image_paths = np.array(data_frame["image"])
tokenized_texts = np.array(tokenized_texts)  # Make sure this is an array

train_images, val_images, train_texts, val_texts = train_test_split(
    image_paths, tokenized_texts, test_size=0.1, random_state=42
)



print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.get_device_details(physical_devices[0])

# Define the Distribute Strategy
strategy = tf.distribute.MirroredStrategy()


# Prepare the dataset 
training_dataset = prepare_dataset(train_images, train_texts, batch_size=6 * strategy.num_replicas_in_sync)
validation_dataset = prepare_dataset(val_images, val_texts, batch_size=6 * strategy.num_replicas_in_sync)


# Check the shapes of a sample batch from the training dataset
sample_train_batch = next(iter(training_dataset))
for k in sample_train_batch:
    print("Training:", k, sample_train_batch[k].shape)

# Check the shapes of a sample batch from the validation dataset
sample_val_batch = next(iter(validation_dataset))
for k in sample_val_batch:
    print("Validation:", k, sample_val_batch[k].shape)



# Load weights from Riffusion, Define Model    
# Download the PyTorch weights for the diffusion model
diffusion_model_pytorch_weights = keras.utils.get_file(
    origin="https://huggingface.co/riffusion/riffusion-model-v1/resolve/main/riffusion-model-v1.ckpt",
    file_hash="99a6eb51c18e16a6121180f3daa69344e571618b195533f67ae94be4eb135a57",
)

with strategy.scope():
    diffusion_model = StableDiffusionPy(RESOLUTION, RESOLUTION, download_weights=False)
    diffusion_model.load_weights_from_pytorch_ckpt(diffusion_model_pytorch_weights)



if __name__ == "__main__":
    # Enable mixed-precision training if the underlying GPU has tensor cores.
    USE_MP = True
    if USE_MP:
        keras.mixed_precision.set_global_policy("mixed_float16")

    with strategy.scope():
        image_encoder = ImageEncoder()
        diffusion_ft_trainer = Trainer(
            diffusion_model=diffusion_model.diffusion_model,
            vae=tf.keras.Model(
                image_encoder.input,
                image_encoder.layers[-2].output,
            ),
            noise_scheduler=NoiseScheduler(),
            use_mixed_precision=USE_MP,
        )
    
        lr = 1e-5
        beta_1, beta_2 = 0.9, 0.999
        weight_decay = (1e-2,)
        epsilon = 1e-08

        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
        )
        diffusion_ft_trainer.compile(optimizer=optimizer, loss="mse")

    log_dir = "logs/fit"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    total_epochs = 20
    ckpt_path = "checkpoints/finetuned_itt_v3_f20.h5"
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path,
        save_weights_only=True,
        monitor="loss",
        mode="min",
    )

    callbacks = [ckpt_callback, tensorboard_callback]
    with open('losses.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Training Loss", "Validation Loss"])

        train_losses=[]
        val_losses = []
        for epoch in range(total_epochs):
            history = diffusion_ft_trainer.fit(training_dataset, validation_data=validation_dataset, epochs=1, callbacks=callbacks)
            train_loss ={":.4f"}.format(history.history['loss'][0])
            val_loss = {":.4f"}.format(history.history['val_loss'][0])
            writer.writerow([epoch+1, train_loss, val_loss])
            print(f"Epoch {epoch + 1}: Training Loss = {train_loss}, Validation Loss = {val_loss}")

    # Set up a style - ggplot gives a nice aesthetic touch
    plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(10, 6))  # Bigger size for clarity

    # Plotting the training and validation losses
    ax.plot(train_losses, 'b', linestyle='-', linewidth=2, label='Training Loss')
    ax.plot(val_losses, 'r', linestyle='--', linewidth=2, label='Validation Loss')

    # Titles, labels, and legend
    ax.set_title('Loss Curve Over Epochs', fontsize=16, fontweight='bold')
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.legend(loc='upper right', fontsize=12)

    # Displaying gridlines
    ax.grid(True, linestyle='--')

    plt.tight_layout()  # Fitted layout
    # Comment plt.show() while running the training on container
    #plt.show() 

    fig.savefig('loss_curve.png', dpi=300, bbox_inches='tight')