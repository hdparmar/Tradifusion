from textwrap import wrap
import os
import csv
import keras_cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from tqdm import tqdm
from sklearn import train_test_split
from prepare_dataset import process_text, prepare_dataset
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder
from stable_diffusion_tf.stable_diffusion import StableDiffusion as StableDiffusionPy
from tensorflow.keras.callbacks import TensorBoard
from AutoEncoderKL import AutoencoderKL, kl_divergence_loss, reconstruction_loss
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

# Collate the tokenized captions into an array
tokenized_texts = np.empty((len(data_frame), MAX_PROMPT_LENGTH))
all_captions = list(data_frame["caption"].values)
for i, caption in enumerate(all_captions):
    tokenized_texts[i] = process_text(caption)

# Dataset preparation
image_paths = np.array(data_frame["image"])
tokenized_texts = np.array(tokenized_texts) 

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

# Train the AutoEncoderKL First and check the reconstrunction 
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
autoencoder = AutoencoderKL()

# Beta, for a fare-trade between losses. 
# We add weights to decide which one of the two losses to have more influence on the model.
beta = 0.7
@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        x_reconstructed, mu, log_var = autoencoder(x)
        rec_loss = reconstruction_loss(x, x_reconstructed)
        kl_loss = kl_divergence_loss(mu, log_var)
        total_loss = rec_loss + beta * kl_loss

    gradients = tape.gradient(total_loss, autoencoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))

    return total_loss, rec_loss, kl_loss

# Store the losses in a CSv file
with open('kl_rc_losses.csv', 'w', newline='') as csvfile:
    fieldnames = ['Epoch', 'Total Loss', 'Rec Loss', 'KL Loss']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# Training loop
epochs = 25  
total_images = len(training_dataset)

try:
    for epoch in range(epochs):
        progress_bar = tqdm(range(total_images), desc=f"Epoch {epoch+1}")
        for batch in training_dataset:
            x = batch['images']  
            total_loss, rec_loss, kl_loss = train_step(x)
            # Update the progress bar with losses
            loss_stats = {"Total Loss": float(total_loss.numpy()), "Rec Loss": float(rec_loss.numpy()), "KL Loss": float(kl_loss.numpy())}
            progress_bar.set_postfix(loss_stats)  

        # Save at the end of each epoch
        autoencoder.save(f"checkpoints/AutoEncoderKL_epoch_{epoch}")
        # Write it into CSV file
        with open('kl_rc_losses.csv', 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'Epoch': epoch, 'Total Loss': total_loss.numpy(), 'Rec Loss': rec_loss.numpy(), 'KL Loss': kl_loss.numpy()})
        print(f"Epoch {epoch}, Total Loss: {total_loss}, Rec Loss: {rec_loss}, KL Loss: {kl_loss}")

except MemoryError:
    print("Hit the memory wall, Reduce batch size.")
except Exception as e:
    print(f"Something unexpected happened Here's the catch: {e}")
finally:
    print("Model Trained!!")

"""
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
    #plt.show() So

    fig.savefig('loss_curve.png', dpi=300, bbox_inches='tight')

    """