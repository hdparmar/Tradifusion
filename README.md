# Tradi-fusion
Tradi-fusion Refined: Evaluating and tuning the Diffusion Model for Irish Traditional Music.

# Focus of the project
The project will investigate the following research questions:
The main question 
- Can the Diffusion model produce good results for generating Irish Traditional music?
    - Yeaapp!!
- How close can we get with Diffusion Model(s)?
    - Pretty close!
- What challenges are involved in fine-tuning the model for Irish Traditional music?
    - Dataset creation, time taken to train for good results and resources.
- Can the fine-tuned model generate Irish Traditional music that is comparable in quality to
human-composed Irish Traditional music? If not, what is the reason?
    - It can produce music similar to Irish Traditional Music, comparable yes but not in the same quality as human-composed music mainly because of downstream methods to conver spectrogram back to audio.

The project is of interest to the field of Music Technology, Culture and Generative AI. It can be of interest to researchers, practitioners, and enthusiasts in these fields who are interested in exploring the possibilities of AI-generated music and its potential applications and limitations.

## Background 


## Dependencies 🛠️
I have useed keras-cv version 0.5.1 becuase it supports tensorflow version 2.11.0.
Make sure to install dependencies using `pip install -r requirements.txt`

## Training 🏋🏽
Train on 512 x 512 Spectrograms of Irish Traditional Tunes. 

The dataset produces 512x512 images!
Main Dataset Card (hugging-face) [hdparmar/irish-traditional-tunes](https://huggingface.co/datasets/hdparmar/irish-traditional-tunes)


The fine-tuning training was done on multiple GPUs (NVIDIA GeForce RTX 3090) with the use of [NVIDIA NGC Tensorflow container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow). 

The advantage of using is to avoid erros with cudnn library and errors concering not finding libdevice library under `/usr/local/cuda`. It also helps with the matching the compatible Tensorflow verison with CUDA and cuDNN. 

### Running on Jupyter Lab 📓
The file `finetune_itt.ipynb` can be used to play with the model, visualise the results and tweak the parameters using the config file `spectrogram.yaml` and see the outcome. Once you are satisfied with that, you can go forward and make a training script.


### Running the Script ▶️
The file `trainerClass.py` is for defining a custom Keras Model for training and testing routines for diffusion model.
The file `train.py` is a trainig script that you can run within a NGC Tensorflow 23.03 container to save the model weights, losses (in a CSV file), and TensorBoard logs.

Provided you can in the NGC Tensorflow container which is running and you have mounted the dataset into Workspace.
You can run the script by `python3 train.py`

## Checkpoints ⛳︎
The various checkpoints will be availble on Hugging Face.
Training files and metrics [TFM](https://huggingface.co/hdparmar/tradfusion-v2-training-files)
Main Model: [tradfusion-v2](https://huggingface.co/hdparmar/tradfusion-v2) 

## To-Do Progress
![](https://geps.dev/progress/45)
- [x] Background study
- [x] Build Dataset 
- [x] [Fine-tune using the Dreambooth approach](https://dreambooth.github.io/)
- [x] Fine-tune in the traditional manner using NGC Container
- [x] Study the model from TensorBoard
- [x] Explore tweaking to produce comparable Irish Traditional Tunes
- [x] Looping and Interpolation 
- [x] Possibility to train on 512x512 images with Gradient Accumulation (but will the forward pass fit in memory!??)
- [ ] [Fine-tune using the Textual-inversion method](https://textual-inversion.github.io/) (Is the classifier approach a way to go!?)
- [x] Come up with a some tweaks and prepare a novel pipeline
- [ ] Deploy a Website for live inference (Use streamlit to deploy the best generated checkpoint)

## Acknowledgments

This project uses the following resources:
- LambdaLabsML's [stable-diffusion-finetuning](https://github.com/LambdaLabsML/examples/tree/main/stable-diffusion-finetuning) to train the model. 
- RunPod for GPUs: [2 x RTX 6000 Ada](https://www.runpod.io/)
- Dataset Obtained using [riffusion-manilab](https://github.com/hdparmar/riffusion-manilab) and adoting it for this project (check the dataset folder).
- Inference pipeline was adopted with courses, resources and document from Hugging Face's [Diffusers](https://huggingface.co/docs/diffusers/index) library and [Riffusion](https://github.com/riffusion/riffusion/tree/main)
    - For specifics, check out the documentation on Diffusion Model, StableDiffusionImg2Img pipeline in Diffusers and Riffusion repo.

Massive Thanks to all the original authors and contributors.



