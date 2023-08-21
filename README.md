# Tradi-fusion
Tradi-fusion Refined: Evaluating and Fine-tuning the Riffusion Model for Irish Traditional Music.

# Focus of the project
The project will investigate the following research questions:
- How can the Riffusion model be fine-tuned for generating Irish Traditional music?
- What challenges are involved in fine-tuning the Riffusion model for Irish Traditional music?
- Can the fine-tuned Riffusion model generate Irish Traditional music that is comparable in quality to
human-composed Irish Traditional music? If not, what is the reason?

The project is of interest to the field of music technology and artificial intelligence. It can be of interest to researchers, practitioners, and enthusiasts in these fields who are interested in exploring the possibilities of AI-generated music and its potential applications.

## Background 


## Dependencies
I have useed keras-cv version 0.5.1 becuase it supports tensorflow version 2.11.0.
Make sure to install dependencies using `pip install -r requirements.txt`

## Training
Train on 256 x 256 Spectrograms of Irish Traditional Tunes. 
Dataset Obtained using [riffusion-manilab](https://github.com/hdparmar/riffusion-manilab)
The dataset produces 512x512 images but in the fine-tuning script we use 256x256 because of OOM Erros (it requires lot of GPU RAM).

The fine-tuning training was done on multiple remote GPUs (NVIDIA GeForece RTX 3090) with the use of [NVIDIA NGC Tensorflow container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow) using Jupyter Lab. 

The advantage of using is to avoid erros with cudnn library and errors concering not finding libdevice library under `/usr/local/cuda`. It also helps with the matching the compatible Tensorflow verison with CUDA and cuDNN. 

### Running on Jupyter Lab
The file `finetune_latest.ipynb` can be used to play with the model, visualise the results and tweak the parameters and see the outcome. Once you are satisfied with that, you can go forward and make a training script.


### Running the Script
The file `trainerClass.py` is for defining a custom Keras Model for training and testing routines for diffusion model.
The file `train.py` is a trainig script that you can run within a NGC Tensorflow 23.03 container to save the model weights, losses (in a CSV file), and TensorBoard logs.

Provided you can in the NGC Tensorflow container which is running and you have mounted the dataset into Workspace.
You can run the script by `python3 train.py`

## Checkpoints 
The various checkpoints will be availble on Hugging Face.

## To-Do Progrss
![](https://geps.dev/progress/50)
- [x] Background study
- [x] Build Dataset 
- [x] [Fine-tune using the Dreambooth approach](https://dreambooth.github.io/)
- [x] Fine-tune in the traditional manner using NGC Container
- [x] Study the model from TensorBoard
- [ ] Looping and Interpolation 
- [ ] Possibility to train on 512x512 images with Gradient Accumulation (but will the forward pass fit in memory!??)
- [ ] [Fine-tune using the Textual-inversion method](https://textual-inversion.github.io/)
- [ ] Come up with a some tweaks model
- [ ] Deploy a Website for live inference (Use streamlit to deploy the best generated checkpoint)





