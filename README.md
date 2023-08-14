# Tradi-fusion
Tradi-fusion Refined: Evaluating and Fine-tuning the Riffusion Model for Irish Traditional Music.

# Focus of the project
The project will investigate the following research questions:
a) How can the Riffusion model be fine-tuned for generating Irish Traditional music?
b) What challenges are involved in fine-tuning the Riffusion model for Irish Traditional music?
c) Can the fine-tuned Riffusion model generate Irish Traditional music that is comparable in quality to
human-composed Irish Traditional music? If not, what is the reason?

The project is of interest to the field of music technology and artificial intelligence. It can be of interest to researchers, practitioners, and enthusiasts in these fields who are interested in exploring the possibilities of AI-generated music and its potential applications.

## Background 


## Dependencies
I have useed keras-cv version 0.5.1 becuase it supports tensorflow version 2.11.0.
Make sure to install dependencies using `pip install requirements.txt`

## Training
The model was trained on multiple remote GPUs (NVIDIA GeForece RTX 3090) with the use of [NVIDIA NGC Tensorflow container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow). 

The advantage of using is to avoid erros with cudnn library and concering not findinf libdevice under `/usr/local/cuda`. It also helps with the matching the Tensorflow compatible verison with CUDA and cuDNN. 

## To-Do Progrss
![](https://geps.dev/progress/33)
- [x] Background study
- [x] Build Dataset 
- [x] [Fine-tune using the Dreambooth approach](https://dreambooth.github.io/)
- [ ] Fine-tune in the traditional manner using NGC Container
- [ ] Study the model from TensorBoard
- [ ] [Fine-tune using the Textual-inversion method](https://textual-inversion.github.io/)
- [ ] Come up with a new model





