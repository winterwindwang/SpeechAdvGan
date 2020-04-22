# Speech Adversarial Examples with GAN
### This is the implemention of *Speech Adversarial Examples Attack with Generative Adversarial Networks*
## Setup
### The trained victim model can be available from [model](https://drive.google.com/open?id=1Jyrpu6dhK4jhq0aB2S8EWh2ycSy2PXVp)
### The trained target(generator) model can be available from [pretrained model](https://drive.google.com/open?id=1bBHAUwbfS34RDe-uk2j903cStBf5Ko0E)
### Notice: victim pretrained model should be placed in checkpoints
###         generator model for wideresnet and sampleCNN should be placed checkpoints/speech_common_generator and checkpoints/music_genres_generator
### 	|-- checkpoints
###			|-- music_genres_generator
###				|-- target_blues
###				|-- target classical
###				|-- ...
###			|-- speech_common_generator
###				|-- target yes
###				|-- target no
###				|-- ...
###			|-- sampleCNN.pth	
###			|-- wideResNet28_10.pth

## Attack Evaluation
### To run:
###     python test_gan.py --data_dir original_speech.wav  --target yes --checkpoint checkpoints
