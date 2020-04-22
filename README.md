Speech Adversarial Examples with GAN
================================

This is the implemention of *Speech Adversarial Examples Attack with Generative Adversarial Networks*

Setup
--------------------------
   The pre-trained victim model is available for use and can be downloaded from [here](https://drive.google.com/open?id=1Jyrpu6dhK4jhq0aB2S8EWh2ycSy2PXVp)
    
   The pre-trained target(generator) model is available for use and can be downloaded from [here](https://drive.google.com/open?id=1bBHAUwbfS34RDe-uk2j903cStBf5Ko0E)
    
   Notice: The directory structure of the checkpoints file is as follows: 

    |-- checkpoints
        |-- music_genres_generator
            |-- target_blues
            |-- target classical
            |-- ...
	    |-- speech_common_generator
            |-- target yes
            |-- target no
            |-- ...
        |-- sampleCNN.pth	
        |-- wideResNet28_10.pth

Attack Evaluation
--------------------------
To run:

  python test_gan.py --data_dir original_speech.wav  --target yes --checkpoint checkpoints
