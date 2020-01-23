# Deep Learning for Speech and Language Processing Project

The project has two parts:

* [Natural Language Understanding](https://github.com/mtfrigo/DL-NLU)

* Speech Emotion Recognition

## Getting Started

This repository implements the task of the final project of the Deep Learning for Speech and Language Processing Class from Universität Stuttgart regarding the SER part.

Speech Emotion Recognition (SER) is the hability to recognize emotion from the audio recording.

Generally for this kind of application the input would be speech recodings (.wav files) and the output various emotions class (happy, sad, excited, depressed, ...)

In our case it is provided a preprocessed inputs (LogMel features) and the expected output is two binary labels: Arousal and Valence. **This approach does not implements a transformation of audio files into a dataset.**

### Prerequisites

* Keras
* pickle
* nltk
* sklearn
* numpy
* pandas
* tensor flow

### Installing

TODO

## Running the tests

TODO


## Code in a nutshell

As said before, the input in this case is already preprocessed and this means our input is already features extracted from audio files.

Each input data has its own features. The feature array is composed of different quantities of inner arrays with length equals 26. It means the inner array will always have the same length (26) while the outer array is totaly random. We have to consider this for the transformation of data.

So, each input data has a dimension of (F, 26) where F is the number of inner arrays. The Neural Network needs all the inputs to have the same length, then we need to pad/truncate (some data will have more and some data will have less quantity of inner arrays of the threshold).

The threshold is totally arbitrary and can be changed by the MAXLEN parameter.

For the labels, we have in the test data two binary labels. For the neural network we transform the 4 possible combinations into 4 different labels. Once having the data parsed we can transform it into one hot vectors.

After having all the inputs with same dimension and labels prepared we can split into train and validation data.

As the model uses 2D Convolutional Layers we have to reshape the input data to  have explicited NUMBER-OF-INPUTS x MAXLEN x BUCKETS x CHANNELS. NUMBER-OF-INPUTS is the number of the input train data; MAXLEN is the parameter mentioned above; BUCKETS is how we call the length of the inner array (26 as we know); CHANNELS we use 1.

Model details are explained below

## Model

I'm testing differents combinations of layers to achieve a better result on the task, so the last model might be different from this explanation.

I used a CNN on my model. Probably there is better approachs for Speech Emotion Recognition, however I'd to improve my knowledge on this type of neural network for future projects.

My model uses a 2D Convolutional Layer as input with 32 Filters and Kernel with 3x3 size. The input shape is NUMBER-OF-INPUTS x MAXLEN x BUCKETS x CHANNELS as mentioned above.

Afterwards comes a 2D Maxpooling layer with 2x2 pool size.

This sequence is repeated, then we have 2x Conv2D and 2x 2D MaxPooling) to improve the results

Then comes a Dropout layer to reduce overfitting.

And a Flatten to reshape.

The Dense layer with the dimension of the hidden layer and relu function activation followed by a Dropout layer.

Finally the Output layer has the dimension of the number of labels, 4 in our case.

# Results

| Model         | 	F1-Score    |
| ------------- |:-------------:|
| ser_model1.h5 | 48.47%        |

