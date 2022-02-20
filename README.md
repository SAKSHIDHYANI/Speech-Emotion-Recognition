# Speech-Emotion-Recognition

Sakshi dhyani

Sakshidhyani73@gmail.com

Contribution: 

1.	Data features extraction 
•	Combining data from RAVDESS speech and song datasets and Toronto emotional speech set. Data sets available in Kaggle:
https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess
https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio 

•	Extraction of features for audio files using Mel Frequency Cepstral Coefficients


2.	Exploratory Data Analysis 

•	Plotting waveform for some random audio files
•	Visualizing MFCC transformation for some audio files
•	Bar graph to visualize the count of various emotions

3.	Models Implementation

•	Models used: 
	SVC – Support Vector Classifier:

SVM is supervised machine learning algorithm used for classification, regression and outlier detection as well. SVM has classifications: SVC, NuSVC, LinearSVC. The       implementation of SVC is based on libsvm. It provides best fit hyperplane to categorize the data. The multiclass support is handled according to a one-vs-one scheme.  

	CNN – Convolutional Neural Network: Class of deep neural network:

Convolutional neural networks are distinguished from other neural networks by their superior performance with image, speech, or audio signal inputs. They have three main types of layers: 
Convolutional layer, pooling layer and fully-connected layer.
The convolutional layer is the first layer of a convolutional network. While convolutional layers can be followed by additional convolutional layers or pooling layers, the fully-connected layer is the final layer. The convolutional layer is the core building block of a CNN, and it is where the majority of computation occurs.

	MLP – Multilayer Perceptron: Class of feed forward artificial neural network

MLP Classifier stands for Multi-layer Perceptron classifier. MLP is a feedforward artificial neural network model that maps input data sets to a set of appropriate outputs. An MLP consists of multiple layers and each layer is fully connected to the following one. The nodes of the layers are neurons with nonlinear activation functions, except for the nodes of the input layer. Between the input and the output layer there may be one or more nonlinear hidden layers.


•	Standardization of data to improve score in both models.
•	Evaluating score for training and testing data for both the models
•	Applying early stopping to balance the number of epochs and reduce overfitting in the models.
•	Accuracy plot and Loss plot for training and testing data for different number of epochs.
•	Saving models for further use

4.	Testing audio data

•	Applied saved CNN model on randomly selected audio files to test the results.

5.	Making model accessible as a web application using streamlit

•	Via Streamlit, created web application to test speech emotion using CNN and MLP models both, providing end user with the interface to directly provide audio file and detect emotion.



6.	Deploying web application created to share.streamlit.io platform

https://share.streamlit.io/sakshidhyani/speech-emotion-recognition/main/app.py 



