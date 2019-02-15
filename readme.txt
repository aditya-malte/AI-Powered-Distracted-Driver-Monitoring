Technologies used:
1)Convolutional Neural Networks
2)Transfer Learning
3)Image Processing

Making an all round driver monitoring tool to detect 
1)Distracted driver(using a mobile)
2)Sleepiness
3)Anomalies in driving style(eg. haste, drunkenness)

This is a code to detect(classify) whether the driver is using a mobile or not
while driving.

The data was taken from "Kaggle's State Farm Distracted Driver Dataset".

Although, it might seem like an easy task of classification, There is a high lack of 
data and a highly similar dataset with low variability(only around a few people were used to make thousands
of images).Besides the images itself are from random frames of videos and have a 
severe trend of being very identical.
Hence I had to use: tranfer learning, heavy dropout, regularization(to lower covariance)
and a host of other techniques like data augmentation.

Future additions:
1)Would add AlphaPose/OpenPose as a feature detectior(instead of this CNN model) in addition to training a
  mobile position localization CNN, to sound alarm if the hand is near a mobile.
2)Sleepiness/Fatigues detection using dlib face pose library using EAR(Eye aspect ratio) and .
3)Addition of LSTM/other time series model to detect anomalies in driving.
