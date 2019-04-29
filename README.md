# final_project_balance
# Pneumonia Diagnosis using Convolutional Neural Networks

## Team members: Amro Elhag, Mounica pokala, Micheal Alrafati, Petros Paterakis

Pneumonia is an inflammatory condition of the lung affecting primarily the small air sacs known as alveoli. Pneumonia is 
usually caused by infection with viruses or bacteria and less commonly by other microorganisms, certain medications and conditions such as autoimmune diseases. 
This project is about diagnosing Pneumonia from X_Rays images of persons lungs. XRay images are imported from 
Kaggle(https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). Datasets contains 5863 images as test, train and val where each has two files of images Normal and Pneumonia.

We used four different models to find the best model that fits. 
Models used are: InceptionV3, Xception, VGG19, InceptionResnet.
By the highest accuracy of the model, we decided VGG19 model to diagnose Pneumonia from XRay images. 
For the most accurate results dataset is categorised into Pneumonia and normal instead of Pneumonia, Bacteria and Virus.
Recall of the model is 0.98 
Precision of the model is 0.76

## Tools/ libraries used:
Python
Keras
Tenserflow
Numpy
Matplotlib
Scikit - learn
Flask
HTML, CSS, Javascript
Tableau
MongoDB

Refer the project ppt for the details that are used in the project. Run index.html, app.py for the webpage where you can upload XRay images and find out the person has Pneumonia or not. 
