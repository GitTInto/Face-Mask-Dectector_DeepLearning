# FACE MASK DECTECTOR: USING DEEP LEARNING FOR COMPUTER VISION

## Abstract
This paper details the processes and methods for building a “Face Mask Detector” for health and safety using computer vision. With the increase in corona virus cases all over the world, wearing mask is one of the ways that Doctors and scientists advise to fight against this pandemic. With the expansion of the COVID-19 pandemic, it is necessary for all people to comply with the health security measures in place and, among them, the use of masks. The work of enforcing this obligation becomes complicated when there is a massive transit of people in the facility., or when this work is carried out by means of video surveillance systems. 

However real-time video analysis through facial recognition, Artificial Intelligence and Machine Learning enabling computer vision can carry out exhaustive detection of people who do not use a sanitary mask or who do so incorrectly, this will disburden the compliance control out of the hands of individuals, by automating face mask detection.

I am using Tenser flow Conv2D to build a model, 4 layers of Conv2D and MaxPooling2D, this serves both to augment the capacity of the network and to further reduce the size of the feature maps, so they aren’t overly large when I reach the Flatten layer. Because I am attacking a binary-classification problem, I will end the network with a single unit (a Dense layer of size 1) and a sigmoid activation. This unit will encode the probability that the network is looking at one class or the other, and RMSprop optimizer for compiling.

To expose the model to more aspects of the data and generalize better, to be able to predict real world scenarios, I am using data augmentation techniques. For training I am using facial images with and without masks downloaded from Kaggle.


## Business Use Case : Problem statement:
We are all getting accustomed to the new norm of wearing mask as we had been living in the pandemic for 2 years now. The World Health Organization, Center of Disease Control and other health, safety organizations and bodies around the world have recemented mask as an efficient way of preventing and fighting COVID-19. (CDC, n.d.)

Adhering to these guidelines, Masks have been mandated around the world in many public places. (Facing Your Face Mask Duties, n.d.). The work of enforcing this had become complicated and demands more workforce during the times of social distancing is not pragmatic and calls for an automated way of encouraging compliance. 

An accurate deep learning algorithm for machine learning for computer vision can be integrated with access control system in office building, hospitals and can also be used in public transport systems can be used to ensure compliance to mask obligation. 


## Data set
Fundamental characteristic of deep learning is that it can find interesting features in the training data on its own, without any need for manual feature engineering. For this we need a set of facial images wearing mask and without wearing mask.

The dataset is obtained from Kaggle. https://www.kaggle.com/prasoonkottarathil/face-mask-lite-dataset?select=without_mask


## Data Preparation
After downloading and uncompressing it, I created a new dataset containing three subsets: a training set with 1,000 samples of each class, a validation set with 500 samples of each class, and a test set with 500 samples of each class.

## Data Pre-Processing
Data should be formatted into appropriately preprocessed floating-point tensors before being fed into the network. Currently, the data sits on a drive as PNG files, so the steps for getting it into the network are roughly as follows:
•	Read the picture files.
•	Decode the PNG content to RGB grids of pixels.
•	Convert these into floating-point tensors.
•	Rescale the pixel values (between 0 and 255) to the [0, 1] interval (as we know, neural networks prefer to deal with small input values).
Details of this implementation is made available in the Appendix.


## Data Augmentation
We could have overfitting which is caused by having too few samples to learn from, rendering you unable to train a model that can generalize to new data. Given infinite data, your model would be exposed to every possible aspect of the data distribution at hand: you would never overfit. 

Data augmentation takes the approach of generating more training data from existing training samples, by augmenting the samples via several random transformations that yield believable-looking images. The goal is that at training time, your model will never see the exact same picture twice. This helps expose the model to more aspects of the data and generalize better. (data augmentation, n.d.)

## Method
For image classification I am using Tenser flow Conv2D to build a model that can identify if in the image the person is wearing a mask or not.

## Building my Network
I am using 4 layers of Conv2D and MaxPooling2D, this serves both to augment the capacity of the network and to further reduce the size of the feature maps, so they aren’t overly large when I reach the Flatten layer. 

Because I am attacking a binary-classification problem, I will end the network with a single unit (a Dense layer of size 1) and a sigmoid activation. This unit will encode the probability that the network is looking at one class or the other.

The depth of the feature map progressively increases in the network from 32 to 128, whereas the size of the features map decreases from to 224 to 12.

For the compilation, I will go with RMSprop optimizer. Because we ended the network with a single sigmoid unit, I will use binary crossentropy as the loss.

## Fitting the model
Let’s fit the model to the data using the generator, using the fit_generator method, the equivalent of fit for data generators. It expects as its first argument a Python generator that will yield batches of inputs and targets indefinitely. Because the data is being generated endlessly, the Keras model needs to know how many samples to draw from the generator before declaring an epoch over. This is the role of the steps_per_epoch argument: after having drawn steps_per_epoch batches from the generator—that is, after having run for steps_per_epoch gradient descent steps—the fitting process will go to the next epoch. In this case, batches are 20 samples, so it will take 100 batches until you see your target of 2,000 samples.

When using fit_generator, you can pass a validation_data argument, much as with the fit method. It’s important to note that this argument is allowed to be a data generator, but it could also be a tuple of Numpy arrays. If I pass a generator as validation_data, then this generator is expected to yield batches of validation data endlessly; thus, I should also specify the validation_steps argument, which tells the process how many batches to draw from the validation generator for evaluation. In This case, I have 500 Mask + 500 No Mask, Total of 1000 Validation samples, so we define validation_steps = 1000/20(batch size of validation_generator). 

As you can see very soon, we reached good accuracy. Now I will plot the accuracy, loss and validation accuracy and validation loss against the epochs.

## Evaluation
As we can see our model accuracy is 99.9 %, Impressive.

## Prediction
I will use the model to predict one batch and display the image alongside the prediction.

## Performance
Classification report and the confusion matrix to get a sense of the performance of the model.

## Conclusion
By the methods I followed, build a good model with very high accuracy of 99.9%.

##
We are now using the model to predict if my face picture - Am i wearing a maks or not ?

## References
An Ethical Framework for Facial Recognition. (n.d.). Retrieved from https://www.ntia.doc.gov/files/ntia/publications/aclu_an_ethical_framework_for_face_recognition.pdf
CDC. (n.d.). Retrieved from https://www.cdc.gov/coronavirus/2019-ncov/prevent-getting-sick/about-face-coverings.html
data augmentation. (n.d.). Retrieved from https://learning.oreilly.com/library/view/deep-learning-with/9781617294433/OEBPS/Text/05.xhtml#ch05lev2sec7
Facing Your Face Mask Duties. (n.d.). Retrieved from https://www.littler.com/publication-press/publication/facing-your-face-mask-duties-list-statewide-orders
kaggle dataset. (n.d.). Retrieved from https://www.kaggle.com/prasoonkottarathil/face-mask-lite-dataset?select=without_mask
The ethical questions that haunt facial-recognition research. (n.d.). Retrieved from https://www.nature.com/articles/d41586-020-03187-3

