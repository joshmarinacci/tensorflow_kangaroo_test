# Handwritten digits

For this lesson I'm recognizing hand written digits. I'm starting with a dataset called the
[MNIST Database](http://yann.lecun.com/exdb/mnist/)  Fortunately I don't have to handle
the data directly. The Tensorflow lesson has already combined them into a special sprite
file so we can download them as a single 10MB png, with some code to automatically split them
into the separate examples.

# the idea

This is a *classification* task. We train the model to classify an input image to one of 10 classes
(the digits 0-9). Each training image is already normalized as 28x28px images, 1bit with a black background.
The shape of each image as a tensor is: `[28, 28, 1]`

The model is a stack of layers. The first two are `conv2d` since we are working with images.
Then theres a `flatten` layer, then the standard 1d `dense` layers.

To train the model the data is split into two datasets. A training set to use as input, and a validation
set we will use to test the model at the end of each epoch. By having separate
validation data those images will never be used as training data, so it won't be biased.

# takeaway

This part worked exactly as I expected machine learning to. The training data is a bunch of 
images that are pre-labeled with the correct answer.  The testing data is different
set of images with labels.  Once trained we can stuff in new images and they will be recognized
and produce an output label along with an accuracy rating.  The accuracy was above 90% for all
digits except eight (76%) and nine (80%). Odd.





