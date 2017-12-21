## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)
[image1]: ./misc/signs-original.png "Original Sign Images"
[image2]: ./misc/dist-all-signs.png "Distribution - All Data Sets"
[image3]: ./misc/dist-training-signs.png "Distribution - Training Data Set"
[image4]: ./misc/signs-aug-norm.png "Augmented and normalized signs"
[image5]: ./misc/signs-aug-only.png "Augmented Only"
[image6]: ./misc/traing-loss.png
[image7]: ./misc/training-accuracy.png
[image8]: ./misc/my-signs-norm.png
[image9]: ./misc/my-signs-raw.png
[image10]: ./misc/feature-map.png

Overview
---
The objective of this project is to build a neural network to classify road signs. The dataset used is the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). 

After training the model we'll try the it against other signs we find on the internet.

A few random examples of the signs in the dataset are below. As you can see the quality is often challenging, much like the real world will be. 

![][image1]

## TL;DR
* My model validation accuracy is repeatably in the 97-98% range.
* My Jupyter notebook "Copy3" is my submitted [project code](
https://github.com/tpak/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier-Copy3.ipynb) -- other notebooks are for personal reference purposes.

I initially got a very simple LeNet based model up and running and as specified it quickly got up and into the 92% validation accuracy range but hit a wall. Simple image augmentation techniques did not want to get LeNet beyond 92% so I started exploring other augmentation techniques. My advisor pointed me to various resources and that is when I ran smack into python, scikit, and cv2 issues that sent me off on a wild goose chase for several days. After augmenting images by rotating, shearing (think of this as tilting the image at various angles), and transforming (moving the signs around randomly so they were not always in the middle of the frame) I was able to get the model up over 94%. That would have been good enough but I kept going now that I had a working foundation and played with adding dropout layers to my model. 

By trial and error and repeated training runs I was able to find that dropping out towards the end of my model and only doing so once seemed to have the most effect. The training rate was also important - too high and the model learns quickly but then overfits. Too low and it never quite gets you there. I tried to add tensorflows learning decay but that blew my project up so I abandoned it, although I may go back to play with it when I have some free time as I believe that would help correct the model fit issue. Also changing the number of filters in the convolutions helped.

In the end I was able to get my model to the 97-98% validation accuracy range fairly consistently. Unfortunately the validation accuracy was better than the accuracy against both the test set and the images I downloaded from the interent. 

Things that I think would be good to explore more in the future:

* Implement a Learning rate decay function 
* Looping the model with varying parameters adn automatically stopping when "best" fit is obtained.
* Different models - I only scratched the surface on changing the model due to time constraints - I think there is a lot of gain in different models
* Grey scale - oddly I never really worked with this while a lof of people do. I felt the colors would allow the model to differentiate signs better but many of the pictures were so bad that perhaps that is wrong. 
* Different augmentation parameters. I settled on a set of augmentation parameters for rotation, shear, and transformation that seemed to help the model learn better but taking this further and exploring each individually to determine it seffects would be good. 

---
## Data Set Summary & Exploration
### Basic Info about our training dataset
* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

### Distribution
As you can see in the histograms and the table below we have a non-uniform distribution of signs in our data sets and each data set roughly approximates the other in terms of distributions. I chose not to augment the data to even this out becaus ethe real world is cruel place to operate in and will vary from day to day and location to location. In retrospect it may be beneficial to do so for training puposes so that the model can learn each type of sign at a deeper level.

![][image2]
![][image3]

Count of each sign class

count    | class description
-------- | -----------
180	|	Speed limit (20km/h)
1980	|	Speed limit (30km/h)
2010	|	Speed limit (50km/h)
1260	|	Speed limit (60km/h)
1770	|	Speed limit (70km/h)
1650	|	Speed limit (80km/h)
360	|	End of speed limit (80km/h)
1290	|	Speed limit (100km/h)
1260	|	Speed limit (120km/h)
1320	|	No passing
1800	|	No passing for vehicles over 3.5 metric tons
1170	|	Right-of-way at the next intersection
1890	|	Priority road
1920	|	Yield
690	|	Stop
540	|	No vehicles
360	|	Vehicles over 3.5 metric tons prohibited
990	|	No entry
1080	|	General caution
180	|	Dangerous curve to the left
300	|	Dangerous curve to the right
270	|	Double curve
330	|	Bumpy road
450	|	Slippery road
240	|	Road narrows on the right
1350	|	Road work
540	|	Traffic signals
210	|	Pedestrians
480	|	Children crossing
240	|	Bicycles crossing
390	|	Beware of ice/snow
690	|	Wild animals crossing
210	|	End of all speed and passing limits
599	|	Turn right ahead
360	|	Turn left ahead
1080	|	Ahead only
330	|	Go straight or right
180	|	Go straight or left
1860	|	Keep right
270	|	Keep left
300	|	Roundabout mandatory
210	|	End of no passing
210	|	End of no passing by vehicles over 3.5 metric


---
### Image pre-processing
The supplied images are quite small - a mere 32x32 pixels. However; this is probably representative of the portion of an image that a real camera would capture in a moving car - i.e. any given object identified as a sign will probably only occupy a small portion of the overall captured frame. While almost 35k signs might seem like a lot the real world is a tough place and cars will approach signs at a nearly infinite number of angles, distances pitches and an equally infinite number of lighting conditions so I chose to augment the data that way. I also chose to use an image processing library (OpenCV) to equalize the histogram of all the images to try and get better visibility into the features. A simple equalization method was suggested and my exploration showed that it had slightly better results initially but when I augmented the size of the data set it was not as good. One suggestion to brighten the images using an OpenCV API did not seem to help my results so it was abandoned for this excercise.  

All augmentation and normalization was applied with a random normal disgtribution to try and re-create the randomness a car would encounter in the real world.

After augmenting the data and normailizing the histograms I added the original training data set (no totation, etc just histogram adjustements) into the final training set bringing the number of images to 208,794.

Examples of images with augmentation:
![][image5]

Examples of images with augmentation and normalization applied:
![][image4]

---
## Neural Network Model
I used an almost pure derivative of the LeNet architecture. 

Input 32x32x3 images

Layer | Type 			| Description
----- | ----------- | -----------
1    | Convolution 	| 12@32x32x3  Output = 12@28x28x3
2    | Convolution 	| 12@28x28x3  Output = 16@10x10
3    | Fully Connected | Input = 400. Output = 120.
4    | Fully Connected | Input = 120. Output = 84. Dropout@ 0.5
5    | Fully Connected | Input = 84. Output = 43

I played with placing dropouts at a variety of individual layers as well as multiple layers. It was interesting to see the accuracy plummet in some scenarios. I ended up settling on only one dropout at layer 4. I also played with the dropout rate and settled at 50%. This seemed to have some effect but not much in the 40-60% range. Above or below that and the accuracy dropped. 

## Neural Network Training
I was able to set up my model to save the most accurate version as it went along during training so that I could let it run longer without worry if the model degraded, which they often do. I chose to save the highest validation accuracy as my chosen method. Another approach might be to choose the model with the lowest Validation Loss once the accuracy is adequate. 

You can see in the below graphs that over time the model appears to gain loss - I'm not sure if this is considered a form of overfitting or not but I think that the model does get worse. In the run featured here my final model saved at iteration 60 with a validation accuracy rate of 98.23%. However as will be discussed below, that was not as accurate as a previous run with only 30 iterations and 97% validationa ccuracy when it came to both test data sets and data sets downloaded from the internet. These graphs clearly show that around iteration 15 the training loss  begins to steady and the validation loss begins to vary wildly and seperates from the training loss. I suspect this would be the best place to save the model - approximately iteration 15 which you can see in the table has validation accuracy of 96.24%. 

The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters.

This model was trainied with the AdamOptimizer using the 30 Epochs of 256 images each, a learning rate of 0.0005 and dropout probability of 0.5. The batch size seems to work better with smaller batches on this model. However; I suspect that if I could ge the tensorflow learning rate decay functions working, a larger batch size would improve the model. 

This particular version of the model was 94.66% accurate against the test data set. 

![][image6]
![][image7]


EPOCH	|	Training Accuracy	|	Validation Accuracy	|	Training Loss	|	Validation Loss
--- |	--- 	|	--- 	|	--- |	--- 
1	|	0.2872	|	0.2773	|	2.5572	|	2.5589
2	|	0.4021	|	0.4181	|	2.0123	|	1.9555
3	|	0.4759	|	0.463	|	1.5437	|	1.4987
4	|	0.5546	|	0.5658	|	1.3278	|	1.3034
5	|	0.6399	|	0.6556	|	1.0979	|	1.0945
6	|	0.723	|	0.7401	|	0.8356	|	0.7885
7	|	0.7905	|	0.8039	|	0.6345	|	0.5984
8	|	0.8497	|	0.8528	|	0.4724	|	0.4821
9	|	0.8828	|	0.8932	|	0.3632	|	0.3454
10	|	0.8998	|	0.9093	|	0.3065	|	0.2771
11	|	0.9236	|	0.9186	|	0.2312	|	0.2598
12	|	0.9261	|	0.9329	|	0.2252	|	0.2184
13	|	0.9462	|	0.9429	|	0.1623	|	0.1855
14	|	0.9513	|	0.9535	|	0.1461	|	0.1545
15	|	0.9594	|	0.9624	|	0.123	|	0.1322
..	|	...	|	...	|	...	|	...
60	|	0.9954	|	0.9844	|	0.0144	|	0.1144
61	|	0.9938	|	0.9823	|	0.0189	|	0.1666
..	|	...	|	...	|	...	|	...
74	|	0.9951	|	0.9782	|	0.0168	|	0.3832
75	|	0.9964	|	0.9769	|	0.0113	|	0.186

--- 
### Test against random images from the internet. 
I found 5 images on the internet and cropped close to square and then resized them to 32x32 pixels using the app "Preview" on my Mac rather than writing resizing code. You can see them here.

My selected images before normalization:
![][image9]

My selected images after normalization:
![][image8]

Running my images through the model resulted in 60% accuracy. 

	Image  1 predicted: 11 actual: 7 Correct prediction: False
	Image  2 predicted: 11 actual: 11 Correct prediction: True
	Image  3 predicted: 13 actual: 13 Correct prediction: True
	Image  4 predicted: 17 actual: 17 Correct prediction: True
	Image  5 predicted: 28 actual: 25 Correct prediction: False
	Accuracy on my signs = 0.6000
	
Not great at all! Previous runs had accuracy of 80%. Interestingly the 100kph sign proved to fail every time. 

---
### Softmax analysis

	TopKV2(values=array([[  9.99935627e-01,   6.43333478e-05,   5.64296804e-11,
	          4.41504811e-14,   2.86955891e-14],
	       [  1.00000000e+00,   1.33470480e-14,   6.83242527e-24,
	          2.12374799e-28,   3.55552431e-35],
	       [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
	          0.00000000e+00,   0.00000000e+00],
	       [  1.00000000e+00,   2.03085866e-21,   7.91097671e-25,
	          9.11259898e-33,   9.82023134e-34],
	       [  5.37557721e-01,   4.62441295e-01,   8.32136209e-07,
	          1.42813448e-07,   1.25235128e-10]], dtype=float32), indices=array([[11, 27, 20, 30, 19],
	       [11, 30, 20, 27, 25],
	       [13,  0,  1,  2,  3],
	       [17, 18, 14,  9, 26],
	       [28, 25, 20, 24, 29]], dtype=int32))

Looking at the softmax dump you can see that the model doesn't even consider the first image which is a 100kph sign to be that at all. It's not even in the top 5. At least on this run the last sign, construction is the second possibility. Other runs yielded similar results for the 100kph sign but more often correctly guessed the construction sign. The 100kph sign was often mis-classified as a 20kph sign although that is not considrered here (value 0). 

---
###  Visualize the Neural Network's State with Test Images
Using the supplied code I was able to dump waht the model was "seeing" on my test images. For now I am only showing layer 1 of our problematic 100kph sign. It's pretty clear even from looking that the model is not able to clearly see what it is looking at with the parameters it was trained with. Remember I changed LeNet to use 12 filters at layer 1, they are here as 0-11.

![][image10]

---
### Conclusion

I wouldn't want this model driving my car!

That aside there are lots of ways as pointed out above to improve on this model. I learned a lot about how tensorflow models work and have some good ideas on how to improve this model.




























