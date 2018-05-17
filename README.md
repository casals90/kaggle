# Kaggle competitions #

<p>The aim of this repository, it's to store the codes and results about Kaggle competitions.</p>

## Titanic: Machine Learning from Disaster ##
	

	<p>The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.</p>

	<p>One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.</p>

	<p>In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.</p>


* <h3>Data</h3>
	<p>The data has been split into two groups training set (train.csv) and test set (test.csv):<p>
	<p>The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.</p>

	<p>The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.</p>


	| Variable | Definition                                 | Key                                            |
	| -------- | ------------------------------------------ | ---------------------------------------------- |
	| Survival | Survival                                   | 0 = No, 1 = Yes                                |
	| Pclass   | Ticket class                               | 1 = 1st, 2 = 2nd, 3 = 3rd                      |
	| Sex      | Sex                                        | male, famele                                   |
	| Age      | Age in years                               |                                                |
	| Sibsp    | # of siblings / spouses aboard the Titanic |                                                |
	| Parch    | # of parents / children aboard the Titanic |                                                |
	| Ticket   | Ticket number                              |                                                |
	| Fare     | Passenger fare                             |                                                |
	| Cabin    | Cabin number                               |                                                |
	| Embarked | Port of Embarkation                        | C = Cherbourg, Q = Queenstown, S = Southampton |

		
* <h3>Results</h3>

	| Variable               | Accuracy training | Accuracy test (kaggle submission)|
	| ---------------------- | ----------------- | -------------------------------- |
	| Support Vector Machine | 	84.91			 |  77.99       					|
	| Random Forest          | 	84.89			 |  77.98       					|

* <h3>Submission</h3>
	<p>
		Score = 77.99<br>
		Position 5466 of 11307<br>
	</p>

[https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)

## Digit Recognizer ##

<p>
	MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.
	The goal of this challange is to predict handwritten digits (0 to 9).
</p>

* <h3>Data</h3>
	<p>The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.</p>

	<p>Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.</p>

	<p>The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.<p>

* <h3>Results</h3>

	| Variable               | Accuracy training | Accuracy test (kaggle submission)|
	| ---------------------- | ----------------- | -------------------------------- |
	| Neural network         | 	97.84			 |  95.64       					|
	| K nearest neighbors    | 	97.36 			 |  97.04    					    |

* <h3>Submission</h3>
	<p>
		Score = 97.04<br>
		Position 1570 of 2504<br>
	</p>

[https://www.kaggle.com/c/digit-recognizer](https://www.kaggle.com/c/digit-recognizer)
