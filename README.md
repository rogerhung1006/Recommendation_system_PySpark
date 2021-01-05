# Movie Recommendation System with PySpark

## Introduction
Recommendation engines are one of the best use cases for big data. It’s fairly easy to collect training data about users’ past preferences at scale, and this data can be used in many domains to recommend users with new content such as movie or course recommendations. Spark is an open source tool and one of the most ideal choices for large-scale recommendations. 
In this project, I managed to build a basic recommendation engine using PySpark, as I am more comfortable with Python. I personalized movie recommendations tailored for each user in the dataset using Alternating Least Squares (ALS) algorithm, and worked with 10 million ratings from 72,000 users on 10,000 movies as of 2018, collected by MovieLens.

## What is Collaborative filtering?
Collaborative filtering is commonly used for recommender systems. These techniques aim to fill in the missing entries of a user-item association matrix. spark.mllib currently supports model-based collaborative filtering, in which users and products are described by a small set of latent factors that can be used to predict missing entries. spark.mllib uses the alternating least squares (ALS) algorithm to learn these latent factors.

## Analysis Process 
- Prepare the Datasets
- Check Sparsity
- Explore and Examine the Datasets
- Cross-Validated ALS Model and Recommendations
</br>

### 1. Prepare the Datasets
I would use two datasets in this project, the movie data and the rating data. The datasets can be found on MovieLens. The movie data has 9,742 movies with the newest one released in 9/2018. 
<p align="center">	
	<img align="middle" src="images/Figure 1. Movie list.png">
<p align="center">
     <i>Figure 1. The movies data contains 50,394 records (after sampling).</i> 
</p>
</br>
<p align="center">	
	<img align="middle" width=700 src="images/Figure 2. Rating.png">
</p>
<p align="center">
  <i>Figure 2. Rating data</i> 
</p>


### 2. Check Sparsity
As ALS works well with sparse datasets, I would like to see how much of the ratings matrix is actually empty. It turned out that the dataframe is 98.9 empty, which is a fairly good percentage for ALS algorithm implementation.
<p align="center">	
	<img align="middle" src="images/Figure 3. Sparsity.png">
</p>
<p align="center">
  <i>Figure 3. The rating data is 98.9% empty</i> 
</p>


### 3. Explore the Dataset
In this section, I took a little dive into the data to get a better understanding of the ratings and movies datasets. Users have at least 5 ratings and on average of 82 ratings. And movies have at least 1 rating with an average of 6 ratings.
<p align="center">	
	<img align="middle" src="images/Figure 4. Data Exploration.png">
</p>
<p align="center">
  <i>Figure 4. Data Exploration</i> 
</p>


One thing to note is that Spark's implementation of ALS algorithm requires that movieIds and userIds be provided as integer or long datatypes. So the datasets need to be prepared accordingly in order for them to function properly with Spark.
<p align="center">	
	<img align="middle" src="images/Figure 5. Schema Examination.png">
</p>
<p align="center">
  <i>Figure 5. Schema Examination</i> 
</p>


### 4. Cross-Validated ALS Model and Recommendations
I first split the data into a training set and testing set and trained the matrix factorization model using the ALS algorithm. Additionally, I tuned the ALS model by sweeping over 27 hyperparameter combinations. All combinations of ranks, lambdas, and iterations are run to see which has the lowest RMSE (Root Mean Squared Error) against the validation model. The model with the lowest RMSE is evaluated against the testing set of data. The model took me 1h 6min 33s to train and the final RMSE is 0.964, much lower than 1.3 of the initial ALS model without tuning. For those who are not familiar with the RMSE metric, an RMSE of 0.964 means that on average the model predicts 0.964 above or below values of the original ratings data. For simplicity, I would show the best model directly. However, the code used in this blog post can be found on GitHub.
<p align="center">	
	<img align="middle" src="images/Figure 6. Best Model and Best Model Parameters.png">
</p>
<p align="center">
  <i>Figure 6. Best Model and Best Model Parameters</i> 
</p>

<p align="center">	
	<img align="middle" src="images/Figure 7. Recommendation Results.png">
</p>
<p align="center">
  <i>Figure 7. Recommendation Results</i> 
</p>

From Fugure 7, we see that how the ALS Model is predicting on and how well it is generalizing to the testing data. And yes, my initial recommendation engine is completed. 



## Getting Recommendations
We can also take a quick look on our recommendations. Now that we have some confidence that the model will provide recommendations that are relevant to users, we can also look at recommendations made to a user and see if they make sense. For example, I looked into user 50's original ratings, and compared them to what my model recommended for them.

And here's my model's recommendations:

Obviously, user 50 is quite into drama and action movie, isn't he?

## Reference
- https://grouplens.org/datasets/movielens/
- Collaborative Filtering - RDD-based API
- Wikipedia-Recommender system
- Building Recommendation Engines with PySpark
