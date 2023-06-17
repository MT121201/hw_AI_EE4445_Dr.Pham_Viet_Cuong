## Homework 1
**Topic:** Write a program implementing the perceptron learning algorithm.

The task at hand is to implement the perceptron learning algorithm to classify whether an animal is a cat or a dog based on its size and weight.\
Numerical features are used as inputs for classification.\
The program is designed to train the perceptron model on a given dataset of animals with their corresponding labels and then make predictions on new test inputs. 

## Homework 3
**Topic:** Write a program implementing feature normalization. Find and demonstrate a real-world example.

I use Python randomly make a dataset for  20 students' CGPA and their Salary, and then normalize it

| CGPA | Salary |
|------|--------|
| 3.14 | 65126  |
| 1.73 | 45567  |
| 2.10 | 71234  |
| 2.59 | 36487  |
| 3.70 | 50211  |

## Homework 4
**Topic:** Write a program implementing the linear regression algorithm to solve a freely chosen problem.\
I using a small simple Predicting Sales Revenue dataset for this problem (d4.csv)

| Advertising Spend (Online Ads) | Advertising Spend (Social Media) | Advertising Spend (Email Marketing) | Sales Revenue |
|-------------------------------|----------------------------------|------------------------------------|---------------|
| 1000                          | 500                              | 200                                | 2500          |
| 800                           | 700                              | 300                                | 2200          |
| 1200                          | 900                              | 400                                | 3200          |
| 1500                          | 600                              | 100                                | 2800          |
| 900                           | 800                              | 250                                | 2100          |
| ...                           | ...                              | ...                     

## Homework 5
**Topic:** Write a program implementing kNN algorithm solving a freely chosen problem

I generate the dataset below for implementing kNN (k=5)  (customer_churn_dataset.csv)

| Customer ID | Gender | Age | Monthly Charges | Total Charges | Number of Calls | Number of Messages | Churn |
|-------------|--------|-----|----------------|---------------|-----------------|--------------------|-------|
| 1           | Male   | 35  | 50             | 500           | 100             | 20                 | 0     |
| 2           | Female | 42  | 70             | 800           | 200             | 50                 | 1     |
| 3           | Male   | 28  | 40             | 300           | 50              | 10                 | 0     |
| 4           | Female | 55  | 80             | 1500          | 300             | 100                | 1     |
| 5           | Male   | 20  | 30             | 100           | 10              | 5                  | 0     |

## Homework 6
**Topic:** Write a program implementing a k-means clustering algorithm solving a freely chosen problem

I perform customer segmentation using the K-means clustering algorithm (k=2) with the simple custom dataset (d6.csv)

| Customer ID | Age | Gender | Annual Income ($) | Spending Score (0-100) |
|-------------|-----|--------|-------------------|-----------------------|
| 1           | 25  | Male   | 40,000            | 75                    |
| 6           | 55  | Female | 100,000           | 65                    |
| 7           | 20  | Male   | 30,000            | 80                    |
| 13          | 50  | Female | 80,000            | 15                    |
| 16          | 28  | Female | 32,000            | 80                    |

## Homework 7
**Topic:** Write a program implementing K++ algorithm (centroid initialization only)

I perform customer segmentation using the K++ algorithm (k=2) with the new distribution custom dataset (d7.csv)

## Homework 8
**Topic:** Use SVM, soft margin SVM, kernel SVM tools solving a freely chosen problem

I used a synthetic dataset generated using the make_blobs function from scikit-learn

## Homework 8
**Topic:** Write a programe implementing Q learning algorithm solving a freely chosen problem (except the problem used as example in the lecture)

I use this matrix to find path From Start(S) reach Goal(W) and avoid Hold(H)

|   | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | S |   |   |   |
| 1 | H |   |   | H |
| 2 |   |   |   |   |
| 3 |   | H |   | W |
