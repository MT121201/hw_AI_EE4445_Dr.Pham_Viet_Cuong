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
I using a small simple Predicting Hotle prices dataset for this problem (sea_hotel.csv)



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

I perform customer segmentation using the K-means clustering algorithm (k=3) with the simple dataset using sklearn make_blobs


## Homework 7
**Topic:** Write a program implementing K++ algorithm (centroid initialization only)

I perform customer segmentation using the K++ algorithm (k=3) with the new distribution custom dataset 

## Homework 8
**Topic:** Use SVM, soft margin SVM, kernel SVM tools solving a freely chosen problem

I used a synthetic dataset generated using the make_blobs function from scikit-learn

## Homework 9
**Topic:** Write a program implementing Q learning algorithm solving a freely chosen problem (except the problem used as example in the lecture)

I use Q_learning to find path From Start(S) reach Goal(G) and avoid Hold(X)

|   | 0 | 1 | 2 | 3 |4|
|---|---|---|---|---|---|
| 0 | S |   |   |   |   |
| 1 | X |   |   | X |   |
| 2 |   |   |   |   |   |
| 3 |   |   |   |   |   |
| 4 |   | X |   |   | G |

## Homework 12
**Topic:** Write a program implementing gradient descent method. Consider at least two non-convex two-variable functions f(x,y), multiple initial points, and various learning rates

I will use the function below, a well-known example of a non-convex function.\
`f(x, y) = x^2 + y^4 - 2xy + x - y`

