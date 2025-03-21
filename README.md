# Custom-Logistic-Regression

This repository compares a custom Logistic Regression implementation of binary classification to Scikit-learn's 
Logistic Regression.
The custom implementation uses Gradient Descent for optimizing the weights, and it is compared to scikit-learn's 
optimized newton-cg solver with l2 regularization.

The comparison is performed on the dummy breast cancer dataset, which is made up of 30 features and 569 observations, and 
is accessed through the sklearn.datasets package

# Usage

Run the evaluation script:

    python evaluate.py

Expected outputs:
- Scatter plot of predicted probabilities of class affiliation with the actual class of each observation represented by 
the color of the points for the custom logistic regression model. Accuracy is displayed on the right-bottom corner of 
the plot
- Same scatter plot but for scikit-learn's logistic regression model

# Results

- This is the scatter plot after using sci-kit learn's logistic regression model using the newton-cg solver and l2 
penalty
    ![Image](https://github.com/user-attachments/assets/86e843c7-51c2-45f5-9f4e-985df8b4784e)

- This is the scatter plot after uisng the custom logistic regression model, which uses Gradient Descent as a solver, 
which might explain why the probability values don't converge as precisely as sci-kit learn's optimized solvers
    ![Image](https://github.com/user-attachments/assets/e946df60-7def-4731-8ada-41a2de826399)

# Repository Structure

This repository contains:

    custom_logistic_regression.py: Implementation of the CustomLogisticRegressionClass
    evaluate.py: Main script for performing the comparisons on the dummy breast cancer Dataset and generating plots
    requirements.txt: List of required Python packages

Python 3.12 version was used
