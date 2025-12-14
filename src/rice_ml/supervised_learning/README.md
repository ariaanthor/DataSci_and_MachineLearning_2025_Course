## Supervised Machine Learning Overview
Supervised learning is a class of machine learning where the goal is to learn a function that maps input data to output labels using labeled examples. These algorithms are trained on a dataset that includes both the inputs and the correct outputs, and the model learns to generalize to new, unseen examples.

This project explores a variety of supervised learning algorithms, each with different strengths and applications:

### The Perceptron
The Perceptron is one of the earliest binary classifiers. It learns a linear decision boundary by adjusting weights through a simple rule based on misclassified examples. Though limited to linearly separable data, it's foundational to understanding more complex models like neural networks.
Dataset used: A 2D projection of the UCI German Credit dataset, with credit risk as the binary target.

### Linear Regression
Linear Regression is used for predicting continuous outcomes. It models the relationship between one or more features and a continuous target variable by fitting a linear equation. It assumes a linear relationship and minimizes the mean squared error between predicted and actual values.
Dataset used: Ames Housing dataset, with house sale price as the target.

### Logistic Regression
Despite its name, Logistic Regression is used for classification. It models the probability that a given input belongs to a particular category using the logistic (sigmoid) function. It's particularly effective for binary classification tasks.
Dataset used: Give Me Some Credit dataset, predicting whether an individual experienced financial distress.

### Neural Networks
Neural Networks are a class of models inspired by the human brain. They consist of layers of interconnected "neurons" and are capable of capturing complex, non-linear relationships in data. Training is typically done via backpropagation and gradient descent.
Dataset used: Give Me Some Credit dataset, used to train a multi-layer perceptron for binary classification.

### K Nearest Neighbors (KNN)
KNN is a non-parametric, instance-based learning algorithm. It makes predictions based on the majority label (classification) or average value (regression) among the k closest training examples in the feature space. It’s simple, interpretable, and effective with well-structured data.
Dataset used: Give Me Some Credit dataset, with scaled numeric features and binary labels.

### Decision Trees / Regression Trees
Decision Trees split the data into regions based on feature thresholds to make predictions. For classification, the leaf nodes correspond to class labels; for regression, they output average values. Trees are interpretable but can overfit without pruning.
Dataset used: Lending Club dataset, used to classify loans as fully paid or charged off.

### Random Forests
Random Forests are ensemble methods that build multiple decision trees using bootstrapped subsets of the data and random subsets of features. The final prediction is made by averaging (regression) or voting (classification) across all trees. This reduces variance and improves generalization.
Dataset used: Lending Club dataset, with the same loan classification task as decision trees.

### Other Ensemble Methods (e.g., Boosting)
Boosting methods, such as AdaBoost and Gradient Boosting, combine weak learners in a sequential manner. Each new model corrects the errors of its predecessors. These methods are powerful for reducing both bias and variance and often yield state-of-the-art performance.
Dataset used: A macroeconomic dataset built from FRED (Federal Reserve Economic Data), used to predict the 10-Year Treasury Yield.