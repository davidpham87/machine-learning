# coding: utf-8

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().magic('matplotlib tk')

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

# Success
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))

# ## Data Exploration

minimum_price = np.min(prices)
maximum_price = np.max(prices)
mean_price = np.mean(prices)
median_price = np.median(prices)
std_price = np.std(prices)

# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${:,.2f}".format(minimum_price))
print("Maximum price: ${:,.2f}".format(maximum_price))
print("Mean price: ${:,.2f}".format(mean_price))
print("Median price ${:,.2f}".format(median_price))
print("Standard deviation of prices: ${:,.2f}".format(std_price))

# ### Question 1 - Feature Observation

# - `'RM'` is the average number of rooms among homes in the neighborhood.
# - `'LSTAT'` is the percentage of homeowners in the neighborhood considered "lower class" (working poor).
# - `'PTRATIO'` is the ratio of students to teachers in primary and secondary schools in the neighborhood.

# ** Using your intuition, for each of the three features above, do you
# ** think that an increase in the value of that feature would lead to an
# ** **increase** in the value of `'MEDV'` or a **decrease** in the value
# ** of `'MEDV'`? Justify your answer for each.**

# **Answer: **

#  - An increase of `RM` should lead to an increase of `MEDV`, as the
#  - number of rom is a proxy for the size of house and it is expected that
#  - bigger house are more expensive.

# - A increase of `LSTAT` should lead to a decrease of `MEDV` as house
# - prices are subject to supply and demand mechanics and it is commonly
# - accepted that consumer are ready to pay premium to live in richer
# - neighborhood. Supposedly, high income area tends to have better
# - infrastructure and schools, explained by the fact that government can
# - collect more taxes.

# - Consequently, an increase `PTRATIO` should lead to a decrease of `MDEV`
# - as this metric is a proxy for the quality of teaching or amount of
# - money invested in education, and parents are ready to pay a premium for
# - houses in area where school are good.

# ----
#
# ## Developing a Model

from sklearn.metrics import r2_score

# Calculates and returns the performance score between
# true and predicted values based on the metric chosen.
performance_metric = r2_score

# ### Question 2 - Goodness of Fit
# Assume that a dataset contains five data points and a model made the following predictions for the target variable:
#
# | True Value | Prediction |
# | :-------------: | :--------: |
# | 3.0 | 2.5 |
# | -0.5 | 0.0 |
# | 2.0 | 2.1 |
# | 7.0 | 7.8 |
# | 4.2 | 5.3 |
#
# Run the code cell below to use the `performance_metric` function and calculate this model's coefficient of determination.

# In[ ]:

# Calculate the performance of this model
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print("Model has a coefficient of determination, R^2, of {:.3f}.".format(score))

# * Would you consider this model to have successfully captured the variation of the target variable?
# * Why or why not?
#
# **Answer:**
# Yes, the model does a good job to capture the variation of the target
# variable. A R2 score of 0.93 means that the variance of the errors from
# predicted values is 7% of the variance of the target variable. So the model
# gives much better predictions than the mean of the target variable.

# ### Implementation: Shuffle and Split Data
# Your next implementation requires that you take the Boston housing dataset and split the data into training and testing subsets. Typically, the data is also shuffled into a random order when creating the training and testing subsets to remove any bias in the ordering of the dataset.
#
# For the code cell below, you will need to implement the following:
# - Use `train_test_split` from `sklearn.cross_validation` to shuffle and split the `features` and `prices` data into training and testing sets.
#   - Split the data into 80% training and 20% testing.
#   - Set the `random_state` for `train_test_split` to a value of your choice. This ensures results are consistent.
# - Assign the train and testing splits to `X_train`, `X_test`, `y_train`, and `y_test`.

# In[ ]:

from sklearn.model_selection import train_test_split
RANDOM_STATE = 42
X_train, X_test, y_train, y_test = (
    train_test_split(features, prices, test_size=0.2, random_state = RANDOM_STATE))
print("Training and testing split was successful.")


# ### Question 3 - Training and Testing
# * What is the benefit to splitting a dataset into some ratio of training
# * and testing subsets for a learning algorithm?

# **Answer: **

# It is desirable to avoid overfitting, as the model can basically memorize
# all the target for a given input if it has enough
# capacity/parameters/complexity.  Splitting the dataset insures that we have
# an unobserved dataset on which the prediction and generlization performance
# of a trained model can be evaluted

# ----
#
# ## Analyzing Model Performance
# ### Learning Curves
# Produce learning curves for varying training set sizes and maximum depths
vs.ModelLearning(features, prices)

# ### Question 4 - Learning the Data
# * Choose one of the graphs above and state the maximum depth for the model.
# * What happens to the score of the training curve as more training points are added? What about the testing curve?
# * Would having more training points benefit the model?
#
# **Answer: **

# Let the model with a max depth of 3 be the selected model. As expected the training curve decreases as more training points are added while the testing curve increases. Adding additional training points would not improve the performance of the model as both curves converges to the score in the neighborhood of 0.8

# ### Complexity Curves
vs.ModelComplexity(X_train, y_train)

# ### Question 5 - Bias-Variance Tradeoff
# * When the model is trained with a maximum depth of 1, does the model suffer from high bias or from high variance?
# * How about when the model is trained with a maximum depth of 10? What visual cues in the graph justify your conclusions?
#
# **Hint:** High bias is a sign of underfitting(model is not complex enough to pick up the nuances in the data) and high variance is a sign of overfitting(model is by-hearting the data and cannot generalize well). Think about which model(depth 1 or 10) aligns with which part of the tradeoff.

# **Answer: **
# The model with a depth of 1 suffer from high bias (but low variance as they have around the same score). In contrast, the model with depth 10 obviously from high variance as the difference between the training and validation score is big. Visual cues for our choices are that the training curves for model with depth 1 achieved a low accuracy, around 0.45, whereas the model of depth 10 had an oustanding training score of almost 0.95, whereas the testing curve was capped at around 0.7, showing that the model failed to achieve the same performance with new observations. This is a sign of over-fitting.

# ### Question 6 - Best-Guess Optimal Model
# * Which maximum depth do you think results in a model that best generalizes to unseen data?
# * What intuition lead you to this answer?
#
# ** Hint: ** Look at the graph above Question 5 and see where the validation scores lie for the various depths that have been assigned to the model. Does it get better with increased depth? At what point do we get our best validation score without overcomplicating our model? And remember, Occams Razor states "Among competing hypotheses, the one with the fewest assumptions should be selected."

# **Answer: **
# A model with a maximum depth of 3 sounds reasonable as the variance is small , and the validation score is as high as with a depth of 4 and higher than a depth of 2. A depth of 3 is prefered to 4 as there is one less dimension in the model.
# -----
#
# ## Evaluating Model Performance
# In this final section of the project, you will construct a model and make a prediction on the client's feature set using an optimized model from `fit_model`.

# ### Question 7 - Grid Search
# * What is the grid search technique?
# * How it can be applied to optimize a learning algorithm?
#
# ** Hint: ** When explaining the Grid Search technique, be sure to touch upon why it is used,  what the 'grid' entails and what the end goal of this method is. To solidify your answer, you can also give an example of a parameter in a model that can be optimized using this approach.

# **Answer: **

# As illustrated by the
# [scikit](http://scikit-learn.org/stable/modules/grid_search.html) function
# `GridSearchCV`, the grid search techinque is a method that compute the best
# hyperparameters among the cartesian product of the (restricted space of)
# hyperparameters. More precisely, candidates for each dimension of
# hyperparameters (in the example of decision tree, there is only one dimension
# the max depth) are selected and then the grid search generate the cartesian
# product of all the dimension of the hyperparameters, whose members are used to
# compute a score. The element whose score is the best is then selected as the
# best set of hyperparameters. This is useful for learning algorithm as it is
# rather simple (and embarassingly parallel) algorithm to implement in order to
# improve the prediction ability of learning models.

# ### Question 8 - Cross-Validation
#
# * What is the k-fold cross-validation training technique?
#
# * What benefit does this technique provide for grid search when optimizing a model?
#
# **Hint:** When explaining the k-fold cross validation technique, be sure to touch upon what 'k' is, how the dataset is split into different parts for training and testing and the number of times it is run based on the 'k' value.
#
# When thinking about how k-fold cross validation helps grid search, think about the main drawbacks of grid search which are hinged upon **using a particular subset of data for training or testing** and how k-fold cv could help alleviate that. You can refer to the [docs](http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation) for your answer.

# **Answer: **

A model is usually assessed by its ability to predict new observations. With
this respect, a part of the original data set is put aside, the so-called
*test* set, is hidden to the model when its parameters are trained. When using,
grid search, we implicity use the test set as a part of the training phase, as
we compare the error of the different hyperparameters on the test set, which
goes against its definition. _k-fold_ cross validation is a technique to
prevent this: instead of splitting the original data set in two unbalanced
quanitity, we add another layer of splitting. The training data set is split
into _k_ sets of approximatively the same size, then _k-1_ sets are used to
train the model and the generalisation error is asssesd on the set left out of
the data input of the model. We repeat the process _k_ times by selecting a
different test set each time in order to get _k_ estimates of the
generalization error. A final estimate is given by taking the average.

This benefits grid search as it reduce the variance of the estimator introduced by
selecting a fix validation set.



# ### Implementation: Fitting a Model
# Your final implementation requires that you bring everything together and
# train a model using the **decision tree algorithm**. To ensure that you are
# producing an optimized model, you will train the model using the grid search
# technique to optimize the `'max_depth'` parameter for the decision tree. The
# `'max_depth'` parameter can be thought of as how many questions the decision
# tree algorithm is allowed to ask about the data before making a
# prediction. Decision trees are part of a class of algorithms called
# *supervised learning algorithms*.
#
# In addition, you will find your implementation is using `ShuffleSplit()` for
# an alternative form of cross-validation (see the `'cv_sets'` variable). While
# it is not the K-Fold cross-validation technique you describe in **Question
# 8**, this type of cross-validation technique is just as useful!. The
# `ShuffleSplit()` implementation below will create 10 (`'n_splits'`) shuffled
# sets, and for each shuffle, 20% (`'test_size'`) of the data will be used as
# the *validation set*. While you're working on your implementation, think
# about the contrasts and similarities it has to the K-fold cross-validation
# technique.
#
# Please note that ShuffleSplit has different parameters in scikit-learn versions 0.17 and 0.18.
# For the `fit_model` function in the code cell below, you will need to
# implement the following:
# - Use
# - [`DecisionTreeRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
# - from `sklearn.tree` to create a decision tree regressor object.
#   - Assign this object to the `'regressor'` variable.
# - Create a dictionary for `'max_depth'` with the values from 1 to 10, and
# - assign this to the `'params'` variable.
# - Use
# - [`make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html)
# - from `sklearn.metrics` to create a scoring function object.
#   - Pass the `performance_metric` function as a parameter to the object.
#   - Assign this scoring function to the `'scoring_fnc'` variable.
# - Use
# - [`GridSearchCV`](http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html)
# - from `sklearn.grid_search` to create a grid search object.
#   - Pass the variables `'regressor'`, `'params'`, `'scoring_fnc'`, and
#   - `'cv_sets'` as parameters to the object.  Assign the `GridSearchCV`
#   - object to the `'grid'` variable.

from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)

    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': list(range(1, 11))}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search cv object --> GridSearchCV()
    # Make sure to include the right parameters in the object:
    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.
    grid = GridSearchCV(regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


# ### Making Predictions

# Once a model has been trained on a given set of data, it can now be used to
# make predictions on new sets of input data. In the case of a *decision tree
# regressor*, the model has learned *what the best questions to ask about the
# input data are*, and can respond with a prediction for the **target
# variable**. You can use these predictions to gain information about data
# where the value of the target variable is unknown — such as data the model
# was not trained on.

# ### Question 9 - Optimal Model
#
# * What maximum depth does the optimal model have? How does this result
# * compare to your guess in **Question 6**?
#
# Run the code block below to fit the decision tree regressor to the training
# data and produce an optimal model.

# In[ ]:

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])


# ** Hint: ** The answer comes from the output of the code snipped above.
#
# **Answer: **
# Parameter 'max_depth' is 4 for the optimal model. As guessed in question 6, this was expected as the model with depth 4 had better prediction abilitiy (at the cost of more overfitting/variance).

# ### Question 10 - Predicting Selling Prices
# Imagine that you were a real estate agent in the Boston area looking to use this model to help price homes owned by your clients that they wish to sell. You have collected the following information from three of your clients:
#
# | Feature | Client 1 | Client 2 | Client 3 |
# | :---: | :---: | :---: | :---: |
# | Total number of rooms in home | 5 rooms | 4 rooms | 8 rooms |
# | Neighborhood poverty level (as %) | 17% | 32% | 3% |
# | Student-teacher ratio of nearby schools | 15-to-1 | 22-to-1 | 12-to-1 |
#
# * What price would you recommend each client sell his/her home at?
# * Do these prices seem reasonable given the values for the respective features?
#
# **Hint:** Use the statistics you calculated in the **Data Exploration**
# **section to help justify your response.  Of the three clients, client 3 has
# **has the biggest house, in the best public school neighborhood with the
# **lowest poverty level; while client 2 has the smallest house, in a
# **neighborhood with a relatively high poverty rate and not the best public
# **schools.
#
# Run the code block below to have your optimized model make predictions for each client's home.

# Produce a matrix for client data
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print "Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price)


# **Answer: **
# The recommended prices are $403k for client 1, $237k for client 2 and $932k for
# client 3. These prices seems reasonable with respect to the exploratory data
# analysis. The third client posses a house in a well frequented neighbourhood
# with low student-teacher ratio in nearby schools and low poverty level. Hence
# it is expected that the price is near the maximum price. In contrast, client 2
# has a house where the two aforementioned measure are much higher and seems to
# illustrate a poor area. Client 1 has a price near the median/mean of the home
# prices, hence it seems reasonable as a prediction.




# ### Sensitivity
# An optimal model is not necessarily a robust model. Sometimes, a model is either too complex or too simple to sufficiently generalize to new data. Sometimes, a model could use a learning algorithm that is not appropriate for the structure of the data given. Other times, the data itself could be too noisy or contain too few samples to allow a model to adequately capture the target variable — i.e., the model is underfitted.
#
# **Run the code cell below to run the `fit_model` function ten times with different training and testing sets to see how the prediction for a specific client changes with respect to the data it's trained on.**

# In[ ]:

vs.PredictTrials(features, prices, fit_model, client_data)


# ### Question 11 - Applicability
#
# * In a few sentences, discuss whether the constructed model should or should not be used in a real-world setting.
#
# **Hint:** Take a look at the range in prices as calculated in the code snippet above. Some questions to answering:
# - How relevant today is data that was collected from 1978? How important is
# - inflation?
# - Are the features present in the data sufficient to describe a home? Do you
#  think factors like quality of apppliances in the home, square feet of the
#  plot area, presence of pool or not etc should factor in?
# - Is the model robust enough to make consistent predictions?
# - Would data collected in an urban city like Boston be applicable in a rural city?
# - Is it fair to judge the price of an individual home based on the characteristics of the entire neighborhood?

# **Answer: **

# In a real world setting, the model should not be used as it is now specified.

# Inflation
# is clearly a big factor and it has been quite nonlinear in the past forty
# years. Maybe original prices could standardized, and then scaled back to actual
# prices with new measures, as the two features are still relevant today. Another
# reason to avoid using the data is the ignorance of the properties of the house
# itself: the size and type of the house, the quality of construction and size of
# the garden should be incorporated as they might have an impact on the price.

# Obviously, the model can predict prices in a similar environment to the data
# from which it was collected: we collected data from the city of Boston and the
# relationships between the dependent dimensions and the prices should remain,
# but the prices classification must vary depending the city or the
# region. Prices in New York should be higher, but in a rural city the features
# might have less impact than some other measures.

# Using characteristics external from the house to determine its prices is a fair
# practice, although one should definitively incorporate more specification of
# the house (a single room flat should be much cheaper than a twelve rooms
# mansion).

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
