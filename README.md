[![Downloads](https://pepy.tech/badge/safe-transformer)](https://pepy.tech/project/safe-transformer)
[![PyPI version](https://badge.fury.io/py/safe-transformer.svg)](https://badge.fury.io/py/safe-transformer)


# SAFE - Surrogate Assisted Feature Extraction

SAFE is a python library that you can use to build better explainable ML models leveraging capabilities of more powerful, black-box models. 
The idea is to use more complicated model - called surrogate model - to extract more information from features, which can be used later to fit some simpler but explainable model.
Input data is divided into intervals or new set of categories, determined by surrogate model, and then it is transformed based on the interval or category each point belonged to.
Library provides you with SafeTransformer class, which implements TransformerMixin interface, so it can be used as a part of the scikit-learn pipeline.
Using this library you can boost simple ML models, by transferring informations from more complicated models.
Article about SAFE on can be found [here](https://arxiv.org/abs/1902.11035).

## Requirements

To install this library run:

```
pip install safe-transformer
```

The only requirement is to have Python 3 installed on your machine.

## Usage with example

Sample code using SAFE transformer as part of scikit-learn pipeline:

```python
from SafeTransformer import SafeTransformer
from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

data = load_boston()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y)

surrogate_model = GradientBoostingRegressor(n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    loss='huber')
surrogate_model = surrogate_model.fit(X_train, y_train)

linear_model = LinearRegression()
safe_transformer = SafeTransformer(surrogate_model, penalty = 0.84)
pipe = Pipeline(steps=[('safe', safe_transformer), ('linear', linear_model)])
pipe = pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
mean_squared_error(y_test, predictions)

```

```bash
13.617733207161479
```

```python
linear_model_standard = LinearRegression()
linear_model_standard = linear_model_standard.fit(X_train, y_train)
standard_predictions = linear_model_standard.predict(X_test)
mean_squared_error(y_test, standard_predictions)
```

```bash
29.27790566931337
```

As you can see you can improve your simple model performance with help of the more powerful, black-box model, keeping the interpretability of the simple model.

You can use any model you like, as long as it has fit and predict methods in case of regression, or fit and predict_proba in case of classification. Data used to fit SAFE transformer needs to be pandas data frame. 

You can also specify penalty and pelt model arguments.

In [examples folder](https://github.com/olagacek/SAFE/tree/master/examples) you can find jupyter notebooks with complete classification and regression examples.

API reference documentation can be found [here](https://plubon.github.io/safe-docs/)

## Algorithm

Our goal is to divide each feature into intervals or new categories and then transform feature values based on the subset they belonged to. 
The division is based on the response of the surrogate model. 
In case of continuous dependent variables for each of them we find changepoints - points that indicate values of variable for which the response of the surrogate model changes quickly. Intervals between changepoints are the basis of the transformation, eg. feature is transformed to categorical variable, where feature values in the same interval form the same category. To find changepoints we need partial dependence plots. 
These plots describe the marginal effect of a given variable (or multiple variables) on an outcome of the model.
In case of categorical variables for each of them we perform hierarchical clustering based on surrogate model responses. Then, based on the biggest similarity in response between categories, they are merged together forming new categories.


Algorithm for performing fit method is illustrated below:

&nbsp;&nbsp;

![*Fit method algorithm*](images/fl.svg)

&nbsp;&nbsp;

Our algorithm works both for regression and classification problems. In case of regression we simply use model response for creating partial dependence plot and hierarchical clustering. As for classification we use predicted probabilities of each class.

### Continuous variable transformation

Here is example of partial dependence plot. It was created for boston housing data frame, variable in example is LSTAT. To get changepoints from partial dependence plots we use [ruptures](http://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/index.html) library and its model [Pelt](http://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/detection/pelt.html).

<img src="images/simple-plot.png" width="425"/> <img src="images/changepoint.png" width="425"/> 

### Categorical variable transformation

In the plot below there is illustrated categorical variable transformation. To create new categories, based on the average model responses, we use scikit-learn [ward algorithm](https://scikit-learn.org/0.15/modules/generated/sklearn.cluster.Ward.html) and to find number of clusters to cut KneeLocator class from [kneed library](https://github.com/arvkevi/kneed) is used.

<img src="images/categorical.png" width="425"/> <img src="images/dendo.png" width="425"/> 

## Model optimization

One of the parameters you can specify is penalty - it has an impact on the number of changepoints that will be created. Here you can see how the quality of the model changes with penalty. For reference results of surrogate and basic model are also in the plot.

&nbsp;&nbsp;
<img src="images/pens.png" alt="Model performance" width="500"/>
&nbsp;&nbsp;

With correctly chosen penalty your simple model can achieve much better accuracy, close to accuracy of surrogate model.

## Variables transformation

If you are interested in how your dataset was changed you can check summary method. 

```python
safe_transformer.summary(variable_name='CRIM')
```

```
Numerical Variable CRIM
Selected intervals:
	[-Inf, 4.90)
	[4.90, 11.14)
	[11.14, 15.59)
	[15.59, 24.50)
	[24.50, 33.40)
	[33.40, 48.54)
	[48.54, Inf)
```

To see transformations of all the variables do not specify variable_name argument.

```python
safe_transformer.summary()
```

```
Numerical Variable CRIM
Selected intervals:
	[-Inf, 4.90)
	[4.90, 11.14)
	[11.14, 15.59)
	[15.59, 24.50)
	[24.50, 33.40)
	[33.40, 48.54)
	[48.54, Inf)
Numerical Variable ZN
Selected intervals:
	[-Inf, 33.53)
	[33.53, Inf)
Numerical Variable INDUS
Selected intervals:
	[-Inf, 2.78)
	[2.78, 3.19)
	[3.19, 4.28)
	[4.28, 10.29)
	[10.29, 26.68)
	[26.68, Inf)
	
.
.
.

Numerical Variable LSTAT
Selected intervals:
	[-Inf, 4.55)
	[4.55, 4.73)
	[4.73, 5.43)
	[5.43, 5.96)
	[5.96, 7.55)
	[7.55, 8.08)
	[8.08, 9.67)
	[9.67, 9.85)
	[9.85, 10.02)
	[10.02, 14.43)
	[14.43, 14.96)
	[14.96, 16.02)
	[16.02, 18.14)
	[18.14, 19.37)
	[19.37, 23.96)
	[23.96, 26.78)
	[26.78, 29.61)
	[29.61, Inf)
```
## References

* [Original Safe algorithm](https://mi2datalab.github.io/SAFE/index.html), implemented in R 
* [ruptures library](https://github.com/deepcharles/ruptures), used for finding changepoints
* [kneed library](https://github.com/arvkevi/kneed), used for cutting hierarchical tree 
* [SAFE article](https://arxiv.org/abs/1902.11035) - article about SAFE algorithm, including benchmark results using SAFE library

The project was made on [research workshops classes](https://github.com/pbiecek/CaseStudies2019W) at the Warsaw University of Technology at the Faculty of Mathematics and Information Science by Aleksandra Gacek and Piotr Luboń.
