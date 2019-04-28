import numpy as np
import ruptures as rpt
from sklearn.base import TransformerMixin
import pandas as pd
from sklearn.exceptions import NotFittedError
from scipy.cluster.hierarchy import ward, cut_tree
from kneed import KneeLocator
import sys


class Variable():

	def __init__(self, name, index):
		self.original_name = name
		self.original_index = index
		self.new_names = None


class NumericVariable(Variable):

	def __init__(self, name, index, penalty, pelt_model, no_changepoint_strategy='median'):
		super().__init__(name, index)
		self.changepoints = []
		self.penalty = penalty
		self.pelt_model = pelt_model
		self.changepoint_values = []
		self.no_changepoint_strategy = no_changepoint_strategy

	def _get_partial_dependence(self, model, X, grid_resolution=1000):
		axes = []
		pdp = []
		points = np.linspace(min(X.loc[:, self.original_name]), max(
		    X.loc[:, self.original_name]), grid_resolution)
		X_copy = X.copy()
		for point in points:
			axes.append(point)
			X_copy.loc[:, self.original_name] = point
			if(hasattr(model, 'predict_proba')):
				predictions = model.predict_proba(X_copy)
			else:
				predictions = model.predict(X_copy)
			val = np.mean(predictions, axis=0)
			pdp.append(val)
		return np.array(pdp), axes

	def fit(self, model, X, verbose):
		if verbose:
			print('Fitting variable:' + str(self.original_name))
		pdp, axis = self._get_partial_dependence(model, X, grid_resolution=1000)
		algo = rpt.Pelt(model=self.pelt_model).fit(pdp)
		self.changepoints = algo.predict(pen=self.penalty)
		self.changepoint_values = [axis[i] for i in self.changepoints[:-1]]
		if not self.changepoint_values and self.no_changepoint_strategy == 'median':
			self.changepoint_values = [np.median(X)]
		changepoint_names = ['%.2f' % self.changepoint_values[i]
		    for i in range(len(self.changepoint_values))] + ['Inf']
		self.new_names = [str(self.original_name) + "_[" + changepoint_names[i] + ", " +
			changepoint_names[i + 1] + ")" for i in range(len(changepoint_names) - 1)]
		return self

	def transform(self, X, verbose):
		if verbose:
			print('Transforming variable:' + str(self.original_name))
		new_data = [len(list(filter(lambda e: x >= e, self.changepoint_values)))
		                for x in X.loc[:, self.original_name]]
		ret = np.zeros([len(new_data), len(self.changepoint_values)])
		for row_num, val in enumerate(new_data):
			if val > 0:
				ret[row_num, val - 1] = 1
		return pd.DataFrame(ret, columns=self.new_names)

	def summary(self):
		summary = 'Numerical Variable ' + self.original_name + '\n'
		summary += 'Selected intervals:\n'
		changepoint_names = ['%.2f' % self.changepoint_values[i]
		    for i in range(len(self.changepoint_values))] + ['Inf']
		interval_names = ['\t[-Inf, ' + changepoint_names[0] + ')']
		interval_names += ["\t[" + changepoint_names[i] + ", " +
			changepoint_names[i + 1] + ")" for i in range(len(changepoint_names) - 1)]
		summary += '\n'.join(interval_names)
		return summary


class CategoricalVariable(Variable):

	def __init__(self, name, index, dummy_names, levels):
		super().__init__(name, index)
		self.dummy_names = dummy_names
		self.axes = None
		self.clusters = None
		self.Z = None
		self.pdp = None
		self.levels = levels
		self.levels.sort()

	def fit(self, model, X, verbose):
		if verbose:
			print('Fitting variable:' + str(self.original_name))
		pdp, names = self._get_partial_dependence(model, X)
		self.pdp = pdp
		self.axes = names
		if pdp.ndim == 1:
			arr = np.reshape(pdp, (len(pdp), 1))
		else:
			arr = pdp
		self.Z = ward(arr)
		if pdp.shape[0] == 3:
			self.clusters = cut_tree(
			    self.Z, height=self.Z[0, 2] - sys.float_info.epsilon)
			self.new_names = []
			for cluster in range(len(np.unique(self.clusters))):
				names = []
				for idx, c_val in enumerate(self.clusters):
					if c_val == cluster:
						if idx == 0:
							names.append('base')
						else:
							names.append(self.dummy_names[idx - 1][len(self.original_name) + 1:])
				self.new_names.append(self.original_name + '_' + "_".join(names))
		elif pdp.shape[0] > 3:
			kneed = KneeLocator(
			    range(self.Z.shape[0]), self.Z[:, 2], direction='increasing', curve='convex')
			if kneed.knee is not None:
				self.clusters = cut_tree(
				    self.Z, height=self.Z[kneed.knee + 1, 2] - sys.float_info.epsilon)
				self.new_names = []
				for cluster in range(len(np.unique(self.clusters))):
					names = []
					for idx, c_val in enumerate(self.clusters):
						if c_val == cluster:
							if idx == 0:
								names.append('base')
							else:
								names.append(self.dummy_names[idx - 1][len(self.original_name) + 1:])
					self.new_names.append(self.original_name + '_' + "_".join(names))
		return self

	def transform(self, X, verbose):
		if verbose:
			print('Transforming variable:' + str(self.original_name))
		dummies = pd.get_dummies(
		    X.loc[:, self.original_name], prefix=self.original_name, drop_first=True)
		if self.clusters is not None:
			ret_len = len(np.unique(self.clusters)) - 1
			ret = np.zeros([X.shape[0], ret_len])
			for row_num in range(dummies.shape[0]):
				if not np.sum(dummies.iloc[row_num, :]) == 0:
					idx = np.argwhere(dummies.iloc[row_num, :] == 1)[0]
					if self.clusters[idx + 1] > 0:
						ret[row_num, self.clusters[idx + 1] - 1] = 1
			return pd.DataFrame(ret, columns=self.new_names[1:])
		return dummies

	def _get_partial_dependence(self, model, X):
		pdp = []
		axes = []
		X_copy = X.copy()
		axes.append('base')
		X_copy.loc[:, self.dummy_names] = 0
		if(hasattr(model, 'predict_proba')):
			predictions = model.predict_proba(X_copy)
		else:
			predictions = model.predict(X_copy)
		val = np.mean(predictions, axis=0)
		pdp.append(val)
		for colname in self.dummy_names:
			axes.append(colname)
			X_copy.loc[:, self.dummy_names] = 0
			X_copy.loc[:, colname] = 1
			if(hasattr(model, 'predict_proba')):
				predictions = model.predict_proba(X_copy)
			else:
				predictions = model.predict(X_copy)
			val = np.mean(predictions, axis=0)
			pdp.append(val)
		return np.array(pdp), axes

	def summary(self):
		summary = 'Categorical Variable ' + self.original_name + '\n'
		summary += 'Created variable levels:\n'
		if self.clusters is None:
			for level in self.levels:
				summary += '\t' + level + ' -> ' + level + '\n'
		else:
			for i in range(len(np.unique(self.clusters))):
				names = [self.axes[index]
				    for index in list(np.argwhere(self.clusters == i)[:, 0])]
				if 'base' in names:
					names[names.index('base')] = self.original_name + '_' + self.levels[0]
				names = [x[len(self.original_name) + 1:] for x in names]
				summary += '\t' + ', '.join(names) + ' -> ' + '_'.join(names) + '\n'
		return summary.rstrip()


class SafeTransformer(TransformerMixin):
	"""
	Transform dataset using outputs of complex model. Safe detects which values of variable strongly influence predctions
	and transforms variable into a new, discrete, with each level corresponding to values for which the reponses of surrogate
	model are similiar. Such variables can be fed to another, interpretable model to obtain enchanced predictions compared to using
	standard dataset.

	:param model: Surrogate model. If model is not fitted, y parameter must be passed to fit method. Regression models should implement predict method, whileclassification models should implement predict_proba for getting predictions. If predict_proba  methode exists model is assumed to be a classification model
	:param penalty: Penalty corresponding to adding a new changepoint. The higher the value of penalty the smaller nunber of levels of transformed variableswill be created (Default value = 3)
	:param pelt_model: Cost function used in pelt algorith, possible values: 'l2', 'l1', 'rbf' (Default value = 'l2')
	:param model_params: Dictionary of paramters passed to fit method of surrogate model. Only used if passed surrogate model is not alreedy fitted.
	
	"""

	categorical_dtypes = ['category', 'object']

	def __init__(self, model, penalty=3, pelt_model='l2', model_params={}, no_changepoint_strategy='median'):
		"""
		Initialize new transformer instance

		:param model: Surrogate model. If model is not fitted, y parameter must be passed to fit method. Regression models should implement predict method, while classification models should implement predict_proba for getting predictions. If predict_proba  methode exists model is assumed to be a classification model
		:param penalty: Penalty corresponding to adding a new changepoint. The higher the value of penalty the smaller nunber of levels of transformed variableswill be created (Default value = 3)
		:param pelt_model: Cost function used in pelt algorith, possible values: 'l2', 'l1', 'rbf' (Default value = 'l2')
		:param model_params: Dictionary of parameters passed to fit method of surrogate model. Only used if passed surrogate model is not alreedy fitted.
		:param no_changepoint_strategy: String specifying strategy to take, when no changepoint was detected. Should be one of: 'median', 'no_value'. If median is chosen, then there will be one changepoint set to 'median' value of a column. If 'no_value' is specified column will be removed. Default value = 'median'.
		"""
		self.variables = []
		self.model = model
		self.penalty = penalty
		self.pelt_model = pelt_model
		self.model_params = model_params
		self.is_fitted = False
		if no_changepoint_strategy != 'median' and no_changepoint_strategy != 'no_value':
			raise ValueError('Incorrect no changepoint strategy value. Should be one of: median or no_value.')
		self.no_changepoint_strategy = no_changepoint_strategy

	def _is_model_fitted(self, data):
		try:
			self.model.predict(data.head(1))
			return True
		except NotFittedError as e:
			return False
		except:
			return False

	def fit(self, X, y=None, verbose=False):
		"""
		Fit the transformer. For continous variables intervals for which reponse of surrogate models does not vary are found. For categorical variables average reponses foreach level are found, and then levels with similiar reponse leveles are marked for merging.
		
		:param X: A pandas data frame of predictors. Columns of dtypes category and object are assumed to be categorical variables, while other columns will be treatedas continous variables
		:param y:  A vector of response. Only used if passed surrogate model is not already fitted to fit the surrogate model.(Default value = None)
		:param verbose:  If true logs will be printed.(Default value = False)

		"""
		if not isinstance(X, pd.DataFrame):
			raise ValueError("Data must be a pandas DataFrame")
		if self.is_fitted:
			raise RuntimeError('Model is already fitted')
		colnames = list(X)
		for idx, name in enumerate(colnames):
			if str(X.loc[:, name].dtype) in self.categorical_dtypes:
				levels = np.unique(X.loc[:, name])
				dummies = pd.get_dummies(X.loc[:, name], prefix=name, drop_first=True)
				dummy_index  = X.columns.get_loc(name)
				X = pd.concat([X.iloc[:,range(dummy_index)], dummies, X.iloc[:, range(dummy_index+1, len(X.columns))]], axis=1)
				self.variables.append(CategoricalVariable(name, idx, list(dummies), levels=levels))
			else:
				self.variables.append(NumericVariable(name, idx, self.penalty, self.pelt_model, self.no_changepoint_strategy))
		if not self._is_model_fitted(X):
			self.model.fit(X, y, **self.model_params)
		for variable in self.variables:
			variable.fit(self.model, X, verbose=verbose)
		self.is_fitted = True
		return self

	def transform(self, X, verbose=False):
		"""
		Transforms a data frame of predictors. Continous variables are transformed into a discrete variable, with each level corresponding to an interval of originalvariable for which reponses of surrogate model did not vary. For categorical variables levels with similiar model response will be merged into one new level.All variable are one-hot encoded in p-1 columns where p is the number of levels (first level is represented by having zeros in each column).
		
		:param X: A pandas date frame to be transformed. Should have the same columns as the one passed  to fit.
		:param verbose:  If true logs will be printed.(Default value = False)

		"""
		if not self.is_fitted:
			raise RuntimeError('Model is not fitted')
		vals = [var.transform(X, verbose).reset_index(drop=True) for var in self.variables]
		return pd.concat(vals , axis=1)

	def summary(self, variable_name=None):
		"""
		Describes how variables were transformed in human readable way. For continous variables intervals corresponding to levels of newly created categorical variable are printed.For categorical variables information about which levels were merged is shown.
		
		:param variable_name: If None summary for all variables will be printed. Otherwise summary will be shown only for the selected variable.  (Default value = None)

		"""
		if variable_name != None:
			summaries = [var.summary() for var in filter(lambda var: var.original_name==variable_name, self.variables)]
		else:
			summaries = [var.summary() for var in self.variables]
		print('\n'.join(summaries))
