import numpy as np
import ruptures as rpt
from sklearn.base import TransformerMixin
import pandas as pd
from sklearn.exceptions import NotFittedError
from scipy.cluster.hierarchy import ward, cut_tree
from kneed import KneeLocator
import sys


class Variable():

	def __init__(self, name ,index):
		self.original_name = name
		self.original_index = index
		self.new_names = None


class NumericVariable(Variable):

	def __init__(self, name, index, penalty, pelt_model):
		super().__init__(name, index)
		self.changepoints = []
		self.penalty = penalty
		self.pelt_model = pelt_model
		self.changepoint_values = []

	def _get_partial_dependence(self, model, X, grid_resolution=1000):
		axes = []
		pdp = []
		points = np.linspace(min(X.loc[:,self.original_name]), max(X.loc[:,self.original_name]), grid_resolution)
		X_copy = X.copy()
		for point in points:
			axes.append(point)
			X_copy.loc[:,self.original_name] = point
			if(hasattr(model, 'predict_proba')):
				predictions = model.predict_proba(X_copy)
			else:
				predictions = model.predict(X_copy)
			val = np.mean(predictions, axis=0)
			pdp.append(val)
		return np.array(pdp), axes

	def fit(self, model, X):
		print('Fitting variable:'+ str(self.original_name))
		pdp, axis = self._get_partial_dependence(model, X, grid_resolution=1000)
		algo = rpt.Pelt(model=self.pelt_model).fit(pdp)
		self.changepoints = algo.predict(pen=self.penalty)
		self.changepoint_values = [axis[i] for i in self.changepoints[:-1]]
		changepoint_names = ['%.2f' % self.changepoint_values[i] for i in range(len(self.changepoint_values))] + ['Inf']
		self.new_names = [str(self.original_name) + "_[" + changepoint_names[i] + ", " + 
			changepoint_names[i+1]+")" for i in range(len(changepoint_names)-1)]
		return self

	def transform(self, X):
		print('Transforming variable:'+str(self.original_name))
		new_data = [len(list(filter(lambda e: x>=e, self.changepoint_values))) for x in X.loc[:,self.original_name]]
		ret = np.zeros([len(new_data), len(self.changepoint_values)])
		for row_num, val in enumerate(new_data):
			if val > 0:
				ret[row_num, val - 1] = 1
		return pd.DataFrame(ret, columns=self.new_names)



class CategoricalVariable(Variable):

	def __init__(self, name, index, dummy_names):
		super().__init__(name, index)
		self.dummy_names = dummy_names
		self.axes = None
		self.clusters = None
		self.Z = None
		self.pdp = None

	def fit(self, model, X):
		print('Fitting variable:'+str(self.original_name))
		pdp, names  = self._get_partial_dependence(model, X)
		self.pdp = pdp
		self.axes = names
		if pdp.ndim == 1:
			arr = np.reshape(pdp, (len(pdp), 1))
		else:
			arr = pdp
		self.Z = ward(arr)
		if pdp.shape[0] == 3:
			self.clusters = cut_tree(self.Z, height=self.Z[0, 2] - sys.float_info.epsilon)
			self.new_names = []
			for cluster in range(len(np.unique(self.clusters))):
				names = []
				for idx, c_val in enumerate(self.clusters):
					if c_val == cluster:
						if idx == 0:
							names.append('base')
						else:
							names.append(self.dummy_names[idx-1][len(self.original_name)+1:])
				self.new_names.append(self.original_name+'_'+"_".join(names))
		elif pdp.shape[0] > 3:
			kneed = KneeLocator(range(self.Z.shape[0]), self.Z[:, 2], direction='increasing', curve='convex')
			if kneed.knee is not None:
				self.clusters = cut_tree(self.Z, height=self.Z[kneed.knee+1, 2] - sys.float_info.epsilon)
				self.new_names = []
				for cluster in range(len(np.unique(self.clusters))):
					names = []
					for idx, c_val in enumerate(self.clusters):
						if c_val == cluster:
							if idx == 0:
								names.append('base')
							else:
								names.append(self.dummy_names[idx-1][len(self.original_name)+1:])
					self.new_names.append(self.original_name+'_'+"_".join(names))
		return self

	def transform(self, X):
		print('Transforming variable:'+str(self.original_name))
		dummies = pd.get_dummies(X.loc[:, self.original_name], prefix=self.original_name, drop_first=True)
		if self.clusters is not None:
			ret_len = len(np.unique(self.clusters)) - 1
			ret = np.zeros([X.shape[0], ret_len])
			for row_num in range(dummies.shape[0]):
				if not np.sum(dummies.iloc[row_num,:]) == 0:
					idx = np.argwhere(dummies.iloc[row_num,:] == 1)[0]
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



class SafeTransformer(TransformerMixin):

	categorical_dtypes = ['category', 'object']

	def __init__(self, model, penalty=3, pelt_model='l2', model_params={}):
		self.variables = []
		self.model = model
		self.penalty = penalty
		self.pelt_model = pelt_model
		self.model_params = model_params

	def _is_model_fitted(self, data):
		try:
			self.model.predict(data.head(1))
			return True
		except NotFittedError as e:
			return False
		except:
			return False

	def fit(self, X, y=None):
		if not isinstance(X, pd.DataFrame):
			raise ValueError("Data must be a pandas DataFrame")
		colnames = list(X)
		for idx, name in enumerate(colnames):
			if str(X.loc[:, name].dtype) in self.categorical_dtypes:
				dummies = pd.get_dummies(X.loc[:, name], prefix=name, drop_first=True)
				dummy_index  = X.columns.get_loc(name)
				X = pd.concat([X.iloc[:,range(dummy_index)], dummies, X.iloc[:, range(dummy_index+1, len(X.columns))]], axis=1)
				self.variables.append(CategoricalVariable(name, idx, list(dummies)))
			else:
				self.variables.append(NumericVariable(name, idx, self.penalty, self.pelt_model))
		if not self._is_model_fitted(X):
			self.model.fit(X, y, **self.model_params)
		for variable in self.variables:
			variable.fit(self.model, X)
		return self

	def transform(self, X):
		vals = [var.transform(X).reset_index(drop=True) for var in self.variables]
		return pd.concat(vals , axis=1)



