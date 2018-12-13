import numpy as np
import ruptures as rpt
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.base import TransformerMixin
import pandas as pd

class SafeTransformer(TransformerMixin):
    
    def __init__(self):
        self.changepoint_values = []
        self.x_dims = 0
    
    def fit(self, X, clf, penalty=3):
        if isinstance(X, pd.DataFrame):
            base_names = list(X)
        else:
            base_names = ['X' + str(i) for i in range(X.shape[1])]
        pdps = []
        axes = []
        self.x_dims = X.shape[1]
        changepoints = []
        for i in range(self.x_dims):
            pdp, axis = self._get_partial_dependence(clf, i, X=X, grid_resolution=1000)
            pdps.append(pdp[0])
            axes.append(axis[0])
        for i, pdp in enumerate(pdps):
            algo = rpt.Pelt(model='l2').fit(pdp)
            my_bkps = algo.predict(pen=penalty) 
            changepoints.append(my_bkps)
        self.changepoint_values = [[axes[n_dim][i-1] for i in changepoints[n_dim]]
                                   for n_dim in range(self.x_dims)]
        changepoint_names = [['%.2f' % self.changepoint_values[i][j] for j in range(len(self.changepoint_values[i]))] + ["+Inf"] for i in range(len(self.changepoint_values))]
        self.names = [[str(base_names[i]) + "_(" + changepoint_names[i][j] + ", " + 
                  changepoint_names[i][j+1]+")" for j in range(len(changepoint_names[i])-1)] for i in range(len(base_names))]
        self.names = [item for sublist in self.names for item in sublist]
        return self
        
    def transform(self, X):
        new_data = [[len(list(filter(lambda e: x>=e, self.changepoint_values[current_dim]))) for x in X.iloc[:,current_dim]] 
                    for current_dim in range(self.x_dims)]
        new_data = pd.DataFrame(new_data).transpose()
        arrays = []
        for idx in new_data:
            ret = np.zeros([len(new_data[idx]), len(self.changepoint_values[idx])])
            for row_num, val in enumerate(new_data[idx]):
                if val > 0:
                    ret[row_num, val - 1] = 1
            arrays.append(ret)
        result = np.concatenate(arrays, axis=1)
        return pd.DataFrame(result, columns=self.names)
    
    def _get_partial_dependence(self, clf, i, X, grid_resolution=1000):
        axes = []
        pdp = []
        points = np.linspace(min(X.iloc[:,i]), max(X.iloc[:,i]), grid_resolution)
        for point in points:
            X_copy = np.copy(X)
            axes.append(point)
            X_copy[:,i] = point
            if(hasattr(clf, 'predict_proba')):
            	predictions = clf.predict_proba(X_copy)
            else:
            	predictions = clf.predict(X_copy)
            val = np.mean(predictions)
            pdp.append(val)
        return [np.array(pdp)], [axes]
                             
