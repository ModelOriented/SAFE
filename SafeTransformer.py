import numpy as np
import ruptures as rpt
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.base import TransformerMixin

class SafeTransformer(TransformerMixin):
    
    def __init__(self):
        self.changepoint_values = []
        self.x_dims = 0
    
    def fit(self, X, clf):
        pdps = []
        axes = []
        self.x_dims = X.shape[1]
        changepoints = []
        for i in range(self.x_dims):
            pdp, axis = partial_dependence(clf, (i), X=X, grid_resolution=1000)
            pdps.append(pdp[0])
            axes.append(axis[0])
        for i, pdp in enumerate(pdps):
            algo = rpt.Pelt(model='l2').fit(pdp)
            my_bkps = algo.predict(pen=1) 
            changepoints.append(my_bkps)
        self.changepoint_values = [[axes[n_dim][i-1] for i in changepoints[n_dim]]
                                   for n_dim in range(self.x_dims)]
        return self
        
    def transform(self, X):
        new_data = [[len(list(filter(lambda e: x>=e, self.changepoint_values[current_dim]))) for x in X[:,current_dim]] 
                    for current_dim in range(self.x_dims)]
        return new_data
