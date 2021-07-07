import json
import numpy as np
from sklearn.mixture import GaussianMixture

# 模型参数模块

def save_json(model_params):
    json_txt = json.dumps(model_params, indent=4)
    f = open('./model/gmm.json', 'w')  #'../../model/gmm.json'   './model/gmm.json'
    f.write(json_txt)

def load_json(filepath):
    f = open(filepath, 'r')
    model_params = json.load(f)
    model_params['means_init'] = np.asarray(model_params['means_init'])
    model_params['weights'] = np.asarray(model_params['weights'])
    model_params['means'] = np.asarray(model_params['means'])
    model_params['covariances'] = np.asarray(model_params['covariances'])
    model_params['precisions'] = np.asarray(model_params['precisions'])
    model_params['precisions_cholesky'] = np.asarray(model_params['precisions_cholesky'])
    model_params['lower_bound'] = np.asarray(model_params['lower_bound']).astype(np.float64)
    return model_params

def save_model(model):
    model_params = model.get_params()
    model_params['means_init'] = model_params['means_init'].tolist()
    model_params['weights'] = model.weights_.tolist()
    model_params['means'] = model.means_.tolist()
    model_params['covariances'] = model.covariances_.tolist()
    model_params['precisions'] = model.precisions_.tolist()
    model_params['precisions_cholesky'] = model.precisions_cholesky_.tolist()
    model_params['converged'] = model.converged_
    model_params['n_iter'] = model.n_iter_
    model_params['lower_bound'] = model.lower_bound_.tolist()
    save_json(model_params)

def load_model(filepath):
    loaded_params = load_json(filepath)
    
    model = GaussianMixture(n_components=loaded_params['n_components'],
            covariance_type=loaded_params['covariance_type'],
            tol=loaded_params['tol'],
            reg_covar=loaded_params['reg_covar'],
            max_iter=loaded_params['max_iter'],
            n_init=loaded_params['n_init'],
            init_params=loaded_params['init_params'],
            weights_init=loaded_params['weights_init'],
            means_init=loaded_params['means_init'],
            precisions_init=loaded_params['precisions_init'],
            random_state=loaded_params['random_state'],
            warm_start=loaded_params['warm_start'],
            verbose=loaded_params['verbose'],
            verbose_interval=loaded_params['verbose_interval'])
    
    model.weights_ = loaded_params['weights']
    model.means_ = loaded_params['means']
    model.covariances_ = loaded_params['covariances']
    model.precisions_ = loaded_params['precisions']
    model.precisions_cholesky_ = loaded_params['precisions_cholesky']
    model.converged_ = loaded_params['converged']
    model.n_iter = loaded_params['n_iter']
    model.lower_bound_ = loaded_params['lower_bound']
    return model
