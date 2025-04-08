from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler, RobustScaler
# from sklearn import cluster, mixture
import numpy as np

# https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
def batched_std_update(std1, std2, mean1, mean2, n1, n2):  
    var_tot = (n1 / (n1 + n2))*std1**2 + (n2 / (n1 + n2))*std2**2 + (n1*n2 / ((n1 + n2)**2))*(mean1 - mean2)**2
    return var_tot**0.5

def batched_mean_update(mean1, mean2, n1, n2):
    return (n1 / (n1 + n2))*mean1 + (n2 / (n1 + n2))*mean2

def batched_update_standardscaler(scaler1, scaler2):
    """Combine two StandardScalars without needing to explicitly load their data into memory

    Args:
        scaler1 (StandardScalar): fitted scaler object with mean, std
        scaler2 (StandardScalar):  fitted scaler object with mean, std

    Returns:
        StandardScalar: updated scalar with new mean and std and new total data size.
    """    
    assert scaler1.n_features_in_ == scaler2.n_features_in_
    new_scaler = StandardScaler()
    new_scaler.n_features_in_ = scaler1.n_features_in_
    new_scaler.scale_ = batched_std_update(scaler1.scale_, scaler2.scale_, scaler1.mean_, scaler2.mean_, scaler1.n_samples_seen_, scaler2.n_samples_seen_)
    new_scaler.var_ = np.copy(new_scaler.scale_)**2
    new_scaler.mean_ = batched_mean_update(scaler1.mean_, scaler2.mean_, scaler1.n_samples_seen_, scaler2.n_samples_seen_)
    new_scaler.n_samples_seen_ = scaler1.n_samples_seen_ + scaler2.n_samples_seen_
    return new_scaler

# for the edge features in the GNN
class LogAugmenter:
    """
    Sign-preserving log transformer with optional standard scaling post process step
    """    
    def __init__(self, ns, nd, standardize = True, jitter = 1e-20):
        self.nstatic = ns
        self.ndynamical = nd
        self.standardize = standardize
        self.ntot = ns + nd
        self.jitter = jitter # to avoid unstably small values abs(x)

    def log_augment(self, X):
        N, d = X.shape
        transformed_matrix = np.zeros((N, 2 * d))

        # For X < 0, set first d entries to 0 and last d entries to log(abs(X))
        negative_mask = X < 0
        transformed_matrix[:, :d][negative_mask] = 0
        transformed_matrix[:, d:][negative_mask] = np.log10(np.abs(X)+ self.jitter)[negative_mask]

        # For X > 0, set first d entries to log(abs(X)) and last d entries to 0
        positive_mask = X > 0
        transformed_matrix[:, :d][positive_mask] = np.log10(np.abs(X)+ self.jitter)[positive_mask]
        transformed_matrix[:, d:][positive_mask] = 0

        return transformed_matrix

    def fit(self, X):
        if not self.standardize:
            return self
        else:
            positive_mask = X > 0
            logx = np.log10(np.abs(X) + self.jitter)
            mdata_pos, mdata_neg = np.ma.masked_array(logx, mask=~positive_mask.astype(bool)), np.ma.masked_array(logx, mask=positive_mask.astype(bool))
            self.std_neg, self.mean_neg = mdata_neg.std(axis=0).data, mdata_neg.mean(axis=0).data
            self.std_pos, self.mean_pos = mdata_pos.std(axis=0).data, mdata_pos.mean(axis=0).data
            self.n_pos = np.sum(positive_mask, axis = 0)
            self.n_neg = np.sum(~positive_mask, axis = 0)
            return self

    def transform(self, X):
        xs = self.log_augment(X[:,:self.nstatic])
        xd = self.log_augment(X[:,self.nstatic:(self.nstatic+self.ndynamical)])
        if not self.standardize:
            return np.hstack((xs, xd))
        else:
            positive_mask = X[:,:self.nstatic] > 0
            xs[:,:self.nstatic] -= self.mean_pos[:self.nstatic]
            xs[:,:self.nstatic] /= self.std_pos[:self.nstatic]
            xs[:,self.nstatic:] -= self.mean_neg[:self.nstatic]
            xs[:,self.nstatic:] /= self.std_neg[:self.nstatic]
            xs[:,:self.nstatic][~positive_mask] = 0
            xs[:,self.nstatic:][positive_mask] = 0

            positive_mask = X[:,self.nstatic:(self.nstatic+self.ndynamical)] > 0
            xd[:,:self.ndynamical] -= self.mean_pos[self.nstatic:(self.nstatic+self.ndynamical)]
            xd[:,:self.ndynamical] /= self.std_pos[self.nstatic:(self.nstatic+self.ndynamical)]
            xd[:,self.ndynamical:] -= self.mean_neg[self.nstatic:(self.nstatic+self.ndynamical)]
            xd[:,self.ndynamical:] /= self.std_neg[self.nstatic:(self.nstatic+self.ndynamical)]
            xd[:,:self.ndynamical][~positive_mask] = 0
            xd[:,self.ndynamical:][positive_mask] = 0
        
            return np.hstack((xs, xd))

    def inverse_transform(self, X):
        raise NotImplementedError
    
    @staticmethod
    def from_two_augmenters(logaug1, logaug2):
        """combine two LogAugmenters, analagous to combine standard scaler function

        Args:
            logaug1 (LogAugmenter)
            logaug2 (LogAugmenter)

        Returns:
            _type_: LogAugmenter
        """        
        assert logaug1.nstatic == logaug2.nstatic
        assert logaug1.ndynamical == logaug2.ndynamical
        assert logaug1.standardize == logaug2.standardize
        if not logaug1.standardize:
            return LogAugmenter(logaug1.nstatic, logaug1.ndynamical, standardize = False)
        else:
             new_log_aug = LogAugmenter(logaug1.nstatic, logaug1.ndynamical, standardize = True)
             new_log_aug.mean_pos = batched_mean_update(logaug1.mean_pos, logaug2.mean_pos, logaug1.n_pos, logaug2.n_pos)
             new_log_aug.mean_neg = batched_mean_update(logaug1.mean_neg, logaug2.mean_neg, logaug1.n_neg, logaug2.n_neg)
             new_log_aug.std_pos = batched_std_update(logaug1.std_pos, logaug2.std_pos, logaug1.mean_pos, logaug2.mean_pos, logaug1.n_pos, logaug2.n_pos)
             new_log_aug.std_neg = batched_std_update(logaug1.std_neg, logaug2.std_neg, logaug1.mean_neg, logaug2.mean_neg, logaug1.n_neg, logaug2.n_neg)
             new_log_aug.n_pos = logaug1.n_pos + logaug2.n_pos
             new_log_aug.n_neg = logaug1.n_neg + logaug2.n_neg
             return new_log_aug