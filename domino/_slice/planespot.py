from typing import Union

import meerkat as mk
import numpy as np
import torch
import torch.optim as optim
from torch.nn.functional import cross_entropy
from tqdm import tqdm

from domino.utils import unpack_args

from abstract import Slicer

## PlaneSpot imports
from sklearn import mixture
import glob
from collections import defaultdict

from domino.utils import convert_to_numpy, unpack_args
import pandas as pd

class PlaneSpotSlicer(Slicer):
    r"""
    Implements PlaneSpot [plumb_2023], a simple SDM that fits a GMM to a 2D model 
    embedding, fit using scvis [ding_2018]. 

    ..  [plumb_2023]
        Gregory Plumb*, Nari Johnson*, Ángel Alexander Cabrera, Ameet Talwalkar.
        Towards a More Rigorous Science of Blindspot Discovery in Image 
        Classification Models. arXiv:2207.04104 [cs] (2023)
        
    ..  [ding_2018]
        Jiarui Ding, Anne Condon, and Sohrab P Shah. 
        Interpretable dimensionality reduction of single cell transcriptome 
        data with deep generative models. 
        Nature communications, 9(1):1–13. (2018)
        
    PREREQUISITES:  Assumes that scvis is installed in the conda environment
        at scvis_conda_env, using the instructions here: 
        https://github.com/shahcompbio/scvis
    """

    def __init__(
        self,
        scvis_conda_env: str, # name of conda environment where scvis is installed
        n_slices: int = 10,
        n_max_mixture_components: int = 33, # maximum number of mixture components
        weight: float = 0.025, # weight hyperparameter
        scvis_config_path = None, # custom scvis config path
        scvis_output_dir = 'scvis', # path to output directory for scvis
        fit_scvis = True # flag to load rather than re-compute the scvis embedding 
    ):
        super().__init__(n_slices=n_slices)
        
        # scvis hyper-parameters
        self.scvis_conda_env = scvis_conda_env
        self.config.scvis_config_path = scvis_config_path
        self.config.scvis_output_dir = scvis_output_dir
        self.fit_scvis = fit_scvis
        
        # GMM hyper-parameters
        self.config.n_max_mixture_components = n_max_mixture_components
        self.config.weight = weight

        self.gmm = None

    def fit(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = None,
        pred_probs: Union[str, np.ndarray] = None,
        losses: Union[str, np.ndarray] = None,
        verbose: bool = True,
        random_state: int = 0, # random state for GMM
        **kwargs
    ):
        embeddings, targets, pred_probs, losses = unpack_args(
            data, embeddings, targets, pred_probs, losses
        )
        
        embeddings, targets, pred_probs = convert_to_numpy(
            embeddings, targets, pred_probs
        )
        
        # 1.  Fit scvis.
        if verbose:
            print('Fitting scvis...')
        
        scvis_embeddings = self._fit_scvis(embeddings.reshape(embeddings.shape[0], embeddings.shape[1]))
        
        # 2.  Fit GMM.
        if verbose:
            print('Fitting GMM...')
            
        self._fit_gmm(scvis_embeddings,
                     pred_probs,
                     random_state,
                     verbose,)

    def predict_proba(
        self,
        data: mk.DataPanel,
        scvis_embeddings: str, # scvis column name
        pred_probs: str, # predicted probabilities column name
    ) -> np.ndarray:
        ''' Returns the probability that each datapoint belongs to the
            top self.n_slices slices.
            
            Note that the probabilities may not sum to 1.
        '''
        
        # Append the scvis embedding and predicted probabilities; normalize
        X = self._combine_embedding(data[scvis_embeddings], data[pred_probs])
        probs_all_components = self.gmm.predict_proba(X)
        
        probs_slices = probs_all_components[:, self.slice_indices]
        return probs_slices
        
    def predict(
        self,
        data: mk.DataPanel,
        scvis_embeddings: str, # scvis column name
        pred_probs: str, # predicted probabilities column name
    ) -> np.ndarray:
        ''' Assigns (or does not assign) each datapoint in data to a slice.
        
            Datapoints that are not assigned to a slice have a returned label
            of np.nan.
        '''
        
        # Append the scvis embedding and predicted probabilities; normalize
        X = self._combine_embedding(data[scvis_embeddings], data[pred_probs])
        hard_predictions = self.gmm.predict(X)
        
        # Re-assign their indices
        return np.array([self._gmm_label_to_slice_label(l) for l in hard_predictions])
        
    def _fit_scvis(
        self, embeddings: np.ndarray
    ):
        ''' Fits an scvis model to the input embedding(s).
        '''
        if self.fit_scvis:
            ### Fit scvis
            
            # Make output directory
            os.system(f'rm -rf {self.config.scvis_output_dir}')
            os.system(f'mkdir {self.config.scvis_output_dir}')

            # Dump the embeddings as a CSV file
            embedding_filepath = f'{self.config.scvis_output_dir}/tmp.tsv'
            embedding_df = pd.DataFrame(embeddings)
            embedding_df.to_csv(embedding_filepath, sep = '\t', index = False)

            # Run scvis using the command line
            # source: https://github.com/shahcompbio/scvis
            command = f'conda run -n {self.scvis_conda_env} scvis train --data_matrix_file {embedding_filepath} --out_dir {self.config.scvis_output_dir}'

            if self.config.scvis_config_path is not None:
                print(self.config.scvis_config_path)
                # Add optional scvis config
                command += f' --config_file {self.config.scvis_config_path}'

            # Run the command (blocking)
            print(command)
            os.system(command)
            print('done')

            # Cleanup
            os.system('rm -rf {}'.format(embedding_filepath))
        
        ### Load and return the scvis embeddings
        return self._load_scvis_embeddings()
    
    def _fit_gmm(
        self, 
        reduced_embeddings: np.ndarray, 
        pred_probs: np.ndarray,
        random_state: int, # random state for sklearn
        verbose: bool = False,
    ):
        ''' Fits an error-aware Gaussian Mixture model to the scvis embeddings 
            and model predictions.
        '''
        # Store the min and max column values to normalize in the future.
        self.min_scvis_vals = np.min(reduced_embeddings, axis = 0)
        self.max_scvis_vals = np.max(reduced_embeddings, axis = 0)

        X = self._combine_embedding(reduced_embeddings, pred_probs)
        
        lowest_bic = np.infty
        bic = []
        n_components_range = range(self.config.n_slices, self.config.n_max_mixture_components)
        for n_components in n_components_range:
            # Fit a GMM with n_components components
            gmm = mixture.GaussianMixture(n_components = n_components, 
                                          covariance_type = 'full', 
                                          random_state = random_state)
            gmm.fit(X)
            
            # Calculate the Bayesian Information Criteria
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
                
        self.gmm = best_gmm
        
        # Assign a score to each mixture component to find the top-k slices
        # Create the map from "group" to "set of points" (recorded as indices)
        hard_preds = self.gmm.predict(X)
        
        cluster_map = defaultdict(list)
        for i, v in enumerate(hard_preds):        
            cluster_map[v].append(i)
            
        if verbose:
            print(f'The best GMM has {len(cluster_map)} components.')
            
        # Score each of those groups
        scores = []
        errors = (1. - pred_probs)
        for i in cluster_map:
            indices = cluster_map[i]
            score = len(indices) * np.mean(errors[indices]) ** 2 # Equivalent to 'number of errors * error rate'
            scores.append((i, score))
        scores = sorted(scores, key = lambda x: -1 * x[1])
        
        # Store the indices of the mixture components with the highest scores
        self.slice_indices = np.array([t[0] for t in scores[:self.config.n_slices]])
        
        if verbose:
            print('Scores:')
            for i, score in scores:
                indices = cluster_map[i]
                print(i, score, len(indices) * np.mean(errors[indices]), np.mean(errors[indices]))
            print()
            
            
    def _gmm_label_to_slice_label(self, gmm_label: int):
        ''' Returns the slice index corresponding to the GMM component 
            index gmm_label.
        
            If the datapoint's GMM component is not in the top self.n_slices 
            slices, returns np.nan instead.
        '''
        slice_idxs = np.argwhere(self.slice_indices == gmm_label)
        
        if len(slice_idxs) > 0:
            return slice_idxs.item()
        else:
            return np.nan
    
    def _load_scvis_embeddings(self) -> np.ndarray:
        ''' Loads and returns pre-computed scvis embeddings from 
            self.config.scvis_output_dir.
        '''
        ### Load and return the scvis embeddings
        search_string = f'{self.config.scvis_output_dir}/*.tsv'
        scvis_embedding_filepath = sorted(glob.glob(search_string), key = len)[0]
        return pd.read_csv(scvis_embedding_filepath, sep = '\t', index_col = 0).values
    
    def _combine_embedding(self, 
                           scvis_reps: np.ndarray, 
                           pred_probs: np.ndarray) -> np.ndarray:
        ''' Normalizes the scvis_reps and appends the predicted probabilities.
        '''
        # Normalize the embeddings using the minimum and maximum column values
        X = np.copy(scvis_reps)
        X -= self.min_scvis_vals
        X /= self.max_scvis_vals
        
        # Append (weighted) predicted probabilities to the embedding
        return np.concatenate((X, self.config.weight * pred_probs.reshape(-1, 1)), axis = 1)