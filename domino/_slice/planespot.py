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
    """

    def __init__(
        self,
        n_slices: int = 10,
        n_max_mixture_components: int = 33, # maximum number of mixture components
        weight: float = 0.025, # weight hyperparameter
        scvis_config_path = None, # custom scvis config path
        scvis_output_dir = 'scvis' # path to output directory for scvis
    ):
        super().__init__(n_slices=n_slices)

        self.config.scvis_config_path = scvis_config_path
        self.config.scvis_output_dir = scvis_output_dir
        
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
        
        scvis_embeddings = self._fit_scvis(embeddings)
        
        # 2.  Fit GMM.
        if verbose:
            print('Fitting GMM...')
            
        self._fit_gmm(scvis_embeddings,
                     pred_probs)
        
        return self

    def predict_proba(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = None,
        pred_probs: Union[str, np.ndarray] = None,
        losses: Union[str, np.ndarray] = None,
    ) -> np.ndarray:
        embeddings, targets, pred_probs, losses = unpack_args(
            data, embeddings, targets, pred_probs, losses
        )

        losses = self._compute_losses(
            pred_probs=pred_probs, targets=targets, losses=losses
        )
        embeddings = torch.tensor(embeddings).to(
            dtype=torch.float, device=self.config.device
        )

        all_weights = []

        for slice_idx in range(self.config.n_slices):
            weights, _, _, _ = md_adversary_weights(
                mean=self.means[slice_idx],
                precision=torch.exp(self.precisions[slice_idx])
                * torch.eye(self.means[slice_idx].shape[0], device=self.config.device),
                x=embeddings,
                losses=losses,
            )
            all_weights.append(weights.cpu().numpy())
        return np.stack(all_weights, axis=1)

    def predict(
        self,
        data: mk.DataPanel,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = None,
        pred_probs: Union[str, np.ndarray] = None,
        losses: Union[str, np.ndarray] = None,
    ) -> np.ndarray:
        probs = self.predict_proba(
            data=data,
            embeddings=embeddings,
            targets=targets,
            pred_probs=pred_probs,
            losses=losses,
        )

        # TODO (Greg): check if this is the preferred way to get hard predictions from
        # probabilities
        return (probs > 0.5).astype(np.int32)

    def _fit_scvis(
        self, embeddings: np.ndarray
    ):
        ''' Fits an scvis model to the input embedding(s).
        '''
        # Make output directory
        os.system(f'rm -rf {self.config.scvis_output_dir}')
        os.system(f'mkdir {self.config.scvis_output_dir}')
    
        # Dump the embeddings as a CSV file
        embedding_filepath = f'{self.config.scvis_output_dir}/tmp.tsv'
        embedding_df = pd.DataFrame(reps)
        embedding_df.to_csv(embedding_filepath, sep = '\t', index = False)

        # Run scvis using the command line
        # source: https://github.com/shahcompbio/scvis
        command = f'scvis train --data_matrix_file {out_tmp} --out_dir {base_dir}'
        
        if scvis_config_path is not None:
            # Add optional scvis config
            command += f' --config_file {self.config.scvis_config_path}'
            
        # Run the command (blocking)
        os.system(command)
        
        # Cleanup
        os.system('rm -rf {}'.format(out_tmp))
        
        # Load and return the scvis embeddings
        search_string = f'{self.config.scvis_output_dir}/*.tsv'
        scvis_embedding_filepath = sorted(glob.glob(search_string), key = len)[0]
        return pd.read_csv(scvis_embedding_filepath, sep = '\t', index_col = 0).values
    
    def _fit_gmm(
        self, reduced_embeddings: np.ndarray, pred_prbs: np.ndarray
    )
        # Normalize the embeddings to have range [0, 1]
        X = np.copy(embedding)
        X -= np.min(X, axis = 0)
        X /= np.max(X, axis = 0)
        
        # Append (weighted) predicted probabilities to the embedding
        X = np.concatenate((X, error_weight * pred_prbs.reshape(-1, 1)), axis = 1)
        
        lowest_bic = np.infty
        bic = []
        n_components_range = range(1, self.config.n_max_mixture_components)

        for n_components in n_components_range:
            # Fit a GMM with n_components components
            gmm = mixture.GaussianMixture(n_components = n_components, covariance_type = 'full')
            gmm.fit(X)
            
            # Calculate the Bayesian Information Criteria
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
                
        self.gmm = best_gmm

