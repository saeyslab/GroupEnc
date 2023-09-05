"""
Copyright 2023 David Novak

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
#################################################
# Definition of GroupEnc model for tabular data #
#################################################
"""

import os
import warnings
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from itertools import combinations

from typing import Optional,Union
import numpy as np
import pandas as pd

from .losses import loss_group, euclidean_distance, loss_kldiv
from .vae import VariationalEncoder, VariationalSampler

class GroupEncNetwork(tf.keras.Model):
    """
    GroupEnc neural net model class

    Parametric dimension-reduction model with a group loss.
    """
    def __init__(
        self,
        data:         Optional[np.ndarray] = None,
        full_dim:     Optional[int] = int,
        enc_shape:    list = [32,64,128,32],
        latent_dim:   int = 2,
        group_size:   int = 4,
        dropout_rate: float = 0.00,
        activation:   str = 'relu',
        **kwargs
    ):
        """
        Instantiate GroupEnc network model for tabular data

        Constructor for a GroupEncNetwork object.

        - data:         optional high-dimensional data coordinate matrix (`full_dim` can be specified instead) (nd.nparray)
        - full_dim:     optional `data` dimensionality `data.shape[1]` (`data` can be specified instead) (int)
        - enc_shape:    list of consecutive node counts defining the size of each layer of the encoder (list of ints)
        - latent_dim:   dimensionality of latent projection of `data` (int)
        - group_size:   group size for group loss (int)
        - dropout_rate: rate of dropout for regularisation (float)
        - activation:   activation function in each node of the encoder and decoder networks: eg. 'selu', 'relu', 'sigmoid' (str)
        - verbose:      print progress messages during instantiation? (bool)
        """
        super(GroupEncNetwork, self).__init__(name='GroupEncNetwork', **kwargs)

        if data is None and full_dim is None:
            raise ValueError('Either `data` or `full_dim` must be specified to build network')
        
        if full_dim is None:
            full_dim = data.shape[1]
        if latent_dim < 1 or latent_dim >= full_dim:
            raise ValueError('Invalid latent representation dimensionality')

        self.full_dim = full_dim
        self.enc_shape = enc_shape
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.encoder = VariationalEncoder(
            latent_dim=self.latent_dim,
            shape=     self.enc_shape,
            dropout=   self.dropout_rate,
            activation=self.activation
        )
        self.sampler = VariationalSampler()
        self.k = group_size
        self.p = list(combinations(np.arange(self.k), 2))

    def call(self, x):
        """
        Run forward pass through VAE network for tabular data

        - x: training data (np.ndarray or tf.Tensor)
        """

        z_mu, z_sigma = self.encoder(x)
        z = self.sampler(z_mu, z_sigma)

        l = []

        if self.w['group'] > 0.0:
            n = tf.shape(x)[0]
            l_group = loss_group(x, z, self.full_dim, self.latent_dim, n, self.k, 1, self.p, euclidean_distance, euclidean_distance, None) * self.w['group']
            self.add_metric(l_group, aggregation='mean', name='group')
            l.append(l_group)

        if self.w['kldiv'] > 0.0:
            l_kldiv = loss_kldiv(z_mu, z_sigma) * self.w['kldiv']
            self.add_metric(l_kldiv, aggregation='mean', name='kldiv')
            l.append(l_kldiv)    

        self.add_loss(l)

        return l

class GroupEnc:
    """
    GroupEnc model (for tabular data)
    """
    def __init__(
        self,
        full_dim:     int = int,
        enc_shape:    list = [32,64,128,32],
        latent_dim:   int = 2,
        dropout_rate: float = 0.,
        activation:   str = 'relu'
    ):
        """
        Create a SGroupVAE dimension-reduction model for tabular data

        - full_dim:     original input data dimensionality (int)
        - enc_shape:    list of consecutive node counts defining the size of each layer of the encoder (list of ints)
        - latent_dim:   target dimensionality of the output data projection (int)
        - dropout_rate: rate of dropout for regularisation (float)
        - activation:   activation function in each node of the encoder and decoder networks: eg. 'selu', 'relu', 'sigmoid' (str)
        """
        self.model = GroupEncNetwork(
            full_dim=    full_dim,
            enc_shape=   enc_shape, 
            latent_dim=  latent_dim,
            dropout_rate=dropout_rate,
            activation=  activation
        )
        self.model.fitted = False

    def __repr__(self):
        return f'GroupEnc tabular data model (fitted={self.model.fitted}, full_dim={self.model.full_dim}, latent_dim={self.model.latent_dim})'

    def reset(self):
        """
        Reset model weights

        Resets the parameters learned in fitting a model, so that it can be re-trained independently again.
        """

        full_dim     = self.model.full_dim
        enc_shape    = self.model.enc_shape
        dec_shape    = self.model.dec_shape
        latent_dim   = self.model.latent_dim
        dropout_rate = self.model.dropout_rate
        activation   = self.model.activation

        self.model = None
        self.model = GroupEncNetwork(
            full_dim=    full_dim,
            enc_shape=   enc_shape,
            dec_shape=   dec_shape,
            latent_dim=  latent_dim,
            dropout_rate=dropout_rate,
            activation=  activation
        )
        self.model.fitted = False

    def fit(
        self,
        X:                 np.ndarray,
        w_group:           float = 1.0,
        w_kldiv:           float = 1.0,
        batch_size:        int = 512,
        n_epochs:          int = 100,
        early_stopping:    bool = False,
        no_reset:          bool = False,
        min_delta:         float = 0.,
        patience:          int = 5,
        learning_rate:     float = 0.001,
        callback_csv:      Optional[str] = None,
        callback_tb:       Optional[str] = None,
        seed:              Optional[int] = 1,
        verbose:           bool = True
    ):
        """
        Fit a GroupEnc model

        Trains GroupEnc model on input high-dimensional tabular data.
        Use the `transform` method to produce an embedding using the trained model.

        - X:                 training data (np.ndarray)
        - w_kldiv:           weight for the group loss term (float)
        - w_kldiv:           weight for the KL-divergence from latent prior term (float)
        - batch_size:        size of each training batch (int)
        - n_epochs:          number of training epochs (or maximum number if early stopping is enabled) (int)
        - early_stopping:    enable early stopping if value of evaluation metric (`monitor_quantity`) does not improve over some training epochs? (bool)
        - no_reset:          (experimental) if model has been trained already, continue training instead of refitting? (bool)
        - min_delta:         minimal change in monitored quantity (float)
        - patience:          number of epochs without improvement which triggers early stopping (int)
        - learning_rate:     Adam optimiser learning rate parameter for training (float)
        - callback_csv:      optional name of CSV file to log value of each loss function term per elapsed epoch (str)
        - callback_tb:       optional name of directory to store intermediate files for logging loss function term values using TensorBoard (str)
        - seed:              optional random seed for reproducibility (int)
        - verbose:           print progress messages? (bool)
        """
        
        self.model.verbose = verbose
        if self.model.fitted and not no_reset:
            self.reset()

        self.model.w = {'group': w_group, 'kldiv': w_kldiv}

        self.model.batch_size = batch_size

        ## Resolve callbacks
        callbacks = []
        if callback_tb is not None:
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=callback_tb))
        if callback_csv is not None:
            if os.path.exists(callback_csv):
                f = open(callback_csv, 'a')
                f.close()
            else:
                colnames = ['group', 'kldiv']
                colnames.append('loss')
                colnames = sorted(colnames)
                colnames.insert(0, 'epoch')
                df = pd.DataFrame(columns=colnames)
                df.to_csv(callback_csv, index=False)
            callbacks.append(CSVLogger(filename=callback_csv, separator=',', append=True))
        if early_stopping:
            callbacks.append(EarlyStopping(monitor='loss', min_delta=min_delta, patience=patience, verbose=1))

        ## Compile model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

        ## Train model on input data
        if seed is not None:
            tf.random.set_seed(seed)
        fit_args = {
            'x':                X,
            'y':                None,
            'batch_size':       batch_size,
            'epochs':           n_epochs,
            'shuffle':          True,
            'callbacks':        callbacks,
            'validation_split': 0.0,
            'verbose':          1 if verbose else 0
        }
        self.model.fit(**fit_args)
        self.model.fitted = True

        pass

    def transform(self, X: np.ndarray):
        """
        Transform data using GroupEnc model

        Using a trained GroupEnc model, generate a data embedding.

        - X: input data (np.ndarray)
        """

        if not self.model.fitted:
            raise AttributeError('Model not trained. Use `fit` first')

        inputs = layers.Input(shape=X.shape[1], )
        outputs = self.model.encoder(inputs)[0]
        enc = tf.keras.models.Model(inputs, outputs)
        proj = enc.predict(X, batch_size=self.model.batch_size)

        return proj

    def fit_transform(
        self,
        X:                 np.ndarray,
        batch_size:        int = 512,
        n_epochs:          int = 100,
        early_stopping:    bool = False,
        no_reset:          bool = False,
        min_delta:         float = 0.,
        patience:          int = 5,
        learning_rate:     float = 0.001,
        callback_csv:      Optional[str] = None,
        callback_tb:       Optional[str] = None,
        seed:              Optional[int] = 1,
        verbose:           bool = True
    ):
        """
        Fit a GroupEnc model and transform input

        Trains GroupEnc model on input high-dimensional tabular data and produces an embedding using the trained model.

        - X:                 training data (np.ndarray)
        - batch_size:        size of each training batch (int)
        - n_epochs:          number of training epochs (or maximum number if early stopping is enabled) (int)
        - early_stopping:    enable early stopping if value of evaluation metric (`monitor_quantity`) does not improve over some training epochs? (bool)
        - no_reset:          (experimental) if model has been trained already, continue training instead of refitting? (bool)
        - monitor_quantity:  quantity (evaluation metric) to monitor for early stopping ('`loss`' or any loss term name) (str)
        - min_delta:         minimal change in monitored quantity (float)
        - patience:          number of epochs without improvement which triggers early stopping (int)
        - learning_rate:     Adam optimiser learning rate parameter for training (float)
        - callback_csv:      optional name of CSV file to log value of each loss function term per elapsed epoch (str)
        - callback_tb:       optional name of directory to store intermediate files for logging loss function term values using TensorBoard (str)
        - seed:              optional random seed for reproducibility (int)
        - verbose:           print progress messages? (bool)
        """
        self.fit(
            X=X, batch_size=batch_size,
            n_epochs=n_epochs, early_stopping=early_stopping, no_reset=no_reset, min_delta=min_delta,
            patience=patience, learning_rate=learning_rate, callback_csv=callback_csv, callback_tb=callback_tb, seed=seed, verbose=verbose
        )
        return self.transform(X=X)
    

