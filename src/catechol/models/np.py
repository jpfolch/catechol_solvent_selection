import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam

from catechol.data.data_labels import get_data_labels_mean_var
from catechol.data.featurizations import featurize_input_df
from catechol.models.base_model import Model

from tqdm import tqdm

import pandas as pd
#References: https://github.com/google-deepmind/neural-processes/blob/master/conditional_neural_process.ipynb, https://github.com/EmilienDupont/neural-processes/blob/master/models.py

class np(nn.Module):
        ## Conditional Neural Process - Garnelo et al 2018 https://arxiv.org/abs/1807.01613
        def __init__(self, x_dim, y_dim, r_dim, z_dim, h_dim, sigma_bound):
            super().__init__()
            self.x_dim = x_dim
            self.y_dim = y_dim
            self.r_dim = r_dim
            self.z_dim = z_dim
            self.h_dim = h_dim

            self.sigma_bound = sigma_bound

            self = self.double()

            # Encoder network
            self.encoder = nn.Sequential(
                nn.Linear(x_dim + y_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, r_dim)
            )

            # Aggregator function (mean pooling)
            self.aggregator = lambda r: torch.mean(r, dim=0, keepdim=True)

            # Latent Variable network blocks
            self.hidden_latent_encoder = nn.Sequential(
                nn.Linear(r_dim, h_dim),
                nn.ReLU()  # Outputs hidden representation
            )

            self.mu_encoder_layer = nn.Sequential(
                nn.Linear(h_dim, z_dim)
            )

            self.sigma_encoder_layer = nn.Sequential(
                nn.Linear(h_dim, z_dim)
            )

            # Decoder network blocks
            self.hidden_decoder = nn.Sequential(
                nn.Linear(x_dim + z_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU(), 
            )

            self.mu_decoder_layer = nn.Sequential(
                nn.Linear(h_dim, y_dim)
            )

            self.sigma_decoder_layer = nn.Sequential(
                nn.Linear(h_dim, y_dim)
            )

        ###### Model Structure ######

        def encode(self, x, y):
            # x, y -> r

            xy = torch.cat([x, y], dim=-1)
            r = self.encoder(xy)

            return r

        def sample_latent(self, r):
            # r -> mu_z, sigma_z

            hidden = self.hidden_latent_encoder(r)
            mu = self.mu_encoder_layer(hidden)
            sigma = self.sigma_bound + (1-self.sigma_bound)*torch.sigmoid(self.sigma_encoder_layer(hidden))

            return mu, sigma

        def decode(self, x, z):
            # x, z -> mu_y, sigma_y
            
            if z.size(0) == 1:
                z = z.repeat(x.size(0), 1)
                
            xz = torch.cat([x, z], dim=-1)

            hidden = self.hidden_decoder(xz)
            mu = self.mu_decoder_layer(hidden)
            sigma = self.sigma_bound + (1-self.sigma_bound) * torch.sigmoid(self.sigma_decoder_layer(hidden))
            
            return mu, sigma

        def forward(self, context_x, context_y, target_x, target_y=None):
            """
            Forward pass of the Neural Process.

            Args:
                context_x (torch.Tensor): Context inputs of shape (context_size, x_dim).
                context_y (torch.Tensor): Context outputs of shape (context_size, y_dim).
                target_x (torch.Tensor): Target inputs of shape (target_size, x_dim).

            Returns:
                torch.Tensor: Predicted target outputs.
            """
            # Encode context points x, y -> r
            r_context = self.aggregator(self.encode(context_x, context_y))

            # Sample latent variable
            mu_context, sigma_context = self.sample_latent(r_context)
            q_context = Normal(mu_context, sigma_context)

            if target_y is not None:
                # Encode target points x, y -> r
                r_target = self.encode(target_x, target_y)
                # Sample latent variable
                mu_target, sigma_target = self.sample_latent(r_context)
                q_target = Normal(mu_target, sigma_target)
                z_training = q_target.rsample()

                y_mu, y_sigma = self.decode(target_x, z_training)
                y_dist = Normal(y_mu, y_sigma)

                return y_dist, q_target, q_context

            else:
                z_predict = q_context.rsample()
                # Decode target points x, z -> y
                y_mu, y_sigma = self.decode(target_x, z_predict)
                y_dist = Normal(y_mu, y_sigma)

                return y_dist, y_mu, y_sigma
            
        def loss(self, target_y, y_dist, q_target, q_context):
            # Compute Loss. source: https://bayesiandeeplearning.org/2018/papers/92.pdf

            # Compute negative log likelihood
            nll = -y_dist.log_prob(target_y).mean()

            # Compute KL divergence
            kl_div = torch.distributions.kl.kl_divergence(q_target, q_context).mean()

            return nll + kl_div

class NPModel(Model):    

    def __init__(self, 
                 r_dim=32, 
                 z_dim=8, 
                 h_dim=64, 
                 sigma_bound=0.1,
                 featurization="spange_descriptors", 
                 optimizer=Adam, 
                 learning_rate=0.1,
                 max_context_proportion=.8,
                 min_context_proportion=.2, 
                 num_epochs=1000, 
                 lr_factor =0.5,
                 lr_patience=100):
        
        super(NPModel, self).__init__()
        
        self.featurization = featurization

        ## Model parameters
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.sigma_bound = sigma_bound

        ## Training parameters
        self.optimize_init = optimizer
        self.learning_rate = learning_rate
        self.max_context_proportion = max_context_proportion
        self.min_context_proportion = min_context_proportion
        self.num_epochs = num_epochs
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience

        ####### Implementation ########

    def featurisation(self, x, y = None):
        train_X_featurized = featurize_input_df(x, self.featurization, remove_constant=True, normalize_feats=True)            
        train_X_tensor = torch.tensor(train_X_featurized.to_numpy(), dtype=torch.float64)

        if y is not None:
            train_Y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float64)
            return train_X_tensor, train_Y_tensor
        else:
            return train_X_tensor
    
    def _train(self, train_X, train_Y):

        # Featurisation
        train_X_tensor, train_Y_tensor = self.featurisation(train_X, train_Y)

        # Load Model
        self.x_dim = train_X_tensor.shape[-1]
        self.y_dim = train_Y_tensor.shape[-1]

        self.model = np(x_dim=self.x_dim, y_dim=self.y_dim, r_dim=self.r_dim, z_dim=self.z_dim, 
                        h_dim=self.h_dim, sigma_bound=self.sigma_bound)
        self.model = self.model.double()
        self.optimizer = self.optimize_init
        self.optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)
        # Learning rate scheduler: ReduceLROnPlateau
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=self.lr_factor, patience=self.lr_patience)
        
        # Store training context
        self.train_context_X = train_X_tensor
        self.train_context_Y = train_Y_tensor

        pbar = tqdm(range(self.num_epochs), desc="Training Progress")

        for epoch in pbar:
            self.optimizer.zero_grad()

            # Create context and target sets
            context_size = torch.randint(
                int(self.min_context_proportion * train_X_tensor.size(0)),
                int(self.max_context_proportion * train_X_tensor.size(0)) + 1,
                (1,)
            ).item()
            context_indices = torch.randperm(train_X_tensor.size(0))[:context_size]
            context_x = train_X_tensor[context_indices]
            context_y = train_Y_tensor[context_indices]
            target_x = train_X_tensor
            target_y = train_Y_tensor 

            # Forward pass
            y_dist, q_target, q_context = self.model.forward(context_x, context_y, target_x, target_y)

            # Compute loss
            loss = self.model.loss(target_y, y_dist, q_target, q_context)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)
            
            # Print loss every 2% epochs
            if (epoch + 1) % int(self.num_epochs/50) == 0:
                pbar.set_description(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}. Current LR: {self.optimizer.param_groups[0]["lr"]:.4f}')     

        
    def _predict(self, test_X: pd.DataFrame):

        target_x = self.featurisation(test_X)

        with torch.no_grad():
            y_dist, mean, var = self.model.forward(self.train_context_X, self.train_context_Y, target_x)
        
        mean_lbl, var_lbl = get_data_labels_mean_var()
        mean_df = pd.DataFrame(mean, columns=mean_lbl)
        var_df = pd.DataFrame(var, columns=var_lbl)
        return  pd.concat([mean_df, var_df], axis=1)
    
    def _ask():
        pass