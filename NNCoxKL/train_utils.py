import torch
import optuna
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from .loss_feiyang import CoxKLLOSS
from .dataset import Data
from .evaluate_utils import Evaluator
from sklearn.model_selection import KFold
import numpy as np
import sksurv
import random
import itertools


def _calculate_delta_tilde(event, time, theta_tilde):

    event = event.view(-1)
    theta_tilde = theta_tilde.view(-1)
    theta_center = theta_tilde - theta_tilde.max()   # stabilize
    stabilized_exp = theta_center.exp()

    n = len(event)  # total number of individuals
    m = len(torch.where(event == 1)[0])  # number of failures (where event == 1)

    failure_time = time[event == 1]  # time of failures

    denominator = 1/torch.flip(torch.flip(stabilized_exp, dims=(0,)).cumsum(0), dims=(0,))
    denominator_kfailures = denominator[torch.where(event == 1)[0]] # Selects the denominator value for each failure.

    device = event.device
    delta_tilde = torch.zeros(n, device=device)
    for k in range(m):
        Y_i = (time >= failure_time[k])
        numerator = Y_i * stabilized_exp
        delta_tilde += numerator * denominator_kfailures[k]

    return delta_tilde

def set_seeds(seed_value=42):
    """Set seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # For CUDA to ensure reproducibility on GPU operations

def hyperparameter_tuning(train_set, val_set, model_base, param_grid, eta, 
                          selection_criterion='loss', seed = 42):

    set_seeds(seed)

    if selection_criterion not in ['loss', 'c-index']:
        raise ValueError("selection_criterion must be either 'loss' or 'c-index'")

    if selection_criterion == 'loss':
        best_performance = float('inf')  
    else:
        best_performance = -float('inf')  # For c-index, higher is better

    best_params = None
    results = []

    # Iterate over all combinations of parameters
    for params in itertools.product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        hidden_dim, learning_rate, num_dr, batch_size, weight_decay, alpha = param_dict['hidden_dim'], param_dict['learning_rate'], param_dict['num_dr'], param_dict['batch_size'], param_dict['weight_decay'], param_dict['alpha']
        num_layer = param_dict['num_layer']
        
        model = model_base(input_dim=train_set.X.shape[1], hidden_dim=hidden_dim, output_dim=1, num_layers=num_layer, num_dr=num_dr)
        trainer = Trainer(model=model, weight_decay=weight_decay)

        _, val_loss_list = trainer.fit(
            train_set=train_set,
            val_set=val_set,
            eta_list=[eta],
            alpha = alpha,
            batch_size=batch_size,  # Example batch size
            num_epoch=300,  # Example number of epochs
            learning_rate=learning_rate,
            patience=50  # Example patience for early stopping
        )

        if selection_criterion == 'loss':
            current_performance = val_loss_list[-1]
        else:
            evaluator = Evaluator()
            c_index, _ = evaluator.eval_single(model, val_set)
            current_performance = c_index

        results.append((param_dict, current_performance))
        
        if selection_criterion == 'loss':
            if current_performance < best_performance:
                best_performance = current_performance
                best_params = param_dict
        else:
            if current_performance > best_performance:
                best_performance = current_performance
                best_params = param_dict

        print(f'Tested {param_dict} with val_{selection_criterion}: {current_performance}')

    return best_params, results


def hyperparameter_tuning_optuna(train_set, val_set, model_base, param_ranges, eta=None,
                                 selection_criterion='loss', n_trials=50, seed=42,
                                 num_epoch=300, patience=50, study_name=None,
                                 metrics_class=None, initial_state_dict=None,
                                 reset_batchnorm=False):
    """
    Hyperparameter tuning using Optuna framework for automatic hyperparameter selection.
    
    Parameters:
    -----------
    train_set : Dataset
        Training dataset
    val_set : Dataset
        Validation dataset
    model_base : class
        Model class to instantiate
    param_ranges : dict
        Dictionary with parameter ranges for Optuna to search. Each key should map to:
        - A single number (int/float) for fixed values
        - A tuple/list of [min, max] for continuous/integer ranges
        - A list of discrete choices (3+ elements) for categorical parameters
        Expected keys: 'hidden_dim', 'learning_rate', 'num_dr', 'batch_size', 'weight_decay', 'num_layer', 'eta'
        Example: {
            'hidden_dim': [32, 256],  # Integer range (min, max)
            'learning_rate': [1e-5, 1e-2],  # Float range (min, max) for log-uniform
            'num_dr': [0.0, 0.5],  # Float range (min, max) for uniform
            'batch_size': [16, 128],  # Integer range (min, max)
            'weight_decay': [1e-6, 1e-3],  # Float range (min, max) for log-uniform
            'num_layer': [1, 4],  # Integer range (min, max)
            'eta': [0.1, 10.0]  # Float range (min, max) for uniform - optional, can tune eta
        }
        Alternative with fixed values and categorical choices:
        {
            'hidden_dim': 128,  # Fixed value
            'num_layer': 2,  # Fixed value
            'batch_size': [16, 32, 64, 128],  # List of discrete choices
            'learning_rate': [1e-5, 1e-2],  # Range
            'eta': [0.1, 0.5, 1.0, 2.0, 5.0],  # Discrete choices for eta
            ...
        }
    eta : float, optional
        Eta parameter for the loss function. If provided and 'eta' is not in param_ranges,
        this fixed value will be used. If 'eta' is in param_ranges, it will be tuned and
        this parameter will be ignored.
    selection_criterion : str, default='loss'
        Criterion for selecting best hyperparameters: 'loss' or 'c-index'
    n_trials : int, default=50
        Number of Optuna trials to run
    seed : int, default=42
        Random seed for reproducibility
    num_epoch : int, default=300
        Number of training epochs per trial
    patience : int, default=50
        Patience for early stopping
    study_name : str, optional
        Name for the Optuna study (for database storage)
    
    Returns:
    --------
    best_params : dict
        Best hyperparameters found
    study : optuna.Study
        Optuna study object containing all trial results
    """
    set_seeds(seed)
    
    if selection_criterion not in ['loss', 'c-index']:
        raise ValueError("selection_criterion must be either 'loss' or 'c-index'")
    
    # Validate that eta is provided either as fixed value or in param_ranges
    if eta is None and 'eta' not in param_ranges:
        raise ValueError("eta must be provided either as a fixed parameter or in param_ranges")
    
    # Create Optuna study
    direction = 'minimize' if selection_criterion == 'loss' else 'maximize'
    study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
    
    def objective(trial):
        """Objective function for Optuna to optimize."""
        # Set seed for this trial to ensure reproducibility and fresh start
        # Use trial number to get different seeds for each trial
        trial_seed = seed + trial.number
        set_seeds(trial_seed)
        
        # Suggest hyperparameters based on param_ranges
        param_dict = {}
        
        # Integer parameters that should use suggest_int
        integer_params = ['hidden_dim', 'batch_size', 'num_layer']
        log_params = ['learning_rate', 'weight_decay', 'alpha']

        for param_name, param_range in param_ranges.items():
            # Handle fixed values (single number or single-element list/tuple)
            if isinstance(param_range, (int, float)):
                # Fixed value - use directly
                param_dict[param_name] = param_range
            elif isinstance(param_range, (list, tuple)) and len(param_range) == 1:
                # Fixed value in a list/tuple
                param_dict[param_name] = param_range[0]
            elif isinstance(param_range, (list, tuple)) and len(param_range) == 2:
                # Range parameter - determine if integer or float
                if param_name in integer_params:
                    # Integer range
                    param_dict[param_name] = trial.suggest_int(
                        param_name, int(param_range[0]), int(param_range[1])
                    )
                elif param_name in log_params:
                    # Use log-uniform for learning rate and weight decay
                    param_dict[param_name] = trial.suggest_float(
                        param_name, param_range[0], param_range[1], log=True
                    )
                else:
                    # Use uniform for other continuous parameters (including eta)
                    param_dict[param_name] = trial.suggest_float(
                        param_name, param_range[0], param_range[1]
                    )
            elif isinstance(param_range, (list, tuple)) and len(param_range) > 2:
                # Categorical parameter (list of discrete choices)
                if all(isinstance(x, int) for x in param_range):
                    # Integer choices - use suggest_int if consecutive, otherwise categorical
                    if param_name in integer_params and max(param_range) - min(param_range) == len(param_range) - 1:
                        # Consecutive integers - use suggest_int
                        param_dict[param_name] = trial.suggest_int(
                            param_name, min(param_range), max(param_range)
                        )
                    else:
                        # Non-consecutive integers - use categorical
                        param_dict[param_name] = trial.suggest_categorical(param_name, param_range)
                else:
                    # Float choices (including eta)
                    param_dict[param_name] = trial.suggest_categorical(param_name, param_range)
            else:
                raise ValueError(f"Invalid param_range for {param_name}: {param_range}")
        
        # Extract parameters
        hidden_dim = param_dict['hidden_dim']
        learning_rate = param_dict['learning_rate']
        num_dr = param_dict['num_dr']
        batch_size = param_dict['batch_size']
        weight_decay = param_dict['weight_decay']
        num_layer = param_dict['num_layer']
        alpha = param_dict.get('alpha', 1.0)
        ridge_lambda = param_dict.get('ridge_lambda', 0.0)
        
        # Extract eta: use from param_dict if tuned, otherwise use fixed eta
        trial_eta = param_dict.get('eta', eta)
        
        model = model_base(
            input_dim=train_set.X.shape[1],
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layer,
            num_dr=num_dr,
            seed_value=trial_seed  # Pass trial-specific seed to model
        )
        if initial_state_dict is not None:
            model.load_state_dict(initial_state_dict)
            if reset_batchnorm:
                for module in model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.reset_running_stats()
        trainer_kwargs = dict(model=model, weight_decay=weight_decay, seed=trial_seed)
        if metrics_class is not None:
            trainer_kwargs["metrics"] = metrics_class()
        trainer = Trainer(**trainer_kwargs)

        # Train model
        _, val_loss_list = trainer.fit(
            train_set=train_set,
            val_set=val_set,
            eta_list=[trial_eta],
            alpha = alpha,
            ridge_lambda=ridge_lambda,
            batch_size=batch_size,
            num_epoch=num_epoch,
            learning_rate=learning_rate,
            patience=patience
        )
        
        # Evaluate performance
        if selection_criterion == 'loss':
            current_performance = val_loss_list[-1]
        else:
            evaluator = Evaluator()
            c_index, _ = evaluator.eval_single(model, val_set)
            current_performance = c_index
        
        print(f'Trial {trial.number}: {param_dict} with val_{selection_criterion}: {current_performance:.6f}')
        
        return current_performance
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get best parameters
    best_params = study.best_params.copy()
    best_performance = study.best_value
    
    print(f'\nBest hyperparameters found: {best_params}')
    print(f'Best {selection_criterion}: {best_performance:.6f}')
    
    return best_params, study


def cross_val_performance(train_set, param_dict, model_base, k=5, seed=42):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    fold_performances = []
    
    for train_index, val_index in kf.split(train_set.X):
        train_fold = train_set.subset(train_index)
        val_fold = train_set.subset(val_index)
        
        hidden_dim = param_dict['hidden_dim']
        learning_rate = param_dict['learning_rate']
        num_dr = param_dict['num_dr']
        batch_size = param_dict['batch_size']
        weight_decay = param_dict['weight_decay']
        num_layer = param_dict['num_layer']
        eta = param_dict['eta']
        
        model = model_base(input_dim=train_set.X.shape[1], hidden_dim=hidden_dim, output_dim=1, num_layers=num_layer, num_dr=num_dr)
        trainer = Trainer(model=model, weight_decay=weight_decay)
        
        _, _ = trainer.fit(
            train_set=train_fold,
            val_set=val_fold,
            eta_list=[eta],
            batch_size=batch_size,
            num_epoch=300,  # Example number of epochs
            learning_rate=learning_rate,
            patience=50  # Example patience for early stopping
        )
        
        evaluator = Evaluator()
        c_index,_ = evaluator.eval_single(model, val_fold)
        fold_performances.append(c_index)
    
    avg_performance = np.mean(fold_performances)
    
    return avg_performance

def hyperparameter_tuning_with_crossval(train_set, model_base, param_grid, seed=42):
    set_seeds(seed)
    
    best_performance = -float('inf')
    best_params = None
    results = []

    # Iterate over all combinations of parameters
    for params in itertools.product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        
        # Perform cross-validation to evaluate the current parameter combination
        current_performance = cross_val_performance(train_set, param_dict, model_base, k=5, seed=seed)
        results.append((param_dict, current_performance))
        
        if current_performance > best_performance:
            best_performance = current_performance
            best_params = param_dict
        
        print(f'Tested {param_dict} with avg val_loss: {current_performance}')
    
    return best_params, results


def hyperparameter_tuning_withEta(train_set, val_set, model_base, param_grid, 
                                  selection_criterion='loss', seed = 42):

    set_seeds(seed)

    if selection_criterion not in ['loss', 'c-index']:
        raise ValueError("selection_criterion must be either 'loss' or 'c-index'")

    if selection_criterion == 'loss':
        best_performance = float('inf')  
    else:
        best_performance = -float('inf')  # For c-index, higher is better

    best_params = None
    results = []

    # Iterate over all combinations of parameters
    for params in itertools.product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        hidden_dim, learning_rate, num_dr, batch_size, weight_decay = param_dict['hidden_dim'], param_dict['learning_rate'], param_dict['num_dr'], param_dict['batch_size'], param_dict['weight_decay']
        num_layer = param_dict['num_layer']
        eta = param_dict['eta']
        
        model = model_base(input_dim=train_set.X.shape[1], hidden_dim=hidden_dim, output_dim=1, num_layers=num_layer, num_dr=num_dr)
        trainer = Trainer(model=model, weight_decay=weight_decay)

        _, val_loss_list = trainer.fit(
            train_set=train_set,
            val_set=val_set,
            eta_list=[eta],
            batch_size=batch_size,  # Example batch size
            num_epoch=300,  # Example number of epochs
            learning_rate=learning_rate,
            patience=50  # Example patience for early stopping
        )

        if selection_criterion == 'loss':
            current_performance = val_loss_list[-1]
        else:
            evaluator = Evaluator()
            c_index, _ = evaluator.eval_single(model, val_set)
            current_performance = c_index

        results.append((param_dict, current_performance))
        
        if selection_criterion == 'loss':
            if current_performance < best_performance:
                best_performance = current_performance
                best_params = param_dict
        else:
            if current_performance > best_performance:
                best_performance = current_performance
                best_params = param_dict

        print(f'Tested {param_dict} with val_{selection_criterion}: {current_performance}')

    return best_params, results

class Trainer:
    def __init__(self, model, metrics=None, optimizer = None, weight_decay = 0, seed = 42, scheduler=None, scheduler_params=None, device=None):
        '''metrics must start from NLLPCHazardLoss, then be others
        '''
        self.set_seeds(seed)
        self.model = model
        if metrics is None:
            self.metrics = CoxKLLOSS()
        else:
            self.metrics = metrics

        if device is None or device == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f'use pytorch-cuda for training on {self.device}')
            self.model.to(self.device)
            self.metrics.to(self.device)
        elif device == 'cpu':
            self.device = torch.device('cpu')
            print(f'use cpu for training on {self.device}')
        else:
            raise ValueError(f"Invalid device: {device}")



        if self.device.type == 'cuda':
            print('use pytorch-cuda for training.')
        # elif self.device.type == 'mps':
        #     print('use pytorch-mps for training.')
        else:
            print('GPU not found! will use cpu for training!')
        self.early_stopping = None

        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(self.model.parameters(), weight_decay=weight_decay)

        if scheduler is not None:
            self.scheduler = scheduler(self.optimizer, **(scheduler_params or {}))
        else:
            self.scheduler = None

    def set_seeds(self, seed_value=42):
        """Set seeds for reproducibility."""
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For CUDA to ensure reproducibility on GPU operations

    def set_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def fit(self, 
            train_set,
            val_set,
            eta_list,
            batch_size,
            num_epoch,
            learning_rate,
            patience,
            eval_c_index=False,
            early_stopping=True,
            test_set=None,
            eval_interval=50,
            **kwargs):
        
        self.set_lr(learning_rate)
        alpha = kwargs.get('alpha', 1.0)
        ridge_lambda = kwargs.get('ridge_lambda', 0.0)
        original_weight_decay = [pg.get('weight_decay', 0.0) for pg in self.optimizer.param_groups]
        if ridge_lambda > 0.0:
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = 0.0

        # train_data1 = Data(x_train, y_train, delta_tilde1)
        if batch_size <= 0:
            batch_size = len(train_set)
        train_drop_last = batch_size < len(train_set)
        self.train_dataloader = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=train_drop_last,
        )

        if val_set is not None:
            self.val_dataloader = DataLoader(dataset=val_set, batch_size=len(val_set), shuffle=False, drop_last=False)

        if eval_c_index:
            self.train_dataloader_full = DataLoader(dataset=train_set, batch_size=len(train_set), shuffle=False, drop_last=True)
            if test_set is not None:
                self.test_dataloader = DataLoader(dataset=test_set, batch_size=len(test_set), shuffle=False, drop_last=True)

        eta = eta_list[0]
        # eta = eta.to(self.device)
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        early_stopping_message_printed = False
        early_stopping_epoch = 0

        self.metrics = self.metrics.to(self.device)


        train_loss_list, val_loss_list = [], []
        c_index_train_list, c_index_val_list = [], []
        c_index_test_list = []
        loss_test_list = []

        if test_set is not None:
            c_index_test, loss_test = self.evaluate_test()
            print(f'Initial C-index on test set: {c_index_test}')
            print(f'Initial Loss on test set: {loss_test}')
            c_index_test_list.append((0, c_index_test))  # Store epoch (0) and C-index
            loss_test_list.append((0, loss_test))  # Store epoch (0) and loss

        try:
            for epoch in range(num_epoch):
                self.model.train()
                running_loss = 0.0

                for _, data in enumerate(self.train_dataloader, 0):
                    inputs, target, delta_tilde_tmp = [d.to(self.device) for d in data]
                    event, time = target[:, 0], target[:, 1]

                    # Sort the batch by the time variable
                    sort_indices = torch.argsort(time)
                    inputs = inputs[sort_indices]
                    event = event[sort_indices]
                    time = time[sort_indices]
                    delta_tilde_tmp = delta_tilde_tmp[sort_indices]

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs).view(-1)
                    loss = self.metrics(
                        log_h=outputs,
                        time=time,
                        event=event,
                        eta=eta,
                        risk_scores=delta_tilde_tmp,
                        alpha=alpha,
                        ridge_lambda=ridge_lambda,
                        model=self.model
                    )

                    if loss != torch.tensor(0.):
                        loss.backward()
                        self.optimizer.step()
                    
                    running_loss += loss.item()
                
                
                train_loss_list.append(running_loss / len(self.train_dataloader))  # Normalize by number of batches

                val_loss = float('nan')
                if val_set is not None: 
                    val_loss = self.evaluate(eta=eta, alpha=alpha, ridge_lambda=ridge_lambda)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if early_stopping and patience_counter >= patience:
                        print(f'Stopping early at epoch {epoch+1}')
                        if best_state is not None: # restore the best model
                            self.model.load_state_dict(best_state)
                        break
                    elif not early_stopping and patience_counter >= patience:
                        if not early_stopping_message_printed:
                            print(f'Early stopping is not enabled. Should stop early at epoch {epoch+1}')
                            early_stopping_message_printed = True
                            early_stopping_epoch = epoch + 1
                    
                    val_loss_list.append(val_loss)

                if val_set is not None:
                    print(f"loss for epoch {epoch} with eta {eta}: {running_loss / len(self.train_dataloader):.3f}, val_loss: {val_loss:.3f}")
                else:
                    print(f"loss for epoch {epoch} with eta {eta}: {running_loss / len(self.train_dataloader):.3f}")

                if eval_c_index:
                    _, c_index_train, c_index_val = self.evaluate(eta = 0, eval_c_index=eval_c_index)
                    c_index_train_list.append(c_index_train)
                    c_index_val_list.append(c_index_val)

                    if test_set is not None and (epoch + 1) % eval_interval == 0:
                        c_index_test, loss_test = self.evaluate_test()
                        print(f'Epoch {epoch+1}, C-index on test set: {c_index_test}')
                        print(f'Epoch {epoch+1}, Loss on test set: {loss_test}')
                        c_index_test_list.append((epoch + 1, c_index_test))  # Store epoch and C-index
                        loss_test_list.append((epoch + 1, loss_test))  # Store epoch and loss
                
                if self.scheduler is not None:
                    self.scheduler.step()
        finally:
            if ridge_lambda > 0.0:
                for param_group, wd in zip(self.optimizer.param_groups, original_weight_decay):
                    param_group['weight_decay'] = wd

        if eval_c_index:
            return train_loss_list, val_loss_list, c_index_train_list, c_index_val_list, c_index_test_list, loss_test_list, early_stopping_epoch
        return train_loss_list, val_loss_list

    def evaluate(self, eta, val=False, eval_c_index=False, alpha=1.0, ridge_lambda=0.0):
        self.model.eval()
        dataloader = self.val_dataloader
        val_loss = 0.0

        if isinstance(eta, (int, float)):
            eta = torch.tensor(eta, dtype=torch.float32, device=self.device)
        elif isinstance(eta, torch.Tensor) and eta.device != self.device:
            eta = eta.to(self.device)

        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                inputs, target, delta_tilde_tmp = [d.to(self.device) for d in data]
                event, time = target[:, 0], target[:, 1]
                
                # Sort the batch by the time variable
                sort_indices = torch.argsort(time)
                inputs = inputs[sort_indices]
                event = event[sort_indices]
                time = time[sort_indices]
                delta_tilde_tmp = delta_tilde_tmp[sort_indices]

                outputs = self.model(inputs).view(-1)
                loss = self.metrics(
                    outputs,
                    time=time,
                    event=event,
                    eta=eta,
                    risk_scores=delta_tilde_tmp,
                    alpha=alpha,
                    ridge_lambda=ridge_lambda,
                    model=self.model
                )
                val_loss += loss.item()

                if eval_c_index:
                    # Move tensors to CPU before converting to numpy
                    # risk_scores = outputs.cpu().numpy()
                    # event_cpu = event.cpu().numpy()
                    # time_cpu = time.cpu().numpy()
                    # c_index_val = sksurv.metrics.concordance_index_censored(event_cpu, time_cpu, risk_scores)[0]

                    risk_np  = outputs.detach().cpu().numpy()
                    event_np = event.detach().cpu().numpy().astype(bool)  # <- cast to bool
                    time_np  = time.detach().cpu().numpy()

                    c_index_val = sksurv.metrics.concordance_index_censored(event_np, time_np, risk_np)[0]

        if eval_c_index:
            with torch.no_grad():
                for i, data in enumerate(self.train_dataloader_full, 0):
                    inputs, target, delta_tilde_tmp = [d.to(self.device) for d in data]
                    event, time = target[:, 0], target[:, 1]
                    
                    # Sort the batch by the time variable
                    sort_indices = torch.argsort(time)
                    inputs = inputs[sort_indices]
                    event = event[sort_indices]
                    time = time[sort_indices]
                    delta_tilde_tmp = delta_tilde_tmp[sort_indices]

                    outputs = self.model(inputs).view(-1)            
                if i != 0:
                    raise ValueError('The validation dataset is not loaded correctly')
                risk_scores = []
                # Move tensors to CPU before converting to numpy
                # risk_scores.extend(outputs.cpu().numpy())
                # event_cpu = event.cpu().numpy()
                # time_cpu = time.cpu().numpy()
                # c_index_train = sksurv.metrics.concordance_index_censored(event_cpu, time_cpu, risk_scores)[0]
                risk_np  = outputs.detach().cpu().numpy()
                event_np = event.detach().cpu().numpy().astype(bool)
                time_np  = time.detach().cpu().numpy()
                c_index_train = sksurv.metrics.concordance_index_censored(event_np, time_np, risk_np)[0]               



        if not eval_c_index:
            return val_loss
        else:
            return val_loss, c_index_train, c_index_val


    def evaluate_test(self):
        self.model.eval()
        dataloader = self.test_dataloader
        with torch.no_grad():
            for data in dataloader:
                inputs, target = [d.to(self.device) for d in data]
                event, time = target[:, 0], target[:, 1]
                print("event:", event[:5])  # Print first 5 samples
                print("time:", time[:5])  # Print first 5 targets

                print(f"Inputs device: {inputs.device}")
                print(f"Model device: {next(self.model.parameters()).device}")
                print(f"Target device: {target.device}")
                
                outputs = self.model(inputs).view(-1)
                print("outputs:", outputs[:5])  # Print first 5 targets
                risk_np  = outputs.detach().cpu().numpy()
                event_np = event.detach().cpu().numpy().astype(bool)
                time_np  = time.detach().cpu().numpy()

                c_index_test = sksurv.metrics.concordance_index_censored(event_np, time_np, risk_np)[0]
                eta_tensor = torch.tensor(0.0, dtype=torch.float32, device=self.device)

                loss_test = self.metrics(outputs, time=time, event=event, eta=eta_tensor, risk_scores=None)
        return c_index_test, loss_test
