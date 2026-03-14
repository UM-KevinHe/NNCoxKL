from collections import defaultdict
from sksurv.metrics import concordance_index_censored, integrated_brier_score, cumulative_dynamic_auc
import numpy as np
import pdb
import torch
import sksurv
import pandas as pd
from sksurv.util import Surv
from sksurv.functions import StepFunction
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator

from NNCoxKL.loss_feiyang import cox_kl_loss

class Evaluator:
    def __init__(self):
        '''the input duration_train should be the raw durations (continuous),
        NOT the discrete index of duration.
        '''
        self.device = torch.device('cpu')

    def eval_single(self, model, test_set, return_rs = False):
        model.to(self.device)
        model.eval()
        with torch.no_grad():

            inputs = test_set.X.to(self.device)
            target = test_set.y.to(self.device)
            
            risk_scores_testing = []

            outputs = model(inputs).view(-1)
            risk_scores_testing.extend(outputs.tolist())
            delta_test, time_test = target[:, 0].bool(), target[:, 1].cpu()
            c_index = concordance_index_censored(delta_test, time_test, risk_scores_testing)[0]

            loss = cox_kl_loss(outputs, time_test, delta_test, 0, risk_scores=None)

        if return_rs:
            return c_index, loss, risk_scores_testing
        return c_index, loss
    
    def eval_risk_scores(self, model, test_set):

        model.to(self.device)
        model.eval()
        with torch.no_grad():
            inputs = test_set.X.to(self.device)
            
            risk_scores_testing = []

            outputs = model(inputs).view(-1)
            risk_scores_testing.extend(outputs.tolist())
            
        return risk_scores_testing

    
    def compute_breslow_estimator_from_train(self, time_train, event_train, risk_scores_train):
        """
        Computes the Baseline Hazard H0(t) using TRAINING data.
        """
        # Sort by time
        order = np.argsort(time_train)
        time_sorted = time_train[order]
        event_sorted = event_train[order]
        risk_scores_sorted = risk_scores_train[order]
        
        # Get unique times where events occurred
        unique_times = np.unique(time_sorted[event_sorted])
        
        baseline_hazard = []
        cum_hazard = 0
        
        exp_risk_scores = np.exp(risk_scores_sorted)
        
        for t in unique_times:
            # Risk set: everyone who survived at least until t
            at_risk_mask = time_sorted >= t
            denominator = np.sum(exp_risk_scores[at_risk_mask])
            
            # Events at exactly time t
            events_at_t = np.sum(event_sorted[time_sorted == t])
            
            hazard_contribution = events_at_t / denominator
            cum_hazard += hazard_contribution
            baseline_hazard.append(cum_hazard)
            
        # Return a StepFunction (standard sksurv tool) for easy interpolation later
        return StepFunction(unique_times, np.array(baseline_hazard))

    
    def eval_ibs(self, model, train_set, test_set, times=None):
        """
        Implementation of IBS requiring both Train (to learn H0) and Test sets.
        """
        model.to(self.device)
        model.eval()
        
        # 1. GET TRAINING RISK SCORES & DATA (To compute Baseline Hazard)
        # This is the "Learning" phase of the evaluator
        with torch.no_grad():
            inputs_train = train_set.X.to(self.device)
            target_train = train_set.y.to(self.device)
            
            # Get risk scores r(x) for training
            # Note: Ensure your model handles the full train batch or loop if too large
            out_train = model(inputs_train).view(-1).cpu().numpy()
            
            delta_train = target_train[:, 0].bool().cpu().numpy()
            time_train = target_train[:, 1].cpu().numpy()
            
            # Learn H0(t) from TRAIN
            baseline_hazard_func = self.compute_breslow_estimator_from_train(
                time_train, delta_train, out_train
            )

        # 2. GET TEST PREDICTIONS
        with torch.no_grad():
            inputs_test = test_set.X.to(self.device)
            target_test = test_set.y.to(self.device)
            
            out_test = model(inputs_test).view(-1).cpu().numpy()
            
            delta_test = target_test[:, 0].bool().cpu().numpy()
            time_test = target_test[:, 1].cpu().numpy()

        # 3. PREPARE SK-SURV ARRAYS
        # IPCW requires training data structure
        y_train_struc = Surv.from_arrays(delta_train, time_train)
        y_test_struc = Surv.from_arrays(delta_test, time_test)

        # 4. DEFINE TIME POINTS FOR EVALUATION
        if times is None:
            # Standard: between 10th and 90th percentile of observed times
            times = np.percentile(time_test, np.linspace(10, 90, 100))

        # 5. CALCULATE SURVIVAL PROBABILITIES FOR TEST SET
        # S(t|x) = exp(-H0(t) * exp(r(x)))
        survival_probs = []
        
        # Get H0(t) at the requested evaluation times
        h0_at_times = baseline_hazard_func(times)
        
        for r_score in out_test:
            # Compute survival curve for this patient
            s_t = np.exp(-h0_at_times * np.exp(r_score))
            survival_probs.append(s_t)
            
        survival_probs = np.array(survival_probs)

        # 6. COMPUTE IBS
        # Note: y_train is passed first!
        ibs = integrated_brier_score(y_train_struc, y_test_struc, survival_probs, times)

        return ibs


    def eval_tdauc(self, model, train_set, test_set, times=None):
        """
        Computes the cumulative dynamic AUC (td-AUC).
        Requires Train set for IPCW (Inverse Probability of Censoring Weights).
        """
        model.to(self.device)
        model.eval()

        # Helper to safely convert Tensor to Numpy
        def to_numpy(y_data):
            if isinstance(y_data, torch.Tensor):
                return y_data.cpu().numpy()
            return y_data

        # Convert targets to numpy for sksurv
        y_train_np = to_numpy(train_set.y)
        y_test_np = to_numpy(test_set.y)

        # 1. Prepare Data Structures for scikit-survival
        y_train_struc = Surv.from_arrays(
            y_train_np[:, 0].astype(bool), 
            y_train_np[:, 1]
        )
        y_test_struc = Surv.from_arrays(
            y_test_np[:, 0].astype(bool), 
            y_test_np[:, 1]
        )

        # 2. Get Risk Scores for Test Data
        with torch.no_grad():
            inputs_test = test_set.X.to(self.device)
            risk_scores_test = model(inputs_test).view(-1).cpu().numpy()

        # 3. Define Time Points
        if times is None:
            # Use 10th to 90th percentile of OBSERVED event times in test set
            # (Using all times can sometimes cause errors at the boundaries)
            observed_times = y_test_np[y_test_np[:, 0].astype(bool), 1]
            if len(observed_times) > 0:
                times = np.percentile(observed_times, np.linspace(10, 90, 15))
            else:
                times = np.percentile(y_test_np[:, 1], np.linspace(10, 90, 15))

        # 4. Calculate td-AUC
        try:
            auc_over_time, mean_auc = cumulative_dynamic_auc(
                y_train_struc, 
                y_test_struc, 
                risk_scores_test, 
                times
            )
        except ValueError as e:
            print(f"Warning: td-AUC calculation skipped due to time range issues: {e}")
            return np.nan

        return mean_auc