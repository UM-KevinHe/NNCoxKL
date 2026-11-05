# NNCoxKL

NNCoxKL implements neural Cox proportional-hazards models that incorporate external risk scores through risk-set distribution matching. The default training objective combines the Breslow Cox partial likelihood with teacher-to-student cross-entropy, which has the same student gradients as KL divergence when the teacher scores and regularization settings are fixed.

The `NNCoxKL` package contains the model architectures, survival dataset wrapper, loss functions, training utilities, and evaluation utilities. Targets use two columns: event indicator and observed time. The `delta_tilde` dataset field supplies teacher risk scores and is required by the training utility.

Keep training, validation, and test cohorts separate, and fit preprocessing and teacher models without held-out outcomes. Leave the training utility's optional test monitoring disabled when selecting models; assess test performance only after selection. Raw regularized losses across different teacher temperatures or regularization weights are not directly comparable as KL values.

This repository contains the core implementation only; datasets, checkpoints, experiment scripts, and tutorials are excluded.
