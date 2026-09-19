# Analysis notes

This experiment uses synthetic data with deliberately embedded relationships. Segment differences are descriptive, not causal evidence.

The model uses a stratified 80/20 split. Median imputation, scaling and encoding are fit only on training rows. ROC-AUC measures ranking; the classification report uses a 0.5 threshold.

Candidate retention experiments: onboarding support for early-tenure customers and contract-value offers for month-to-month customers. Validate these hypotheses on real, consented data and a randomized experiment before acting.
