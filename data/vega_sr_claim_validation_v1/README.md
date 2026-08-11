{
  "name": "vega_sr_claim_validation_v1",
  "case_count": 48,
  "targets": [
    "observer",
    "critic"
  ],
  "cases_per_target": 24,
  "n_train": 256,
  "n_val": 128,
  "n_test": 1024,
  "note": "VEGA-SR causal claim-validation dataset with observer and critic target groups.",
  "target_counts": {
    "critic": 24,
    "observer": 24
  },
  "structure_counts": {
    "critic|exp_log": 2,
    "critic|exp_trig": 2,
    "critic|friedman_interaction": 2,
    "critic|log_sqrt": 2,
    "critic|polynomial_trig": 2,
    "critic|rational": 2,
    "critic|rational_interaction": 2,
    "critic|rational_trig": 2,
    "critic|sparse_interaction": 2,
    "critic|sqrt_rational": 2,
    "critic|trig_interaction": 2,
    "critic|trig_polynomial": 2,
    "observer|exp_log": 2,
    "observer|exp_trig": 2,
    "observer|friedman_interaction": 2,
    "observer|log_sqrt": 2,
    "observer|polynomial_trig": 2,
    "observer|rational": 2,
    "observer|rational_interaction": 2,
    "observer|rational_trig": 2,
    "observer|sparse_interaction": 2,
    "observer|sqrt_rational": 2,
    "observer|trig_interaction": 2,
    "observer|trig_polynomial": 2
  },
  "true_variable_counts": {
    "critic|2": 8,
    "critic|3": 8,
    "critic|4": 8,
    "observer|2": 8,
    "observer|3": 8,
    "observer|4": 8
  }
}