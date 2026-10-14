# Nested Sampling, Explained by a Cosmologist

Code accompanying the Medium article **"Nested Sampling, Explained by a Cosmologist"** by Samit Ganguly.

The script reproduces every number and figure in the article:

1. **Two-peaked toy likelihood**: shows how the live points climb both peaks at once, and checks the evidence against the analytic value (ln Z = −4.48).
2. **Evidence accumulation**: plots ln L against the shrinking prior volume X, and L·ΔX to show where the evidence comes from.
3. **ΛCDM vs wCDM on synthetic supernovae**: 200 fake SNe Ia generated from ΛCDM (Ω_m = 0.3, w = −1). wCDM fits slightly better, but the Bayesian evidence still prefers ΛCDM.

## Run it

```bash
pip install -r requirements.txt
python nested_sampling_demo.py
```

It takes about a minute on a laptop and writes `fig1_shrinking.png`, `fig2_evidence.png`, `fig3_cosmo.png` and `synthetic_sne.txt` to the current folder. The evidences and Bayes factor are printed to the terminal.

Random seeds are fixed, so you should get the same numbers as the article (small differences across dynesty versions are possible).

## Reading

- J. Skilling, *Nested Sampling for General Bayesian Computation*, Bayesian Analysis 1(4), 2006
- J. S. Speagle, *dynesty*, MNRAS 493, 2020
