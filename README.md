Installation of Environment Specific Dependencies:
```
conda install -c conda-forge openbabel
conda install -c conda-forge qvina
pip install -r requirements.txt
```

Results on the regression task.

| Target | Ridge | Lasso | XGBoost | GP (exact) | GP (sparse) | MPNN  | Attentive FP | Char RNN (SMILES) | Char RNN (SELFIES) |
|--------|-------|-------|---------|------------|-------------|-------|--------------|-------------------|--------------------|
| ESR2   | 0.421 | 0.416 | 0.497   | 0.441      | 0.508       | 0.506 | 0.627        | 0.6446            | 0.6193             |
| F2     | 0.672 | 0.663 | 0.688   | 0.705      | 0.744       | 0.798 | 0.880        | 0.8771            | 0.8662             |
| KIT    | 0.604 | 0.594 | 0.674   | 0.637      | 0.684       | 0.755 | 0.806        | 0.8095            | 0.78               |
| PARP1  | 0.706 | 0.700 | 0.723   | 0.743      | 0.772       | 0.815 | 0.910        | 0.907             | 0.8973             |
| PGR    | 0.242 | 0.245 | 0.345   | 0.291      | 0.387       | 0.324 | 0.678        | 0.6733            | 0.6478             |
