# peppercall

# Setup project
```bash
conda env create -n peppercall --file env.yml
```

# Train the model
```bash
python train.py --out-steps 12 --aggregation MS
```

# Open MLflow service
```bash
mlflow ui
```