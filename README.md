# Continual learning for computer security

Repo with code for paper "Continual learning for computer security"

Codebase for this project uses [avalanche](https://avalanche.continualai.org/) for continual learning and [foolbox](https:L//github.com/bethgelab/foolbox) for generation of adversarial learning. Experiments can be runed by executing `main.py` file with poper commandline arguments. 
Here are examples of run:

```
python main.py --method="replay" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="replay run 1" --experiment="domain incremental" --seed=1 --device="cuda:0"
```

```
python main.py --method="gdumb" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="gdumb run 1" --experiment="domain incremental" --seed=1 --device="cuda:1"
```

```
python main.py --method="lwf" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="lwf run 1" --experiment="domain incremental" --seed=1 --device="cuda:1"
```

You can obtain list of all comandline args with:

```
python main.py -h
```

The results are stored with mlflow library in `mlruns` folder. To run mlflow ui please [install mlflow](https://mlflow.org/docs/latest/quickstart.html#installing-mlflow) and then run in main repo directory:

```
mlflow ui
```


# Repo structure

List of all most important modules and scripts in our repo:

 - `data` folder contains code for loading datasets
 - `methods` directory contains code for custom implementations of CL algorithms
 - `utils` contains code primarly for plot generation. It also has implementation of TsAIL adversarial attack
 - `adversarial.py` generates new dataset with adversarial examples
 - `main.py` is script for running experiments

# Citation policy

tbd

<!-- If you use our code please cite this paper:

```
some bibtex
``` -->
