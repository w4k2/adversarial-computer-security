#! /bin/bash

python main.py --method="naive" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="naive" --experiment="domain incremental" --seed=1 --device="cuda:0"
python main.py --method="naive" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="naive" --experiment="domain incremental" --seed=2 --device="cuda:0"
python main.py --method="naive" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="naive" --experiment="domain incremental" --seed=3 --device="cuda:0"
python main.py --method="naive" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="naive" --experiment="domain incremental" --seed=4 --device="cuda:0"
python main.py --method="naive" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="naive" --experiment="domain incremental" --seed=5 --device="cuda:0"

python main.py --method="cumulative" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="cumulative" --experiment="domain incremental" --seed=1 --device="cuda:0"
python main.py --method="cumulative" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="cumulative" --experiment="domain incremental" --seed=2 --device="cuda:0"
python main.py --method="cumulative" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="cumulative" --experiment="domain incremental" --seed=3 --device="cuda:0"
python main.py --method="cumulative" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="cumulative" --experiment="domain incremental" --seed=4 --device="cuda:0"
python main.py --method="cumulative" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="cumulative" --experiment="domain incremental" --seed=5 --device="cuda:0"

python main.py --method="ewc" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="ewc" --experiment="domain incremental" --seed=1 --device="cuda:0"
python main.py --method="ewc" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="ewc" --experiment="domain incremental" --seed=2 --device="cuda:0"
python main.py --method="ewc" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="ewc" --experiment="domain incremental" --seed=3 --device="cuda:0"
python main.py --method="ewc" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="ewc" --experiment="domain incremental" --seed=4 --device="cuda:0"
python main.py --method="ewc" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="ewc" --experiment="domain incremental" --seed=5 --device="cuda:0"

python main.py --method="agem" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="agem" --experiment="domain incremental" --seed=1 --device="cuda:0"
python main.py --method="agem" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="agem" --experiment="domain incremental" --seed=2 --device="cuda:0"
python main.py --method="agem" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="agem" --experiment="domain incremental" --seed=3 --device="cuda:0"
python main.py --method="agem" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="agem" --experiment="domain incremental" --seed=4 --device="cuda:0"
python main.py --method="agem" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="agem" --experiment="domain incremental" --seed=5 --device="cuda:0"

python main.py --method="replay" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="replay" --experiment="domain incremental" --seed=1 --device="cuda:0"
python main.py --method="replay" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="replay" --experiment="domain incremental" --seed=2 --device="cuda:0"
python main.py --method="replay" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="replay" --experiment="domain incremental" --seed=3 --device="cuda:0"
python main.py --method="replay" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="replay" --experiment="domain incremental" --seed=4 --device="cuda:0"
python main.py --method="replay" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="replay" --experiment="domain incremental" --seed=5 --device="cuda:0"


