#! /bin/bash

python main.py --method="naive" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="naive run 1" --experiment="domain incremental" --seed=1 --device="cuda:0"
python main.py --method="naive" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="naive run 2" --experiment="domain incremental" --seed=2 --device="cuda:0"
python main.py --method="naive" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="naive run 3" --experiment="domain incremental" --seed=3 --device="cuda:0"
python main.py --method="naive" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="naive run 4" --experiment="domain incremental" --seed=4 --device="cuda:0"
python main.py --method="naive" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="naive run 5" --experiment="domain incremental" --seed=5 --device="cuda:0"

python main.py --method="cumulative" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="cumulative run 1" --experiment="domain incremental" --seed=1 --device="cuda:0"
python main.py --method="cumulative" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="cumulative run 2" --experiment="domain incremental" --seed=2 --device="cuda:0"
python main.py --method="cumulative" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="cumulative run 3" --experiment="domain incremental" --seed=3 --device="cuda:0"
python main.py --method="cumulative" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="cumulative run 4" --experiment="domain incremental" --seed=4 --device="cuda:0"
python main.py --method="cumulative" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="cumulative run 5" --experiment="domain incremental" --seed=5 --device="cuda:0"

python main.py --method="ewc" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="ewc run 1" --experiment="domain incremental" --seed=1 --device="cuda:0"
python main.py --method="ewc" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="ewc run 2" --experiment="domain incremental" --seed=2 --device="cuda:0"
python main.py --method="ewc" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="ewc run 3" --experiment="domain incremental" --seed=3 --device="cuda:0"
python main.py --method="ewc" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="ewc run 4" --experiment="domain incremental" --seed=4 --device="cuda:0"
python main.py --method="ewc" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="ewc run 5" --experiment="domain incremental" --seed=5 --device="cuda:0"

python main.py --method="agem" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="agem run 1" --experiment="domain incremental" --seed=1 --device="cuda:0"
python main.py --method="agem" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="agem run 2" --experiment="domain incremental" --seed=2 --device="cuda:0"
python main.py --method="agem" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="agem run 3" --experiment="domain incremental" --seed=3 --device="cuda:0"
python main.py --method="agem" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="agem run 4" --experiment="domain incremental" --seed=4 --device="cuda:0"
python main.py --method="agem" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="agem run 5" --experiment="domain incremental" --seed=5 --device="cuda:0"

python main.py --method="replay" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="replay run 1" --experiment="domain incremental" --seed=1 --device="cuda:0"
python main.py --method="replay" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="replay run 2" --experiment="domain incremental" --seed=2 --device="cuda:0"
python main.py --method="replay" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="replay run 3" --experiment="domain incremental" --seed=3 --device="cuda:0"
python main.py --method="replay" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="replay run 4" --experiment="domain incremental" --seed=4 --device="cuda:0"
python main.py --method="replay" --interactive_logger=0 --device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="replay run 5" --experiment="domain incremental" --seed=5 --device="cuda:0"


