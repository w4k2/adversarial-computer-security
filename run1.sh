#! /bin/bash

python main.py --method="lwf" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="lwf run 1" --experiment="domain incremental" --seed=1 --device="cuda:1"
python main.py --method="lwf" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="lwf run 2" --experiment="domain incremental" --seed=2 --device="cuda:1"
python main.py --method="lwf" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="lwf run 3" --experiment="domain incremental" --seed=3 --device="cuda:1"
python main.py --method="lwf" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="lwf run 4" --experiment="domain incremental" --seed=4 --device="cuda:1"
python main.py --method="lwf" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="lwf run 5" --experiment="domain incremental" --seed=5 --device="cuda:1"

python main.py --method="mir" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="mir run 1" --experiment="domain incremental" --seed=1 --device="cuda:1"
python main.py --method="mir" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="mir run 2" --experiment="domain incremental" --seed=2 --device="cuda:1"
python main.py --method="mir" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="mir run 3" --experiment="domain incremental" --seed=3 --device="cuda:1"
python main.py --method="mir" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="mir run 4" --experiment="domain incremental" --seed=4 --device="cuda:1"
python main.py --method="mir" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="mir run 5" --experiment="domain incremental" --seed=5 --device="cuda:1"

python main.py --method="icarl" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="icarl run 1" --experiment="domain incremental" --seed=1 --device="cuda:1"
python main.py --method="icarl" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="icarl run 2" --experiment="domain incremental" --seed=2 --device="cuda:1"
python main.py --method="icarl" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="icarl run 3" --experiment="domain incremental" --seed=3 --device="cuda:1"
python main.py --method="icarl" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="icarl run 4" --experiment="domain incremental" --seed=4 --device="cuda:1"
python main.py --method="icarl" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="icarl run 5" --experiment="domain incremental" --seed=5 --device="cuda:1"

python main.py --method="si" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="si run 1" --experiment="domain incremental" --seed=1 --device="cuda:1"
python main.py --method="si" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="si run 2" --experiment="domain incremental" --seed=2 --device="cuda:1"
python main.py --method="si" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="si run 3" --experiment="domain incremental" --seed=3 --device="cuda:1"
python main.py --method="si" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="si run 4" --experiment="domain incremental" --seed=4 --device="cuda:1"
python main.py --method="si" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="si run 5" --experiment="domain incremental" --seed=5 --device="cuda:1"

python main.py --method="gdumb" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="gdumb run 1" --experiment="domain incremental" --seed=1 --device="cuda:1"
python main.py --method="gdumb" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="gdumb run 2" --experiment="domain incremental" --seed=2 --device="cuda:1"
python main.py --method="gdumb" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="gdumb run 3" --experiment="domain incremental" --seed=3 --device="cuda:1"
python main.py --method="gdumb" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="gdumb run 4" --experiment="domain incremental" --seed=4 --device="cuda:1"
python main.py --method="gdumb" --interactive_logger=0 --n_experiences=20 --adversarial_attacks="same" --n_epochs=10 --run_name="gdumb run 5" --experiment="domain incremental" --seed=5 --device="cuda:1"
