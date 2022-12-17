#! /bin/bash

python main.py --method="lwf" --interactive_logger=0 ----device="cuda:1" --n_experiences=20 --adversarial_attacks="same" --n_epochs=5 --run_name="lwf"
python main.py --method="mir" --interactive_logger=0 ----device="cuda:1" --n_experiences=20 --adversarial_attacks="same" --n_epochs=5 --run_name="mir"
python main.py --method="icarl" --interactive_logger=0 ----device="cuda:1" --n_experiences=20 --adversarial_attacks="same" --n_epochs=5 --run_name="icarl"
