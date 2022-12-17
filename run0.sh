#! /bin/bash

# python main.py --method="naive" --interactive_logger=0 ----device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=5 --run_name="naive"
python main.py --method="ewc" --interactive_logger=0 ----device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=5 --run_name="ewc"
python main.py --method="agem" --interactive_logger=0 ----device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=5 --run_name="agem"
python main.py --method="replay" --interactive_logger=0 ----device="cuda:0" --n_experiences=20 --adversarial_attacks="same" --n_epochs=5 --run_name="replay"

