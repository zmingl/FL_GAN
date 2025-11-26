# import argparse

# from gan_fl.config import build_argparser
# from gan_fl.trainer.federated_trainer import run_federated_training

import gan_fl.trainer.federated_trainer as fl_trainer
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from gan_fl.config import build_argparser
    args = build_argparser().parse_args()
    fl_trainer.run_federated_training(args)