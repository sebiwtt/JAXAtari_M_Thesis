# =============================================================================
# CLI entry point for the continual-RL benchmark
# =============================================================================
# Thin hydra wrapper; the actual orchestration is agent-agnostic and lives in
# framework/runner.py. The agent is selected by the AGENT config key ("ppo" is
# the reference PPO + CL-method stack; see agents/). To run programmatically
# instead, use framework.run_benchmark - see FRAMEWORK.md.
#
#   python ppo_crl_continual.py sequence=pong_dyn4 method=ewc modality=oc
#   python ppo_crl_continual.py AGENT=random sequence=pong_dyn4 TRACK=False
# =============================================================================

import sys

import hydra
from omegaconf import OmegaConf

from framework.runner import run_from_config
from tools.config_groups import rewrite_sequence_argv


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    print("Config:\n", OmegaConf.to_yaml(OmegaConf.create(config)))
    run_from_config(config)


# sequence configs live in config/sequence/<game>/; accept the flat
# "sequence=pong_dyn4" spelling as well as hydra's "pong/pong_dyn4".
if __name__ == "__main__":
    rewrite_sequence_argv(sys.argv)
    main()
