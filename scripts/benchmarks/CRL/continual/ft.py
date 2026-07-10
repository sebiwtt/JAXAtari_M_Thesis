# =============================================================================
# FT - naive finetuning (the no-mitigation baseline)
# =============================================================================
# Params are carried forward from task to task and nothing else is done; the
# base-class defaults already implement this. `uses_train_hooks = False` keeps
# train()'s compiled update byte-identical to plain PPO.

from continual.base import CLMethod


class FT(CLMethod):
    name = "ft"
    uses_train_hooks = False
