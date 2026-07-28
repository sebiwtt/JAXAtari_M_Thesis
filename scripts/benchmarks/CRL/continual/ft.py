# =============================================================================
# FT - naive finetuning (the no-mitigation baseline)
# =============================================================================

from continual.base import CLMethod


class FT(CLMethod):
    name = "ft"
    uses_train_hooks = False
