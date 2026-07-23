import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.breakout.breakout_mod_plugins import (
    SpeedModeMod,
    SmallPaddleMod,
    BigPaddleMod,
    BallDriftMod,
    BallGravityMod,
    BallColorMod,
    BlockColorMod,
    PlayerColorMod,
    ChangePaddleColorMod,
    ChangeBallColorMod,
    ChangeBackgroundColorMod,
    SwapScoreColorMod,
    GreyscaleThemeMod,
    RoundPaddleMod,
    BallLossPenaltyMod,
    FlattenRowValuesMod,
    TopRowOnlyMod,
    SurvivalRewardMod,
    EveryKContactsMod,
    BottomRowFirstMod,
    FasterBallMod,
    SlowerBallMod,
    FasterPaddleMod,
    SlowerPaddleMod,
    RandomServeMod,
    BallSpeedX2Mod,
    BallSpeedX3Mod,
    BallSpeedX4Mod,
    BallSpeedX5Mod,
)

class BreakoutEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Breakout.
    It simply inherits all logic from JaxAtariModController and defines the REGISTRY.
    """

    REGISTRY = {
        # Visual
        "change_paddle_color": ChangePaddleColorMod,
        "change_ball_color": ChangeBallColorMod,
        "change_background_color": ChangeBackgroundColorMod,
        "swap_score_color": SwapScoreColorMod,
        "greyscale_theme": GreyscaleThemeMod,
        "round_paddle": RoundPaddleMod,
        # Vis. mods that were already available
        "ball_color": BallColorMod,
        "block_color": BlockColorMod,
        "player_color": PlayerColorMod,

        # Dynamics
        "faster_ball": FasterBallMod,
        "slower_ball": SlowerBallMod,
        "faster_paddle": FasterPaddleMod,
        "slower_paddle": SlowerPaddleMod,
        "random_serve": RandomServeMod,
        "ball_drift": BallDriftMod,

        # Magnitude sequence (ball_speed_xN): same mod, incrementally faster ball
        "ball_speed_x2": BallSpeedX2Mod,
        "ball_speed_x3": BallSpeedX3Mod,
        "ball_speed_x4": BallSpeedX4Mod,
        "ball_speed_x5": BallSpeedX5Mod,
        # Dyn. mods that were already available
        "ball_gravity": BallGravityMod,

        # Reward
        "ball_loss_penalty": BallLossPenaltyMod,
        "flatten_row_values": FlattenRowValuesMod,
        "top_row_only": TopRowOnlyMod,
        "survival_reward": SurvivalRewardMod,
        "every_k_contacts": EveryKContactsMod,
        "bottom_row_first": BottomRowFirstMod,

        # Misc.
        "small_paddle": SmallPaddleMod,
        "big_paddle": BigPaddleMod,
        
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "breakout", "sprites")

    def __init__(self,
                 env,
                 mods_config: list = [],
                 allow_conflicts: bool = False
                 ):

        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY
        )
