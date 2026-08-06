import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.tennis.tennis_mod_plugins import (
    RandomBallSpeedWrapper,
    RandomWalkSpeedWrapper,
    FastPlayerMod,
    SuperGravityMod,
    LazyEnemyMod,
    HighBounceMod,
    FastEnemyMod,
    ClayCourtMod,
    GrassCourtMod,
    HardCourtMod,
    NightMod,
    GrayscaleMod,
    InvertedColorsMod,
    SlowPlayerMod,
    FloatyBallMod,
    LowBounceMod,
    ShortShotsMod,
)

class TennisEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Tennis.
    It simply inherits all logic from JaxAtariModController and defines the REGISTRY.
    """

    REGISTRY = {
        "random_ball_speed": RandomBallSpeedWrapper,
        "random_walk_speed": RandomWalkSpeedWrapper,
        "fast_player": FastPlayerMod,
        "super_gravity": SuperGravityMod,
        "lazy_enemy": LazyEnemyMod,
        "high_bounce": HighBounceMod,
        "fast_enemy": FastEnemyMod,
        "clay_court": ClayCourtMod,
        "grass_court": GrassCourtMod,
        "hard_court": HardCourtMod,
        "night_mode": NightMod,
        "grayscale": GrayscaleMod,
        "inverted_colors": InvertedColorsMod,

        # CRL dyn4 pattern mods
        "slow_player": SlowPlayerMod,
        "floaty_ball": FloatyBallMod,
        "low_bounce": LowBounceMod,
        "short_shots": ShortShotsMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "tennis", "sprites")

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
