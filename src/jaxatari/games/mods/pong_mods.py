import os
from functools import partial
import jax
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.pong.pong_mod_plugins import LazyEnemyMod, RandomEnemyMod, AlwaysZeroScoreMod, LinearMovementMod, ShiftPlayerMod, ShiftEnemyMod, NoFireMod, ChangeBackgroundColorMod, ChangePlayerColorMod, SwapPaddleColorsMod, ChangeBallColorMod, ChangeScoreColorMod, GrayscaleThemeMod, FastBallMod, SlowBallMod, FastPaddleMod, SlowPaddleMod, RandomServeMod, BallGravityMod, BallDriftMod, ScaleRewardMod, RewardPerHitMod, TimePenaltyMod, AsymmetricRewardMod, InvertedRewardMod, _RenderNoiseMod, RenderNoise20Mod, RenderNoise40Mod, RenderNoise60Mod, RenderNoise80Mod, apply_render_noise, BallSpeedX2Mod, BallSpeedX3Mod, BallSpeedX4Mod, BallSpeedX5Mod

class PongEnvMod(JaxAtariModController):    
    """
    Game-specific Mod Controller for Pong.
    It simply inherits all logic from JaxAtariModController and defines the PONG_MOD_REGISTRY.
    """

    REGISTRY = {
        "lazy_enemy": LazyEnemyMod,
        "random_enemy": RandomEnemyMod,
        "zero_score": AlwaysZeroScoreMod,
        "linear_movement": LinearMovementMod,
        "shift_player": ShiftPlayerMod,
        "shift_enemy": ShiftEnemyMod,
        "no_fire": NoFireMod,
        "change_player_color": ChangePlayerColorMod,
        # New mods for CRL Sequences
        # Visual
        "change_background_color": ChangeBackgroundColorMod,
        "swap_paddle_colors": SwapPaddleColorsMod,
        "change_ball_color": ChangeBallColorMod,
        "change_score_color": ChangeScoreColorMod,
        "grayscale_theme": GrayscaleThemeMod,
        # Dynamics
        "fast_ball": FastBallMod,
        "slow_ball": SlowBallMod,
        "fast_paddle": FastPaddleMod,
        "slow_paddle": SlowPaddleMod,
        "random_serve": RandomServeMod,
        "ball_gravity": BallGravityMod,
        "ball_drift": BallDriftMod,
        # Reward
        "scale_reward": ScaleRewardMod,
        "reward_per_hit": RewardPerHitMod,
        "time_penalty": TimePenaltyMod,
        "asymmetric_reward": AsymmetricRewardMod,
        "inverted_reward": InvertedRewardMod,
        # Visual-Mag (scaled render noise)
        "render_noise_20": RenderNoise20Mod,
        "render_noise_40": RenderNoise40Mod,
        "render_noise_60": RenderNoise60Mod,
        "render_noise_80": RenderNoise80Mod,
        # Dynamics-Mag (scaled ball speed)
        "ball_speed_x2": BallSpeedX2Mod,
        "ball_speed_x3": BallSpeedX3Mod,
        "ball_speed_x4": BallSpeedX4Mod,
        "ball_speed_x5": BallSpeedX5Mod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "pong", "sprites")

    def __init__(self,
                 env,
                 mods_config: list = [],
                 allow_conflicts: bool = False
                 ):

        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY  # for pong this is the only specific part, but other games might need to do execute some other logic in the constructor.
        )

        # render() cannot be patched by a plugin (it lives on both env and
        # renderer), so the level is read here and applied in render() below.
        self._render_noise_level = 0.0
        for mod_key in mods_config:
            plugin_class = self.REGISTRY.get(mod_key)
            if isinstance(plugin_class, type) and issubclass(plugin_class, _RenderNoiseMod):
                self._render_noise_level = max(self._render_noise_level, plugin_class._NOISE_LEVEL)

    @partial(jax.jit, static_argnames=['self'])
    def render(self, state):
        raster = self._env.render(state)  # clean frame from the base renderer
        if self._render_noise_level > 0.0:
            raster = apply_render_noise(raster, state.key, self._render_noise_level)
        return raster
