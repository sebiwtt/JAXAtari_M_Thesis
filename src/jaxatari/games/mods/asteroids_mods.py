import os
from functools import partial
import jax
import jax.numpy as jnp
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.asteroids.asteroids_mod_plugins import DontShootMod, MatrixMod, InstantTurnMod, ChangeShipColorMod, ChangeAsteroidColorMod, ChangeBackgroundColorMod, ChangeScoreColorMod, GrayscaleThemeMod, NoFlickerMod, FasterAsteroidsMod, SlowerAsteroidsMod, AsteroidSpeedX2Mod, AsteroidSpeedX3Mod, AsteroidSpeedX4Mod, AsteroidSpeedX5Mod, FasterShipMod, SlowerShipMod, RandomizeAsteroidSpawnMod, MoreAsteroidsMod, ShipInertiaMod, LifeLossPenaltyMod, FlattenAsteroidValuesMod, LargeAsteroidOnlyMod, SmallAsteroidOnlyMod, WaveClearBonusMod, EveryKKillsMod, SurvivalRewardMod

class AsteroidsEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Asteroids.
    """

    REGISTRY = {
        # Visual
        "change_ship_color": ChangeShipColorMod,
        "change_asteroid_color": ChangeAsteroidColorMod,
        "change_background_color": ChangeBackgroundColorMod,
        "change_score_color": ChangeScoreColorMod,
        "grayscale_theme": GrayscaleThemeMod,
        "no_flicker": NoFlickerMod,
        "matrix_theme": MatrixMod,

        # Dynamics
        "faster_asteroids": FasterAsteroidsMod,
        "slower_asteroids": SlowerAsteroidsMod,

        # Magnitude sequence (asteroid_speed_xN): same mod, incrementally faster
        "asteroid_speed_x2": AsteroidSpeedX2Mod,
        "asteroid_speed_x3": AsteroidSpeedX3Mod,
        "asteroid_speed_x4": AsteroidSpeedX4Mod,
        "asteroid_speed_x5": AsteroidSpeedX5Mod,
        "faster_ship": FasterShipMod,
        "slower_ship": SlowerShipMod,
        "randomize_asteroid_spawn": RandomizeAsteroidSpawnMod,
        "more_asteroids": MoreAsteroidsMod,
        "ship_inertia": ShipInertiaMod,
        "dont_shoot": DontShootMod,
        "instant_turn": InstantTurnMod,

        # Reward
        "life_loss_penalty": LifeLossPenaltyMod,
        "flatten_asteroid_values": FlattenAsteroidValuesMod,
        "large_asteroid_only": LargeAsteroidOnlyMod,
        "small_asteroid_only": SmallAsteroidOnlyMod,
        "wave_clear_bonus": WaveClearBonusMod,
        "every_k_kills": EveryKKillsMod,
        "survival_reward": SurvivalRewardMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "asteroids", "sprites")

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

        # render() cannot be patched by a plugin (it lives on both env and
        # renderer), so no_flicker is detected here and applied in render() below.
        self._no_flicker = any(
            isinstance(self.REGISTRY.get(mod_key), type) and issubclass(self.REGISTRY[mod_key], NoFlickerMod)
            for mod_key in mods_config
        )

    @partial(jax.jit, static_argnames=['self'])
    def render(self, state):
        if not self._no_flicker:
            return self._env.render(state)

        # The base renderer gates two sprite groups on step_counter parity:
        # {ship, missile1, missile2} draw only on even frames, {asteroids, death
        # animations} only on odd frames (Atari hardware-flicker emulation).
        # Render once at each forced parity (all other state fields untouched)
        # and merge both groups' foreground pixels onto the true background.
        even_state = state.replace(step_counter=state.step_counter - (state.step_counter % 2))
        odd_state = state.replace(step_counter=even_state.step_counter + 1)

        frame_even = self._env.render(even_state)
        frame_odd = self._env.render(odd_state)

        renderer = self._env.renderer
        bg = renderer.jr.render_from_palette(renderer.BACKGROUND, renderer.PALETTE)

        is_fg_even = jnp.any(frame_even != bg, axis=-1, keepdims=True)
        is_fg_odd = jnp.any(frame_odd != bg, axis=-1, keepdims=True)

        return jnp.where(is_fg_even, frame_even, jnp.where(is_fg_odd, frame_odd, bg))
