import os
import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.seaquest.seaquest_mod_plugins import (
    DisableEnemiesMod, NoDiversMod, EnemyMinesMod, FireBallsMod, UnlimitedOxygenMod,
    GravityMod, RandomColorEnemiesMod,
    ChangeSubColorMod, ChangeEnemyColorMod, ChangeWaterColorMod, ChangeScoreColorMod, GrayscaleThemeMod,
    FasterEnemiesMod, SlowerEnemiesMod, FasterSubMod, SlowerSubMod, FasterOxygenDrainMod,
    DenseSpawnMod, RandomizeSpawnMod, DiverSpawnRateMod,
)

class SeaquestEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Seaquest.
    It simply inherits all logic from JaxAtariModController and defines the REGISTRY.
    """

    REGISTRY = {
        # ------------------------------------------------------------------ #
        # Dynamic: change how the game plays (entities / oxygen / control)
        # ------------------------------------------------------------------ #
        "disable_enemies": DisableEnemiesMod,     # zero out sharks/subs/missiles
        "no_divers": NoDiversMod,                 # remove divers entirely
        "unlimited_oxygen": UnlimitedOxygenMod,   # oxygen never drains (easier)
        "gravity": GravityMod,                    # sub is pulled downward each frame
        # Speed axes (post-step; enemy speed has no constant to override)
        "faster_enemies": FasterEnemiesMod,       # sharks/subs move 2x faster
        "slower_enemies": SlowerEnemiesMod,       # sharks/subs move at half speed
        "faster_sub": FasterSubMod,               # player sub moves 2x faster
        "slower_sub": SlowerSubMod,               # player sub moves at half speed
        # Oxygen
        "faster_oxygen_drain": FasterOxygenDrainMod,  # oxygen depletes 2x faster
        # Spawns
        "dense_spawn": DenseSpawnMod,             # always 3-enemy formations
        "randomize_spawn": RandomizeSpawnMod,     # formation randomized per wave, not by difficulty
        "diver_spawn_rate": DiverSpawnRateMod,    # divers spawn in all 4 lanes (more rescues)

        # ------------------------------------------------------------------ #
        # Visual: sprite / colour swaps (no gameplay change on their own)
        # ------------------------------------------------------------------ #
        # Single-element recolours + grayscale theme (the `vis4` set)
        "change_sub_color": ChangeSubColorMod,        # player sub + torpedo
        "change_enemy_color": ChangeEnemyColorMod,    # sharks (via SHARK_DIFFICULTY_COLORS) + enemy subs
        "change_water_color": ChangeWaterColorMod,    # background water tones
        "change_score_color": ChangeScoreColorMod,    # score digits + life indicator
        "grayscale_theme": GrayscaleThemeMod,         # whole scene desaturated

        # Sprite swaps / other visual
        "mines": EnemyMinesMod,                       # sharks + subs -> mine sprite
        "random_color_enemies": RandomColorEnemiesMod,# per-shark random colours (logic in render() below)
        "fireballs": FireBallsMod,                    # NOTE: stub (empty class) - not implemented yet

        # ------------------------------------------------------------------ #
        # Reward: reshape what the agent is rewarded for (none yet)
        # ------------------------------------------------------------------ #
        # --- planned (see config/sequence/seaquest_rew4.yaml) ---
        # life_loss_penalty / diver_scoring_only / surface_load_bonus
        # flatten_enemy_values / penalize_diver_shoot

        # ------------------------------------------------------------------ #
        # Not yet implemented (kept as reference from earlier drafts)
        # ------------------------------------------------------------------ #
        # "peaceful_enemies": PeacefulEnemiesMod,
        # "lethal_divers": LethalDiversMod,
        # "polluted_water": PollutedWaterMod,
        # "fireball": ReplaceTorpedoWithFireBallMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "seaquest", "sprites")

    def __init__(self,
                 env,
                 mods_config: list = [],
                 allow_conflicts: bool = False
                 ):
        self._has_random_color = "random_color_enemies" in mods_config
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY
        )

    @partial(jax.jit, static_argnames=['self'])
    def render(self, state):
        if self._has_random_color:
            renderer = self._env.renderer
            jr = renderer.jr
            raster = renderer.BACKGROUND
            step_counter = state.step_counter

            # Player
            player_anim_idx = (step_counter % 12) // 4
            raster = jr.render_at(
                raster, state.player_x, state.player_y,
                renderer.SHAPE_MASKS['player_sub'][player_anim_idx],
                flip_horizontal=state.player_direction == self._env.consts.FACE_LEFT,
                flip_offset=renderer.FLIP_OFFSETS['player_sub']
            )
            
            torp = state.player_missile_position
            raster = jax.lax.cond(
                torp[2] != 0,
                lambda r: jr.render_at_clipped(r, torp[0], torp[1], renderer.SHAPE_MASKS['player_torp'],
                                            flip_horizontal=torp[2] == self._env.consts.FACE_LEFT),
                lambda r: r,
                raster
            )

            raster = renderer._draw_divers(raster, state)
            
            # Enemy Torpedoes
            raster = jax.lax.fori_loop(
                0, state.enemy_missile_positions.shape[0],
                lambda i, r: renderer.render_object_sequentially(r, state.enemy_missile_positions[i], renderer.SHAPE_MASKS['enemy_torp'][None, ...], jnp.zeros(2, dtype=jnp.int32), 0),
                raster
            )

            # Sharks - Custom Color Logic
            shark_anim_idx = jax.lax.select((step_counter % 24) < 16, 0, 1)
            base_shark_masks = renderer.SHAPE_MASKS['shark_base']
            
            def draw_shark(i, r):
                hash_val = (i * 17 + (step_counter // 200)) % 8
                shark_color_id = renderer.SHARK_COLOR_MAP[hash_val]
                recolored_shark_masks = jnp.where(base_shark_masks != jr.TRANSPARENT_ID, shark_color_id, base_shark_masks)
                return renderer.render_object_sequentially(r, state.shark_positions[i], recolored_shark_masks, renderer.FLIP_OFFSETS['shark_base'], shark_anim_idx)
                
            raster = jax.lax.fori_loop(0, state.shark_positions.shape[0], draw_shark, raster)
            
            # UI Elements
            score_digits = jr.int_to_digits(state.score, max_digits=6)
            raster = jr.render_label(raster, 58, 18, score_digits, renderer.SHAPE_MASKS['digits'], spacing=8, max_digits=6)
            
            raster = jr.render_indicator(raster, 14, 28, state.lives, renderer.SHAPE_MASKS['life_indicator'], spacing=10, max_value=3)
            
            # Collected divers blink when there are 6 of them
            visible_divers = jax.lax.select(
                jnp.logical_and(state.divers_collected == 6, (state.step_counter % 8) >= 4),
                0,
                state.divers_collected
            )
            raster = jr.render_indicator(raster, 49, 178, visible_divers, renderer.SHAPE_MASKS['diver_indicator'], spacing=10, max_value=6)

            raster = jr.render_bar(raster, 49, 170, state.oxygen, 64, 63, 5, renderer.OXYGEN_COLOR_ID, renderer.OXYGEN_BAR_BG_COLOR_ID)

            raster = jr.draw_rects(
                raster,
                positions=jnp.array([[0, 0]]),
                sizes=jnp.array([[8, renderer.config.game_dimensions[0]]]),
                color_id=renderer.BACKGROUND[0, 0]
            )
            
            return jr.render_from_palette(raster, renderer.PALETTE)

        return self._env.render(state)
