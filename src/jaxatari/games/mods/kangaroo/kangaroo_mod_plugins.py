import jax
import jax.numpy as jnp
import chex
from functools import partial
from typing import Tuple

from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.games.jax_kangaroo import JaxKangaroo, KangarooState, PlayerState, LevelState, KangarooConstants
from jaxatari.games.kangaroo_levels import LevelConstants, Kangaroo_Level_1, Kangaroo_Level_2, Kangaroo_Level_3
from jaxatari.environment import JAXAtariAction as Action

# --- 1. Internal Mods (Group 1) ---
class NoBellMod(JaxAtariInternalModPlugin):
    """
    Internal mod to disable the Bell.
    Patches '_bell_step'.
    """
    
    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _bell_step(self, state: KangarooState):
        """
        No-op override for _bell_step.
        Returns 0 for the timer and False for the respawn flag, 
        effectively disabling the bell mechanics.
        """
        return jnp.zeros_like(state.level.bell_timer), jnp.array(False)


    @partial(jax.jit, static_argnums=(0,))
    def _draw_bell(self, raster: jnp.ndarray, state: KangarooState):
        """
        Overrides the KangarooRenderer._draw_bell method.
        Draws a static sprite (no animation) shifted 4 pixels up.
        """

        return raster


class NoFruitMod(JaxAtariInternalModPlugin):
    """
    Internal mod to remove Fruits.
    Patches '_fruits_step'.
    """   

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _fruits_step(self, state: KangarooState):
        """
        Override for _fruits_step to remove fruits.
        """
        # We must still call _bell_step because the environment logic 
        # normally chains these together.
        bell_timer, _ = self._env._bell_step(state)

        return (
            jnp.zeros((), dtype=jnp.int32),                             # Score addition
            jnp.zeros_like(state.level.fruit_actives, dtype=jnp.bool_), # Set actives to False (Hides them visually)
            state.level.fruit_stages,                                   # Keep stages (irrelevant since inactive)
            bell_timer                                                  # Pass through the bell timer
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _draw_single_fruit(self, i, raster, state: KangarooState):
        """
        Overrides the KangarooRenderer._draw_fruits method.
        Does not draw any fruits.
        """
        return raster


class NoMonkeyMod(JaxAtariInternalModPlugin):
    """
    Internal mod to disable monkeys.
    This patches the environment's '_monkey_controller' method.
    """
    
    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _monkey_controller(self, state: KangarooState, punching: chex.Array):
        """
        No-op override for _monkey_controller.
        """
        score_addition = jnp.zeros((), dtype=jnp.int32)
        
        return (
            state.level.monkey_states,       
            state.level.monkey_positions,    
            state.level.monkey_throw_timers, 
            score_addition,                  
            state.level.coco_positions,      
            state.level.coco_states,         
            jnp.array(False),                
        )

class NoFallingCoconutMod(JaxAtariInternalModPlugin):
    """
    Internal mod to disable the single falling coconut.
    This patches the environment's '_falling_coconut_controller' method.
    """
    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _falling_coconut_controller(self, 
                                    state: KangarooState, 
                                    punching: chex.Array
                                    ):
        """
        No-op override for _falling_coconut_controller.
        """
        return (
            state.level.falling_coco_position, 
            state.level.falling_coco_dropping, 
            state.level.falling_coco_counter,  
            state.level.falling_coco_skip_update,
            jnp.zeros((), dtype=jnp.int32),     
        )


class NoThrownCoconutMod(JaxAtariInternalModPlugin):
    """
    Internal mod to disable thrown coconuts.
    This patches the environment's '_update_coco_state' method to prevent 
    coconuts from transitioning to active states (1 or 2).
    """
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_coco_state(
        self,
        old_m_state: chex.Array,
        new_m_state: chex.Array,
        old_m_timer: chex.Array,
        new_m_timer: chex.Array,
        c_state: chex.Array,
        c_pos_x: chex.Array,
    ) -> chex.Array:
        """
        Override to prevent coconut state updates.
        Returns 0 (non-existent) regardless of monkey state.
        """
        return jnp.array(0, dtype=jnp.int32)


class AlwaysHighCoconutMod(JaxAtariInternalModPlugin):
    """
    Internal mod to force coconuts to always spawn at the 'head' (high) position.
    """
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_coco_positions(
        self,
        new_c_state: chex.Array,
        old_c_state: chex.Array,
        stepc: chex.Array,
        old_c_pos: chex.Array,
        new_m_pos: chex.Array,
        spawn_position: chex.Array,
    ) -> chex.Array:
        
        return jnp.where(
            new_c_state == 2,
            # --- Flight Logic (Unchanged) ---
            jnp.where(
                stepc % 2 == 0,
                jnp.array([old_c_pos[0] - 2, old_c_pos[1]]),
                old_c_pos,
            ),
            # --- Spawn Logic (Modified) ---
            jnp.where(
                (new_c_state == 1) & (old_c_state == 0),
                jnp.array(
                    [
                        new_m_pos[0] - 6,
                        new_m_pos[1] - 5 
                    ]
                ),
                old_c_pos,
            ),
        )

class FirstLevelOnlyMod(JaxAtariInternalModPlugin):
    """
    Internal mod to force the game to always stay on level 1.
    This patches the environment's '_level_transition_controller' method.
    """
    conflicts_with = ["second_level_only", "third_level_only"]

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _next_level(self, state: KangarooState):
        RESET_AFTER_TICKS = 256

        counter = state.levelup_timer
        counter_start = state.level_finished & (counter == 0)
        counter = jnp.where((counter > 0) | counter_start, counter + 1, counter)
        reset_timer_done = counter == RESET_AFTER_TICKS
        counter = jnp.where(counter > RESET_AFTER_TICKS, 0, counter)

        reset_coords = jnp.where(reset_timer_done, jnp.array(True), jnp.array(False))
        levelup = jnp.where(reset_timer_done, jnp.array(True), jnp.array(False))

        current_level = jnp.where(levelup, 1, state.current_level)

        return current_level, counter, reset_coords, levelup


class SecondLevelOnlyMod(JaxAtariInternalModPlugin):
    """
    Internal mod to force the game to always stay on level 2.
    This patches the environment's '_level_transition_controller' method.
    """
    conflicts_with = ["first_level_only", "third_level_only", "center_ladders", "invert_ladders", "flame_trap"]

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _next_level(self, state: KangarooState):
        RESET_AFTER_TICKS = 256

        counter = state.levelup_timer
        counter_start = state.level_finished & (counter == 0)
        counter = jnp.where((counter > 0) | counter_start, counter + 1, counter)
        reset_timer_done = counter == RESET_AFTER_TICKS
        counter = jnp.where(counter > RESET_AFTER_TICKS, 0, counter)

        reset_coords = jnp.where(reset_timer_done, jnp.array(True), jnp.array(False))
        levelup = jnp.where(reset_timer_done, jnp.array(True), jnp.array(False))

        current_level = jnp.where(levelup, 2, state.current_level)

        return current_level, counter, reset_coords, levelup

    @partial(jax.jit, static_argnums=(0,))
    def reset_level(self, next_level=1) -> KangarooState:
        next_level = 2
        level_constants = Kangaroo_Level_2
        main_consts = KangarooConstants()
        new_state = KangarooState(
            player=PlayerState(
                x=jnp.array(main_consts.PLAYER_START_X),
                y=jnp.array(main_consts.PLAYER_START_Y),
                vel_x=jnp.array(0),
                is_crouching=jnp.array(False),
                is_jumping=jnp.array(False),
                is_climbing=jnp.array(False),
                jump_counter=jnp.array(0),
                orientation=jnp.array(1),
                jump_base_y=jnp.array(main_consts.PLAYER_START_Y),
                landing_base_y=jnp.array(main_consts.PLAYER_START_Y),
                height=jnp.array(main_consts.PLAYER_HEIGHT),
                jump_orientation=jnp.array(0),
                climb_base_y=jnp.array(main_consts.PLAYER_START_Y),
                climb_counter=jnp.array(0),
                punch_left=jnp.array(False),
                punch_right=jnp.array(False),
                cooldown_counter=jnp.array(0),
                chrash_timer=jnp.array(0),
                is_crashing=jnp.array(False),
                last_stood_on_platform_y=jnp.array(1000),
                walk_animation=jnp.array(0),
                punch_counter=jnp.array(0),
                needs_release=jnp.array(False),
            ),
            level=LevelState(
                bell_position=level_constants.bell_position,
                bell_timer=jnp.array(0),
                fruit_positions=level_constants.fruit_positions,
                fruit_actives=jnp.ones(3, dtype=jnp.bool_),
                fruit_stages=jnp.zeros(3, dtype=jnp.int32),
                ladder_positions=level_constants.ladder_positions,
                ladder_sizes=level_constants.ladder_sizes,
                platform_positions=level_constants.platform_positions,
                platform_sizes=level_constants.platform_sizes,
                child_position=level_constants.child_position,
                child_timer=jnp.array(0),
                child_velocity=jnp.array(1),
                timer=jnp.array(2000),  # to be modified
                falling_coco_position=jnp.array([13, -1]),
                falling_coco_dropping=jnp.array(False),
                falling_coco_counter=jnp.array(0),
                falling_coco_skip_update=jnp.array(False),
                step_counter=jnp.array(0),
                monkey_states=jnp.zeros(4, dtype=jnp.int32),
                monkey_positions=jnp.array([[152, 5], [152, 5], [152, 5], [152, 5]]),
                monkey_throw_timers=jnp.zeros(4, dtype=jnp.int32),
                spawn_protection=jnp.array(True),
                coco_positions=jnp.array(
                    [[-10, -10], [-10, -10], [-10, -10], [-10, -10]]
                ),
                coco_states=jnp.zeros(4, dtype=jnp.int32),
                spawn_position=jnp.array(False),
                bell_animation=jnp.array(0),
            ),
            score=jnp.array(0),
            current_level=next_level,
            level_finished=jnp.array(False),
            levelup_timer=jnp.array(0),
            reset_coords=jnp.array(False),
            levelup=jnp.array(False),
            lives=jnp.array(3),
        )
        return new_state

class ThirdLevelOnlyMod(JaxAtariInternalModPlugin):
    """
    Internal mod to force the game to always stay on level 3.
    This patches the environment's '_level_transition_controller' method.
    """
    conflicts_with = ["first_level_only", "second_level_only", "center_ladders", "invert_ladders", "flame_trap"]

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _next_level(self, state: KangarooState):
        RESET_AFTER_TICKS = 256

        counter = state.levelup_timer
        counter_start = state.level_finished & (counter == 0)
        counter = jnp.where((counter > 0) | counter_start, counter + 1, counter)
        reset_timer_done = counter == RESET_AFTER_TICKS
        counter = jnp.where(counter > RESET_AFTER_TICKS, 0, counter)

        reset_coords = jnp.where(reset_timer_done, jnp.array(True), jnp.array(False))
        levelup = jnp.where(reset_timer_done, jnp.array(True), jnp.array(False))

        current_level = jnp.where(levelup, 3, state.current_level)

        return current_level, counter, reset_coords, levelup
    
    @partial(jax.jit, static_argnums=(0,))
    def reset_level(self, next_level=1) -> KangarooState:
        next_level = 3
        level_constants = Kangaroo_Level_3
        main_consts = KangarooConstants()
        new_state = KangarooState(
            player=PlayerState(
                x=jnp.array(main_consts.PLAYER_START_X),
                y=jnp.array(main_consts.PLAYER_START_Y),
                vel_x=jnp.array(0),
                is_crouching=jnp.array(False),
                is_jumping=jnp.array(False),
                is_climbing=jnp.array(False),
                jump_counter=jnp.array(0),
                orientation=jnp.array(1),
                jump_base_y=jnp.array(main_consts.PLAYER_START_Y),
                landing_base_y=jnp.array(main_consts.PLAYER_START_Y),
                height=jnp.array(main_consts.PLAYER_HEIGHT),
                jump_orientation=jnp.array(0),
                climb_base_y=jnp.array(main_consts.PLAYER_START_Y),
                climb_counter=jnp.array(0),
                punch_left=jnp.array(False),
                punch_right=jnp.array(False),
                cooldown_counter=jnp.array(0),
                chrash_timer=jnp.array(0),
                is_crashing=jnp.array(False),
                last_stood_on_platform_y=jnp.array(1000),
                walk_animation=jnp.array(0),
                punch_counter=jnp.array(0),
                needs_release=jnp.array(False),
            ),
            level=LevelState(
                bell_position=level_constants.bell_position,
                bell_timer=jnp.array(0),
                fruit_positions=level_constants.fruit_positions,
                fruit_actives=jnp.ones(3, dtype=jnp.bool_),
                fruit_stages=jnp.zeros(3, dtype=jnp.int32),
                ladder_positions=level_constants.ladder_positions,
                ladder_sizes=level_constants.ladder_sizes,
                platform_positions=level_constants.platform_positions,
                platform_sizes=level_constants.platform_sizes,
                child_position=level_constants.child_position,
                child_timer=jnp.array(0),
                child_velocity=jnp.array(1),
                timer=jnp.array(2000),  # to be modified
                falling_coco_position=jnp.array([13, -1]),
                falling_coco_dropping=jnp.array(False),
                falling_coco_counter=jnp.array(0),
                falling_coco_skip_update=jnp.array(False),
                step_counter=jnp.array(0),
                monkey_states=jnp.zeros(4, dtype=jnp.int32),
                monkey_positions=jnp.array([[152, 5], [152, 5], [152, 5], [152, 5]]),
                monkey_throw_timers=jnp.zeros(4, dtype=jnp.int32),
                spawn_protection=jnp.array(True),
                coco_positions=jnp.array(
                    [[-10, -10], [-10, -10], [-10, -10], [-10, -10]]
                ),
                coco_states=jnp.zeros(4, dtype=jnp.int32),
                spawn_position=jnp.array(False),
                bell_animation=jnp.array(0),
            ),
            score=jnp.array(0),
            current_level=next_level,
            level_finished=jnp.array(False),
            levelup_timer=jnp.array(0),
            reset_coords=jnp.array(False),
            levelup=jnp.array(False),
            lives=jnp.array(3),
        )
        return new_state

# --- 2. Post-Step Mod (Group 2) ---

class PinChildMod(JaxAtariPostStepModPlugin):
    """
    Post-step mod to pin the child kangaroo in place.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: KangarooState, new_state: KangarooState):
        """
        Called *after* the main step. Overwrites the child's
        position with its static starting position.
        """
        # Get the level constants to find the child's start position
        level_constants = self._env._get_level_constants(new_state.current_level)
        
        # Pin the child's position and velocity
        pinned_level_state = new_state.level.replace(
            child_position=level_constants.child_position, #
            child_velocity=jnp.array(0) # Also stop its velocity
        )
        
        return new_state.replace(level=pinned_level_state)


class RenderDebugInfo(JaxAtariInternalModPlugin):
    """
    Patches the render hook to draw the player's X,Y coords
    over the UI.
    """

    @partial(jax.jit, static_argnums=(0,))
    def _render_hook_post_ui(self, raster, state: KangarooState):
        """
        This function patches the hook in KangarooRenderer.
        'self_env' is the JaxKangaroo instance, so we use
        'self_env.renderer.jr' to access the utils with correct config.
        """
        jr = self._env.renderer.jr
        masks = self._env.renderer.SHAPE_MASKS["score_digits"]
        
        # Draw Player X Position to frame
        x_digits = jr.int_to_digits(state.player.x, max_digits=3)
        raster = jr.render_label(raster, 10, 10, x_digits, masks, 8, 3)
        
        return raster


class ReplaceChildWithMonkeyMod(JaxAtariInternalModPlugin):
    """
    Replaces the child sprite with a monkey sprite.
    """

    asset_overrides = {
        "child": "ape"
    }

    @partial(jax.jit, static_argnums=(0,))
    def _render_hook_post_ui(self, raster, state: KangarooState):
        """
        This function patches the hook in KangarooRenderer.
        'self_env' is the JaxKangaroo instance, so we use
        'self_env.renderer.jr' to access the utils with correct config.
        """
        
        # do nothing
        
        return raster

# Multiple Plugins to change bell to a fire (bundled into *modpack* in the kangaroo_mods.py file)

class ReplaceFruitWithCoin(JaxAtariInternalModPlugin):
    asset_overrides = {
        "fruit": {
            'name': 'fruit',
            'type': 'group',
            'files': ['coin.npy', 'coin2.npy', 'coin3.npy', 'coin4.npy']
        }
    }
    
class ReplaceFruitWithDiamond(JaxAtariInternalModPlugin):
    asset_overrides = {
        "fruit": {
            'name': 'fruit',
            'type': 'group',
            'files': ['diamond.npy', 'ruby.npy', 'emerald.npy', 'amethyst.npy']
        }
    }

class ReplaceCoconutWithFireball(JaxAtariInternalModPlugin):
    asset_overrides = {
        "coconut": {
            'name': 'coconut',
            'type': 'single',
            'file': 'fireball.npy'
        }
    }
    constants_overrides = {
        "THROWN_COCONUT_WIDTH": 16,
        "THROWN_COCONUT_HEIGHT": 12,
    }


class ReplaceCoconutWithWasp(JaxAtariInternalModPlugin):
    asset_overrides = {
        "coconut": {
            'name': 'coconut',
            'type': 'single',
            'file': 'wasp.npy'
        }
    }
    constants_overrides = {
        "THROWN_COCONUT_WIDTH": 16,
        "THROWN_COCONUT_HEIGHT": 12,
    }

class ReplaceCoconutWithHoneyBee(JaxAtariInternalModPlugin):
    asset_overrides = {
        "coconut": {
            'name': 'coconut',
            'type': 'single',
            'file': 'honey_bee.npy'
        }
    }
    constants_overrides = {
        "THROWN_COCONUT_WIDTH": 16,
        "THROWN_COCONUT_HEIGHT": 12,
    }

class ReplaceMonkeyWithTankMod(JaxAtariInternalModPlugin):
    asset_overrides = {
        "ape": {
            'name': 'ape',
            'type': 'group',
            'files': ['tank_15x8.npy']
        }
    }

class ReplaceMonkeyWithChickenMod(JaxAtariInternalModPlugin):
    asset_overrides = {
        "ape": {
            'name': 'ape',
            'type': 'group',
            'files': ['chicken.npy']
        }
    }

class ReplaceMonkeyWithDangerSignMod(JaxAtariInternalModPlugin):
    asset_overrides = {
        "ape": {
            'name': 'ape',
            'type': 'group',
            'files': ['danger_sign.npy']
        }
    }

class ReplaceMonkeyWithDragonMod(JaxAtariInternalModPlugin):
    asset_overrides = {
        "ape": {
            'name': 'ape',
            'type': 'group',
            'files': ['dragon.npy']
        }
    }

class ReplaceMonkeyWithPolarbearMod(JaxAtariInternalModPlugin):
    asset_overrides = {
        "ape": {
            'name': 'ape',
            'type': 'group',
            'files': ['polarbear.npy']
        }
    }

class ReplaceMonkeyWithSnakeMod(JaxAtariInternalModPlugin):
    asset_overrides = {
        "ape": {
            'name': 'ape',
            'type': 'group',
            'files': ['snake.npy']
        }
    }

# --- MOD A: Replace Bell Sprite + Patch Animation ---
class ReplaceBellWithCactusMod(JaxAtariInternalModPlugin):
    """
    Replaces the 'bell' asset group with a new 'cactus' asset group.
    
    Expects 'cactus.npy' to exist.
    """
    
    # 1. Swap the assets (using Method 2: manually overriding a key in the asset_overrides dict)
    asset_overrides = {
        "bell": {
            'name': 'bell', 
            'type': 'group',
            'files': ['cactus_tall.npy']
        }
    }
    @partial(jax.jit, static_argnums=(0,))
    def _draw_bell(self, raster: jnp.ndarray, state: KangarooState):
        """
        Overrides the KangarooRenderer._draw_bell method.
        Draws a static sprite (no animation) shifted 4 pixels up.
        """
        jr = self._env.renderer.jr
        
        # CHANGED: Removed flicker logic. Hardcoded to index 0 for a static image.
        flame_idx = 0 
        
        # We use "bell" as the key because our override mapped to it
        flame_mask = self._env.renderer.SHAPE_MASKS["bell"][flame_idx]
        flame_offset = self._env.renderer.FLIP_OFFSETS["bell"]
        
        # Keep original logic for *when* to draw
        should_draw_flame = (state.level.bell_position[0] != -1) & ~jnp.any(state.level.fruit_stages == 3)
        
        # CHANGED: Adjusted Y position logic inside render_at
        raster = jax.lax.cond(should_draw_flame,
            lambda r: jr.render_at(
                r, 
                state.level.bell_position[0].astype(int), 
                # CHANGED: Subtract 4 from Y to move it up
                state.level.bell_position[1].astype(int) - 8, 
                flame_mask, 
                flip_horizontal=jnp.array(False),
                flip_offset=flame_offset
            ),
            lambda r: r, 
            raster
        )
        return raster



class ReplaceBellWithFlameMod(JaxAtariInternalModPlugin):
    """
    Replaces the 'bell' asset group with a new 'flame' asset group
    and patches the _draw_bell render hook to make it animate constantly.
    
    Expects 'flame_0.npy' and 'flame_1.npy' to exist.
    """
    
    # 1. Swap the assets (using Method 2: manually overriding a key in the asset_overrides dict)
    asset_overrides = {
        "bell": {
            'name': 'bell', 
            'type': 'group',
            'files': ['flame_0.npy', 'flame_1.npy']
        }
    }

    # 2. Patch the new animation hook
    @partial(jax.jit, static_argnums=(0,))
    def _draw_bell(self, raster: jnp.ndarray, state: KangarooState):
        """
        Overrides the KangarooRenderer._draw_bell method.
        This logic ignores the bell_animation timer and uses the
        step_counter for a constant flicker.
        """
        jr = self._env.renderer.jr
        
        # Animate using the global step counter for a constant flicker
        is_flicker_frame = (state.level.step_counter % 16) < 8
        flame_idx = jax.lax.select(is_flicker_frame, 1, 0) # 1 for frame 1, 0 for frame 0
        
        # We use "bell" as the key because our override mapped to it
        flame_mask = self._env.renderer.SHAPE_MASKS["bell"][flame_idx]
        flame_offset = self._env.renderer.FLIP_OFFSETS["bell"]
        
        # Keep original logic for *when* to draw
        should_draw_flame = (state.level.bell_position[0] != -1) & ~jnp.any(state.level.fruit_stages == 3)
        
        raster = jax.lax.cond(should_draw_flame,
            lambda r: jr.render_at(
                r, 
                state.level.bell_position[0].astype(int), 
                state.level.bell_position[1].astype(int), 
                flame_mask, 
                flip_horizontal=jnp.array(False), # No flipping
                flip_offset=flame_offset
            ),
            lambda r: r, 
            raster
        )
        return raster

class ReplaceBellWithDangerSignMod(JaxAtariInternalModPlugin):
    """
    Replaces the 'bell' asset group with a new 'danger_sign' asset group.
    
    Expects 'danger_sign.npy' to exist.
    """
    
    # 1. Swap the assets (using Method 2: manually overriding a key in the asset_overrides dict)
    asset_overrides = {
        "bell": {
            'name': 'bell', 
            'type': 'group',
            'files': ['danger_sign.npy']
        }
    }
    @partial(jax.jit, static_argnums=(0,))
    def _draw_bell(self, raster: jnp.ndarray, state: KangarooState):
        """
        Overrides the KangarooRenderer._draw_bell method.
        Draws a static sprite (no animation) shifted 4 pixels up.
        """
        jr = self._env.renderer.jr
        
        # CHANGED: Removed flicker logic. Hardcoded to index 0 for a static image.
        flame_idx = 0 
        
        # We use "bell" as the key because our override mapped to it
        flame_mask = self._env.renderer.SHAPE_MASKS["bell"][flame_idx]
        flame_offset = self._env.renderer.FLIP_OFFSETS["bell"]
        
        # Keep original logic for *when* to draw
        should_draw_flame = (state.level.bell_position[0] != -1) & ~jnp.any(state.level.fruit_stages == 3)
        
        # CHANGED: Adjusted Y position logic inside render_at
        raster = jax.lax.cond(should_draw_flame,
            lambda r: jr.render_at(
                r, 
                state.level.bell_position[0].astype(int), 
                state.level.bell_position[1].astype(int), 
                flame_mask, 
                flip_horizontal=jnp.array(False),
                flip_offset=flame_offset
            ),
            lambda r: r, 
            raster
        )
        return raster

class ReplaceLadderWithChainMod(JaxAtariInternalModPlugin):
    """
    Replaces the ladder sprites with grey chain sprites.
    Chains are drawn as a 4-pixel wide alternating pattern (Inner vs Outer pixels).
    """
    NEW_CHAIN_COLOR = (128, 128, 128) # Grey

    # Create the procedural asset (1x1 pixel RGBA sprite)
    custom_color_rgba = jnp.array([[[
        NEW_CHAIN_COLOR[0],  # R
        NEW_CHAIN_COLOR[1],  # G
        NEW_CHAIN_COLOR[2],  # B
        255  # Alpha (fully opaque)
    ]]], dtype=jnp.uint8)

    # Add via asset_overrides
    asset_overrides = {
        'custom_chain_color': {
            'name': 'custom_chain_color',
            'type': 'procedural',
            'data': custom_color_rgba
        }
    }
    
    @partial(jax.jit, static_argnums=(0,))
    def _draw_ladders(self, raster: jnp.ndarray, state):
        """
        Draws chains: A 4px wide pattern alternating between inner connectors and outer loops.
        """
        # 1. Access Data from the Environment State
        positions = state.level.ladder_positions
        sizes = state.level.ladder_sizes
        
        # Access the renderer's calculated color ID for the chain
        chain_color = self._env.renderer.COLOR_TO_ID.get(self.NEW_CHAIN_COLOR, 0)
            
        # 2. Get dimensions
        h, w = raster.shape
        
        # 3. Create the meshgrid for vectorization
        yy, xx = jnp.mgrid[:h, :w]

        # 4. Define Visual Constants
        chain_visual_width = 4
        half_width = 2
        segment_height = 2  # Height of one link segment

        def _create_single_chain_mask(pos, size):
            """Generates a boolean mask for a single chain."""
            # Only draw if the ladder exists (x != -1)
            is_active = pos[0] != -1
            
            # Geometry
            x, y = pos
            hitbox_width, height = size

            y -= 8
            height += 12
            
            # Calculate Center
            center_x = x + (hitbox_width // 2)
            
            # Start drawing x (shift left by 2 to center the 4px chain)
            draw_x_start = center_x - half_width
            
            # --- Bounding Box Logic ---
            dx = xx - draw_x_start
            dy = yy - y
            
            # Check if pixel is within the 4px wide x height bounding box
            in_box = (dx >= 0) & (dx < chain_visual_width) & \
                     (dy >= 0) & (dy < height)
            
            # --- Chain Pattern Logic ---
            segment_idx = dy // segment_height
            
            # Logic:
            # Inner pixels (dx=1, dx=2) represent the vertical connector
            # Outer pixels (dx=0, dx=3) represent the sides of the link loop
            is_inner_pixel = (dx == 1) | (dx == 2)
            is_outer_pixel = (dx == 0) | (dx == 3)
            
            # Alternating Pattern:
            # Even segments (0, 2...) -> Draw Inner (Connector)
            # Odd segments  (1, 3...) -> Draw Outer (Loop Sides)
            segment_is_even = (segment_idx % 2) == 0
            
            is_chain_pixel = (segment_is_even & is_inner_pixel) | \
                             (~segment_is_even & is_outer_pixel)
            
            # Combine
            return in_box & is_chain_pixel & is_active

        # 5. Vectorize
        all_masks = jax.vmap(_create_single_chain_mask)(positions, sizes)
        
        # 6. Collapse
        combined_mask = jnp.any(all_masks, axis=0)
        
        # 7. Apply to Raster
        return jnp.where(combined_mask, jnp.asarray(chain_color, dtype=raster.dtype), raster)


class ReplaceLadderWithRopeMod(JaxAtariInternalModPlugin):
    """
    Replaces the ladder sprites with rope sprites.
    Ropes are drawn as a 2-pixel wide zig-zag pattern centered on the original ladder position.
    """
    NEW_LADDER_COLOR = (149, 75, 49)

    # Create the procedural asset (1x1 pixel RGBA sprite)
    custom_color_rgba = jnp.array([[[
        NEW_LADDER_COLOR[0],  # R
        NEW_LADDER_COLOR[1],  # G
        NEW_LADDER_COLOR[2],  # B
        255  # Alpha (fully opaque)
    ]]], dtype=jnp.uint8)

    # Add via asset_overrides (can add new assets, not just override existing ones!)
    asset_overrides = {
        'custom_ladder_color': {
            'name': 'custom_ladder_color',
            'type': 'procedural',
            'data': custom_color_rgba
        }, 
        "kangaroo": {
            'name': 'kangaroo', 
            'type': 'group',
            'files': ['kangaroo.npy', 'kangaroo_dead.npy', 'kangaroo_rope_climb.npy', 'kangaroo_ducking.npy', 'kangaroo_jump.npy', 'kangaroo_boxing.npy', 'kangaroo_walk.npy', 'kangaroo_jump_high.npy']
        }
    }
    
    @partial(jax.jit, static_argnums=(0,))
    def _draw_ladders(self, raster: jnp.ndarray, state: KangarooState):
        """
        Draws ropes: a 2-pixel wide zig-zag pattern.
        """
        # 1. Access Data from the Environment State
        positions = state.level.ladder_positions
        sizes = state.level.ladder_sizes
        
        # Access the renderer's calculated color ID for the ladder/rope
        rope_color = self._env.renderer.COLOR_TO_ID.get(self.NEW_LADDER_COLOR, 0)
            
        # 2. Get dimensions
        h, w = raster.shape
        
        # 3. Create the meshgrid for vectorization
        # yy corresponds to row indices, xx to column indices
        yy, xx = jnp.mgrid[:h, :w]

        # 4. Define Visual Constants
        rope_visual_width = 2 
        segment_height = 4  # How many pixels tall one 'twist' of the rope is

        def _create_single_rope_mask(pos, size):
            """Generates a boolean mask for a single rope."""
            # Only draw if the ladder exists (x != -1)
            is_active = pos[0] != -1
            
            # Geometry
            x, y = pos
            hitbox_width, height = size

            y -= 8
            height += 8
            
            # Calculate Center: The rope hangs in the middle of the ladder hitbox
            center_x = x + (hitbox_width // 2)
            
            # Start drawing x (shift left by 1 to center the 2px rope)
            draw_x_start = center_x - 1
            
            # --- Bounding Box Logic ---
            # Determine relative coordinates to the top-left of the rope
            dx = xx - draw_x_start
            dy = yy - y
            
            # Check if pixel is within the 2px wide x height bounding box
            in_box = (dx >= 0) & (dx < rope_visual_width) & \
                     (dy >= 0) & (dy < height)
            
            # --- Zig-Zag Pattern Logic ---
            # We determine the 'segment' index based on Y position.
            # Example: Rows 0-1 are segment 0, Rows 2-3 are segment 1.
            segment_idx = dy // segment_height
            
            # Check if the pixel is on the left side (dx=0) or right side (dx=1)
            is_left_pixel = (dx == 0)
            
            # Pattern: 
            # Even segments (0, 2, 4...) -> Draw Left Pixel
            # Odd segments (1, 3, 5...)  -> Draw Right Pixel
            segment_is_even = (segment_idx % 2) == 0
            
            # Draw if (Even Segment AND Left Pixel) OR (Odd Segment AND Right Pixel)
            # This is equivalent to checking if the boolean values are equal
            is_rope_pixel = (segment_is_even == is_left_pixel)
            
            # Combine: Must be in bounding box, match the pattern, and be active
            return in_box & is_rope_pixel & is_active

        # 5. Vectorize: Apply logic to all ladders simultaneously
        # resulting shape: (Num_Ladders, Height, Width)
        all_masks = jax.vmap(_create_single_rope_mask)(positions, sizes)
        
        # 6. Collapse: Combine all ladder masks into one single layer
        combined_mask = jnp.any(all_masks, axis=0)
        
        # 7. Apply to Raster: Where mask is True, paint the rope color
        return jnp.where(combined_mask, jnp.asarray(rope_color, dtype=raster.dtype), raster)


        # Vectorize over all ropes in the array
        all_masks = jax.vmap(_create_single_rope_mask)(pos_scaled, size_scaled)
        
        # Combine all rope masks into one layer
        combined_mask = jnp.logical_or.reduce(all_masks, axis=0)
        
        # Apply to raster
        return jnp.where(combined_mask, jnp.asarray(self.LADDER_COLOR_ID, raster.dtype), raster)


# --- MOD B: Make the "Flame" (Bell) Lethal ---

class LethalFlameMod(JaxAtariPostStepModPlugin):
    """
    Post-step mod that kills the player if they touch the bell.
    
    This runs *after* the main step and overrides the final
    state if a collision is detected.
    Using this instead of overriding the _bell_step method because kangaroo has a central function that handles collisions
    which might overwrite internal changes that happen before it executes.
    Other possibility would have been to either override the _bell_step method and the _lives_controller method or to insert a dedicated hook.
    """
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: KangarooState, new_state: KangarooState):
        """
        Called by the wrapper after the main step is complete.
        """
        # 1. Check for collision between player and bell
        is_colliding = self._env._entities_collide(
            new_state.player.x,
            new_state.player.y,
            self._env.consts.PLAYER_WIDTH,
            new_state.player.height,
            new_state.level.bell_position[0],
            new_state.level.bell_position[1],
            self._env.consts.BELL_WIDTH,
            self._env.consts.BELL_HEIGHT,
        )
        
        # 2. If colliding and not already crashing, set crash state
        #    and remove one life.
        return jax.lax.cond(
            is_colliding & ~new_state.player.is_crashing,
            
            # --- If True (Kill the player AND remove life) ---
            lambda state: self._trigger_crash_and_lose_life(state),
            
            # --- If False (Do nothing) ---
            lambda state: state,
            
            # Pass in the state
            new_state
        )

    @partial(jax.jit, static_argnums=(0,))
    def _trigger_crash_and_lose_life(self, state: KangarooState) -> KangarooState:
        """
        Returns a new state with the player set to 'is_crashing'
        and the life count decremented.
        """
        # 1. Set the player to crashing
        crashed_player_state = state.player.replace(
            is_crashing=jnp.array(True)
        )
        
        # 2. Decrement the life count
        new_lives = state.lives - 1
        
        # 3. Return the new state with both changes
        return state.replace(
            player=crashed_player_state,
            lives=new_lives
        )


class SpawnOnSecondFloorMod(JaxAtariInternalModPlugin):
    """Mod to spawn the player on the second level position."""
    # overwrite constants
    constants_overrides = {
        "PLAYER_START_Y": 52,
    }


# --- Ladder Modification Mods ---
def _center_ladders(level_constants):
    """Center all ladders horizontally on the screen while keeping their y positions."""
    # Screen width is 160, ladder width is 8
    SCREEN_WIDTH = 160
    # Center x position: (160 - 8) / 2 = 76
    center_x = 76
    
    # Keep invalid positions (-1) as invalid
    is_valid = level_constants.ladder_positions[:, 0] >= 0
    
    # Create new positions: center x, keep original y
    original_y = level_constants.ladder_positions[:, 1]
    centered_positions = jnp.where(
        is_valid[:, jnp.newaxis],
        jnp.stack([jnp.full_like(original_y, center_x), original_y], axis=1),
        level_constants.ladder_positions  # Keep invalid positions as -1
    )
    
    # Center also the platforms accordingly
    platform_original_y = level_constants.platform_positions[:, 1]
    platform_centered_x = (SCREEN_WIDTH - level_constants.platform_sizes[:, 0]) // 2
    centered_platform_positions = jnp.where(
        (level_constants.platform_sizes[:, 0] < 128)[:, None], # Only center small platforms (x > 16)
        jnp.stack([jnp.full_like(platform_original_y, platform_centered_x), platform_original_y], axis=1),
        level_constants.platform_positions
    )

    return LevelConstants(
        ladder_positions=centered_positions,
        ladder_sizes=level_constants.ladder_sizes,
        platform_positions=centered_platform_positions,
        platform_sizes=level_constants.platform_sizes,
        fruit_positions=level_constants.fruit_positions,
        bell_position=level_constants.bell_position,
        child_position=level_constants.child_position,
    )

class CenterLaddersMod(JaxAtariInternalModPlugin):
    """
    Internal mod to center all ladder positions horizontally on the screen.
    All ladders will be perfectly aligned at x=76 (center of 160px screen).
    Uses constants_overrides to directly modify LEVEL_1, LEVEL_2, LEVEL_3.
    """
    # Create modified level constants with centered ladders
    _level1_centered = _center_ladders(Kangaroo_Level_1)
    _level2_centered = _center_ladders(Kangaroo_Level_2)
    _level3_centered = _center_ladders(Kangaroo_Level_3)
    
    # Override constants directly
    constants_overrides = {
        "LEVEL_1": _level1_centered,
        "LEVEL_2": _level2_centered,
        "LEVEL_3": _level3_centered,
    }

def _invert_ladders(level_constants):
    """Invert ladder positions horizontally on the screen."""
    # Screen width is 160, ladder width is 8
    screen_width = 160
    ladder_width = 8
    
    # Invert x positions: new_x = screen_width - ladder_width - original_x
    inverted_x = screen_width - ladder_width - level_constants.ladder_positions[:, 0]
    
    inverted_positions = jnp.stack([inverted_x, level_constants.ladder_positions[:, 1]], axis=1)

    inverted_platform_x = screen_width - level_constants.platform_positions[:, 0] - level_constants.platform_sizes[:, 0]

    inverted_platform_positions = jnp.stack([inverted_platform_x, level_constants.platform_positions[:, 1]], axis=1)

    inverted_bell_x = screen_width - level_constants.bell_position[0] - 6  # Bell width is 6

    inverted_bell_position = jnp.array([inverted_bell_x, level_constants.bell_position[1]])
    
    return LevelConstants(
        ladder_positions=inverted_positions,
        ladder_sizes=level_constants.ladder_sizes,
        platform_positions=inverted_platform_positions,
        platform_sizes=level_constants.platform_sizes,
        fruit_positions=level_constants.fruit_positions,
        bell_position=inverted_bell_position,
        child_position=level_constants.child_position,
    )

class InvertLaddersMod(JaxAtariInternalModPlugin):
    """
    Internal mod to invert all ladder positions horizontally on the screen.
    Uses constants_overrides to directly modify LEVEL_1, LEVEL_2, LEVEL_3.
    """
    
    # Create modified level constants with inverted ladders
    _level1_inverted = _invert_ladders(Kangaroo_Level_1)
    _level2_inverted = _invert_ladders(Kangaroo_Level_2)
    _level3_inverted = _invert_ladders(Kangaroo_Level_3)
    
    # Override constants directly
    constants_overrides = {
        "LEVEL_1": _level1_inverted,
        "LEVEL_2": _level2_inverted,
        "LEVEL_3": _level3_inverted,
    }


# Create modified level constants with four ladders
def _add_fourth_ladder(level_constants, level_number):
    # Get existing ladder positions and sizes
    ladder_positions = level_constants.ladder_positions
    ladder_sizes = level_constants.ladder_sizes
    
    # Identify the third ladder (index 2)
    fourth_ladder_pos = jnp.array([132, 84])  # Default position for the fourth ladder
    fourth_ladder_size = jnp.array([8, 36])  # Standard ladder size
    

    # Append the fourth ladder
    new_ladder_positions = jnp.vstack([ladder_positions, fourth_ladder_pos])
    new_ladder_sizes = jnp.vstack([ladder_sizes, fourth_ladder_size])
    return LevelConstants(
        ladder_positions=new_ladder_positions,
        ladder_sizes=new_ladder_sizes,
        platform_positions=level_constants.platform_positions,
        platform_sizes=level_constants.platform_sizes,
        fruit_positions=level_constants.fruit_positions,
        bell_position=level_constants.bell_position,
        child_position=level_constants.child_position,
    )

class FourLaddersMod(JaxAtariInternalModPlugin):
    """
    Internal mod to add a fourth ladder to each level.
    The fourth ladder is placed symmetrically to the third ladder.
    """
    _level1_with_four = _add_fourth_ladder(Kangaroo_Level_1, 0)
    _level2_with_four = _add_fourth_ladder(Kangaroo_Level_2, 1)
    _level3_with_four = _add_fourth_ladder(Kangaroo_Level_3, 2)
    
    constants_overrides = {
        "LEVEL_1": _level1_with_four,
        "LEVEL_2": _level2_with_four,
        "LEVEL_3": _level3_with_four,
    }


def flame_trap(level_constants):
    """Moves the flame to the first floor position."""
    return LevelConstants(
        ladder_positions=level_constants.ladder_positions,
        ladder_sizes=level_constants.ladder_sizes,
        platform_positions=level_constants.platform_positions,
        platform_sizes=level_constants.platform_sizes,
        fruit_positions=level_constants.fruit_positions,
        bell_position=jnp.array([100, 113]),  # First floor position
        child_position=level_constants.child_position,
    )

class FlameTrapMod(JaxAtariInternalModPlugin):
    """
    Internal mod to place the flame (bell) on the way to the fruit at each level.
    """

    _level1_centered = _center_ladders(Kangaroo_Level_1)
    _level2_centered = _center_ladders(Kangaroo_Level_2)
    _level3_centered = _center_ladders(Kangaroo_Level_3)
    constants_overrides = {
        "LEVEL_1": flame_trap(_level1_centered),
        "LEVEL_2": flame_trap(_level2_centered),
        "LEVEL_3": flame_trap(_level3_centered),
    }


# --- 3. Dynamic difficulty mods (parallels to freeway/pong dyn sequences) ---
# These four change *how the game behaves* (movement / spawn dynamics) rather
# than how it looks. They are the members of the kangaroo `dyn4` sequence:
#   change_kangaroo_speed, change_monkey_speed, randomize_coconuts, jump_gravity

class ChangeKangarooSpeedMod(JaxAtariInternalModPlugin):
    """
    Makes the kangaroo walk faster horizontally by overriding MOVEMENT_SPEED
    (base 1 px per move; the player only advances every 3rd frame, so this scales
    the walking speed directly). Counterpart in spirit to freeway's
    change_car_speed / faster_player. Default: 2x.

    The player's LEFT_CLIP/RIGHT_CLIP bounds still clamp the position, so the
    kangaroo cannot walk off-screen; 2 px steps stay well inside the ladder
    hitboxes, so climbing is unaffected.
    """
    constants_overrides = {"MOVEMENT_SPEED": 2}


class ChangeMonkeySpeedMod(JaxAtariPostStepModPlugin):
    """
    Makes the monkeys patrol faster horizontally (default 2x), so they reach
    their throw position sooner and hurl coconuts more often.

    Monkeys only move on 1-in-16 frames, and their vertical descent snaps to
    exact platform y-values (the base game compares monkey_lower_y == 172/124/76
    with '=='), so amplifying vertical motion would break their platform
    transitions. This post-step mod therefore only amplifies the horizontal walk
    -- monkey states 2 (left) and 4 (right), whose x thresholds use <= / >= -- by
    adding (_SPEED - 1) extra px in the travel direction on the frames a monkey
    actually took its base 3 px step. Same idea as freeway's _FasterCarsMod, but
    horizontal-only.
    """
    _SPEED = 2

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: KangarooState, new_state: KangarooState) -> KangarooState:
        pos = new_state.level.monkey_positions
        dx = pos[:, 0] - prev_state.level.monkey_positions[:, 0]
        walking = jnp.abs(dx) == 3  # a normal horizontal patrol step (state 2/4 only)
        extra = jnp.where(walking, jnp.sign(dx) * 3 * (self._SPEED - 1), 0)
        # Keep monkeys within their patrol range so the <=107 / >=146 throw/turn
        # thresholds still fire; overshooting a threshold is fine (they use <=/>=).
        new_x = jnp.clip(pos[:, 0] + extra, 0, 152).astype(pos.dtype)
        new_positions = pos.at[:, 0].set(new_x)
        return new_state.replace(level=new_state.level.replace(monkey_positions=new_positions))


class RandomizeCoconutsMod(JaxAtariInternalModPlugin):
    """
    Randomizes each thrown coconut's launch height (head vs. foot) instead of the
    base game's fixed head/foot alternation, so the player can no longer predict
    from the last throw whether the next one must be ducked or jumped.

    Patches _update_coco_positions (the same hook as high_thrown_coconuts). The
    spawn height is a coin flip keyed on the throw frame + monkey x, so it is
    deterministic per rollout (JAX-pure, no external RNG state) but varies from
    throw to throw. Flight logic (the -2 px horizontal drift) is left unchanged.
    """

    @partial(jax.jit, static_argnums=(0,))
    def _update_coco_positions(
        self,
        new_c_state: chex.Array,
        old_c_state: chex.Array,
        stepc: chex.Array,
        old_c_pos: chex.Array,
        new_m_pos: chex.Array,
        spawn_position: chex.Array,
    ) -> chex.Array:
        c = self._env.consts
        # Per-throw coin flip: keyed on the frame and the throwing monkey's x.
        seed = (stepc.astype(jnp.uint32) * jnp.uint32(97)
                + new_m_pos[0].astype(jnp.uint32) * jnp.uint32(13))
        spawn_high = jax.random.bernoulli(jax.random.PRNGKey(seed))

        spawn_y = jnp.where(
            spawn_high,
            new_m_pos[1] - 5,                                              # head height
            new_m_pos[1] + c.MONKEY_HEIGHT - c.THROWN_COCONUT_HEIGHT,      # foot height
        )

        return jnp.where(
            new_c_state == 2,
            # --- Flight Logic (unchanged) ---
            jnp.where(
                stepc % 2 == 0,
                jnp.array([old_c_pos[0] - 2, old_c_pos[1]]),
                old_c_pos,
            ),
            # --- Spawn Logic (randomized height) ---
            jnp.where(
                (new_c_state == 1) & (old_c_state == 0),
                jnp.array([new_m_pos[0] - 6, spawn_y]),
                old_c_pos,
            ),
        )


class JumpGravityMod(JaxAtariInternalModPlugin):
    """
    Changes the kangaroo's jump dynamics: a floatier, higher "low-gravity" jump.

    Kangaroo's jump is scripted, not physics-based -- _player_jump_controller
    drives the player's y through a fixed per-frame offset profile (offset_for)
    keyed on jump_counter (1..41), and the landing/cancel logic depends on that
    counter timeline and on the tail reaching exactly -8 (one platform, 8 px, up).
    This override raises the *peak* of that profile so jumps float higher and
    hang longer, while KEEPING the -8 tail and the 41-frame timeline so all
    platform-landing and jump-cancel logic (and thus level traversal) still works.

    Only the `offset_for` height values differ from the base method; everything
    else is copied verbatim so the surrounding jump/cancel logic is preserved.

    NOTE: this is the "reduced gravity" direction (higher, floatier). A "heavier"
    (lower/faster-falling) jump is riskier because it can make platforms
    unreachable and soft-lock a level, so it is deliberately not the default.
    """

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _player_jump_controller(self, state: KangarooState, jump_pressed: chex.Array, ladder_intersect: chex.Array):
        consts = self._env.consts
        player_y = state.player.y
        jump_counter = state.player.jump_counter
        is_jumping = state.player.is_jumping

        is_crouch_jumping = state.player.is_crouching & jump_pressed
        crouch_height_adjustment = consts.PLAYER_HEIGHT - 16  # Crouch height is 16
        player_y_for_jump = jnp.where(is_crouch_jumping, player_y - crouch_height_adjustment, player_y)

        cooldown_condition = state.player.cooldown_counter > 0
        jump_start = (
            jump_pressed
            & ~is_jumping
            & ~ladder_intersect
            & ~cooldown_condition
            & ((player_y + consts.PLAYER_HEIGHT) > 28)
        )

        jump_counter = jnp.where(jump_start, 0, jump_counter)
        jump_orientation = jnp.where(
            jump_start, state.player.orientation, state.player.jump_orientation
        )
        jump_base_y = jnp.where(jump_start, player_y_for_jump, state.player.jump_base_y)
        new_landing_base_y = jump_base_y

        platform_y_below_player = self._env._get_y_of_platform_below_player(state)

        new_landing_base_y = jnp.where(
            is_jumping
            & ((platform_y_below_player - consts.PLAYER_HEIGHT) == (jump_base_y - 8))
            & ~jump_start,
            platform_y_below_player - consts.PLAYER_HEIGHT,
            new_landing_base_y,
        )

        new_landing_base_y = jnp.where(
            is_jumping
            & ((platform_y_below_player - consts.PLAYER_HEIGHT) == (jump_base_y + 8))
            & ~jump_start,
            platform_y_below_player - consts.PLAYER_HEIGHT,
            new_landing_base_y,
        )

        is_jumping = is_jumping | jump_start

        jump_counter = jnp.where(is_jumping, jump_counter + 1, jump_counter)

        # --- MODIFIED: raised peak for a floatier, lower-gravity jump. The -8
        # --- tail (count 33-39) and 41-frame timeline are kept so the +8-px
        # --- platform landing and jump-cancel logic below still line up.
        def offset_for(count):
            conditions = [
                (count > 0) & (count <= 8),
                (count > 8) & (count < 16),
                (count >= 16) & (count <= 24),
                (count > 24) & (count <= 32),
                (count > 32) & (count < 40),
            ]
            values = [
                -1,
                -10,   # base: -8
                -14,   # base: -8
                -20,   # base: -16 (higher peak)
                -8,    # base: -8  (kept: one platform up, for landing)
            ]
            return jnp.select(conditions, values, default=0)

        jump_cancel_up = (
            is_jumping
            & (player_y >= new_landing_base_y)
            & (new_landing_base_y < jump_base_y)
            & (jump_counter > 32)
        )

        jump_cancel_down = (
            is_jumping
            & ((player_y + 1) == jump_base_y)
            & (new_landing_base_y == (jump_base_y + 8))
            & (jump_counter >= 40)
        )

        jump_cancel = jump_cancel_up | jump_cancel_down

        jump_counter = jnp.where(jump_cancel, 40, jump_counter)
        jump_base_y = jnp.where(jump_cancel, new_landing_base_y, jump_base_y)
        new_y = jnp.where(jump_cancel, new_landing_base_y, player_y)
        new_cooldown_counter = jnp.where(jump_cancel, 8, state.player.cooldown_counter)

        total_offset = offset_for(jump_counter)
        new_y = jnp.where(is_jumping & ~jump_cancel, jump_base_y + total_offset, new_y)

        jump_complete = jump_counter >= 41
        is_jumping = jnp.where(jump_complete, False, is_jumping)
        jump_counter = jnp.where(jump_complete, 0, jump_counter)

        return_value = (
            new_y,
            jump_counter,
            is_jumping,
            jump_base_y,
            new_landing_base_y,
            jump_orientation,
            new_cooldown_counter,
        )

        return jax.lax.cond(
            state.levelup_timer == 0,
            lambda: return_value,
            lambda: (
                state.player.y,
                state.player.jump_counter,
                state.player.is_jumping,
                state.player.jump_base_y,
                state.player.landing_base_y,
                state.player.jump_orientation,
                state.player.cooldown_counter,
            ),
        )

# ============================================================================ #
# 4. Visual mods: single-element recolours + a grayscale theme
#    (parallels the freeway/pong change_*_color mods). The kangaroo and monkey
#    sprites each use a single baked body colour, so a selective source->target
#    recolour swaps them cleanly -- no sprite-swap fallback needed.
# ============================================================================ #
import os
import numpy as np
from jaxatari.rendering.jax_rendering_utils import (
    JaxRenderingUtils, RendererConfig, get_base_sprite_dir,
)

_jr = JaxRenderingUtils(RendererConfig())
_SPRITE_DIR = os.path.join(get_base_sprite_dir(), "kangaroo")


def _load(fname):
    return _jr.loadFrame(os.path.join(_SPRITE_DIR, fname))


# Source colours baked into the base sprites (verified directly from the .npy).
_KANGAROO_SRC  = (223, 183, 85)
_MONKEY_SRC    = (227, 151, 89)
_BACKDROP_SRC  = (80, 0, 132)     # purple wall behind everything
_STRUCTURE_SRC = (162, 98, 33)    # platforms / ladders / coconuts ("wood" brown)
_SCORE_SRC     = (160, 171, 79)

# New colours (tweak here). Each recolour mod touches only its own element.
_NEW_KANGAROO_COLOR = (66, 135, 245)   # blue
_NEW_MONKEY_COLOR   = (190, 70, 190)   # magenta
_NEW_BACKDROP_COLOR = (26, 58, 92)     # deep blue-grey
_NEW_SCORE_COLOR    = (0, 200, 200)    # cyan

# Group file lists (must mirror get_default_asset_config in jax_kangaroo.py).
_KANGAROO_FILES = ['kangaroo.npy', 'kangaroo_dead.npy', 'kangaroo_climb.npy', 'kangaroo_ducking.npy',
                   'kangaroo_jump.npy', 'kangaroo_boxing.npy', 'kangaroo_walk.npy', 'kangaroo_jump_high.npy']
_APE_FILES   = ['ape_standing.npy', 'ape_climb_left.npy', 'ape_moving.npy', 'throwing_ape.npy', 'ape_climb_right.npy']
_BELL_FILES  = ['bell.npy', 'ringing_bell.npy']
_FRUIT_FILES = ['strawberry.npy', 'tomato.npy', 'cherry.npy', 'pineapple.npy']
_CHILD_FILES = ['child.npy', 'child_jump.npy']

_SCORE_PATHS = [os.path.join(_SPRITE_DIR, f'score_{i}.npy') for i in range(10)]
_TIME_PATHS  = [os.path.join(_SPRITE_DIR, f'time_{i}.npy') for i in range(10)]


def _recolor_group(files, src, tgt):
    """Selective source->target recolour for each frame of a sprite group."""
    rule = [{'source': src, 'target': tgt}]
    return [_jr.perform_recoloring(_load(f), rule) for f in files]


class ChangeKangarooColorMod(JaxAtariInternalModPlugin):
    """Recolours the kangaroo (single baked body colour). Default: blue."""
    asset_overrides = {
        "kangaroo": {
            'name': 'kangaroo', 'type': 'group',
            'data': _recolor_group(_KANGAROO_FILES, _KANGAROO_SRC, _NEW_KANGAROO_COLOR),
        }
    }


class ChangeMonkeyColorMod(JaxAtariInternalModPlugin):
    """Recolours the monkeys (ape sprite group). Default: magenta."""
    asset_overrides = {
        "ape": {
            'name': 'ape', 'type': 'group',
            'data': _recolor_group(_APE_FILES, _MONKEY_SRC, _NEW_MONKEY_COLOR),
        }
    }


class ChangeBackgroundColorMod(JaxAtariInternalModPlugin):
    """
    Recolours only the purple backdrop wall. The platforms/ladders (brown) and
    the black shading are left untouched so the level structure still reads and
    the renderer's platform/ladder colour-ID lookups keep resolving. Default:
    deep blue-grey.
    """
    asset_overrides = {
        "background": {
            'name': 'background', 'type': 'background',
            'data': _jr.perform_recoloring(
                _load('background.npy'),
                [{'source': _BACKDROP_SRC, 'target': _NEW_BACKDROP_COLOR}],
            ),
        }
    }


class ChangeScoreColorMod(JaxAtariInternalModPlugin):
    """
    Recolours the whole bottom UI -- score digits, timer digits and the life
    icons -- which all share the same baked colour (160,171,79). Default: cyan.
    """
    _rule = [{'source': _SCORE_SRC, 'target': _NEW_SCORE_COLOR}]
    asset_overrides = {
        "score_digits": {
            'name': 'score_digits', 'type': 'digits',
            'data': _jr.perform_recoloring(_jr._load_and_pad_digits_from_paths(_SCORE_PATHS), _rule),
        },
        "time_digits": {
            'name': 'time_digits', 'type': 'digits',
            'data': _jr.perform_recoloring(_jr._load_and_pad_digits_from_paths(_TIME_PATHS), _rule),
        },
        "lives": {
            'name': 'lives', 'type': 'single',
            'data': _jr.perform_recoloring(_load('kangaroo_lives.npy'), _rule),
        },
    }


def _to_gray(arr, keep=None):
    """
    Luminance-grayscale an RGBA sprite array (alpha preserved). If `keep` is a
    source RGB, pixels of exactly that colour are left unchanged -- used to keep
    the structural brown intact so the renderer can still identify platforms and
    ladders by it.
    """
    a = np.array(arr, dtype=np.uint8)
    r, g, b = a[..., 0].copy(), a[..., 1].copy(), a[..., 2].copy()
    lum = np.round(0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    a[..., 0] = a[..., 1] = a[..., 2] = lum
    if keep is not None:
        m = (r == keep[0]) & (g == keep[1]) & (b == keep[2])
        a[..., 0] = np.where(m, r, a[..., 0])
        a[..., 1] = np.where(m, g, a[..., 1])
        a[..., 2] = np.where(m, b, a[..., 2])
    return jnp.asarray(a)


def _gray_group(files):
    return [_to_gray(_load(f)) for f in files]


class GrayscaleThemeMod(JaxAtariInternalModPlugin):
    """
    Desaturates the whole scene to luminance grayscale.

    The "wood" brown (162,98,33) is deliberately kept: the renderer draws the
    platforms and ladders by looking that exact colour up in the palette
    (PLATFORM_COLOR_ID / LADDER_COLOR_ID = COLOR_TO_ID.get((162,98,33))), so
    grayscaling it away would drop those lookups to palette id 0 (the backdrop)
    and the platforms/ladders would vanish. Keeping it in the background (and in
    the coconuts, which share that colour) leaves the level structure readable as
    brown "wood" on an otherwise grayscale scene.
    """
    asset_overrides = {
        "background":      {'name': 'background', 'type': 'background',
                            'data': _to_gray(_load('background.npy'), keep=_STRUCTURE_SRC)},
        "ape":             {'name': 'ape', 'type': 'group', 'data': _gray_group(_APE_FILES)},
        "kangaroo":        {'name': 'kangaroo', 'type': 'group', 'data': _gray_group(_KANGAROO_FILES)},
        "bell":            {'name': 'bell', 'type': 'group', 'data': _gray_group(_BELL_FILES)},
        "fruit":           {'name': 'fruit', 'type': 'group', 'data': _gray_group(_FRUIT_FILES)},
        "child":           {'name': 'child', 'type': 'group', 'data': _gray_group(_CHILD_FILES)},
        "coconut":         {'name': 'coconut', 'type': 'single',
                            'data': _to_gray(_load('coconut.npy'), keep=_STRUCTURE_SRC)},
        "falling_coconut": {'name': 'falling_coconut', 'type': 'single',
                            'data': _to_gray(_load('falling_coconut.npy'), keep=_STRUCTURE_SRC)},
        "lives":           {'name': 'lives', 'type': 'single', 'data': _to_gray(_load('kangaroo_lives.npy'))},
        "score_digits":    {'name': 'score_digits', 'type': 'digits',
                            'data': _to_gray(_jr._load_and_pad_digits_from_paths(_SCORE_PATHS))},
        "time_digits":     {'name': 'time_digits', 'type': 'digits',
                            'data': _to_gray(_jr._load_and_pad_digits_from_paths(_TIME_PATHS))},
    }


# ============================================================================ #
# 5. Reward-shaping mods (parallels the freeway/pong/asteroids reward mods).
#    All override _get_reward(previous_state, state) and recompute the reward
#    from state deltas -- the base reward is `state.score - previous_state.score`.
# ============================================================================ #

class LifeLossPenaltyMod(JaxAtariInternalModPlugin):
    """
    Penalizes losing a life, on top of the normal score-delta reward, to shift the
    optimum away from greedy point-grabbing toward staying alive.

    A life is lost on exactly the frame `lives` decrements (the base game does this
    once per death, when `remove_live & ~is_crashing` fires), so a lives delta is a
    clean single-frame death signal. Default penalty is on the order of one monkey
    punch (200), so a death roughly cancels a couple of kills.
    """
    _PENALTY = 200

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: KangarooState, state: KangarooState):
        base = state.score - previous_state.score
        lives_lost = jnp.maximum(previous_state.lives - state.lives, 0)
        return (base - lives_lost * self._PENALTY).astype(jnp.int32)


class RewardPerFloorMod(JaxAtariInternalModPlugin):
    """
    Dense progress reward: +_PER_FLOOR for each floor climbed upward (and
    -_PER_FLOOR per floor lost), replacing the sparse point score. Potential-based
    on the player's current floor, so bobbing up and down a ladder nets zero --
    only net upward progress toward Joey (the top platform) is rewarded.

    The floor index is derived from `last_stood_on_platform_y` (the platform the
    player last stood on: 172 ground .. 28 top, ~48 px apart), which is stable
    while jumping/climbing and only advances once a new floor is actually reached.
    Death/level-transition teleports back to the ground are gated out so they don't
    read as a huge negative (the player is respawned/frozen on those frames).
    """
    _PER_FLOOR = 1

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: KangarooState, state: KangarooState):
        # 172 (ground) -> 0, 124 -> 1, 76 -> 2, 28 (top / Joey) -> 3
        def floor(player):
            return jnp.clip(jnp.round((172 - player.last_stood_on_platform_y) / 48.0), 0, 4).astype(jnp.int32)

        delta = floor(state.player) - floor(previous_state.player)
        # Ignore the ground teleport on death / level change / levelup freeze.
        teleported = (
            (state.lives < previous_state.lives)
            | state.player.is_crashing
            | (previous_state.levelup_timer != 0)
            | (state.levelup_timer != 0)
            | (state.current_level != previous_state.current_level)
        )
        delta = jnp.where(teleported, 0, delta)
        return (delta * self._PER_FLOOR).astype(jnp.int32)


class ReachJoeyOnlyMod(JaxAtariInternalModPlugin):
    """
    Sparse goal-only reward: +_REWARD the moment the player reaches Joey (the baby
    kangaroo) at the top of the level, and 0 for everything else -- fruit, punches,
    bell and the time bonus are all ignored. The whole task collapses to "get to
    the top".

    Reaching the top is the base game's `level_finished` flag (set when the player
    stands on the final platform, where the child sits); the rising edge
    `level_finished & ~prev.level_finished` credits it once per level.
    """
    _REWARD = 1000

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: KangarooState, state: KangarooState):
        reached = state.level_finished & ~previous_state.level_finished
        return jnp.where(reached, self._REWARD, 0).astype(jnp.int32)


class FruitOnlyMod(JaxAtariInternalModPlugin):
    """
    Rewards only fruit collection; punches, the bell, and the level/time bonus give
    nothing. Encourages a pure fruit-forager policy.

    A fruit is collected on the frame its `fruit_actives` flag flips True->False
    (the base game sets it False on the collision that scores it; respawns only go
    False->True, and level resets set every flag True, so True->False is uniquely a
    collection). The credited value matches the game's own `100 * 2**stage`, read
    from the pre-collection stage.
    """

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: KangarooState, state: KangarooState):
        collected = previous_state.level.fruit_actives & ~state.level.fruit_actives
        values = 100 * (2 ** previous_state.level.fruit_stages)
        return jnp.sum(jnp.where(collected, values, 0)).astype(jnp.int32)


class PunchOnlyScoringMod(JaxAtariInternalModPlugin):
    """
    Rewards only punching enemies -- monkeys and the single falling coconut (200
    each, matching the base scores) -- and nothing for fruit, the bell or the level
    bonus. Encourages an aggressive boxing policy.

    The punch outcomes are recomputed exactly as the base game does inside
    `_monkey_controller` / `_falling_coconut_controller`: the fist box is built
    from the *previous* frame's player pose (which is what those controllers see),
    the targets from the previous frame's positions, and `punching` from this
    frame's resolved punch flags (which already encode the game's can-punch rules).
    """

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: KangarooState, state: KangarooState):
        c = self._env.consts
        p = previous_state.player
        punching = state.player.punch_left | state.player.punch_right

        fist_w, fist_h = 3, 4
        fist_x = jnp.where(p.orientation > 0, p.x + c.PLAYER_WIDTH, p.x - fist_w)
        fist_y = p.y + 8

        def monkey_hit(m_x, m_y, m_state):
            return (
                self._env._entities_collide(
                    fist_x, fist_y, fist_w, fist_h,
                    m_x, m_y, c.MONKEY_WIDTH, c.MONKEY_HEIGHT,
                )
                & (m_state != 0)
                & punching
            )

        monkeys_hit = jax.vmap(monkey_hit, in_axes=(0, 0, 0))(
            previous_state.level.monkey_positions[:, 0],
            previous_state.level.monkey_positions[:, 1],
            previous_state.level.monkey_states,
        )
        monkey_reward = jnp.sum(monkeys_hit) * 200

        coco_hit = (
            self._env._entities_collide_with_threshold(
                fist_x, fist_y, fist_w, fist_h,
                previous_state.level.falling_coco_position[0],
                previous_state.level.falling_coco_position[1],
                c.FALLING_COCONUT_WIDTH, c.FALLING_COCONUT_HEIGHT,
                0.01,
            )
            & punching
        )
        coco_reward = jnp.where(coco_hit, 200, 0)

        return (monkey_reward + coco_reward).astype(jnp.int32)


class SurvivalRewardMod(JaxAtariInternalModPlugin):
    """
    Rewards +_PER_STEP for every step taken, regardless of fruit, punches, the
    bell or reaching Joey -- replaces the score-based reward entirely, turning the
    task into "stay alive as long as possible".

    The episode ends only once the player is out of lives (`_get_done` fires at
    lives <= 0), and the training loop stops calling step() at done, so a flat
    per-step reward already integrates to time-alive -- no extra lives check needed.
    """
    _PER_STEP = 1

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: KangarooState, state: KangarooState):
        return jnp.array(self._PER_STEP, dtype=jnp.int32)


# ============================================================================ #
# 6. Magnitude sequence (coconut_speed_xN): the same dynamics mod scaled to
#    incrementally harder levels -- thrown coconuts fly N px/frame (base 1),
#    shrinking the player's duck/jump reaction window. Kangaroo's analog of
#    freeway car_speed_xN / pong+breakout ball_speed_xN / asteroids asteroid_speed_xN.
# ============================================================================ #

class _FasterThrownCoconutsMod(JaxAtariInternalModPlugin):
    """
    Base for the magnitude sequence. Patches _update_coco_positions so a thrown
    coconut in flight moves _SPEED px/frame instead of the base 2 px every other
    frame (= 1 px/frame). Only the flight branch changes; the spawn branch is
    copied verbatim from the base game so head/foot launch height is unchanged.

    Tunnel-safe up to x5: the base checks thrown-coconut/player collision every
    frame (in _lives_controller), coconut width 2 + player width 8 gives a ~10 px
    overlap window, so a step of up to ~9 px/frame can't skip the player between
    checks -- x5 = 5 px is well under that, so faster coconuts stay lethal rather
    than harmlessly phasing through.

    Registered only via its concrete xN subclasses below (the base is not in the
    registry). The controller patches _update_coco_positions by inspecting the
    plugin instance's methods, so the inherited override is picked up per subclass
    with its own _SPEED.
    """
    _SPEED = 2  # px/frame (base is 1)

    @partial(jax.jit, static_argnums=(0,))
    def _update_coco_positions(
        self,
        new_c_state: chex.Array,
        old_c_state: chex.Array,
        stepc: chex.Array,
        old_c_pos: chex.Array,
        new_m_pos: chex.Array,
        spawn_position: chex.Array,
    ) -> chex.Array:
        c = self._env.consts
        return jnp.where(
            new_c_state == 2,
            # --- Flight Logic (sped up: _SPEED px every frame, no %2 gate) ---
            jnp.array([old_c_pos[0] - self._SPEED, old_c_pos[1]]),
            # --- Spawn Logic (unchanged from base) ---
            jnp.where(
                (new_c_state == 1) & (old_c_state == 0),
                jnp.array(
                    [
                        new_m_pos[0] - 6,
                        jnp.where(
                            spawn_position,
                            new_m_pos[1] - 5,
                            new_m_pos[1] + c.MONKEY_HEIGHT - c.THROWN_COCONUT_HEIGHT,
                        ),
                    ]
                ),
                old_c_pos,
            ),
        )


class CoconutSpeedX2Mod(_FasterThrownCoconutsMod):
    """Thrown-coconut speed x2."""
    _SPEED = 2


class CoconutSpeedX3Mod(_FasterThrownCoconutsMod):
    """Thrown-coconut speed x3."""
    _SPEED = 3


class CoconutSpeedX4Mod(_FasterThrownCoconutsMod):
    """Thrown-coconut speed x4."""
    _SPEED = 4


class CoconutSpeedX5Mod(_FasterThrownCoconutsMod):
    """Thrown-coconut speed x5."""
    _SPEED = 5


# ============================================================================ #
# 7. Control-restriction dynamic mods: disable one of the player's abilities.
#    no_punch / no_crouch strip the FIRE / DOWN button from the action and then
#    delegate to the base _player_step (so the other 250 lines of movement logic
#    are reused, not copied); no_jump neutralises the jump controller directly
#    while leaving ladder climbing (also on UP) intact.
# ============================================================================ #

def _build_action_remap(mapping):
    """Length-18 lookup (indexed by ALE action code 0..17) that rewrites the keys
    in `mapping` to their values and leaves every other action unchanged."""
    arr = list(range(18))
    for frm, to in mapping.items():
        arr[int(frm)] = int(to)
    return jnp.array(arr, dtype=jnp.int32)


# FIRE removed: every *FIRE action -> its non-fire counterpart.
_STRIP_FIRE = _build_action_remap({
    Action.FIRE: Action.NOOP,
    Action.UPFIRE: Action.UP,
    Action.RIGHTFIRE: Action.RIGHT,
    Action.LEFTFIRE: Action.LEFT,
    Action.DOWNFIRE: Action.DOWN,
    Action.UPRIGHTFIRE: Action.UPRIGHT,
    Action.UPLEFTFIRE: Action.UPLEFT,
    Action.DOWNRIGHTFIRE: Action.DOWNRIGHT,
    Action.DOWNLEFTFIRE: Action.DOWNLEFT,
})

# DOWN removed: every DOWN* action -> its non-down counterpart.
_STRIP_DOWN = _build_action_remap({
    Action.DOWN: Action.NOOP,
    Action.DOWNRIGHT: Action.RIGHT,
    Action.DOWNLEFT: Action.LEFT,
    Action.DOWNFIRE: Action.FIRE,
    Action.DOWNRIGHTFIRE: Action.RIGHTFIRE,
    Action.DOWNLEFTFIRE: Action.LEFTFIRE,
})


class NoPunchMod(JaxAtariInternalModPlugin):
    """
    Disables punching. The FIRE button is stripped from the action before the
    normal player step runs, so the kangaroo can never box -- monkeys and coconuts
    can no longer be destroyed for points or self-defence.

    Implemented by remapping every *FIRE action to its non-fire counterpart and
    delegating to the base JaxKangaroo._player_step (movement is unchanged); the
    downstream `punching` flag the monkey/coconut controllers read is derived from
    the punch state this produces, so it is False throughout.
    """

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _player_step(self, state: KangarooState, action: chex.Array):
        action = _STRIP_FIRE[jnp.clip(action.astype(jnp.int32), 0, 17)]
        return JaxKangaroo._player_step(self._env, state, action)


class NoCrouchMod(JaxAtariInternalModPlugin):
    """
    Disables crouching/ducking. The DOWN button is stripped from the action before
    the normal player step runs, so the kangaroo can no longer duck under high
    thrown coconuts.

    Side effect: DOWN also drives climbing *down* ladders, so that is disabled too
    -- harmless for reaching Joey (the objective is always upward). Implemented by
    remapping every DOWN* action to its non-down counterpart (FIRE is preserved,
    e.g. DOWNFIRE -> FIRE) and delegating to the base JaxKangaroo._player_step.
    """

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _player_step(self, state: KangarooState, action: chex.Array):
        action = _STRIP_DOWN[jnp.clip(action.astype(jnp.int32), 0, 17)]
        return JaxKangaroo._player_step(self._env, state, action)


class NoJumpMod(JaxAtariInternalModPlugin):
    """
    Disables jumping. The jump controller is neutralised so a jump can never start
    (is_jumping stays False, y is left to the normal platform/gravity logic), which
    removes the player's ability to hop over low coconuts or jump up to ring the
    bell.

    Ladder climbing is untouched: it is handled by a separate controller that also
    reads UP, so the player can still climb to Joey -- the level stays winnable.
    Neutralising the jump controller (rather than stripping UP) is what keeps
    climbing working.
    """

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _player_jump_controller(self, state: KangarooState, jump_pressed: chex.Array, ladder_intersect: chex.Array):
        return (
            state.player.y,
            state.player.jump_counter,
            state.player.is_jumping,
            state.player.jump_base_y,
            state.player.landing_base_y,
            state.player.jump_orientation,
            state.player.cooldown_counter,
        )
