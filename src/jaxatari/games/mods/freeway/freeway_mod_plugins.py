import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.modification import JaxAtariPostStepModPlugin, JaxAtariInternalModPlugin
from jaxatari.games.jax_freeway import FreewayState


class StopAllCarsMod(JaxAtariPostStepModPlugin):
    """Stops all cars randomly with probability 0.4"""
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: FreewayState, new_state: FreewayState) -> FreewayState:
        """
        This function is called by the wrapper *after*
        the main step is complete.
        Access the environment via self._env (set by JaxAtariModWrapper).
        """
        key = jax.random.PRNGKey(new_state.time)
        chance = 0.4
        random_bool = jax.random.bernoulli(key, chance)
        
        new_cars = jax.lax.cond(
            random_bool,
            lambda _: prev_state.cars,  # Keep the previous cars' x positions if the random condition is met
            lambda _: new_state.cars,  # Otherwise, use the current cars' x positions
            operand=None
        )
        
        return new_state.replace(cars=new_cars)


class StaticCarsMod(JaxAtariPostStepModPlugin):
    """Stops all cars after spawning"""
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: FreewayState, new_state: FreewayState) -> FreewayState:
        """
        This function is called by the wrapper *after*
        the main step is complete.
        Access the environment via self._env (set by JaxAtariModWrapper).
        """
        # Always keep the previous cars' x positions
        return new_state.replace(cars=prev_state.cars)

class HallOfFameMod(JaxAtariInternalModPlugin):
    """
    Spawns the cars to make a "hall of fame" formation.
    """
    constants_overrides = {
        "lane_phase_offset": [
            -101, -101, -101, -101, -101, 
            35, 35, 35, 35, 35
            ]
    }

class SlowCarsMod(JaxAtariInternalModPlugin):
    """
    Halves the speed of all cars by making them move twice as far
    on each update, via multiplying their update step by 2.
    This is done by overriding the `CAR_UPDATES` constant.
    """
    constants_overrides = {
        "CAR_UPDATES": 
        [
            -10,  # Lane 0
            -8,   # Lane 1
            -6,   # Lane 2
            -4,   # Lane 3
            -2,   # Lane 4
            2,    # Lane 5
            4,    # Lane 6
            6,    # Lane 7
            8,    # Lane 8
            10,   # Lane 9
        ]
    }


class BlackCarsMod(JaxAtariInternalModPlugin):
    """Makes all cars black by overriding CAR_COLORS constant"""
    
    # Override CAR_COLORS to make all cars black
    # Using (0, 0, 0) for pure black. None means use original color.
    constants_overrides = {
        "CAR_COLORS": [
            (0, 0, 0),  # Lane 0 - black
            (0, 0, 0),  # Lane 1 - black
            (0, 0, 0),  # Lane 2 - black
            (0, 0, 0),  # Lane 3 - black
            (0, 0, 0),  # Lane 4 - black
            (0, 0, 0),  # Lane 5 - black
            (0, 0, 0),  # Lane 6 - black
            (0, 0, 0),  # Lane 7 - black
            (0, 0, 0),  # Lane 8 - black
            (0, 0, 0),  # Lane 9 - black
        ]
    }

class InvertSpeed(JaxAtariInternalModPlugin):
    """Inverts the speed of all cars by overriding CAR_UPDATES constant"""
    
    # Override CAR_UPDATES to invert the directions of all cars
    constants_overrides = {
        "CAR_UPDATES": [
            5,  # Lane 0
            4,  # Lane 1
            3,  # Lane 2
            2,  # Lane 3
            1,  # Lane 4
            -1,   # Lane 5
            -2,   # Lane 6
            -3,   # Lane 7
            -4,   # Lane 8
            -5,   # Lane 9
        ]
    }


class CenterCarsOnResetMod(JaxAtariPostStepModPlugin):
    """
    Positions all cars in the center of the screen when the environment resets.
    """
    
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: FreewayState):
        """
        Called after reset to modify initial state.
        Positions all cars in the center of the screen horizontally.
        """
        # Calculate center x position
        center_x = self._env.consts.screen_width // 2
        
        # Create new cars array with all cars at center x position
        # Keep the original y positions (lane positions)
        centered_cars = state.cars.at[:, 0].set(center_x)
        
        # Return modified observation and state
        modified_state = state.replace(cars=centered_cars)
        return obs, modified_state


import os
from jaxatari.rendering.jax_rendering_utils import JaxRenderingUtils, RendererConfig, get_base_sprite_dir

# Initialize utilities
_jr = JaxRenderingUtils(RendererConfig())
_bike_path = os.path.join(get_base_sprite_dir(), "freeway", "bike.npy")
_bike_array = _jr.loadFrame(_bike_path)

# Define distinct color pairs for 10 lanes (Biker, Motorbike)
_color_pairs = [
    ((255, 0, 0), (0, 0, 255)),       # Lane 0: Red / Blue
    ((0, 255, 0), (255, 255, 0)),     # Lane 1: Green / Yellow
    ((255, 0, 255), (0, 255, 255)),   # Lane 2: Magenta / Cyan
    ((255, 128, 0), (128, 0, 255)),   # Lane 3: Orange / Purple
    ((255, 255, 255), (0, 0, 0)),     # Lane 4: White / Black
    ((0, 0, 255), (255, 0, 0)),       # Lane 5: Blue / Red
    ((255, 255, 0), (0, 255, 0)),     # Lane 6: Yellow / Green
    ((0, 255, 255), (255, 0, 255)),   # Lane 7: Cyan / Magenta
    ((128, 0, 255), (255, 128, 0)),   # Lane 8: Purple / Orange
    ((0, 0, 0), (255, 255, 255)),     # Lane 9: Black / White
]

_recolored_bikes = []
for _biker_color, _motorbike_color in _color_pairs:
    _rule = [
        {'source': (80, 184, 57), 'target': _biker_color},
        {'source': (32, 167, 32), 'target': _biker_color},
        {'source': (234, 61, 49), 'target': _motorbike_color},
        {'source': (255, 32, 32), 'target': _motorbike_color}
    ]
    _recolored_bikes.append(_jr.perform_recoloring(_bike_array, _rule))


class FrogMod(JaxAtariInternalModPlugin):
    """Replaces the player sprites with frog sprites."""
    asset_overrides = {
        "player": {
            'name': 'player', 'type': 'group',
            'files': ['frog_hit.npy', 'frog_walk.npy', 'frog_idle.npy']
        }
    }


class BikesMod(JaxAtariInternalModPlugin):
    """Replaces all cars with uniquely colored bike sprites."""
    asset_overrides = {
        'car_dark_red': {'name': 'car_dark_red', 'type': 'procedural', 'data': _recolored_bikes[0]},
        'car_light_green': {'name': 'car_light_green', 'type': 'procedural', 'data': _recolored_bikes[1]},
        'car_dark_green': {'name': 'car_dark_green', 'type': 'procedural', 'data': _recolored_bikes[2]},
        'car_light_red': {'name': 'car_light_red', 'type': 'procedural', 'data': _recolored_bikes[3]},
        'car_blue': {'name': 'car_blue', 'type': 'procedural', 'data': _recolored_bikes[4]},
        'car_brown': {'name': 'car_brown', 'type': 'procedural', 'data': _recolored_bikes[5]},
        'car_light_blue': {'name': 'car_light_blue', 'type': 'procedural', 'data': _recolored_bikes[6]},
        'car_red': {'name': 'car_red', 'type': 'procedural', 'data': _recolored_bikes[7]},
        'car_green': {'name': 'car_green', 'type': 'procedural', 'data': _recolored_bikes[8]},
        'car_yellow': {'name': 'car_yellow', 'type': 'procedural', 'data': _recolored_bikes[9]},
    }


_bg_path = os.path.join(get_base_sprite_dir(), "freeway", "background.npy")
_bg_array = _jr.loadFrame(_bg_path)

_lane_color_rule = [
    {'source': (214, 214, 214), 'target': (0, 0, 0)},       # Lane separation black
    {'source': (252, 252, 84), 'target': (255, 0, 0)}       # Double lane separation red
]
_recolored_bg = _jr.perform_recoloring(_bg_array, _lane_color_rule)

class NewLaneColorsMod(JaxAtariInternalModPlugin):
    """Makes the lane separation black and the double lane separation red."""
    asset_overrides = {
        'background': {
            'name': 'background',
            'type': 'background',
            'data': _recolored_bg
        }
    }

_score_paths = [os.path.join(get_base_sprite_dir(), "freeway", f"score_{i}.npy") for i in range(10)]
_score_array = _jr._load_and_pad_digits_from_paths(_score_paths)
_green_score_rule = [{'source': (228, 111, 111), 'target': (0, 255, 0)}]
_recolored_score = _jr.perform_recoloring(_score_array, _green_score_rule)

class GreenScoreMod(JaxAtariInternalModPlugin):
    """Makes the score digits green."""
    asset_overrides = {
        'score_digits': {
            'name': 'score_digits',
            'type': 'digits',
            'data': _recolored_score
        }
    }


# --- "change_X_color" visual mods (parallels to the Pong color mods) ---------
# Source colors baked into the base sprites.
_PLAYER_SRC = (252, 252, 84)     # chicken yellow
_ROAD_SRC = (142, 142, 142)      # main road grey
_ROAD_SRC_LIGHT = (170, 170, 170)  # lighter road/curb grey
_SCORE_SRC = (228, 111, 111)     # default score salmon

# New colors (tweak here). Each mod recolors only its own element.
_NEW_PLAYER_COLOR = (30, 144, 255)   # dodger blue
_NEW_CAR_COLOR = (200, 60, 200)      # magenta (all cars)
_NEW_ROAD_COLOR = (60, 95, 80)       # dark teal
_NEW_SCORE_COLOR = (0, 220, 220)     # cyan


def _lighter(rgb, factor=1.35):
    return tuple(min(255, int(round(c * factor))) for c in rgb)


# Player is a 3-sprite group; recolor each (order matches the default asset config).
_player_files = ['player_hit.npy', 'player_walk.npy', 'player_idle.npy']
_recolored_player = [
    _jr.perform_recoloring(
        _jr.loadFrame(os.path.join(get_base_sprite_dir(), "freeway", _f)),
        [{'source': _PLAYER_SRC, 'target': _NEW_PLAYER_COLOR}],
    )
    for _f in _player_files
]


class ChangePlayerColorMod(JaxAtariInternalModPlugin):
    """Changes the player (chicken) color. Default: dodger blue."""
    asset_overrides = {
        'player': {'name': 'player', 'type': 'group', 'data': _recolored_player}
    }


class ChangeCarColorMod(JaxAtariInternalModPlugin):
    """Changes every car to a single color via CAR_COLORS. Default: magenta."""
    constants_overrides = {"CAR_COLORS": [_NEW_CAR_COLOR] * 10}


_recolored_road_bg = _jr.perform_recoloring(
    _bg_array,
    [
        {'source': _ROAD_SRC, 'target': _NEW_ROAD_COLOR},
        {'source': _ROAD_SRC_LIGHT, 'target': _lighter(_NEW_ROAD_COLOR)},
    ],
)


class ChangeRoadColorMod(JaxAtariInternalModPlugin):
    """Changes the road surface color (both grey tones). Default: dark teal."""
    asset_overrides = {
        'background': {'name': 'background', 'type': 'background', 'data': _recolored_road_bg}
    }


_recolored_score_new = _jr.perform_recoloring(
    _score_array, [{'source': _SCORE_SRC, 'target': _NEW_SCORE_COLOR}]
)


class ChangeScoreColorMod(JaxAtariInternalModPlugin):
    """Changes the (default) score digit color. Default: cyan. End-game blink
    colors are applied separately by the renderer and are unaffected."""
    asset_overrides = {
        'score_digits': {'name': 'score_digits', 'type': 'digits', 'data': _recolored_score_new}
    }


# --- Dynamics mods -----------------------------------------------------------
class ChangeCarSpeedMod(JaxAtariInternalModPlugin):
    """
    Makes all cars faster by roughly halving their movement periods.

    In Freeway |CAR_UPDATES[i]| is the number of frames between 1px moves for
    lane i (smaller = faster; sign = direction). Halving the periods (min 1,
    which is the 1px/frame cap) about doubles each lane's speed while keeping its
    direction and the fast/slow gradient. Counterpart to the existing slow_cars.
    """
    constants_overrides = {
        "CAR_UPDATES": [-3, -2, -2, -1, -1, 1, 1, 2, 2, 3]  # default: -5..-1, 1..5
    }


class FasterPlayerMod(JaxAtariPostStepModPlugin):
    """
    Makes the chicken move faster vertically (default: 2x).

    The base game moves the chicken +/-1 px per frame, hardcoded inside step(),
    so there is no constant to override. Instead this post-step mod amplifies
    each *voluntary* move (cooldown == 0, i.e. not stunned / thrown back / in the
    post-score freeze) by (_SPEED - 1) extra pixels. A 1px overshoot is not
    collision-checked until the next frame, but cars are 10px tall so the chicken
    cannot tunnel through one at these speeds.
    """
    _SPEED = 2  # pixels per voluntary move (base is 1)

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: FreewayState, new_state: FreewayState) -> FreewayState:
        c = self._env.consts
        dy = new_state.chicken_y - prev_state.chicken_y
        voluntary = new_state.cooldown == 0
        extra = jnp.where(voluntary, dy * (self._SPEED - 1), 0)
        new_y = jnp.clip(
            new_state.chicken_y + extra,
            c.top_border,
            c.bottom_border + c.chicken_height - 1,
        ).astype(new_state.chicken_y.dtype)
        return new_state.replace(chicken_y=new_y)


class SlowerPlayerMod(JaxAtariPostStepModPlugin):
    """
    Makes the chicken move slower vertically (half speed): it advances only on
    every other frame of a held move, holding position on the rest.

    Same post-step approach as FasterPlayerMod. A voluntary move is reverted
    (position held at the previous frame's value) when walking_frames is even, so
    it advances on the odd frames (1, 3, 5, 7) -- the first frame of any move
    (walking_frames == 1) always advances, so single taps still register.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: FreewayState, new_state: FreewayState) -> FreewayState:
        voluntary = new_state.cooldown == 0
        skip = (new_state.walking_frames % 2) == 0
        new_y = jnp.where(voluntary & skip, prev_state.chicken_y, new_state.chicken_y)
        return new_state.replace(chicken_y=new_y.astype(new_state.chicken_y.dtype))


class ChangeCarSpawningMod(JaxAtariInternalModPlugin):
    """
    Changes the car spawn layout: instead of the default (all cars clustered at
    the right edge), spread them evenly across the road so the chicken faces
    distributed traffic from the start. Done via lane_phase_offset, which is
    added to each lane's base spawn x at reset.
    """
    # base spawn x is 152 for lanes 0-4 and 0 for lanes 5-9; these offsets place
    # the cars at evenly spaced x = 8, 24, 40, ... 152 across the screen width.
    constants_overrides = {
        "lane_phase_offset": [-144, -128, -112, -96, -80, 88, 104, 120, 136, 152]
    }


class InvertCarsMod(InvertSpeed):
    """
    Inverts every car's travel direction while preserving each lane's speed
    (negates CAR_UPDATES): lanes 0-4 now move right and lanes 5-9 move left.
    """
    pass


# --- Reward mods -------------------------------------------------------------
def _lanes_crossed(chicken_y, lane_borders):
    """Number of lanes the chicken has crossed (0 at the bottom, num_lanes at the
    top). Moving up decreases y, so more borders satisfy y < border."""
    return jnp.sum(chicken_y < lane_borders).astype(jnp.int32)


class RewardPerLaneMod(JaxAtariInternalModPlugin):
    """
    Dense progress reward: +_PER_LANE for each lane advanced upward (and
    -_PER_LANE for each lane lost, e.g. when thrown back), replacing the sparse
    score reward. Potential-based, so bobbing up and down nets zero -- only net
    progress is rewarded. On the scoring frame the chicken teleports back to the
    bottom, so the final lane is credited directly instead of the reset delta.
    """
    _PER_LANE = 1

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: FreewayState, state: FreewayState):
        borders = self._env.consts.lane_borders
        delta = _lanes_crossed(state.chicken_y, borders) - _lanes_crossed(previous_state.chicken_y, borders)
        scored = state.score > previous_state.score
        return jnp.where(scored, self._PER_LANE, delta * self._PER_LANE).astype(jnp.int32)


class CollisionPenaltyMod(JaxAtariInternalModPlugin):
    """
    Adds a penalty of -_PENALTY each time the chicken is hit by a car, on top of
    the normal +1 for reaching the top. A fresh hit is the frame the cooldown
    jumps to (throw_back_frames + stun_frames) from 0.
    """
    _PENALTY = 1

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: FreewayState, state: FreewayState):
        c = self._env.consts
        hit_cooldown = c.throw_back_frames + c.stun_frames
        collided = jnp.logical_and(previous_state.cooldown == 0, state.cooldown == hit_cooldown)
        base = state.score - previous_state.score
        return (base - collided.astype(jnp.int32) * self._PENALTY).astype(jnp.int32)


class RewardMiddleLaneMod(JaxAtariInternalModPlugin):
    """
    Adds a milestone bonus of +_BONUS for crossing into the upper half of the
    road (reaching the middle lane), on top of the normal +1 for reaching the
    top. Potential-based across the middle line, so bobbing across it nets zero.
    """
    _BONUS = 1

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: FreewayState, state: FreewayState):
        borders = self._env.consts.lane_borders
        middle = self._env.consts.num_lanes // 2
        above_new = _lanes_crossed(state.chicken_y, borders) >= middle
        above_prev = _lanes_crossed(previous_state.chicken_y, borders) >= middle
        scored = state.score > previous_state.score
        base = state.score - previous_state.score
        # ignore the score-reset teleport (chicken drops below the middle again)
        middle_term = jnp.where(
            scored, 0, (above_new.astype(jnp.int32) - above_prev.astype(jnp.int32)) * self._BONUS
        )
        return (base + middle_term).astype(jnp.int32)


class CleanCrossingMod(JaxAtariInternalModPlugin):
    """
    Rewards +1 for reaching the top only if the chicken was not hit at any point
    during that crossing; a crossing where the chicken was hit gives 0. Reads the
    base game's hit_since_reset flag (accumulated over the crossing, cleared on
    score), so it is all-or-nothing per crossing.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: FreewayState, state: FreewayState):
        scored = state.score > previous_state.score
        clean = jnp.logical_not(previous_state.hit_since_reset)
        return jnp.where(jnp.logical_and(scored, clean), 1, 0).astype(jnp.int32)
