from jaxatari.modification import JaxAtariInternalModPlugin


class RapidFireMod(JaxAtariInternalModPlugin):
    """Increase active bullet caps for ship, saucer, and enemies to 4."""

    constants_overrides = {
        "MAX_ACTIVE_PLAYER_BULLETS_MAP": 4,
        "MAX_ACTIVE_PLAYER_BULLETS_LEVEL": 4,
        "MAX_ACTIVE_PLAYER_BULLETS_ARENA": 4,
        "MAX_ACTIVE_SAUCER_BULLETS": 4,
        "MAX_ACTIVE_ENEMY_BULLETS": 8,  # 2 bullets per enemy * 4 enemies
    }


class ZeroGravityMod(JaxAtariInternalModPlugin):
    """Disable all gravity from sun, planets, and reactors."""

    constants_overrides = {
        "SOLAR_GRAVITY": 0.0,
        "PLANETARY_GRAVITY": 0.0,
        "REACTOR_GRAVITY": 0.0,
    }


class HyperGravityMod(JaxAtariInternalModPlugin):
    """Increase all gravity from sun, planets, and reactors substantially."""

    constants_overrides = {
        "SOLAR_GRAVITY": 0.132,  # 0.044 * 3
        "PLANETARY_GRAVITY": 0.0096,  # 0.0032 * 3
        "REACTOR_GRAVITY": 0.001,  # 0.0001 * 10
    }


class FuelCrisisMod(JaxAtariInternalModPlugin):
    """Increase fuel consumption rate by 5x."""

    constants_overrides = {
        "FUEL_CONSUME_THRUST": 20.0,
        "FUEL_CONSUME_SHIELD_TRACTOR": 50.0,
    }

class HarmlessEnemiesMod(JaxAtariInternalModPlugin):
    """Make all enemies harmless by disabling their bullets."""

    constants_overrides = {
        "MAX_ACTIVE_ENEMY_BULLETS": 0,
        "MAX_ACTIVE_SAUCER_BULLETS": 0,
    }


class ValuableReactorMod(JaxAtariInternalModPlugin):
    """Populate reactor level with 3 enemies and 2 fuel tanks."""

    constants_overrides = {
        "ALLOW_TRACTOR_IN_REACTOR": True,
        "REACTOR_LEVEL_LAYOUT": (
            {"type": 5, "coords": (104, 104)},   # ENEMY_ORANGE
            {"type": 6, "coords": (56, 144)},   # ENEMY_GREEN
            {"type": 39, "coords": (80, 18)},  # ENEMY_ORANGE_FLIPPED
            {"type": 12, "coords": (66, 88)}, # FUEL_TANK
        ),
    }


class AntiGravityMod(JaxAtariInternalModPlugin):
    """Reverse gravity from sun, planets, and reactors."""

    constants_overrides = {
        "SOLAR_GRAVITY": -0.005,
        "PLANETARY_GRAVITY": -0.0032,
        "REACTOR_GRAVITY": -0.0001,
    }


class HighSpeedMod(JaxAtariInternalModPlugin):
    """Make the ship faster by increasing thrust power and max speed."""

    constants_overrides = {
        "THRUST_POWER": 0.075,
        "MAX_SPEED": 6.0,
    }


class InfiniteFuelMod(JaxAtariInternalModPlugin):
    """Disable fuel consumption."""

    constants_overrides = {
        "FUEL_CONSUME_THRUST": 0.0,
        "FUEL_CONSUME_SHIELD_TRACTOR": 0.0,
    }


class SlowEnemiesMod(JaxAtariInternalModPlugin):
    """Decrease the movement speed of saucers and bullets."""

    constants_overrides = {
        "SAUCER_SPEED_MAP": 0.09,
        "SAUCER_SPEED_ARENA": 0.18,
        "SAUCER_BULLET_SPEED": 1.0,
        "ENEMY_BULLET_SPEED": 0.65,
    }


class LongRangeTractorMod(JaxAtariInternalModPlugin):
    """Increase the range of the tractor beam."""

    constants_overrides = {
        "TRACTOR_BEAM_RANGE": 50.0,
    }

class NeonMod(JaxAtariInternalModPlugin):
    """
    Changes colors to bright neon variants.
    """
    constants_overrides = {
        "RECOLOR_RULES": (
            {"source": (101, 183, 217), "target": (255, 20, 147)},
            {"source": (198, 108, 58), "target": (0, 255, 0)},
            {"source": (72, 160, 72), "target": (255, 255, 0)},
            {"source": (223, 183, 85), "target": (0, 255, 255)},
        )
    }

class RedAlertMod(JaxAtariInternalModPlugin):
    """
    Makes all terrain red/orange for a high-alert aesthetic.
    """
    constants_overrides = {
        "RECOLOR_RULES": (
            {"source": (223, 183, 85), "target": (255, 50, 50)},
            {"source": (84, 160, 197), "target": (220, 40, 40)},
            {"source": (66, 72, 200), "target": (200, 30, 30)},
            {"source": (213, 130, 74), "target": (255, 0, 0)},
        )
    }

class GrayscaleMod(JaxAtariInternalModPlugin):
    """
    Converts the visual palette to grayscale.
    """
    constants_overrides = {
        "RECOLOR_RULES": (
            {"source": (223, 183, 85), "target": (150, 150, 150)},
            {"source": (84, 160, 197), "target": (120, 120, 120)},
            {"source": (66, 72, 200), "target": (80, 80, 80)},
            {"source": (228, 111, 111), "target": (140, 140, 140)},
            {"source": (213, 130, 74), "target": (160, 160, 160)},
            {"source": (101, 183, 217), "target": (220, 220, 220)},
            {"source": (198, 108, 58), "target": (110, 110, 110)},
            {"source": (72, 160, 72), "target": (90, 90, 90)},
        )
    }

class InvertedColorsMod(JaxAtariInternalModPlugin):
    """
    Inverts the primary colors.
    """
    constants_overrides = {
        "RECOLOR_RULES": (
            {"source": (101, 183, 217), "target": (154, 72, 38)}, 
            {"source": (223, 183, 85), "target": (32, 72, 170)},  
            {"source": (84, 160, 197), "target": (171, 95, 58)},
        )
    }


# --- CRL dyn4 pattern mods (enemy speed, ship handling, fire rate, spawn) ----
# Note: jax_gravitar freezes many combat constants at module level
# (_DEFAULT_CONSTS at import time), so constants_overrides silently do nothing
# for those (e.g. SAUCER_SPEED_*, SAUCER_SPAWN/RESPAWN delays,
# SAUCER_FIRE_INTERVAL_FRAMES). Mods below therefore either use constants that
# ARE read via self.consts at runtime, or patch the state post-step.
from functools import partial
import jax
import jax.numpy as jnp
from jaxatari.modification import JaxAtariPostStepModPlugin


class FastSaucersMod(JaxAtariPostStepModPlugin):
    """Doubles saucer movement speed (counterpart to slow_enemies).

    The saucer speed constants are module-frozen (see note above), so this
    post-step mod amplifies the pixel step the base game took this frame
    (seaquest faster_enemies approach): each genuine per-frame move (saucer
    alive on both frames, small delta) is doubled. Spawns/despawns (large
    position jumps) are left alone; base speeds are 0.18/0.36 px per frame, so
    the 2.0 threshold cleanly separates the two."""
    _MULT = 2.0

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        prev_s, new_s = prev_state.saucer, new_state.saucer
        dx = new_s.x - prev_s.x
        dy = new_s.y - prev_s.y
        genuine = prev_s.alive & new_s.alive & (jnp.abs(dx) <= 2.0) & (jnp.abs(dy) <= 2.0)
        scale = self._MULT - 1.0
        new_saucer = new_s.replace(
            x=jnp.where(genuine, new_s.x + dx * scale, new_s.x),
            y=jnp.where(genuine, new_s.y + dy * scale, new_s.y),
        )
        return new_state.replace(saucer=new_saucer)


class SluggishShipMod(JaxAtariInternalModPlugin):
    """Halves ship thrust power and max speed (counterpart to high_speed):
    escaping gravity wells takes far more deliberate thrusting."""
    constants_overrides = {
        "THRUST_POWER": 0.015,
        "MAX_SPEED": 1.25,
    }


class RapidEnemyFireMod(JaxAtariInternalModPlugin):
    """Bunker enemies on planet levels fire twice as often (cooldown 10 -> 5).
    The saucer's fire interval is module-frozen (see note above) and stays at
    its default."""
    constants_overrides = {
        "ENEMY_FIRE_COOLDOWN_FRAMES": 5,
    }


class EarlySaucerMod(JaxAtariPostStepModPlugin):
    """Saucer harassment starts almost immediately and returns quickly: the
    spawn/respawn countdown is capped at 50 frames (base: 200 initial, 540
    after a kill). The delay constants are module-frozen (see note above), so
    the state timer is clamped post-step instead; the ~alive guard in the base
    spawn logic keeps the while-alive sentinel harmless."""
    _CAP = 50

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return new_state.replace(
            saucer_spawn_timer=jnp.minimum(new_state.saucer_spawn_timer, self._CAP)
        )
