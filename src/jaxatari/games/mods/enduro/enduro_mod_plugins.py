import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.modification import JaxAtariInternalModPlugin

class StartInCurveMod(JaxAtariInternalModPlugin):
    """
    Starts the game directly in the curve by setting straight_km_start to 0.
    """
    constants_overrides = {
        "straight_km_start": 0.0
    }

class StartInMaxCurveMod(JaxAtariInternalModPlugin):
    """
    Starts the game in a full curve.
    """
    constants_overrides = {
        "straight_km_start": 0.0,
        "initial_track_top_x_curve_offset": 50.0
    }

class FilledRoadMod(JaxAtariInternalModPlugin):
    """
    Displays the full filled road instead of only the side.
    """
    constants_overrides = {
        "render_full_road": True
    }

class SnowWeatherMod(JaxAtariInternalModPlugin):
    """
    Forces the weather to snow.
    """
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = None):
        obs, state = self._env.__class__.reset(self._env, key)
        state = state.replace(weather_index=jnp.array(self._env.consts.snow_weather_index, dtype=jnp.int32))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        # Force snow before step so steering logic uses it
        state = state.replace(weather_index=jnp.array(self._env.consts.snow_weather_index, dtype=jnp.int32))
        obs, state, reward, done, info = self._env.__class__.step(self._env, state, action)
        # Force snow after step in case it was changed
        state = state.replace(weather_index=jnp.array(self._env.consts.snow_weather_index, dtype=jnp.int32))
        return obs, state, reward, done, info

class NightWeatherMod(JaxAtariInternalModPlugin):
    """
    Forces the weather to night.
    """
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = None):
        obs, state = self._env.__class__.reset(self._env, key)
        state = state.replace(weather_index=jnp.array(self._env.consts.night_weather_index, dtype=jnp.int32))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        # Force night before step
        state = state.replace(weather_index=jnp.array(self._env.consts.night_weather_index, dtype=jnp.int32))
        obs, state, reward, done, info = self._env.__class__.step(self._env, state, action)
        # Force night after step in case it was changed
        state = state.replace(weather_index=jnp.array(self._env.consts.night_weather_index, dtype=jnp.int32))
        return obs, state, reward, done, info

class FogWeatherMod(JaxAtariInternalModPlugin):
    """
    Forces the weather to fog.
    """
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = None):
        obs, state = self._env.__class__.reset(self._env, key)
        state = state.replace(weather_index=jnp.array(self._env.consts.fog_weather_index, dtype=jnp.int32))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        # Force fog before step
        state = state.replace(weather_index=jnp.array(self._env.consts.fog_weather_index, dtype=jnp.int32))
        obs, state, reward, done, info = self._env.__class__.step(self._env, state, action)
        # Force fog after step in case it was changed
        state = state.replace(weather_index=jnp.array(self._env.consts.fog_weather_index, dtype=jnp.int32))
        return obs, state, reward, done, info

class DayWeatherMod(JaxAtariInternalModPlugin):
    """
    Forces the weather to day.
    """
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = None):
        obs, state = self._env.__class__.reset(self._env, key)
        state = state.replace(weather_index=jnp.array(self._env.consts.day_weather_index, dtype=jnp.int32))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        state = state.replace(weather_index=jnp.array(self._env.consts.day_weather_index, dtype=jnp.int32))
        obs, state, reward, done, info = self._env.__class__.step(self._env, state, action)
        state = state.replace(weather_index=jnp.array(self._env.consts.day_weather_index, dtype=jnp.int32))
        return obs, state, reward, done, info

class SunsetWeatherMod(JaxAtariInternalModPlugin):
    """
    Forces the weather to sunset.
    """
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = None):
        obs, state = self._env.__class__.reset(self._env, key)
        state = state.replace(weather_index=jnp.array(self._env.consts.sunset_weather_index, dtype=jnp.int32))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        state = state.replace(weather_index=jnp.array(self._env.consts.sunset_weather_index, dtype=jnp.int32))
        obs, state, reward, done, info = self._env.__class__.step(self._env, state, action)
        state = state.replace(weather_index=jnp.array(self._env.consts.sunset_weather_index, dtype=jnp.int32))
        return obs, state, reward, done, info

class DawnWeatherMod(JaxAtariInternalModPlugin):
    """
    Forces the weather to dawn.
    """
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = None):
        obs, state = self._env.__class__.reset(self._env, key)
        state = state.replace(weather_index=jnp.array(self._env.consts.dawn_weather_index, dtype=jnp.int32))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        state = state.replace(weather_index=jnp.array(self._env.consts.dawn_weather_index, dtype=jnp.int32))
        obs, state, reward, done, info = self._env.__class__.step(self._env, state, action)
        state = state.replace(weather_index=jnp.array(self._env.consts.dawn_weather_index, dtype=jnp.int32))
        return obs, state, reward, done, info

class SpeedAndXPosHudMod(JaxAtariInternalModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def _render_cars_to_pass(self, raster: jnp.ndarray, state) -> jnp.ndarray:
        # Call original renderer method
        from jaxatari.games.jax_enduro import EnduroRenderer
        raster = EnduroRenderer._render_cars_to_pass(self._env.renderer, raster, state)
        
        # Now add speed and X position HUD at the top
        digit_sprites = self._env.renderer.SHAPE_MASKS['digits_black']
        spacing = digit_sprites.shape[2] + 2
        
        # 1. Speed (top left)
        speed = state.player_speed.astype(jnp.int32)
        s_hundreds = (speed // 100) % 10
        s_tens = (speed // 10) % 10
        s_ones = speed % 10
        
        raster = jax.lax.cond(
            speed >= 100,
            lambda r: self._env.renderer.jr.render_at(r, 10, 5, digit_sprites[s_hundreds]),
            lambda r: r,
            raster
        )
        
        raster = jax.lax.cond(
            speed >= 10,
            lambda r: self._env.renderer.jr.render_at(r, 10 + spacing, 5, digit_sprites[s_tens]),
            lambda r: r,
            raster
        )
        
        raster = self._env.renderer.jr.render_at(raster, 10 + 2 * spacing, 5, digit_sprites[s_ones])
        
        # 2. X Position (top right)
        x_pos = state.player_x.astype(jnp.int32)
        x_hundreds = (x_pos // 100) % 10
        x_tens = (x_pos // 10) % 10
        x_ones = x_pos % 10
        
        start_x = 120
        
        raster = jax.lax.cond(
            x_pos >= 100,
            lambda r: self._env.renderer.jr.render_at(r, start_x, 5, digit_sprites[x_hundreds]),
            lambda r: r,
            raster
        )
        
        raster = jax.lax.cond(
            x_pos >= 10,
            lambda r: self._env.renderer.jr.render_at(r, start_x + spacing, 5, digit_sprites[x_tens]),
            lambda r: r,
            raster
        )
        
        raster = self._env.renderer.jr.render_at(raster, start_x + 2 * spacing, 5, digit_sprites[x_ones])
        
        return raster

class ShortDaysMod(JaxAtariInternalModPlugin):
    """
    Iterates the weather state every 16km driven instead of based on time.
    Also scales the number of cars to pass (divided by 10).
    """
    constants_overrides = {
        "weather_cycle_distance": 16.0,
        "initial_position": 20,
        "next_day_car_position": 30
    }

class NoOpponentsMod(JaxAtariInternalModPlugin):
    """
    Disables all opponent cars.
    """
    constants_overrides = {
        "opponent_density": 0.0,
        "opponent_density_increment": 0.0
    }


# --- CRL dyn4 pattern mods (traffic density/speed, curve physics, acceleration)
class DenseTrafficMod(JaxAtariInternalModPlugin):
    """Doubles opponent car density (0.17 -> 0.34) and raises the per-level
    density cap so traffic stays dense on later levels (spawn-density axis)."""
    constants_overrides = {
        "opponent_density": 0.34,
        "max_opponent_density": 0.5,
    }


class FasterOpponentsMod(JaxAtariInternalModPlugin):
    """Doubles opponent car speed (2 -> 4) and raises the per-level speed cap
    (5 -> 8) so the difference persists across levels. Player max speed (9)
    still exceeds it, so passing cars stays possible."""
    constants_overrides = {
        "opponent_speed": 4,
        "max_opponent_speed": 8.0,
    }


class SharpCurvesMod(JaxAtariInternalModPlugin):
    """Triples how hard curves pull the car sideways (curve_rate 0.05 -> 0.15);
    the physics counterpart to the start_in_curve layout mods."""
    constants_overrides = {"curve_rate": 0.15}


class SluggishAccelerationMod(JaxAtariInternalModPlugin):
    """Car takes twice as long to gain each speed phase (accel_intervals
    doubled); braking is unchanged, so recovering from collisions is costly."""
    constants_overrides = {
        "accel_intervals": jnp.array([30, 60, 60, 60, 60, 80, 120, 120, 120], dtype=jnp.int32),
    }
