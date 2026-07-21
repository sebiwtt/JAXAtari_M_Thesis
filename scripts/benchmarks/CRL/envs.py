# =============================================================================
# Environment factory for the JAXtari CRL benchmark
# =============================================================================
# Returns a thunk (zero-arg closure) building one fully wrapped env, later
# vmapped over NUM_ENVS by the trainer / eval.

import jaxatari
from jaxatari.wrappers import (
    AtariWrapper,
    FlattenObservationWrapper,
    LogWrapper,
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
)


def make_env(env_id, seed, num_envs, mods=[], pixel_based=True, native_downscaling=True, smooth_image=True, grayscale=False, eval=False):
    def thunk():
        active_mods = mods
        if not eval and isinstance(active_mods, (list, tuple)) and len(active_mods) > 1:
            active_mods = []

        # jaxatari.make expects None (no mods) or a non-empty list.
        if isinstance(active_mods, (list, tuple)) and len(active_mods) == 0:
            mods_arg = None
        else:
            mods_arg = active_mods

        env = jaxatari.make(env_id, mods=mods_arg)

        # episodic_life/clip_reward are train-only tricks; eval sees true boundaries/reward.
        env = AtariWrapper(
                env,
                sticky_actions=0.0,
                episodic_life=not eval,
                first_fire=True,
                noop_max=30,
                full_action_space=False,
        )
        if pixel_based:
            env = PixelObsWrapper(
                env,
                do_pixel_resize=True,
                pixel_resize_shape=(84, 84),
                grayscale=grayscale,
                use_native_downscaling=native_downscaling,
                smooth_image=smooth_image,
                frame_stack_size=4,
                frame_skip=4,
                max_pooling=True,
                clip_reward=not eval,
            )
        else:
            env = FlattenObservationWrapper(
                NormalizeObservationWrapper(
                    ObjectCentricWrapper(
                        env,
                        frame_stack_size=4,
                        frame_skip=4,
                        clip_reward=not eval,
                    )
                )
            )
        env = LogWrapper(env)
        env.num_envs = num_envs
        env.single_action_space = env.action_space
        env.single_observation_space = env.observation_space
        env.is_vector_env = True
        return env
    return thunk
