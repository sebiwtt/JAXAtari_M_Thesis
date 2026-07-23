import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.freeway.freeway_mod_plugins import StopAllCarsMod, StaticCarsMod, SlowCarsMod, BlackCarsMod, CenterCarsOnResetMod, HallOfFameMod, BikesMod, FrogMod, NewLaneColorsMod, GreenScoreMod, ChangePlayerColorMod, ChangeCarColorMod, ChangeRoadColorMod, ChangeScoreColorMod, ChangeCarSpeedMod, FasterPlayerMod, SlowerPlayerMod, ChangeCarSpawningMod, InvertCarsMod, CarSpeedX2Mod, CarSpeedX3Mod, CarSpeedX4Mod, CarSpeedX5Mod, RewardPerLaneMod, CollisionPenaltyMod, RewardMiddleLaneMod, CleanCrossingMod

class FreewayEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Freeway.
    It simply inherits all logic from JaxAtariModController and defines the REGISTRY.
    """

    REGISTRY = {
        # Visual
        "change_player_color": ChangePlayerColorMod,
        "change_car_color": ChangeCarColorMod,
        "change_road_color": ChangeRoadColorMod,
        "change_score_color": ChangeScoreColorMod,
        #Vis. mods that were already available
        "black_cars": BlackCarsMod,
        "bikes": BikesMod,
        "frog": FrogMod,
        "new_lane_colors": NewLaneColorsMod,
        "green_score": GreenScoreMod,
        "change_sprites": ["frog", "bikes", "new_lane_colors", "green_score"],
    
        # Dynamic
        "change_car_speed": ChangeCarSpeedMod,
        "faster_player": FasterPlayerMod,
        "slower_player": SlowerPlayerMod,
        "change_car_spawning": ChangeCarSpawningMod,
        "invert_cars": InvertCarsMod,

        # Magnitude sequence (car_speed_xN): same mod, incrementally faster cars
        "car_speed_x2": CarSpeedX2Mod,
        "car_speed_x3": CarSpeedX3Mod,
        "car_speed_x4": CarSpeedX4Mod,
        "car_speed_x5": CarSpeedX5Mod,

        # Dyn. mods that were already available
        "stop_all_cars": StopAllCarsMod,
        "static_cars": StaticCarsMod,
        "slow_cars": SlowCarsMod,
        "center_cars_on_reset": CenterCarsOnResetMod,
        "hall_of_fame": ["_hall_of_fame_start", "static_cars"],
        "_hall_of_fame_start": HallOfFameMod,

        # Reward
        "reward_per_lane": RewardPerLaneMod,
        "collision_penalty": CollisionPenaltyMod,
        "reward_middle_lane": RewardMiddleLaneMod,
        "clean_crossing": CleanCrossingMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "freeway", "sprites")

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
