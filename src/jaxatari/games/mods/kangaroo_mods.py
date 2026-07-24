import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.kangaroo.kangaroo_mod_plugins import (
    NoMonkeyMod, NoFallingCoconutMod, NoThrownCoconutMod, NoBellMod, NoFruitMod,
    AlwaysHighCoconutMod, PinChildMod, RenderDebugInfo, ReplaceChildWithMonkeyMod, ReplaceBellWithCactusMod,
    ReplaceBellWithFlameMod, ReplaceLadderWithRopeMod, ReplaceLadderWithChainMod, ReplaceMonkeyWithTankMod,
    LethalFlameMod, SpawnOnSecondFloorMod, FlameTrapMod, CenterLaddersMod, InvertLaddersMod,
    FirstLevelOnlyMod, SecondLevelOnlyMod, ThirdLevelOnlyMod, FourLaddersMod, ReplaceCoconutWithFireball,
    ReplaceCoconutWithHoneyBee, ReplaceCoconutWithWasp, ReplaceMonkeyWithChickenMod, ReplaceMonkeyWithDragonMod,
    ReplaceMonkeyWithDangerSignMod, ReplaceMonkeyWithPolarbearMod, ReplaceMonkeyWithSnakeMod, ReplaceBellWithDangerSignMod,
    ReplaceFruitWithCoin, ReplaceFruitWithDiamond,
    ChangeKangarooSpeedMod, ChangeMonkeySpeedMod, RandomizeCoconutsMod, JumpGravityMod,
    ChangeKangarooColorMod, ChangeMonkeyColorMod, ChangeBackgroundColorMod,
    ChangeScoreColorMod, GrayscaleThemeMod,
    LifeLossPenaltyMod, RewardPerFloorMod, ReachJoeyOnlyMod, FruitOnlyMod, PunchOnlyScoringMod,
    SurvivalRewardMod,
    CoconutSpeedX2Mod, CoconutSpeedX3Mod, CoconutSpeedX4Mod, CoconutSpeedX5Mod,
    NoPunchMod, NoCrouchMod, NoJumpMod,
)

# --- 3. The Registry ---
KANGAROO_MOD_REGISTRY = {
    # ------------------------------------------------------------------ #
    # Dynamic: change how the game plays (movement / spawn / difficulty)
    # ------------------------------------------------------------------ #
    # New dynamic mods (the `dyn4` sequence)
    "change_kangaroo_speed": ChangeKangarooSpeedMod,   # player walks faster
    "change_monkey_speed": ChangeMonkeySpeedMod,       # monkeys patrol faster
    "randomize_coconuts": RandomizeCoconutsMod,        # random head/foot throw height
    "jump_gravity": JumpGravityMod,                    # floatier, higher jump arc

    # Magnitude sequence (coconut_speed_xN): same mod, incrementally faster
    # thrown coconuts -> shorter duck/jump reaction window.
    "coconut_speed_x2": CoconutSpeedX2Mod,
    "coconut_speed_x3": CoconutSpeedX3Mod,
    "coconut_speed_x4": CoconutSpeedX4Mod,
    "coconut_speed_x5": CoconutSpeedX5Mod,

    # Remove / disable entities
    "no_bell": NoBellMod,
    "no_fruit": NoFruitMod,
    "no_monkey": NoMonkeyMod,
    "no_falling_coconut": NoFallingCoconutMod,
    "no_thrown_coconut": NoThrownCoconutMod,
    "no_danger": ["no_monkey", "no_falling_coconut"], # bundle into a modpack
    "pin_child": PinChildMod,

    # Control restrictions (disable a player ability)
    "no_punch": NoPunchMod,     # FIRE stripped: cannot box monkeys/coconuts
    "no_crouch": NoCrouchMod,   # DOWN stripped: cannot duck (also no climbing down)
    "no_jump": NoJumpMod,       # jump disabled (climbing still works)

    # Coconut / spawn behaviour
    "high_thrown_coconuts": AlwaysHighCoconutMod,

    # Spawn / hazard placement
    "spawn_on_second_floor": SpawnOnSecondFloorMod,
    "_lethal_bell": LethalFlameMod,
    "_flame_trap": FlameTrapMod,
    "lethal_flame": ["_lethal_bell", "replace_bell_with_flame"], # bundle into a modpack
    "flame_trap": ["_lethal_bell", "replace_bell_with_flame", "_flame_trap"], # modpack
    "cactus_trap": ["_lethal_bell", "replace_bell_with_cactus", "_flame_trap"], # modpack
    "danger_trap": ["_lethal_bell", "replace_bell_with_danger_sign", "_flame_trap"], # modpack

    # ------------------------------------------------------------------ #
    # Reward: reshape what the agent is rewarded for
    # ------------------------------------------------------------------ #
    "life_loss_penalty": LifeLossPenaltyMod,     # score - penalty per life lost
    "reward_per_floor": RewardPerFloorMod,       # dense: reward climbing toward Joey
    "reach_joey_only": ReachJoeyOnlyMod,         # sparse: only reaching the top scores
    "fruit_only": FruitOnlyMod,                  # only fruit collection scores
    "punch_only_scoring": PunchOnlyScoringMod,   # only punching monkeys/coconut scores
    "survival_reward": SurvivalRewardMod,        # +1 per step alive, ignores all scoring

    # ------------------------------------------------------------------ #
    # Layout / level structure
    # ------------------------------------------------------------------ #
    "center_ladders": CenterLaddersMod,
    "invert_ladders": InvertLaddersMod,
    "four_ladders": FourLaddersMod,
    "first_level_only": FirstLevelOnlyMod,
    "second_level_only": SecondLevelOnlyMod,
    "third_level_only": ThirdLevelOnlyMod,

    # ------------------------------------------------------------------ #
    # Visual: sprite / colour swaps (no gameplay change on their own)
    # ------------------------------------------------------------------ #
    # Single-element recolours + grayscale theme (the `vis4` set)
    "change_kangaroo_color": ChangeKangarooColorMod,
    "change_monkey_color": ChangeMonkeyColorMod,
    "change_background_color": ChangeBackgroundColorMod,
    "change_score_color": ChangeScoreColorMod,
    "grayscale_theme": GrayscaleThemeMod,

    # Sprite swaps
    "replace_child_with_monkey": ReplaceChildWithMonkeyMod,
    "replace_bell_with_flame": ReplaceBellWithFlameMod,
    "replace_bell_with_cactus": ReplaceBellWithCactusMod,
    "replace_bell_with_danger_sign": ReplaceBellWithDangerSignMod,
    "ropes": ReplaceLadderWithRopeMod,
    "chains": ReplaceLadderWithChainMod,
    "tanks": ReplaceMonkeyWithTankMod,
    "replace_coconut_fireball": ReplaceCoconutWithFireball,
    "replace_coconut_honey_bee": ReplaceCoconutWithHoneyBee,
    "replace_coconut_wasp": ReplaceCoconutWithWasp,
    "collectable_coins": ReplaceFruitWithCoin,
    "collectable_diamonds": ReplaceFruitWithDiamond,

    # Enemy reskins (helpers + modpacks). Most disable thrown coconuts so the
    # reskinned enemy doesn't throw; `dragons` instead reskins the projectile.
    "_chickens": ReplaceMonkeyWithChickenMod,
    "_dragons": ReplaceMonkeyWithDragonMod,
    # "_danger_signs": ReplaceMonkeyWithDangerSignMod,
    "_polarbears": ReplaceMonkeyWithPolarbearMod,
    "_snakes": ReplaceMonkeyWithSnakeMod,
    "chickens": ["no_thrown_coconut", "_chickens"], # modpack
    "dragons": ["replace_coconut_fireball", "_dragons"], # dragons throw fireballs
    # "danger_signs": ["no_thrown_coconut", "_danger_signs"], # modpack
    "polarbears": ["no_thrown_coconut", "_polarbears"], # modpack
    "snakes": ["no_thrown_coconut", "_snakes"], # modpack

    # ------------------------------------------------------------------ #
    # Debug
    # ------------------------------------------------------------------ #
    "render_debug_info": RenderDebugInfo,
}

class KangarooEnvMod(JaxAtariModController):
    """
    Game-specific (Group 1) Mod Controller for Kangaroo.
    It inherits all logic from JaxAtariModController and defines
    the REGISTRY.
    """

    REGISTRY = KANGAROO_MOD_REGISTRY

    # Define the path relative to this file (mod sprites fallback)
    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "kangaroo", "sprites")

    def __init__(self,
                 env,
                 mods_config: list = [],
                 allow_conflicts: bool = True
                 ):
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY
        )
