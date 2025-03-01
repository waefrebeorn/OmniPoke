import time
import utils
import config
import torch
from transformers import AutoModelForCausalLM
import cv2
import numpy as np
import os
from PIL import Image
import re

class Vision:
    """
    Vision class for Pokémon Blue that uses the Moondream model as an OCR engine and scene descriptor.
    It extracts environmental cues, Pokémon names, and item information and updates the game_state
    (inventory and party) when dedicated menu screens are detected.
    """
    # The 151 original Pokémon names.
    POKEMON_NAMES = {
        "Bulbasaur", "Ivysaur", "Venusaur", "Charmander", "Charmeleon", "Charizard",
        "Squirtle", "Wartortle", "Blastoise", "Caterpie", "Metapod", "Butterfree",
        "Weedle", "Kakuna", "Beedrill", "Pidgey", "Pidgeotto", "Pidgeot", "Rattata",
        "Raticate", "Spearow", "Fearow", "Ekans", "Arbok", "Pikachu", "Raichu",
        "Sandshrew", "Sandslash", "Nidoran♀", "Nidorina", "Nidoqueen", "Nidoran♂",
        "Nidorino", "Nidoking", "Clefairy", "Clefable", "Vulpix", "Ninetales",
        "Jigglypuff", "Wigglytuff", "Zubat", "Golbat", "Oddish", "Gloom", "Vileplume",
        "Paras", "Parasect", "Venonat", "Venomoth", "Diglett", "Dugtrio", "Meowth",
        "Persian", "Psyduck", "Golduck", "Mankey", "Primeape", "Growlithe", "Arcanine",
        "Poliwag", "Poliwhirl", "Poliwrath", "Abra", "Kadabra", "Alakazam", "Machop",
        "Machoke", "Machamp", "Bellsprout", "Weepinbell", "Victreebel", "Tentacool",
        "Tentacruel", "Geodude", "Graveler", "Golem", "Ponyta", "Rapidash", "Slowpoke",
        "Slowbro", "Magnemite", "Magneton", "Farfetch'd", "Doduo", "Dodrio", "Seel",
        "Dewgong", "Grimer", "Muk", "Shellder", "Cloyster", "Gastly", "Haunter", "Gengar",
        "Onix", "Drowzee", "Hypno", "Krabby", "Kingler", "Voltorb", "Electrode",
        "Exeggcute", "Exeggutor", "Cubone", "Marowak", "Hitmonlee", "Hitmonchan",
        "Lickitung", "Koffing", "Weezing", "Rhyhorn", "Rhydon", "Chansey", "Tangela",
        "Kangaskhan", "Horsea", "Seadra", "Goldeen", "Seaking", "Staryu", "Starmie",
        "Mr. Mime", "Scyther", "Jynx", "Electabuzz", "Magmar", "Pinsir", "Tauros",
        "Magikarp", "Gyarados", "Lapras", "Ditto", "Eevee", "Vaporeon", "Jolteon",
        "Flareon", "Porygon", "Omanyte", "Omastar", "Kabuto", "Kabutops", "Aerodactyl",
        "Snorlax", "Articuno", "Zapdos", "Moltres", "Dratini", "Dragonair", "Dragonite",
        "Mewtwo", "Mew"
    }

    def __init__(self, emulator=None):
        self.emulator = emulator
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load Moondream model in full float16 precision.
        self.model = AutoModelForCausalLM.from_pretrained(
            config.VISION_MODEL_PATH,
            revision="2025-01-09",
            trust_remote_code=True,
            device_map={"": self.device},
            torch_dtype=torch.float16
        )
        if torch.cuda.is_available() and config.TORCH_COMPILE:
            self.model = torch.compile(self.model)
        utils.log(f"Moondream model initialized on: {self.device} with full float16 precision")

        self.valid_buttons = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]

        self.base_prompt = """
Describe this Pokémon Blue game screen in detail. Focus on:
1. The current game state (e.g., title screen, dialogue, battle, menu, overworld, cave, dark cave, gym).
2. All visible text—including menu options, any Pokémon names, and item names.
3. Character and NPC positions.
4. Environmental details (indoors, outdoors, cave, water).
5. Any battle information (e.g., Pokémon names, HP, moves) if present.
6. Any other interactive elements.
Be comprehensive but concise.
"""

        self.specialized_prompts = {
            "CAVE_DETECTION": """
The scene appears to be inside a cave. Describe the cave's features (e.g., rocky, dark areas, tunnel-like structures)
and any visible cave entrances.
""",
            "DARK_CAVE_DETECTION": """
The scene appears to be in a dark cave. Describe the low-light conditions and any indications that Flash is needed.
"""
        }

        self.consecutive_title_screens = 0
        self.frames_captured = 0
        self.last_frame_hash = None
        self.current_environment = "UNKNOWN"
        self.environment_confidence = 0
        self.environment_history = []
        self.max_history = 5
        self.last_cave_state = None
        self.walkthrough_path = [
            "PALLET_TOWN", "ROUTE_1", "VIRIDIAN_CITY", "POKEMON_CENTER",
            "OAKS_LAB", "VIRIDIAN_FOREST", "MT MOON", "PEWTER_CITY"
        ]
        self.found_pokemon = []
        self.found_ghost = False

        # Game state: inventory and party.
        self.game_state = {
            "inventory": {},
            "party": []
        }
        # Complete item catalog.
        self.ITEM_CATALOG = {
            "ANTIDOTE": {"description": "Cures poison", "price": "100", "source": "PokéMart (Viridian Forest)"},
            "AWAKENING": {"description": "Cures sleep", "price": "250", "source": "Pokémon Tower"},
            "BICYCLE": {"description": "Allows faster travel", "price": "1,000,000", "source": "Cerulean Bike Shop"},
            "BIKE VOUCHER": {"description": "Exchange for a FREE Bike", "price": None, "source": "Vermillion Pokémon Fan Club"},
            "BURN HEAL": {"description": "Heals burns", "price": "250", "source": "PokéMart"},
            "CALCIUM": {"description": "Increases Special level", "price": None, "source": "Silph Co."},
            "CARBOS": {"description": "Increases Speed level", "price": "9800", "source": "Celadon Dept. Store"},
            "CARD KEY": {"description": "Unlocks Silph Co. doors", "price": None, "source": "Silph Co. (5th Floor)"},
            "COIN": {"description": "Slot Machine money", "price": "50", "source": "Game Corner"},
            "COIN CASE": {"description": "Holds coins", "price": None, "source": "Celadon Restaurant"},
            "DIRE HIT": {"description": "Increases attack effectiveness", "price": None, "source": "Celadon Dept. Store"},
            "DOME FOSSIL": {"description": "Evolves into Kabuto", "price": None, "source": "Mt. Moon"},
            "ELIXER": {"description": "Restores 10 PP to all moves", "price": None, "source": "Pokémon Tower or PokéMart"},
            "ESCAPE ROPE": {"description": "Used to escape an area", "price": None, "source": "PokéMart"},
            "ETHER": {"description": "Restores 10 PP to one move", "price": None, "source": "Various Areas"},
            "EXP. ALL": {"description": "Shares experience points", "price": None, "source": "Route 15"},
            "FIRE STONE": {"description": "Evolves Fire Pokémon", "price": "2100", "source": "Celadon Dept. Store"},
            "FRESH WATER": {"description": "Restores 50 HP", "price": "200", "source": "Celadon Dept. Store Roof"},
            "FULL HEAL": {"description": "Cures all ailments", "price": None, "source": "PokéMart"},
            "FULL RESTORE": {"description": "Cures ailments and restores all HP", "price": None, "source": "PokéMart / Safari Zone"},
            "GOLD TEETH": {"description": "Helps Warden to speak", "price": None, "source": "Safari Zone 3"},
            "GOOD ROD": {"description": "Used for fishing", "price": None, "source": "Fuchsia City"},
            "GREAT BALL": {"description": "Better catch rate than Pokéball", "price": None, "source": "PokéMart"},
            "GUARD SPEC.": {"description": "Disables Special Attacks", "price": "700", "source": "Celadon Dept. Store"},
            "HELIX FOSSIL": {"description": "Evolves into Omanyte", "price": None, "source": "Mt. Moon"},
            "HP UP": {"description": "Increases HP by 1", "price": None, "source": "Various Areas"},
            "HYPER POTION": {"description": "Restores 200 HP", "price": "1500", "source": "PokéMart / Game Corner"},
            "ICE HEAL": {"description": "Heals frozen Pokémon", "price": "250", "source": "PokéMart"},
            "IRON": {"description": "Increases Defense level", "price": None, "source": "Game Corner (Route 12)"},
            "ITEM FINDER": {"description": "Finds hidden items", "price": None, "source": "Route 11"},
            "LEAF STONE": {"description": "Evolves Grass Pokémon", "price": "2100", "source": "Celadon Dept. Store"},
            "LEMONADE": {"description": "Restores 80 HP", "price": "350", "source": "Celadon Dept. Store Roof"},
            "LIFT KEY": {"description": "Activates Game Corner Elevator", "price": None, "source": "Game Corner"},
            "MASTER BALL": {"description": "Catches Pokémon 100% of time", "price": None, "source": "Silph Co."},
            "MAX ELIXER": {"description": "Restores all PP", "price": None, "source": "Cerulean Cave"},
            "MAX ETHER": {"description": "Restores all PP of one move", "price": None, "source": "S.S. Anne"},
            "MAX POTION": {"description": "Restores all HP", "price": None, "source": "PokéMart (S.S. Anne)"},
            "MAX REPEL": {"description": "Prevents wild Pokémon from attacking", "price": "700", "source": "PokéMart"},
            "MAX REVIVE": {"description": "Revives fainted Pokémon and restores all HP", "price": None, "source": "Silph Co. / Safari Zone"},
            "MOON STONE": {"description": "Evolves certain Pokémon", "price": None, "source": "Mt. Moon"},
            "NUGGET": {"description": "Can be sold for money", "price": None, "source": "Various (Nugget Bridge, Game Corner)"},
            "OAK'S PARCEL": {"description": "Exchange for a Pokédex", "price": None, "source": "Viridian City-PokéMart"},
            "OLD AMBER": {"description": "Evolves into Aerodactyl", "price": None, "source": "Pewter City-Museum"},
            "OLD ROD": {"description": "Used for fishing", "price": None, "source": "Vermillion City"},
            "PARALYZE HEAL": {"description": "Heals paralysis", "price": "200", "source": "PokéMart"},
            "POKÉBALL": {"description": "Used to catch Pokémon", "price": "200", "source": "PokéMart (Viridian Forest)"},
            "POKÉDEX": {"description": "Records Pokémon data", "price": None, "source": "Pallet Town"},
            "POKÉDOLL": {"description": "Distracts opponents; appeals to girls", "price": "1000", "source": "Celadon Dept. Store"},
            "POKÉFLUTE": {"description": "Wakes sleeping Pokémon", "price": None, "source": "Pokémon Tower"},
            "POTION": {"description": "Restores 20 HP", "price": None, "source": "PokéMart"},
            "PP UP": {"description": "Increases PP by 1", "price": None, "source": "Cerulean Cave"},
            "PROTEIN": {"description": "Increases Attack level", "price": None, "source": "Celadon Dept. Store / Silph Co."},
            "RARE CANDY": {"description": "Raises level by 1", "price": None, "source": "Mt. Moon / S.S. Anne / Game Corner"},
            "REPEL": {"description": "Prevents wild Pokémon from attacking", "price": "350", "source": "PokéMart"},
            "REVIVE": {"description": "Revives a fainted Pokémon", "price": "1500", "source": "PokéMart"},
            "S.S. TICKET": {"description": "Used to board S.S. Anne", "price": None, "source": "Sea Cottage"},
            "SAFARI BALL": {"description": "Used in the Safari Zone", "price": "1000", "source": "Pokémon Mansion"},
            "SECRET KEY": {"description": "Unlocks Cinnabar Island Gym", "price": None, "source": "Pokémon Mansion"},
            "SILPH SCOPE": {"description": "Identifies ghosts", "price": None, "source": "Game Corner"},
            "SODA POP": {"description": "Restores 60 HP", "price": "300", "source": "Celadon Dept. Store Roof"},
            "SUPER POTION": {"description": "Restores 50 HP", "price": "700", "source": "PokéMart / Game Corner"},
            "SUPER REPEL": {"description": "Prevents wild Pokémon from attacking", "price": "500", "source": "PokéMart"},
            "SUPER ROD": {"description": "Used for fishing", "price": None, "source": "Route 12"},
            "THUNDER STONE": {"description": "Evolves Pikachu", "price": "2100", "source": "Celadon Dept. Store"},
            "TOWN MAP": {"description": "Shows your current location", "price": None, "source": "Rival's House"},
            "ULTRA BALL": {"description": "Better catch rate than a Pokéball", "price": "1200", "source": "PokéMart (Cerulean Cave)"},
            "WATER STONE": {"description": "Evolves certain Water Pokémon", "price": "2100", "source": "Celadon Dept. Store"},
            "X ACCURACY": {"description": "Boosts Accuracy temporarily", "price": "950", "source": "Celadon Dept. Store / Silph Co."},
            "X ATTACK": {"description": "Boosts Attack temporarily", "price": "500", "source": "Celadon Dept. Store"},
            "X DEFEND": {"description": "Boosts Defense temporarily", "price": "550", "source": "Celadon Dept. Store"},
            "X SPECIAL": {"description": "Boosts Special temporarily", "price": "350", "source": "Celadon Dept. Store"},
            "X SPEED": {"description": "Boosts Speed temporarily", "price": "350", "source": "Celadon Dept. Store"}
        }
    
    def update_inventory_from_text(self, text):
        """
        Scan the text for an inventory header (e.g., "BACKPACK" or "ITEMS"). If found,
        parse lines formatted as "ITEM_NAME x NUMBER" and update game_state['inventory'].
        """
        if not re.search(r'\b(backpack|items)\b', text.lower()):
            return
        text_upper = text.upper()
        for item in self.ITEM_CATALOG:
            pattern = rf'\b{item.upper()}\b\s*x\s*(\d+)'
            match = re.search(pattern, text_upper)
            if match:
                count = int(match.group(1))
                self.game_state["inventory"][item] = count
                utils.log(f"Set inventory for {item} to {count}")
    
    def extract_party(self, text):
        """
        If the text appears to be from the party screen (contains "PARTY"),
        extract Pokémon names and update game_state['party'].
        """
        if not re.search(r'\bparty\b', text.lower()):
            return
        party_list = re.findall(r'\b(' + '|'.join(self.POKEMON_NAMES).upper() + r')\b', text.upper())
        self.game_state["party"] = party_list
        utils.log(f"Updated party: {', '.join(party_list)}")
    
    def capture_screen(self):
        if not self.emulator:
            utils.log("WARNING: No emulator provided to Vision. Returning blank image.")
            return Image.new('RGB', (160, 144), color=(0, 0, 0))
        self.frames_captured += 1
        frame = self.emulator.capture_screen()
        if self.frames_captured % config.FRAME_LOGGING_FREQUENCY == 0:
            utils.save_frame(frame, f"frame_{self.frames_captured}.png")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    def _hash_image(self, image):
        img_small = image.resize((32, 32)).convert('L')
        img_array = np.array(img_small)
        return hash(img_array.tobytes())
    
    def parse_battle_text(self, caption):
        """
        Scan the caption for Pokémon names and ghost-related keywords.
        Returns (list_of_detected_pokemon, ghost_detected_flag).
        """
        found = []
        caption_upper = caption.upper()
        for name in self.POKEMON_NAMES:
            if name.upper() in caption_upper:
                found.append(name)
        ghost_detected = False
        if "GHOST" in caption_upper or "SILPH SCOPE" in caption_upper:
            ghost_detected = True
        return found, ghost_detected

    def detect_environment(self, description):
        """
        Analyze the description to determine the environment.
        Returns cave-related states only if keywords are present; otherwise, OVERWORLD.
        """
        description_upper = description.upper()
        if "PROFESSOR" in description_upper or "OAK" in description_upper:
            return "INTRO_SEQUENCE"
        if any(term in description_upper for term in ["CAVE", "MT MOON", "ROCK TUNNEL", "VICTORY ROAD"]):
            if any(term in description_upper for term in ["DARK CAVE", "LIMITED VISIBILITY", "FLASH NEEDED", "VERY LOW LIGHT"]):
                return "DARK_CAVE"
            return "CAVE"
        if any(term in description_upper for term in ["POKEMON CENTER", "POKÉMON CENTER", "NURSE JOY", "HEALING"]):
            return "POKEMON CENTER"
        if any(term in description_upper for term in ["MART", "SHOP", "STORE", "CLERK", "BUY"]):
            return "POKEMON_MART"
        if "GYM" in description_upper:
            return "GYM"
        if any(term in description_upper for term in ["HOUSE", "BUILDING", "INSIDE", "INTERIOR", "INDOOR"]):
            return "BUILDING"
        if any(term in description_upper for term in ["BATTLE", "FIGHT", "POKEMON VS", "POKÉMON VS"]):
            return "BATTLE"
        if any(term in description_upper for term in ["TITLE SCREEN", "BLUE VERSION", "PRESS START"]):
            return "TITLE SCREEN"
        if any(term in description_upper for term in ["TEXT BOX", "DIALOGUE", "TALKING", "MESSAGE"]):
            return "DIALOGUE"
        if any(term in description_upper for term in ["MENU", "OPTIONS", "ITEMS", "POKEMON LIST", "POKÉMON LIST"]):
            return "MENU"
        return "OVERWORLD"
    
    def update_environment_tracking(self, detected_env):
        self.environment_history.append(detected_env)
        if len(self.environment_history) > self.max_history:
            self.environment_history.pop(0)
        env_counts = {}
        for env in self.environment_history:
            env_counts[env] = env_counts.get(env, 0) + 1
        most_common_env = max(env_counts.items(), key=lambda x: x[1])
        confidence = most_common_env[1] / len(self.environment_history)
        if confidence >= 0.6:
            prev_env = self.current_environment
            self.current_environment = most_common_env[0]
            self.environment_confidence = confidence
            if self.current_environment in ["CAVE", "DARK_CAVE"]:
                self.last_cave_state = self.current_environment
            if prev_env != self.current_environment:
                utils.log(f"Environment change: {prev_env} -> {self.current_environment} (confidence: {confidence:.2f})")
        return self.current_environment
    
    def get_game_state_text(self):
        image = self.capture_screen()
        prompt_to_use = self.base_prompt
        with torch.inference_mode():
            caption = self.model.query(image, prompt_to_use)["answer"]
        utils.log(f"Moondream Screen Description: {caption}")
        # Update inventory and party if applicable.
        self.update_inventory_from_text(caption)
        self.extract_party(caption)
        detected_env = self.detect_environment(caption)
        if detected_env in ["CAVE", "DARK_CAVE", "MT MOON", "ROCK_TUNNEL", "VICTORY ROAD"]:
            utils.log("Cave-related keywords detected; using specialized cave detection prompt.")
            with torch.inference_mode():
                caption = self.model.query(image, self.specialized_prompts["CAVE_DETECTION"])["answer"]
            utils.log(f"Revised Cave Screen Description: {caption}")
            detected_env = self.detect_environment(caption)
        found, ghost_flag = self.parse_battle_text(caption)
        if found:
            utils.log("Detected Pokémon via OCR: " + ", ".join(found))
            self.found_pokemon = found
        else:
            self.found_pokemon = []
        self.found_ghost = ghost_flag
        if ghost_flag:
            utils.log("Ghost indicator detected in OCR text.")
        self.update_environment_tracking(detected_env)
        if "title screen" in caption.lower() or "blue version" in caption.lower():
            self.consecutive_title_screens += 1
        else:
            self.consecutive_title_screens = 0
        return caption
    
    def get_context_aware_prompt(self, task_instruction=None):
        base_prefix = "You are playing Pokémon Blue. "
        base_suffix = "\nReply with exactly one button from: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT."
        env_instructions = {
            "DARK_CAVE": "You are in a dark cave with limited visibility. Navigate carefully using directional keys and menus.",
            "CAVE": "You are in a cave. Choose a button to navigate (explore, interact, etc.).",
            "MT MOON": "You are in Mt. Moon. Choose a button to navigate through the cave.",
            "ROCK_TUNNEL": "You are in Rock Tunnel. Choose a button to bypass obstacles.",
            "VICTORY ROAD": "You are in Victory Road. Choose a button to avoid obstacles.",
            "POKEMON CENTER": "You are in a Pokémon Center. Choose a button to navigate the healing menu.",
            "POKEMON_MART": "You are in a Pokémon Mart. Choose a button to browse the menu.",
            "GYM": "You are in a Pokémon Gym. Choose a button to navigate the gym.",
            "OVERWORLD": "You are exploring the overworld. Choose a button to move or interact.",
            "BATTLE": "You are in a battle menu. Choose a button to select your move.",
            "TRAINER_BATTLE": "You are in a trainer battle. Choose a button to select the best move.",
            "WILD_BATTLE": "You are in a wild battle. Choose a button to fight.",
            "DIALOGUE": "You are in a dialogue screen. Choose a button to select a dialogue option.",
            "MENU": "You are in a menu. Choose a button to navigate the menu.",
            "GHOST": "You are encountering a ghost. If your backpack shows you have the Sylph Scope, choose the button to use it. Otherwise, choose the button to run."
        }
        current_env = self.current_state
        if self.found_ghost:
            current_env = "GHOST"
        env_instruction = env_instructions.get(current_env, "Choose a button to continue.")
        walkthrough_hint = ""
        if self.walkthrough_path:
            walkthrough_hint = "Remember your walkthrough path."
        if task_instruction:
            prompt = f"{base_prefix}{walkthrough_hint}\nFollow this instruction: {task_instruction}{base_suffix}"
        else:
            prompt = f"{base_prefix}{env_instruction} {walkthrough_hint}{base_suffix}"
        return prompt

    def get_next_action(self, task_instruction=None):
        game_state = self.get_game_state_text()
        utils.log(f"Game state from vision: {game_state}")
        return self.get_context_aware_prompt(task_instruction)
