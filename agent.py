"""A role-based Season 1 agent for Skirmish at Crane Reach.

Each unit runs a separate instance of this class. Read ``environment.md`` beside this file for
the rules, helpers, and later-season mechanics. Prepare episode state in ``reset``; the
constructor takes no arguments. Movement always comes from the action mask: this agent scores
complete legal paths rather than recreating movement rules.
"""

from __future__ import annotations

import heapq
import random
from typing import Iterable

from sandbox.crane import action, me, paths, tile, units, visible
from sandbox.observation_types import AxialPosition, SkirmishAction, SkirmishObservation, VisibleUnit


Position = dict[str, int]
PathOption = tuple[int, AxialPosition]
Key = tuple[int, int]

# Axial offsets for direction digits 1-6 (environment.md path encoding table).
HEX_OFFSETS = ((1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1))
TERRAIN_COST = {"grass": 1, "hill": 2}
FEATURE_COST = {"none": 0, "forest": 1, "marsh": 2, "waste": 0}
# Never-reached sentinel for route lookups, kept far above any real turn count.
UNREACHABLE = 10_000
# Memory horizons in rounds (= this unit's activations). Enemy sightings guide searching for 3,
# but only sightings up to 2 old count for flee/attack decisions. Allies are tracked for 2.
ENEMY_MEMORY_ROUNDS = 3
THREAT_MEMORY_ROUNDS = 2
ALLY_MEMORY_ROUNDS = 2
# A tile is "tankable" when expected incoming damage < HP * fraction. Starting values, untuned:
# archers keep a wide margin (fragile, they kite), melee only needs to survive.
TANK_FRACTION = {"archer": 0.5, "footman": 1.0, "cavalry": 1.0}
# One type order for both attacking and protecting (user spec): archer > footman > cavalry.
TYPE_RANK = {"archer": 3, "footman": 2, "cavalry": 1}
# Formation (user spec): archers at the back, footmen wall 1 tile in front of them, cavalry the
# front line 3 tiles in front. Tiles measured from the nearest allied archer toward the enemy.
FORMATION_DEPTH = {"footman": 1, "cavalry": 3, "archer": 0}
# Allies farther than this do not block our route planning (the action mask still decides legality).
ALLY_BLOCK_RADIUS = 2
# Activations on the same tile, while trying to get somewhere, before we count as stuck.
STUCK_ACTIVATIONS = 2
# Chat: at most this many direct messages per activation, each within the 200-character cap.
CHAT_RECIPIENTS = 5
CHAT_TEXT_LIMIT = 200
TYPE_CODES = {"f": "footman", "a": "archer", "c": "cavalry"}
# Only answer chat help alerts from allies within this many tiles.
ALERT_RADIUS = 12
# Whether a received focus broadcast raises that archer in target order (the broadcast is still
# sent either way, since its "M" position lets lost allies regroup).
USE_FOCUS_CALLS = False
# Off: against Naive the focus broadcast cost ~0.5 on seat 1 (its archer sighting spread to every
# ally's threat memory). Kept switchable to re-test against stronger rivals.
SEND_BROADCAST = False
# How long a relayed "biggest group" report stays usable for merging groups.
GROUP_MEMORY_ROUNDS = 3


class Agent:
    """Use archer kiting and local melee support to contest the center."""

    def reset(self, seed, observation) -> None:
        # Called once before each match. The opening observation is available here for
        # precomputation outside the decision clock. Each unit's memory remains private.
        self._activation = 0
        self._rng = random.Random(seed)
        # Memory records: {"type", "position", "hit_points", "round"}, stamped with the game round so
        # age is activations since the sighting (each unit acts once per round). Chat reports can
        # write the same records later.
        self._enemy_memory: dict[str, dict] = {}
        self._ally_memory: dict[str, dict] = {}
        self._known_dead: dict[str, int] = {}
        self._round = 0
        self._still = 0
        self._last_position: Key | None = None
        # Chat: roster lookups (messages address players, not unit ids) and conversation state.
        rosters = observation["observation"]["rosters"]
        own_side = me.side(observation)
        own_id = me.unit_id(observation)
        # Blue mirrors its tiebreaks (see _tie) so neither side leans one way on ties.
        self._mirror = own_side == "blue"
        self._tie_cache: dict[int, int] = {}
        self._enemy_side = "blue" if own_side == "red" else "red"
        self._player_of = {entry["unit_id"]: entry["player"] for entry in rosters[own_side]}
        self._unit_of = {entry["player"]: entry["unit_id"] for entry in rosters[own_side]}
        self._ally_roster = [entry for entry in rosters[own_side] if entry["unit_id"] != own_id]
        self._talked: set[str] = set()
        self._prev_seen_allies: set[str] = set()
        self._report: str | None = None
        self._recipients: list[str] = []
        # Allies that reported "enemy attacking me": unit_id -> {"type", "position", "round"}.
        self._ally_alerts: dict[str, dict] = {}
        # Broadcast focus calls on dangerous enemy archers: enemy unit_id -> round called.
        self._focus: dict[str, int] = {}
        self._broadcast: str | None = None
        self._named_target: str | None = None
        # Biggest allied group known (own or relayed by chat): {"position", "size", "round"}.
        self._best_group: dict | None = None
        # Entry cost per tile, None for water/void/wasteland (wasteland is never entered).
        self._cost_grid: dict[Key, int | None] = {}
        self._water: set[Key] = set()
        self._marsh: set[Key] = set()
        cover: set[Key] = set()
        side = observation["observation"]["battlefield"]["side"]
        for r in range(side):
            for q in range(side):
                cell = tile.terrain_at(observation, {"q": q, "r": r})
                if cell["terrain"] in TERRAIN_COST and cell["feature"] != "waste":
                    self._cost_grid[(q, r)] = TERRAIN_COST[cell["terrain"]] + FEATURE_COST.get(cell["feature"], 0)
                else:
                    self._cost_grid[(q, r)] = None
                if cell["terrain"] == "water":
                    self._water.add((q, r))
                if cell["feature"] == "marsh":
                    self._marsh.add((q, r))
                if cell["terrain"] == "hill" or cell["feature"] == "forest":
                    cover.add((q, r))
        # Static terrain reads for the preference: open ground within 2 tiles (edge, water, and marsh
        # close it in) and whether forest or hill is on or next to a tile.
        self._openness: dict[Key, int] = {}
        self._near_cover: dict[Key, bool] = {}
        for key, cost in self._cost_grid.items():
            if cost is None:
                continue
            ring2 = {n2 for n1 in self._neighbor_keys(key) for n2 in (n1, *self._neighbor_keys(n1))} - {key}
            self._openness[key] = sum(
                1 for spot in ring2 if self._cost_grid.get(spot) is not None and spot not in self._marsh
            )
            self._near_cover[key] = key in cover or any(spot in cover for spot in self._neighbor_keys(key))

    def act(self, observation: SkirmishObservation) -> SkirmishAction:
        self._activation += 1
        enemies = visible.enemies(observation)
        allies = visible.allies(observation)
        # Per-activation context shared by the route and terrain helpers.
        self._enemies_now = enemies
        self._allies_now = allies
        self._blocked = {self._key(unit["position"]) for unit in (*enemies, *allies)}
        here = me.position(observation)
        # Our own route planning: enemies always block, allies only nearby (farther ones will have
        # moved by the time we get there). Enemy reach (_reach) still uses the full _blocked set.
        self._route_blocked = {self._key(unit["position"]) for unit in enemies} | {
            self._key(ally["position"]) for ally in allies if self._distance(here, ally["position"]) <= ALLY_BLOCK_RADIUS
        }
        # Anti-stuck: consecutive activations spent on the same tile.
        self._still = self._still + 1 if self._last_position == self._key(here) else 0
        self._last_position = self._key(here)
        self._turn_cache: dict[tuple, dict[Key, int]] = {}
        self._incoming_cache: dict[Key, list[tuple[Key, float]]] = {}
        self._reach_cache: dict[tuple[str, Key], set[Key]] = {}
        self._threatened: list[tuple[VisibleUnit, VisibleUnit]] | None = None
        self._sources: list[tuple[str, Key, float, bool]] | None = None
        self._round = observation["observation"]["round"]
        self._here = me.position(observation)
        self._remember(observation, enemies, allies)
        paths = self._paths(observation)
        if me.unit_type(observation) == "archer":
            order = self._act_archer(observation, paths, enemies, allies)
        else:
            order = self._act_melee(observation, paths, enemies, allies)
        self._prepare_report(observation)
        return order

    def chat(self, inbox: list[dict]) -> list[dict] | None:
        """Read allies' reports into memory, then send this activation's report.

        Runs after act, so what we read is first used at our next activation. Direct messages only
        (a broadcast would also reach the enemy).
        """
        for message in inbox:
            self._read_report(message)
        messages = []
        if self._report and self._recipients:
            self._talked.update(self._recipients)
            messages = [{"to": player, "text": self._report} for player in self._recipients]
        if self._broadcast and SEND_BROADCAST:
            # The one broadcast allowed per activation; the enemy sees it too (see _prepare_report).
            messages.append({"to": None, "text": self._broadcast})
        return messages or None

    def _prepare_report(self, observation: SkirmishObservation) -> None:
        """Build the outgoing text and pick its recipients (user spec).

        Content priority: enemies attacking me > other enemies I see (archer > footman > cavalry,
        nearest first) > me. Format "R<round>|A <code>.<q>.<r>.<hp> ...|S ...|M ...", where a code is
        a type letter plus roster index (enemy "c2" = <enemy side>_cavalry_2), cut to 200 chars.
        Recipient priority (visible allies only): allies that cannot see a reported enemy > allies
        that just came into view > allies not messaged yet, each group archer > footman > cavalry;
        at most CHAT_RECIPIENTS per activation.
        """
        here = me.position(observation)
        here_key = self._key(here)
        enemies = sorted(
            self._enemies_now,
            key=lambda enemy: (-TYPE_RANK[enemy["type"]], self._distance(here, enemy["position"])),
        )
        attacking = [enemy for enemy in enemies if self._strike_from(enemy["type"], self._key(enemy["position"]), here_key)[0]]
        seen = [enemy for enemy in enemies if enemy not in attacking]
        sections = [
            ("A", [self._entry(enemy["unit_id"], enemy["position"], enemy["hit_points"]) for enemy in attacking]),
            ("S", [self._entry(enemy["unit_id"], enemy["position"], enemy["hit_points"]) for enemy in seen]),
            ("M", [self._entry(me.unit_id(observation), here, me.hit_points(observation))]),
        ]
        # Group relay: note our own group (us + allies in sight), then pass on the biggest group we
        # know of as "B <q>.<r>.<size>.<round>" so separate groups can find each other and merge.
        members = [here, *(ally["position"] for ally in self._allies_now)]
        own_center = {
            "q": round(sum(position["q"] for position in members) / len(members)),
            "r": round(sum(position["r"] for position in members) / len(members)),
        }
        self._note_group(own_center, len(members), self._round)
        best = self._best_group
        sections.append(("B", [f"{best['position']['q']}.{best['position']['r']}.{best['size']}.{best['round']}"]))
        text = f"R{self._round}"
        for tag, entries in sections:
            added = False
            for entry in entries:
                piece = (f"|{tag} " if not added else " ") + entry
                if len(text) + len(piece) > CHAT_TEXT_LIMIT:
                    break
                text += piece
                added = True
        self._report = text

        # Focus broadcast: our named target is an enemy archer that can hit us or a visible ally
        # next activation. A broadcast reaches every living ally (no recipient cap) but also the
        # enemy, who only learns we are focusing an archer whose position it already knows.
        # It also carries our own position ("M"), so lost allies with nobody in sight can find us.
        self._broadcast = None
        named = next((enemy for enemy in enemies if enemy["unit_id"] == self._named_target), None)
        if named is not None and named["type"] == "archer" and (
            named in attacking or any(enemy["unit_id"] == named["unit_id"] for _, enemy in self._threatened_allies())
        ):
            self._broadcast = (
                f"R{self._round}|F "
                + self._entry(named["unit_id"], named["position"], named["hit_points"])
                + "|M "
                + self._entry(me.unit_id(observation), here, me.hit_points(observation))
            )

        def usual(ally: dict) -> tuple:
            return (-TYPE_RANK[ally["type"]], self._distance(here, ally["position"]))

        def vision_of(ally: dict) -> int:
            on_hill = tile.terrain_at(observation, ally["position"])["terrain"] == "hill"
            return units.STATS[ally["type"]].vision + on_hill

        reported = attacking + seen
        allies = sorted(self._allies_now, key=usual)
        blind = [
            ally for ally in allies
            if reported and any(self._distance(ally["position"], enemy["position"]) > vision_of(ally) for enemy in reported)
        ]
        seen_ids = {ally["unit_id"] for ally in allies}
        newcomers = [ally for ally in allies if ally["unit_id"] not in self._prev_seen_allies and ally not in blind]
        self._prev_seen_allies = seen_ids
        chosen = [ally["unit_id"] for ally in (*blind, *newcomers)]
        # Visible allies only (user spec): an ally out of sight may already be dead.
        untalked = [
            ally["unit_id"]
            for ally in allies
            if self._player_of.get(ally["unit_id"]) not in self._talked and ally["unit_id"] not in chosen
        ]
        self._recipients = [self._player_of[unit_id] for unit_id in (*chosen, *untalked)][:CHAT_RECIPIENTS]

    @staticmethod
    def _entry(unit_id: str, position: AxialPosition, hit_points: int) -> str:
        _, kind, index = unit_id.split("_")
        return f"{kind[0]}{index}.{position['q']}.{position['r']}.{hit_points}"

    def _read_report(self, message: dict) -> None:
        """Merge one ally report into memory; anything malformed is ignored."""
        sender = self._unit_of.get(message.get("from"))
        text = message.get("text")
        if sender is None or not isinstance(text, str) or not text.startswith("R"):
            return
        try:
            sections = text.split("|")
            sent_round = int(sections[0][1:])
            attacked = False
            sender_position = None
            for section in sections[1:]:
                tag, _, body = section.partition(" ")
                if tag == "B":
                    q, r, size, group_round = (int(part) for part in body.split(".", 3))
                    self._note_group({"q": q, "r": r}, size, group_round)
                    continue
                for entry in body.split():
                    code, q, r, hit_points = entry.split(".")
                    kind = TYPE_CODES[code[0]]
                    record = {"type": kind, "position": {"q": int(q), "r": int(r)}, "hit_points": int(hit_points), "round": sent_round}
                    if tag == "M":
                        sender_position = record["position"]
                        known = self._ally_memory.get(sender)
                        if known is None or known["round"] < sent_round:
                            self._ally_memory[sender] = record
                        continue
                    attacked = attacked or tag == "A"
                    unit_id = f"{self._enemy_side}_{kind}_{code[1:]}"
                    if tag == "F":
                        self._focus[unit_id] = max(sent_round, self._focus.get(unit_id, -1))
                    if self._known_dead.get(unit_id, -1) >= sent_round:
                        continue
                    known = self._enemy_memory.get(unit_id)
                    if known is None or known["round"] < sent_round:
                        self._enemy_memory[unit_id] = record
            if attacked and sender_position is not None:
                sender_type = sender.split("_")[1]
                self._ally_alerts[sender] = {"type": sender_type, "position": sender_position, "round": sent_round}
        except (ValueError, IndexError, KeyError):
            return

    def _note_group(self, position: Position, size: int, seen_round: int) -> None:
        """Keep the biggest fresh group (fresher wins ties); stale entries are replaced."""
        best = self._best_group
        if (
            best is None
            or self._round - best["round"] > GROUP_MEMORY_ROUNDS
            or size > best["size"]
            or (size == best["size"] and seen_round >= best["round"])
        ):
            self._best_group = {"position": position, "size": size, "round": seen_round}

    def _bigger_group(self, observation: SkirmishObservation) -> Position | None:
        """A fresh group bigger than ours that is out of our sight, to merge with; else None."""
        best = self._best_group
        if best is None or self._round - best["round"] > GROUP_MEMORY_ROUNDS:
            return None
        own_size = 1 + len(self._allies_now)
        vision = units.STATS[me.unit_type(observation)].vision
        if best["size"] > own_size and self._distance(self._here, best["position"]) > vision:
            return self._copy(best["position"])
        return None

    def _fresh_alert(self) -> Position | None:
        """Nearest-priority ally that recently reported being attacked (archer > footman > cavalry)."""
        alerts = [
            alert
            for unit_id, alert in self._ally_alerts.items()
            if self._round - alert["round"] <= THREAT_MEMORY_ROUNDS
            and unit_id not in self._known_dead
            and self._distance(self._here, alert["position"]) <= ALERT_RADIUS
        ]
        if not alerts:
            return None
        best = min(alerts, key=lambda alert: (-TYPE_RANK[alert["type"]], self._distance(self._here, alert["position"])))
        return self._copy(best["position"])

    def _remember(self, observation: SkirmishObservation, enemies: Iterable[VisibleUnit], allies: Iterable[VisibleUnit]) -> None:
        """Refresh what we see, then use sight to correct or drop older records."""
        for unit in enemies:
            self._enemy_memory[unit["unit_id"]] = self._record(unit)
            self._known_dead.pop(unit["unit_id"], None)
        for unit in allies:
            self._ally_memory[unit["unit_id"]] = self._record(unit)
        here = me.position(observation)
        vision = units.STATS[me.unit_type(observation)].vision + (tile.terrain_at(observation, here)["terrain"] == "hill")
        self._audit(self._enemy_memory, {unit["unit_id"] for unit in enemies}, here, vision, ENEMY_MEMORY_ROUNDS)
        self._audit(self._ally_memory, {unit["unit_id"] for unit in allies}, here, vision, ALLY_MEMORY_ROUNDS)

    def _record(self, unit: VisibleUnit) -> dict:
        return {
            "type": unit["type"],
            "position": self._copy(unit["position"]),
            "hit_points": unit["hit_points"],
            "round": self._round,
        }

    def _audit(self, memory: dict[str, dict], seen_ids: set[str], here: AxialPosition, vision: int, horizon: int) -> None:
        """Expire old records; drop any whose last tile we can see empty of it.

        If the unit also could not have walked out of our sight since (one move per round, plus
        this round), it can only have died, so it is marked dead.
        """
        for unit_id, record in list(memory.items()):
            if unit_id in seen_ids:
                continue
            age = self._round - record["round"]
            if age > horizon:
                del memory[unit_id]
                continue
            distance = self._distance(here, record["position"])
            if distance <= vision:
                del memory[unit_id]
                reach = units.STATS[record["type"]].movement_points * (age + 1)
                if distance + reach <= vision:
                    self._known_dead[unit_id] = self._round

    def _recent(self, memory: dict[str, dict], rounds: int, kind: str | None = None) -> list[dict]:
        """Records at most `rounds` old, optionally of one type, freshest first."""
        records = [
            record
            for record in memory.values()
            if self._round - record["round"] <= rounds and (kind is None or record["type"] == kind)
        ]
        return sorted(records, key=lambda record: -record["round"])

    def _act_archer(self, observation: SkirmishObservation, paths: list[PathOption], enemies: list[VisibleUnit], allies: list[VisibleUnit]) -> SkirmishAction:
        melee_threats = [enemy for enemy in enemies if enemy["type"] in {"cavalry", "footman"}]
        here = me.position(observation)
        # Army scale: "under threat" means any known enemy can reach and hit this tile.
        under_threat = self._incoming(observation, here) > 0
        target = self._select_target(observation, paths, enemies)
        firing_paths = self._attack_paths(paths, target, "archer") if target is not None else []
        safe_firing = [option for option in firing_paths if self._incoming(observation, option[1]) == 0]
        tankable_firing = [option for option in firing_paths if self._can_tank(observation, option[1])]

        if safe_firing:
            # A firing tile that no enemy can reach this round exists: kite while
            # preserving the shot, preferring the option farthest from the target.
            path = max(
                safe_firing,
                key=lambda option: (
                    self._distance(option[1], target["position"]),
                    self._terrain_preference(observation, option[1]),
                    -self._tie(option[0]),
                ),
            )
            return self._order(path[0], target["unit_id"], observation)

        if tankable_firing:
            # Every firing tile is reachable, but some take a hit we can tank: pick the least-risky
            # firing tile (lowest expected incoming damage), then the farthest from the target.
            path = min( 
                tankable_firing,
                key=lambda option: (
                    self._incoming(observation, option[1]),
                    -self._terrain_preference(observation, option[1]),
                    -self._distance(option[1], target["position"]),
                    self._tie(option[0]),
                ),
            ) 
            return self._order(path[0], target["unit_id"], observation)

        if under_threat or firing_paths:
            # No safe or tankable firing tile exists: escaping outranks preserving the shot, so
            # search every legal path, not only firing-range ones (still shoot if the end allows).
            threat_pairs = [(enemy["type"], enemy["position"]) for enemy in melee_threats]
            best_path = self._flee(observation, paths, threat_pairs, keep_in_range=target)
            end = next(end for path_id, end in paths if path_id == best_path)
            name = (
                target["unit_id"]
                if target is not None and self._distance(end, target["position"]) <= units.STATS["archer"].attack_range
                else None
            )
            return self._order(best_path, name, observation)

        # Merge groups (no enemy in sight): join a bigger group heard about by chat.
        bigger = None if enemies else self._bigger_group(observation)
        if bigger is not None:
            return self._order(self._toward(observation, paths, bigger), None, observation)
        enemy_goal = self._last_seen_goal("archer")
        if allies:
            # Formation: archers advance only behind the melee line once an enemy front is known.
            front = self._front_point(observation)
            rear_paths = (self._behind_melee(paths, allies, front) if front is not None else paths) or paths
            if enemy_goal is None and front is None:
                return self._order(self._wander_to_center(observation, paths, allies), None, observation)
            return self._order(
                self._advance_with_visible_ally(observation, rear_paths, allies, enemy_goal or tile.at_center(observation)),
                None,
                observation,
            )
        ally_goal = self._ally_goal(observation, allies)
        if enemy_goal or ally_goal:
            return self._order(self._toward(observation, paths, enemy_goal or ally_goal), None, observation)
        return self._order(self._archer_center_path(observation, paths), None, observation)

    def _act_melee(self, observation: SkirmishObservation, paths: list[PathOption], enemies: list[VisibleUnit], allies: list[VisibleUnit]) -> SkirmishAction:
        unit_type = me.unit_type(observation)
        attack_range = units.STATS[unit_type].attack_range
        ally_archers = self._ally_archer_positions(allies)
        # Hill and across-water enemies stay in `enemies` (still threats) but are not engaged.
        engageable = [enemy for enemy in enemies if self._melee_can_engage(observation, enemy, ally_archers)]
        adjacent = [enemy for enemy in engageable if self._distance(me.position(observation), enemy["position"]) <= attack_range]
        if adjacent:
            target = self._select_target(observation, paths, adjacent)
            stay = next(option for option in paths if option[0] == 0)
            return self._attack_or_flee(observation, paths, [stay], target, enemies)

        here = me.position(observation)
        uphill = [
            enemy
            for enemy in enemies
            if self._distance(here, enemy["position"]) <= 1
            and tile.terrain_at(observation, enemy["position"])["terrain"] == "hill"
        ]
        if uphill:
            # Next to an enemy on a hill we will not attack: step out of its reach and wait for it
            # to come down (or for our archers), rather than taking its +1 downhill strikes.
            threats = [(enemy["type"], enemy["position"]) for enemy in enemies]
            return self._order(self._flee(observation, paths, threats), None, observation)

        archer_threats = self._archer_threats(observation, enemies)
        # Footmen (12 HP vs 2-damage arrows) only give way to archers they cannot tank; otherwise
        # they keep advancing with the line instead of fleeing every archer within 8 tiles.
        if archer_threats and unit_type == "footman" and self._can_tank(observation, here):
            archer_threats = []
        if archer_threats:
            opportunistic = self._select_target(observation, paths, [enemy for enemy in engageable if enemy["type"] in {"cavalry", "footman"}])
            if opportunistic is not None:
                attack_paths = self._attack_paths(paths, opportunistic, unit_type)
                if attack_paths:
                    return self._attack_or_flee(observation, paths, attack_paths, opportunistic, enemies)
            return self._archer_response(observation, paths, archer_threats)

        # Target order: one-hit kill > attackable now > archer > footman > cavalry (see _target_utility).
        target = self._select_target(observation, paths, engageable)
        if target is not None:
            attack_paths = self._attack_paths(paths, target, unit_type)
            if attack_paths:
                return self._attack_or_flee(observation, paths, attack_paths, target, enemies)
        return self._order(self._melee_move(observation, paths, enemies, allies, engageable, ally_archers), None, observation)

    def _melee_move(
        self,
        observation: SkirmishObservation,
        paths: list[PathOption],
        enemies: list[VisibleUnit],
        allies: list[VisibleUnit],
        engageable: list[VisibleUnit],
        ally_archers: list[Position],
    ) -> int:
        """Where melee walks when it has no strike this turn. Move order (user spec):
        enemies attacking allies > shield wall > allies > enemies > center.
        Protect priority is archer > footman > cavalry (_threatened_allies order).
        """
        here = me.position(observation)
        own_type = me.unit_type(observation)
        mp = units.STATS[own_type].movement_points
        engageable_ids = {enemy["unit_id"] for enemy in engageable}

        threatened = self._threatened_allies()
        if threatened:
            ally, enemy = threatened[0]
            # Engageable attacker: step into the gap between it and the ally (this is how cavalry
            # screens our archers). Hill/across-water attacker: stay beside the ally for the archers.
            if enemy["unit_id"] in engageable_ids:
                goal = self._screen_tile(ally["position"], enemy["position"])
            else:
                goal = self._copy(ally["position"])
            return self._toward(observation, paths, goal)
        alert = self._fresh_alert()
        if alert is not None:
            # An ally out of our sight reported "enemy attacking me" by chat: go help.
            return self._toward(observation, paths, alert)

        if own_type == "footman" and enemies:
            # Only nearby footmen (within 4), so the wall forms locally instead of across the map.
            partners = [ally for ally in allies if ally["type"] == "footman" and self._distance(here, ally["position"]) <= 4]
            if partners and not any(self._distance(here, ally["position"]) == 1 for ally in partners):
                partner = min(partners, key=lambda ally: self._distance(here, ally["position"]))
                return self._toward(observation, paths, partner["position"])

        # Merge groups: with no enemy in sight, a bigger group we heard about by chat (and cannot see)
        # outranks our local grouping, so small groups join into one army.
        bigger = None if enemies else self._bigger_group(observation)
        if bigger is not None:
            return self._toward(observation, paths, bigger)

        # Allies step, as a formation: with an allied archer to anchor on and a known enemy front,
        # cavalry holds the front line 3 tiles ahead of the archer and footmen wall 1 tile ahead.
        # Without that, just regroup with the nearest ally (archer first) when beyond our MP radius.
        group = self._protect_goal(observation, allies)
        front = self._front_point(observation)
        anchor = group
        if group is not None and front is not None and ally_archers:
            archer = min(ally_archers, key=lambda position: self._distance(here, position))
            anchor = self._formation_slot(archer, front, FORMATION_DEPTH[own_type])
            if self._distance(here, anchor) > 1:
                return self._toward(observation, paths, anchor)
        elif group is not None and self._distance(here, group) > mp:
            return self._toward(observation, paths, group)

        target = self._select_target(observation, paths, engageable)
        if target is not None:
            safer_paths = self._safe_paths(paths, [target])
            # Advance on the enemy together: only ends that stay within our MP of the formation
            # anchor and that we can tank, so the group moves as one at a safe distance.
            if anchor is not None:
                grouped = [
                    option
                    for option in (safer_paths or paths)
                    if self._distance(option[1], anchor) <= mp and self._can_tank(observation, option[1])
                ]
                return self._toward(observation, grouped or [(0, here)], target["position"])
            return self._toward(observation, safer_paths or paths, target["position"])
        if ally_archers and len(engageable) < len(enemies):
            # Enemies left to the archers (on a hill or across water): hold near the closest archer.
            guard = min(ally_archers, key=lambda position: self._distance(here, position))
            return self._toward(observation, paths, guard)
        enemy_goal = self._last_seen_non_archer_goal() or self._last_seen_goal("archer")
        if enemy_goal is not None:
            return self._toward(observation, paths, enemy_goal)
        center = tile.at_center(observation)
        if allies:
            return self._wander_to_center(observation, paths, allies)
        return self._toward(observation, paths, center)

    def _wander_to_center(self, observation: SkirmishObservation, paths: list[PathOption], allies: list[VisibleUnit]) -> int:
        """Nothing known (task 9 spec): head for the center, but near allies wander loosely, never
        leaving the closest ally's view distance, so the army spreads out to search together.
        """
        here = me.position(observation)
        center = tile.at_center(observation)
        closest = min(allies, key=lambda ally: self._distance(here, ally["position"]))
        reach = units.STATS[closest["type"]].vision
        linked = [option for option in paths if self._distance(option[1], closest["position"]) <= reach] or paths
        turns = self._route_turns(observation, [center])
        best = min(self._arrival(turns, option[1], center) for option in linked)
        # Wander: any linked end that is at most one activation slower than the best, chosen at random.
        loose = [option for option in linked if self._arrival(turns, option[1], center) <= best + 1]
        return self._rng.choice(sorted(loose, key=lambda option: self._tie(option[0])))[0]

    def _front_point(self, observation: SkirmishObservation) -> Position | None:
        """Where the enemy front is: middle of visible enemies, else the freshest remembered one."""
        if self._enemies_now:
            q = round(sum(enemy["position"]["q"] for enemy in self._enemies_now) / len(self._enemies_now))
            r = round(sum(enemy["position"]["r"] for enemy in self._enemies_now) / len(self._enemies_now))
            return {"q": q, "r": r}
        return self._search_goal(self._recent(self._enemy_memory, ENEMY_MEMORY_ROUNDS))

    def _formation_slot(self, archer: AxialPosition, front: AxialPosition, depth: int) -> Position:
        """Free tile about `depth` steps from our archer toward the enemy front."""
        distance = self._distance(archer, front)
        slot = self._hex_lerp(archer, front, min(depth, distance) / distance) if distance else self._key(archer)
        here = self._key(self._here)
        candidates = [slot, *self._neighbor_keys(slot)]
        free = [
            spot
            for spot in candidates
            if self._cost_grid.get(spot) is not None and spot not in self._marsh and (spot not in self._blocked or spot == here)
        ]
        if not free:
            return self._copy(archer)
        best = min(
            free,
            key=lambda spot: (self._key_distance(spot, slot), self._key_distance(spot, self._key(front)), self._tie_spot(spot)),
        )
        return {"q": best[0], "r": best[1]}

    def _behind_melee(self, paths: list[PathOption], allies: Iterable[VisibleUnit], front: AxialPosition) -> list[PathOption]:
        """Archer paths that stay at least a tile behind our most forward visible melee ally."""
        melee = [ally for ally in allies if ally["type"] in {"footman", "cavalry"}]
        if not melee:
            return paths
        lead = min(self._distance(ally["position"], front) for ally in melee)
        return [option for option in paths if self._distance(option[1], front) >= lead + 1]

    def _protect_goal(self, observation: SkirmishObservation, allies: Iterable[VisibleUnit]) -> Position | None:
        """Ally to group with: archer > footman > cavalry, nearest first; memory if none in sight."""
        here = me.position(observation)
        choices = [(ally["type"], ally["position"]) for ally in allies] or [
            (record["type"], record["position"]) for record in self._recent(self._ally_memory, ALLY_MEMORY_ROUNDS)
        ]
        if not choices:
            return None
        best = min(choices, key=lambda choice: (-TYPE_RANK[choice[0]], self._distance(here, choice[1])))
        return self._copy(best[1])

    def _attack_or_flee(
        self,
        observation: SkirmishObservation,
        paths: list[PathOption],
        attack_paths: list[PathOption],
        target: VisibleUnit,
        enemies: list[VisibleUnit],
    ) -> SkirmishAction:
        """Attack from the best tile if we can tank what reaches it (or the hit kills); else flee.

        Protecting an allied archer in trouble always attacks. Cavalry uses its own choice order.
        """
        if me.unit_type(observation) == "cavalry":
            return self._cavalry_choice(observation, paths, attack_paths, target, enemies)
        best = self._best_attack_path(observation, attack_paths, target)
        if (
            self._can_tank(observation, best[1])
            or self._can_kill_this_turn(observation, paths, target)
            or self._protecting(target, {"archer"})
        ):
            return self._order(best[0], target["unit_id"], observation)
        threats = [(enemy["type"], enemy["position"]) for enemy in enemies]
        return self._order(self._flee(observation, paths, threats), None, observation)

    def _cavalry_choice(
        self,
        observation: SkirmishObservation,
        paths: list[PathOption],
        attack_paths: list[PathOption],
        target: VisibleUnit,
        enemies: list[VisibleUnit],
    ) -> SkirmishAction:
        """Cavalry choice order (user spec): one-hit kill > tank one turn and kill next with no other
        enemy able to reach us > flee if a safer tile exists > attack.

        Protecting any ally the target can strike skips the caution and attacks.
        """
        best = self._best_attack_path(observation, attack_paths, target)
        attack = self._order(best[0], target["unit_id"], observation)
        if self._can_kill_this_turn(observation, paths, target) or self._protecting(target, TYPE_RANK):
            return attack
        here = me.position(observation)
        target_key = self._key(target["position"])
        first_hit = self._expected_damage(observation, "cavalry", best[1], target, start=here)
        follow_up = self._expected_damage(observation, "cavalry", best[1], target)
        incoming = self._incoming(observation, best[1])
        others_near = self._incoming(observation, best[1], exclude=target_key) > 0
        if incoming < me.hit_points(observation) and target["hit_points"] <= first_hit + follow_up and not others_near:
            return attack
        threats = [(enemy["type"], enemy["position"]) for enemy in enemies]
        flee_path = self._flee(observation, paths, threats)
        flee_end = next(end for path_id, end in paths if path_id == flee_path)
        if self._incoming(observation, flee_end) < incoming:
            return self._order(flee_path, None, observation)
        return attack

    def _threat_sources(self, observation: SkirmishObservation) -> list[tuple[str, Key, float, bool]]:
        """Known enemies as (type, tile, weight, on_hill), cached per activation.

        An enemy that already acted this round strikes before our next activation only if the next
        shuffle puts it first (~half the time), so it counts 0.5. Unseen enemies remembered within
        the 2-round threat horizon also count 0.5, since they may have moved.
        """
        if self._sources is not None:
            return self._sources
        sources = []
        for enemy in self._enemies_now:
            weight = 0.5 if enemy.get("has_acted") else 1.0
            on_hill = tile.terrain_at(observation, enemy["position"])["terrain"] == "hill"
            sources.append((enemy["type"], self._key(enemy["position"]), weight, on_hill))
        seen_ids = {enemy["unit_id"] for enemy in self._enemies_now}
        for unit_id, record in self._enemy_memory.items():
            if unit_id not in seen_ids and self._round - record["round"] <= THREAT_MEMORY_ROUNDS:
                on_hill = tile.terrain_at(observation, record["position"])["terrain"] == "hill"
                sources.append((record["type"], self._key(record["position"]), 0.5, on_hill))
        self._sources = sources
        return sources

    def _reach(self, kind: str, source: Key) -> set[Key]:
        """Tiles an enemy of `kind` on `source` can end its walk on in one activation.

        Same rules as engine movement.walk: first step any cost, later steps within MP; other units
        block, except our own tile (we are moving off it). Cached per activation.
        """
        cache_key = (kind, source)
        cached = self._reach_cache.get(cache_key)
        if cached is not None:
            return cached
        mp = units.STATS[kind].movement_points
        blocked = self._blocked - {source, self._key(self._here)}
        reach = {source}
        spent = {source: 0}
        frontier = [(0, source)]
        while frontier:
            cost, key = heapq.heappop(frontier)
            if cost > spent[key]:
                continue
            for neighbor in self._neighbor_keys(key):
                enter = self._cost_grid.get(neighbor)
                if enter is None or neighbor in blocked:
                    continue
                total = cost + enter
                if key == source or total <= mp:
                    reach.add(neighbor)
                if total <= mp and total < spent.get(neighbor, UNREACHABLE):
                    spent[neighbor] = total
                    heapq.heappush(frontier, (total, neighbor))
        self._reach_cache[cache_key] = reach
        return reach

    def _strike_from(self, kind: str, source: Key, target: Key) -> tuple[bool, bool]:
        """(can strike `target` next activation, can do it with a charge) for an enemy on `source`.

        Travel-aware: a melee enemy must actually walk to a tile next to the target, so water and
        blocked tiles count. Cavalry charges if one of those tiles is >= 3 from its start.
        Archers shoot over anything, so only their short walk matters.
        """
        stats = units.STATS[kind]
        distance = self._key_distance(source, target)
        if distance > stats.movement_points + stats.attack_range:
            return False, False
        if kind == "archer":
            if distance <= stats.attack_range:
                return True, False
            reach = self._reach(kind, source)
            return any(self._key_distance(spot, target) <= stats.attack_range for spot in reach), False
        reach = self._reach(kind, source)
        spots = [spot for spot in self._neighbor_keys(target) if spot in reach]
        if not spots:
            return False, False
        return True, any(self._key_distance(source, spot) >= 3 for spot in spots)

    def _incoming(self, observation: SkirmishObservation, end: AxialPosition, exclude: Key | None = None) -> float:
        """Expected damage to us on `end` before our next activation, from every enemy that can reach it.

        Reach is travel-aware (see _strike_from); it was plain hex distance before. Melee attackers
        end next to us on unknown ground, so only our tile's hill/forest/shield counts; an archer is
        assumed to shoot from where it stands. Cavalry charge counts when _strike_from allows it and
        our tile does not deny it. `exclude` leaves out the enemy standing on that tile.
        """
        return sum(value for source, value in self._incoming_parts(observation, end) if source != exclude)

    def _incoming_parts(self, observation: SkirmishObservation, end: AxialPosition) -> list[tuple[Key, float]]:
        key = self._key(end)
        cached = self._incoming_cache.get(key)
        if cached is not None:
            return cached
        cell = tile.terrain_at(observation, end)
        on_hill, in_forest = cell["terrain"] == "hill", cell["feature"] == "forest"
        abilities = bool(observation["observation"]["parameters"]["unit_abilities"])
        shielded = abilities and me.unit_type(observation) == "footman" and any(
            ally["type"] == "footman" and self._distance(end, ally["position"]) == 1 for ally in self._allies_now
        )
        parts = []
        for kind, source, weight, source_on_hill in self._threat_sources(observation):
            can_strike, can_charge = self._strike_from(kind, source, key)
            if not can_strike:
                continue
            stats = units.STATS[kind]
            distance = self._key_distance(source, key)
            damage = stats.damage
            shooting_from_hill = kind == "archer" and source_on_hill and distance <= stats.attack_range
            if shooting_from_hill and not on_hill:
                damage += 1
            if on_hill and not shooting_from_hill:
                damage -= 1
            if kind == "archer" and in_forest and distance > 1:
                damage -= 1
            if kind == "cavalry" and abilities and can_charge and not in_forest and not shielded:
                damage += 2
            if shielded:
                damage -= 1
            parts.append((source, weight * max(1, damage)))
        self._incoming_cache[key] = parts
        return parts

    def _threatened_allies(self) -> list[tuple[VisibleUnit, VisibleUnit]]:
        """(ally, enemy) pairs where a visible enemy can strike a visible ally next activation.

        Ordered by protect priority archer > footman > cavalry, then the ally nearest to us.
        """
        if self._threatened is not None:
            return self._threatened
        pairs = []
        for ally in self._allies_now:
            ally_key = self._key(ally["position"])
            for enemy in self._enemies_now:
                if self._strike_from(enemy["type"], self._key(enemy["position"]), ally_key)[0]:
                    pairs.append((ally, enemy))
        pairs.sort(key=lambda pair: (-TYPE_RANK[pair[0]["type"]], self._distance(self._here, pair[0]["position"])))
        self._threatened = pairs
        return pairs

    def _protecting(self, target: VisibleUnit, ally_types: Iterable[str]) -> bool:
        """True if `target` can strike a visible ally of one of these types next activation."""
        ally_types = set(ally_types)
        return any(
            enemy["unit_id"] == target["unit_id"] and ally["type"] in ally_types
            for ally, enemy in self._threatened_allies()
        )

    def _screen_tile(self, ally_position: AxialPosition, enemy_position: AxialPosition) -> Position:
        """The free tile beside the ally that faces the enemy: standing there blocks or intercepts."""
        here, enemy = self._key(self._here), self._key(enemy_position)
        spots = [
            spot
            for spot in self._neighbor_keys(self._key(ally_position))
            if self._cost_grid.get(spot) is not None and spot not in self._marsh and (spot not in self._blocked or spot == here)
        ]
        if not spots:
            return self._copy(ally_position)
        best = min(spots, key=lambda spot: (self._key_distance(spot, enemy), self._tie_spot(spot)))
        return {"q": best[0], "r": best[1]}

    def _can_tank(self, observation: SkirmishObservation, end: AxialPosition) -> bool:
        """True if the expected incoming damage on `end` stays under our tank threshold."""
        own_type = me.unit_type(observation)
        return self._incoming(observation, end) < me.hit_points(observation) * TANK_FRACTION[own_type]

    def _melee_can_engage(self, observation: SkirmishObservation, enemy: VisibleUnit, ally_archers: list[Position]) -> bool:
        """Melee never picks an enemy on a hill (archers take it, or we wait for it to come down).
        Enemies across water are left to archers too, unless we have no archers left."""
        if tile.terrain_at(observation, enemy["position"])["terrain"] == "hill":
            return False
        return not (ally_archers and self._water_between(me.position(observation), enemy["position"]))

    def _ally_archer_positions(self, allies: Iterable[VisibleUnit]) -> list[Position]:
        """Allied archers in sight, else ones remembered (ally memory lasts 2 activations)."""
        seen = [self._copy(ally["position"]) for ally in allies if ally["type"] == "archer"]
        if seen:
            return seen
        return [self._copy(record["position"]) for record in self._recent(self._ally_memory, ALLY_MEMORY_ROUNDS, "archer")]

    def _water_between(self, first: AxialPosition, second: AxialPosition) -> bool:
        """True if the straight hex line between two tiles crosses water."""
        steps = self._distance(first, second)
        return any(self._hex_lerp(first, second, index / steps) in self._water for index in range(1, steps))

    def _hex_lerp(self, first: AxialPosition, second: AxialPosition, t: float) -> Key:
        """The tile a fraction `t` of the way along the straight hex line from first to second."""
        # A tiny nudge keeps the line off exact tile borders so rounding is consistent. Blue nudges
        # the mirrored way, so both sides round lines identically on the point-symmetric field.
        nudge = -1 if self._mirror else 1
        q0, r0 = first["q"] + nudge * 1e-6, first["r"] + nudge * 2e-6
        q = q0 + (second["q"] - q0) * t
        r = r0 + (second["r"] - r0) * t
        s = -q - r
        rq, rr, rs = round(q), round(r), round(s)
        dq, dr, ds = abs(rq - q), abs(rr - r), abs(rs - s)
        if dq > dr and dq > ds:
            rq = -rr - rs
        elif dr > ds:
            rr = -rq - rs
        return (rq, rr)

    def _cavalry_engage_margin(self, observation: SkirmishObservation, enemy_archer: VisibleUnit) -> float | None:
        """EV calc: HP cavalry expects to keep after killing this archer (None if unreachable).

        Army scale: each turn of the fight costs the damage of every enemy that can reach the
        archer's tile, not just the archer's own arrows.
        """
        archer_pos = enemy_archer["position"]
        here = me.position(observation)
        # Activations until cavalry stands next to the archer (turn-based routes, not raw cost).
        sides = [spot for spot in tile.neighbors(archer_pos).values() if self._cost_grid.get(self._key(spot)) is not None]
        turns = self._route_turns(observation, sides)
        turns_to_adjacent = turns.get(self._key(here))
        if turns_to_adjacent is None:
            return None
        turns_to_close = max(1, turns_to_adjacent)
        archer = {"type": "archer", "position": archer_pos}
        # The first hit can charge if the final approach walks >= 3 tiles; staying adjacent never does.
        charge_start = here if turns_to_close > 1 or self._distance(here, archer_pos) >= 4 else None
        first_hit = self._expected_damage(observation, "cavalry", archer_pos, archer, start=charge_start)
        follow_up = self._expected_damage(observation, "cavalry", archer_pos, archer)
        remaining_hp = enemy_archer.get("hit_points", units.STATS["archer"].hit_points) - first_hit
        turns_to_kill = 1 + max(0, -(-remaining_hp // follow_up))
        dmg_from_archer = self._expected_damage(observation, "archer", archer_pos, self._self_unit(observation))
        per_turn = max(dmg_from_archer, self._incoming(observation, archer_pos))
        projected_loss = per_turn * max(0, turns_to_close + turns_to_kill - 1)
        return me.hit_points(observation) - projected_loss

    def _archer_response(self, observation: SkirmishObservation, paths: list[PathOption], threats: list[tuple[str, Position]]) -> SkirmishAction:
        safe_paths = self._safe_memory_paths(paths, threats)
        if me.unit_type(observation) == "footman":
            return self._order(self._flee(observation, safe_paths or paths, threats), None, observation)
        # Cavalry: if a favorable trade exists with a visible archer, engage it; else flee.
        ally_archers = self._ally_archer_positions(visible.allies(observation))
        archers = [
            enemy
            for enemy in visible.enemies(observation)
            if enemy["type"] == "archer" and self._melee_can_engage(observation, enemy, ally_archers)
        ]
        # Compare every engageable archer and take the best trade, not just the first one seen.
        SAFETY_FLOOR = 3
        margins = [(self._cavalry_engage_margin(observation, archer), archer) for archer in archers]
        margins = [(margin, archer) for margin, archer in margins if margin is not None]
        best = max(margins, key=lambda pair: pair[0], default=None)
        if best is not None and best[0] > SAFETY_FLOOR:
            target = best[1]
            attack_paths = self._attack_paths(paths, target, "cavalry")
            if attack_paths:
                return self._order(self._best_attack_path(observation, attack_paths, target)[0], target["unit_id"], observation)
            return self._order(self._toward(observation, paths, target["position"]), None, observation)
        return self._order(self._flee(observation, safe_paths or paths, threats), None, observation)

    def _advance_with_visible_ally(
        self, observation: SkirmishObservation, paths: list[PathOption], allies: Iterable[VisibleUnit], goal: Position
    ) -> int:
        """Advance toward goal while keeping at least one currently visible ally in vision."""
        vision = units.STATS[me.unit_type(observation)].vision
        linked_paths = [
            path
            for path in paths
            if any(self._distance(path[1], ally["position"]) <= vision for ally in allies)
        ]
        # Staying put is always linked to an ally already visible, but retain the general fallback
        # so this helper remains safe if future rules change what visibility means.
        return self._toward(observation, linked_paths or paths, goal)

    def _paths(self, observation: SkirmishObservation) -> list[PathOption]:
        """Legal paths from the mask, minus any that walk into wasteland or end in marsh.

        Staying (path 0) is always kept, even on marsh, so the list is never empty.
        """
        here = me.position(observation)
        check_waste = bool(observation["observation"]["parameters"].get("wasteland"))
        options = []
        # Marsh-ending paths are kept aside for _toward's marsh-deadlock fallback only.
        self._marsh_paths: list[PathOption] = []
        for path_id in action.legal_paths(observation):
            end = tile.at_path_end(here, path_id)
            if path_id and self._key(end) in self._marsh:
                if not (check_waste and self._enters_waste(observation, here, path_id)):
                    self._marsh_paths.append((path_id, end))
                continue
            if check_waste and path_id and self._enters_waste(observation, here, path_id):
                continue
            options.append((path_id, end))
        return options

    @staticmethod
    def _enters_waste(observation: SkirmishObservation, here: AxialPosition, path_id: int) -> bool:
        position = here
        for direction in paths.decode(path_id):
            position = tile.neighbors(position)[direction]
            if tile.terrain_at(observation, position)["feature"] == "waste":
                return True
        return False

    def _route_turns(
        self,
        observation: SkirmishObservation,
        goals: Iterable[AxialPosition],
        marsh_stop: bool = False,
        allies_passable: bool = False,
    ) -> dict[Key, int]:
        """Fewest activations from each tile to any goal tile, for this unit's movement points.

        Replaces the Season 2 Dijkstra on total move cost: arrival time matters, not MP spent.
        Per engine movement.walk, one activation reaches any adjacent tile whatever it costs, or any
        tile whose entered costs sum to at most MP. Goals seed at 0 even when occupied; other
        visible units block. Built in layers: layer k is every tile one activation from layer k-1.
        Marsh can be walked through but never ends a turn (our own rule, not the engine's), unless
        `marsh_stop` is set for the marsh-deadlock fallback. Allies only block within
        ALLY_BLOCK_RADIUS, or not at all with `allies_passable` (anti-stuck re-plan).
        """
        goal_keys = frozenset(self._key(goal) for goal in goals)
        cache_key = (goal_keys, marsh_stop, allies_passable)
        cached = self._turn_cache.get(cache_key)
        if cached is not None:
            return cached
        mp = units.STATS[me.unit_type(observation)].movement_points
        here = self._key(me.position(observation))
        blocked = {self._key(enemy["position"]) for enemy in self._enemies_now} if allies_passable else self._route_blocked

        def standable(key: Key) -> bool:
            return key == here or (self._cost_grid.get(key) is not None and key not in blocked)

        def can_stop(key: Key) -> bool:
            return key == here or (standable(key) and (marsh_stop or key not in self._marsh))

        turns = {key: 0 for key in goal_keys}
        layer = list(goal_keys)
        step = 0
        while layer:
            step += 1
            # Reverse bounded Dijkstra: spent[T] = MP needed to walk from T into the layer.
            spent = {key: 0 for key in layer}
            frontier = [(0, key) for key in layer]
            found: set[Key] = set()
            while frontier:
                cost, key = heapq.heappop(frontier)
                if cost > spent[key]:
                    continue
                enter = self._cost_grid.get(key)
                if enter is None:
                    if key not in goal_keys:
                        continue
                    # An impassable goal (e.g. water at center) is approached, so only adjacency counts.
                    enter = mp + 1
                for neighbor in self._neighbor_keys(key):
                    if not standable(neighbor):
                        continue
                    total = cost + enter
                    # cost == 0: key is a layer tile, and a single step is legal whatever it costs.
                    if neighbor not in turns and (cost == 0 or total <= mp) and can_stop(neighbor):
                        found.add(neighbor)
                    if total <= mp and total < spent.get(neighbor, UNREACHABLE):
                        spent[neighbor] = total
                        heapq.heappush(frontier, (total, neighbor))
            for key in found:
                turns[key] = step
            layer = list(found)
        self._turn_cache[cache_key] = turns
        return turns

    def _arrival(self, turns: dict[Key, int], end: AxialPosition, goal: AxialPosition) -> int:
        return turns.get(self._key(end), UNREACHABLE + self._distance(end, goal))

    def _terrain_preference(self, observation: SkirmishObservation, end: AxialPosition) -> int:
        """Tiebreak between equally fast options; higher is better.

        Wasteland never appears (filtered in _paths). Marsh is avoided. Forest is preferred when no
        visible enemy is adjacent, since cover only works against attacks from beyond 1 tile. A hill
        is preferred when we would strike a lower enemy from it. Grass is neutral. A footman also
        prefers standing next to an allied footman (shield wall) once enemies are in sight.
        With enemies in sight we also stay near forest/hill and avoid tiles boxed in by the board
        edge, water, or marsh, so there is always room to retreat.
        """
        cell = tile.terrain_at(observation, end)
        enemies = self._enemies_now
        own_type = me.unit_type(observation)
        score = 0
        if cell["feature"] == "marsh":
            score -= 2
        if cell["feature"] == "forest" and all(self._distance(end, enemy["position"]) > 1 for enemy in enemies):
            score += 2
        if cell["terrain"] == "hill":
            reach = units.STATS[own_type].attack_range
            if any(
                self._distance(end, enemy["position"]) <= reach
                and tile.terrain_at(observation, enemy["position"])["terrain"] != "hill"
                for enemy in enemies
            ):
                score += 2
        if own_type == "footman" and enemies and any(
            ally["type"] == "footman" and self._distance(end, ally["position"]) == 1 for ally in self._allies_now
        ):
            score += 1
        if enemies:
            key = self._key(end)
            if self._near_cover.get(key):
                score += 1
            # 18 tiles lie within 2 steps; under half open is a trap, under three quarters is cramped.
            openness = self._openness.get(key, 0)
            if openness <= 9:
                score -= 3
            elif openness <= 13:
                score -= 1
        return score

    def _best_attack_path(self, observation: SkirmishObservation, attack_paths: Iterable[PathOption], target: VisibleUnit) -> PathOption:
        """Strike from the tile that hits hardest (charge, high ground), then by terrain preference."""
        here = me.position(observation)
        own_type = me.unit_type(observation)
        return max(
            attack_paths,
            key=lambda option: (
                self._expected_damage(observation, own_type, option[1], target, start=here),
                self._terrain_preference(observation, option[1]),
                -self._distance(option[1], target["position"]),
                -self._tie(option[0]),
            ),
        )

    def _target_utility(self, observation: SkirmishObservation, paths: list[PathOption], enemy: VisibleUnit) -> tuple:
        """Target order (user spec): one-hit kill > attackable this turn > focus fire > archer >
        footman > cavalry, nearer on ties. Replaces the Season 2 weighted score; counter-attack risk
        is no longer scored here, the attack/flee rules (_incoming, with travel-aware charge) handle it.

        Focus fire (G5): an enemy already being attacked (hurt, or next to one of our units) is
        preferred, so everyone who can see it piles onto the same one.
        """
        own_type = me.unit_type(observation)
        own_range = units.STATS[own_type].attack_range
        attack_ends = [end for _, end in paths if self._distance(end, enemy["position"]) <= own_range]
        attackable = bool(attack_ends)
        here = me.position(observation)
        best_damage = max((self._expected_damage(observation, own_type, end, enemy, start=here) for end in attack_ends), default=0)
        killable = attackable and enemy["hit_points"] <= best_damage
        engaged = (
            enemy["hit_points"] < units.STATS[enemy["type"]].hit_points
            or any(self._distance(ally["position"], enemy["position"]) <= 1 for ally in self._allies_now)
            # A broadcast focus call from the last round counts as "being attacked" too.
            or (USE_FOCUS_CALLS and self._round - self._focus.get(enemy["unit_id"], -99) <= 1)
        )
        return (killable, attackable, engaged, TYPE_RANK[enemy["type"]], -self._distance(here, enemy["position"]))

    def _select_target(self, observation: SkirmishObservation, paths: list[PathOption], candidates: Iterable[VisibleUnit]) -> VisibleUnit | None:
        candidates = list(candidates)
        if not candidates:
            return None
        return max(candidates, key=lambda enemy: self._target_utility(observation, paths, enemy))

    def _attack_paths(self, paths: Iterable[PathOption], target: VisibleUnit, attacker_type: str) -> list[PathOption]:
        return [path for path in paths if self._distance(path[1], target["position"]) <= units.STATS[attacker_type].attack_range]

    def _expected_damage(
        self,
        observation: SkirmishObservation,
        attacker_type: str,
        attacker_position: AxialPosition,
        target: VisibleUnit,
        start: AxialPosition | None = None,
    ) -> int:
        """Expected damage with hill/forest terrain and, when enabled, abilities (mirrors combat.damage).

        ``start`` is where the attacker began its activation; charge is only counted when it is given.
        """
        attacker_terrain = tile.terrain_at(observation, attacker_position)["terrain"]
        defender_tile = tile.terrain_at(observation, target["position"])
        abilities = bool(observation["observation"]["parameters"]["unit_abilities"])
        shielded = abilities and self._shield_wall(observation, target)
        # Forest cover and a shield wall each deny the charge outright.
        charging = (
            abilities
            and attacker_type == "cavalry"
            and start is not None
            and self._distance(start, attacker_position) >= 3
            and defender_tile["feature"] != "forest"
            and not shielded
        )
        total = units.STATS[attacker_type].damage
        if charging:
            total += 2
        if attacker_terrain == "hill" and defender_tile["terrain"] != "hill":
            total += 1
        if attacker_terrain != "hill" and defender_tile["terrain"] == "hill":
            total -= 1
        if defender_tile["feature"] == "forest" and self._distance(attacker_position, target["position"]) > 1:
            total -= 1
        if shielded:
            total -= 1
        return max(1, total)

    def _shield_wall(self, observation: SkirmishObservation, target: VisibleUnit) -> bool:
        """A footman next to another footman of its own side (as far as this unit can see)."""
        if target.get("type") != "footman":
            return False
        seen = [self._self_unit(observation), *visible.allies(observation), *visible.enemies(observation)]
        return any(
            other["side"] == target.get("side")
            and other["type"] == "footman"
            and other["unit_id"] != target.get("unit_id")
            and self._distance(other["position"], target["position"]) == 1
            for other in seen
        )

    @staticmethod
    def _self_unit(observation: SkirmishObservation) -> VisibleUnit:
        """This unit in the same shape as a visible unit, so damage helpers can target it."""
        return {
            "unit_id": me.unit_id(observation),
            "side": me.side(observation),
            "type": me.unit_type(observation),
            "position": me.position(observation),
            "hit_points": me.hit_points(observation),
        }

    def _can_kill_this_turn(self, observation: SkirmishObservation, paths: list[PathOption], enemy: VisibleUnit) -> bool:
        attack_range = units.STATS[me.unit_type(observation)].attack_range
        attacker_type = me.unit_type(observation)
        here = me.position(observation)
        return any(
            self._distance(end, enemy["position"]) <= attack_range
            and enemy["hit_points"] <= self._expected_damage(observation, attacker_type, end, enemy, start=here)
            for _, end in paths
        )

    def _safe_paths(self, paths: Iterable[PathOption], threats: Iterable[VisibleUnit]) -> list[PathOption]:
        return self._safe_memory_paths(paths, [(threat["type"], threat["position"]) for threat in threats])

    def _safe_memory_paths(self, paths: Iterable[PathOption], threats: Iterable[tuple[str, Position]]) -> list[PathOption]:
        threats = list(threats)
        return [path for path in paths if all(self._distance(path[1], position) > self._threat_range(kind) for kind, position in threats)]

    def _archer_threats(
        self, observation: SkirmishObservation, enemies: Iterable[VisibleUnit]
    ) -> list[tuple[str, Position]]:
        here = me.position(observation)
        threats = [
            (enemy["type"], enemy["position"])
            for enemy in enemies
            if enemy["type"] == "archer"
            and self._distance(here, enemy["position"]) <= self._threat_range("archer")
        ]
        # Unseen archers count as threats only while the sighting is fresh enough to act on (2 rounds).
        visible_ids = {enemy["unit_id"] for enemy in enemies}
        for unit_id, record in self._enemy_memory.items():
            if (
                record["type"] == "archer"
                and unit_id not in visible_ids
                and self._round - record["round"] <= THREAT_MEMORY_ROUNDS
                and self._distance(here, record["position"]) <= self._threat_range("archer")
            ):
                threats.append(("archer", record["position"]))
        return threats

    def _ally_goal(self, observation: SkirmishObservation, allies: Iterable[VisibleUnit]) -> Position | None:
        choices = [(ally["type"], ally["position"]) for ally in allies] or [
            (record["type"], record["position"]) for record in self._recent(self._ally_memory, ALLY_MEMORY_ROUNDS)
        ]
        if not choices:
            return None
        return self._copy(min(choices, key=lambda choice: (self._distance(me.position(observation), choice[1]), choice[0] != "cavalry"))[1])

    def _last_seen_goal(self, unit_type: str) -> Position | None:
        """Search goal: freshest remembered enemy of this type (up to 3 rounds), nearest on ties."""
        return self._search_goal(self._recent(self._enemy_memory, ENEMY_MEMORY_ROUNDS, unit_type))

    def _last_seen_non_archer_goal(self) -> Position | None:
        # Was a random pick each activation, which made units dither between stale sightings.
        records = [record for record in self._recent(self._enemy_memory, ENEMY_MEMORY_ROUNDS) if record["type"] != "archer"]
        return self._search_goal(records)

    def _search_goal(self, records: list[dict]) -> Position | None:
        if not records:
            return None
        best = min(records, key=lambda record: (-record["round"], self._distance(self._here, record["position"])))
        return self._copy(best["position"])

    def _toward(self, observation: SkirmishObservation, paths: Iterable[PathOption], goal: Position) -> int:
        """Fewest activations to goal first; terrain preference breaks ties, then nearness.

        Anti-stuck: after STUCK_ACTIVATIONS on one tile while still short of the goal, re-plan with
        allies passable and take a move rather than staying. Marsh deadlock: if no route avoids
        stopping in marsh (e.g. a footman facing a marsh band), allow marsh stops this once.
        """
        center = tile.at_center(observation)
        paths = list(paths)
        here = me.position(observation)
        stuck = False
        if self._still >= STUCK_ACTIVATIONS:
            turns = self._route_turns(observation, [goal])
            stuck = self._arrival(turns, here, goal) > 1
        turns = self._route_turns(observation, [goal], allies_passable=stuck)
        if stuck:
            moving = [option for option in paths if option[0] != 0]
            paths = moving or paths

        def key(option: PathOption) -> tuple:
            return (
                self._arrival(turns, option[1], goal),
                -self._terrain_preference(observation, option[1]),
                self._distance(option[1], goal),
                self._distance(option[1], center),
                self._tie(option[0]),
            )

        best = min(paths, key=key)
        if self._arrival(turns, best[1], goal) >= UNREACHABLE and self._marsh_paths:
            turns = self._route_turns(observation, [goal], marsh_stop=True, allies_passable=stuck)
            best = min([*paths, *self._marsh_paths], key=key)
        return best[0]

    def _flee(
        self,
        observation: SkirmishObservation,
        paths: Iterable[PathOption],
        threats: Iterable[tuple[str, Position]],
        keep_in_range: VisibleUnit | None = None,
    ) -> int:
        paths = list(paths)
        threats = list(threats)
        here = me.position(observation)
        center = tile.at_center(observation)
        center_turns = self._route_turns(observation, [center])
        # Army scale: first minimise expected incoming damage from every known enemy, then keep the
        # old distance margin from the named threats as a tiebreak.
        least_incoming = min(self._incoming(observation, option[1]) for option in paths)
        paths = [option for option in paths if self._incoming(observation, option[1]) == least_incoming]
        if keep_in_range is not None:
            # Archer flee: among equally safe tiles, prefer ones that still keep the target in range.
            reach = units.STATS[me.unit_type(observation)].attack_range
            in_range = [option for option in paths if self._distance(option[1], keep_in_range["position"]) <= reach]
            paths = in_range or paths
        safety_margin = lambda option: min(
            (self._distance(option[1], position) - self._threat_range(kind) for kind, position in threats),
            default=0,
        )
        best_margin = max(safety_margin(option) for option in paths)
        safest_paths = [option for option in paths if safety_margin(option) == best_margin]
        def center_progress(option):
            # Measured in activations to the center now, not total move cost.
            return self._arrival(center_turns, here, center) - self._arrival(center_turns, option[1], center)
        best_center_progress = max(center_progress(option) for option in safest_paths)

        if best_center_progress <= 0:
            last_ally = self._ally_goal(observation, [])
            if last_ally is not None:
                return min(
                    safest_paths,
                    key=lambda option: (
                        self._distance(option[1], last_ally),
                        -self._terrain_preference(observation, option[1]),
                        -sum(self._distance(option[1], position) for _, position in threats),
                        self._tie(option[0]),
                    ),
                )[0]

        return max(
            safest_paths,
            key=lambda option: (
                center_progress(option),
                self._terrain_preference(observation, option[1]),
                sum(self._distance(option[1], position) for _, position in threats),
                -self._tie(option[0]),
            ),
        )[0]

    def _archer_center_path(self, observation: SkirmishObservation, paths_to_score: list[PathOption]) -> int:
        """Reach center without needlessly leading with the archer in the opening."""
        center = tile.at_center(observation)
        best_distance = min(self._distance(end, center) for _, end in paths_to_score)
        comparable = [
            option
            for option in paths_to_score
            if self._distance(option[1], center) <= best_distance + 1
        ]
        forward = me.direction(observation)
        return min(
            comparable,
            key=lambda option: (
                paths.decode(option[0]).count(forward),
                self._distance(option[1], center),
                self._tie(option[0]),
            ),
        )[0]

    def _order(self, path_id: int, target_id: str | None, observation: SkirmishObservation) -> SkirmishAction:
        self._named_target = target_id
        if path_id == 0:
            return action.stay(target_id, observation) if target_id else action.stay()
        return action.move(path_id, target_id, observation) if target_id else action.move(path_id)

    def _tie(self, path_id: int) -> int:
        """Side-neutral last tiebreak for paths: Blue compares each path as its point-mirror.

        Raw path ids favour directions 1-2 (northeast/east), which is toward the enemy for Red but
        away from it for Blue. Mirroring each step (digit d -> d+3) makes Blue break ties exactly as
        Red would in the mirrored position; Red is unchanged.
        """
        if not self._mirror or path_id == 0:
            return path_id
        mirrored = self._tie_cache.get(path_id)
        if mirrored is None:
            mirrored = paths.encode(tuple((digit + 2) % 6 + 1 for digit in paths.decode(path_id)))
            self._tie_cache[path_id] = mirrored
        return mirrored

    def _tie_spot(self, key: Key) -> Key:
        """Side-neutral last tiebreak for tiles: Blue orders tiles as their point-mirror (-q, -r)."""
        return (-key[0], -key[1]) if self._mirror else key

    @staticmethod
    def _key(position: AxialPosition) -> Key:
        return (position["q"], position["r"])

    @staticmethod
    def _key_distance(first: Key, second: Key) -> int:
        dq, dr = first[0] - second[0], first[1] - second[1]
        return (abs(dq) + abs(dr) + abs(dq + dr)) // 2

    @staticmethod
    def _neighbor_keys(key: Key) -> list[Key]:
        return [(key[0] + dq, key[1] + dr) for dq, dr in HEX_OFFSETS]

    @staticmethod
    def _copy(position: AxialPosition) -> Position:
        return {"q": position["q"], "r": position["r"]}

    @staticmethod
    def _distance(first: AxialPosition, second: AxialPosition) -> int:
        return tile.distance(first, second)

    @staticmethod
    def _threat_range(unit_type: str) -> int:
        stats = units.STATS[unit_type]
        return stats.movement_points + stats.attack_range

    # Original template TODO notes, retained as course-reference milestones:
    # TODO(you) (completed): this unit stood still when forward was blocked. The agent now scores
    # every legal path and can take an alternate route toward its current strategic goal.
    # TODO(you) (completed): walking toward the nearest enemy was the entire strategy. Archer,
    # cavalry, and footman now use separate kiting, flee/engage, and support behavior.
    # TODO(you) (completed): only single steps were tried. Legal paths of up to four steps are now
    # evaluated, so cavalry can use its full movement allowance.

    # Optional: a reinforcement-learning hook called after every step with that step's
    # transition. Its time counts against the timing and episode budget. The order argument is
    # what act returned. It is named order so it does not shadow the action helpers.
    #
    # def learn(self, observation, order: SkirmishAction, reward: float, terminated: bool) -> None:
    #     ...

    # Optional: messaging. Season settings enable it from Season 3 onward. When enabled, chat runs
    # after a unit chooses its order and receives messages that arrived since its previous
    # activation. Return each message with a recipient and text. Use None to broadcast to both
    # sides, or a player id such as "player_2", not a unit id, to send directly to one ally. The
    # rosters in the observation map each player to its unit. By default, text is limited to 200
    # characters.
    # A direct message reaches its allied unit at its next activation, after that unit chooses its
    # own order. Every message is recorded and shown in replays, so nothing you send is ever secret.
    # Return nothing to stay silent.
    #
    # def chat(self, inbox: list[dict]) -> list[dict] | None:
    #     ...