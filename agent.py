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
        self._turn_cache: dict[frozenset[Key], dict[Key, int]] = {}
        self._incoming_cache: dict[Key, float] = {}
        self._sources: list[tuple[str, Key, float, bool]] | None = None
        self._round = observation["observation"]["round"]
        self._here = me.position(observation)
        self._remember(observation, enemies, allies)
        paths = self._paths(observation)
        if me.unit_type(observation) == "archer":
            return self._act_archer(observation, paths, enemies, allies)
        return self._act_melee(observation, paths, enemies, allies)

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
                    -option[0],
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
                    option[0],
                ),
            )
            return self._order(path[0], target["unit_id"], observation)

        if under_threat or firing_paths:
            # No safe or tankable firing tile exists: escaping outranks preserving the shot, so
            # search every legal path, not only firing-range ones (still shoot if the end allows).
            threat_pairs = [(enemy["type"], enemy["position"]) for enemy in melee_threats]
            best_path = self._flee(observation, paths, threat_pairs)
            end = next(end for path_id, end in paths if path_id == best_path)
            name = (
                target["unit_id"]
                if target is not None and self._distance(end, target["position"]) <= units.STATS["archer"].attack_range
                else None
            )
            return self._order(best_path, name, observation)

        enemy_goal = self._last_seen_goal("archer")
        if allies:
            return self._order(
                self._advance_with_visible_ally(observation, paths, allies, enemy_goal or tile.at_center(observation)),
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

        archer_threats = self._archer_threats(observation, enemies)
        if archer_threats:
            opportunistic = self._select_target(observation, paths, [enemy for enemy in engageable if enemy["type"] in {"cavalry", "footman"}])
            if opportunistic is not None:
                attack_paths = self._attack_paths(paths, opportunistic, unit_type)
                if attack_paths:
                    return self._attack_or_flee(observation, paths, attack_paths, opportunistic, enemies)
            return self._archer_response(observation, paths, archer_threats)

        target = self._select_target(observation, paths, [enemy for enemy in engageable if enemy["type"] in {"cavalry", "footman"}])
        if target is not None:
            attack_paths = self._attack_paths(paths, target, unit_type)
            if attack_paths:
                return self._attack_or_flee(observation, paths, attack_paths, target, enemies)
            if self._threatens_visible_ally(observation, target):
                return self._order(self._toward(observation, paths, target["position"]), None, observation)
            safer_paths = self._safe_paths(paths, [target])
            return self._order(self._toward(observation, safer_paths or paths, target["position"]), None, observation)
        if ally_archers and len(engageable) < len(enemies):
            # Enemies left to the archers (on a hill or across water): hold near the closest archer.
            here = me.position(observation)
            guard = min(ally_archers, key=lambda position: self._distance(here, position))
            return self._order(self._toward(observation, paths, guard), None, observation)
        return self._melee_rendezvous(observation, paths, allies)

    def _attack_or_flee(
        self,
        observation: SkirmishObservation,
        paths: list[PathOption],
        attack_paths: list[PathOption],
        target: VisibleUnit,
        enemies: list[VisibleUnit],
    ) -> SkirmishAction:
        """Attack from the best tile if we can tank what reaches it (or the hit kills); else flee."""
        best = self._best_attack_path(observation, attack_paths, target)
        if self._can_tank(observation, best[1]) or self._can_kill_this_turn(observation, paths, target):
            return self._order(best[0], target["unit_id"], observation)
        threats = [(enemy["type"], enemy["position"]) for enemy in enemies]
        return self._order(self._flee(observation, paths, threats), None, observation)

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

    def _incoming(self, observation: SkirmishObservation, end: AxialPosition) -> float:
        """Expected damage to us on `end` before our next activation, from every enemy that can reach it.

        Reach is movement + attack range (hex distance; water detours are not modelled yet). Melee
        attackers end next to us on unknown ground, so only our tile's hill/forest/shield counts;
        an archer is assumed to shoot from where it stands. Cavalry is assumed to charge when it
        starts far enough to walk 3 tiles and our tile allows it.
        """
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
        total = 0.0
        for kind, source, weight, source_on_hill in self._threat_sources(observation):
            stats = units.STATS[kind]
            distance = self._key_distance(source, key)
            if distance > stats.movement_points + stats.attack_range:
                continue
            damage = stats.damage
            shooting_from_hill = kind == "archer" and source_on_hill and distance <= stats.attack_range
            if shooting_from_hill and not on_hill:
                damage += 1
            if on_hill and not shooting_from_hill:
                damage -= 1
            if kind == "archer" and in_forest and distance > 1:
                damage -= 1
            if kind == "cavalry" and abilities and distance >= 4 and not in_forest and not shielded:
                damage += 2
            if shielded:
                damage -= 1
            total += weight * max(1, damage)
        self._incoming_cache[key] = total
        return total

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
        # A tiny nudge keeps the line off exact tile borders so rounding is consistent.
        q0, r0 = first["q"] + 1e-6, first["r"] + 2e-6
        for index in range(1, steps):
            t = index / steps
            q = q0 + (second["q"] - q0) * t
            r = r0 + (second["r"] - r0) * t
            s = -q - r
            rq, rr, rs = round(q), round(r), round(s)
            dq, dr, ds = abs(rq - q), abs(rr - r), abs(rs - s)
            if dq > dr and dq > ds:
                rq = -rr - rs
            elif dr > ds:
                rr = -rq - rs
            if (rq, rr) in self._water:
                return True
        return False

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

    def _melee_rendezvous(self, observation: SkirmishObservation, paths: list[PathOption], allies: list[VisibleUnit]) -> SkirmishAction:
        enemy_goal = self._last_seen_non_archer_goal()
        if allies:
            return self._order(
                self._advance_with_visible_ally(observation, paths, allies, enemy_goal or tile.at_center(observation)),
                None,
                observation,
            )
        ally_goal = self._ally_goal(observation, allies)
        if ally_goal is not None:
            return self._order(self._toward(observation, paths, ally_goal), None, observation)
        return self._order(self._toward(observation, paths, enemy_goal or tile.at_center(observation)), None, observation)

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
        for path_id in action.legal_paths(observation):
            end = tile.at_path_end(here, path_id)
            if path_id and self._key(end) in self._marsh:
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

    def _route_turns(self, observation: SkirmishObservation, goals: Iterable[AxialPosition]) -> dict[Key, int]:
        """Fewest activations from each tile to any goal tile, for this unit's movement points.

        Replaces the Season 2 Dijkstra on total move cost: arrival time matters, not MP spent.
        Per engine movement.walk, one activation reaches any adjacent tile whatever it costs, or any
        tile whose entered costs sum to at most MP. Goals seed at 0 even when occupied; other
        visible units block. Built in layers: layer k is every tile one activation from layer k-1.
        Marsh can be walked through but never ends a turn (our own rule, not the engine's).
        """
        goal_keys = frozenset(self._key(goal) for goal in goals)
        cached = self._turn_cache.get(goal_keys)
        if cached is not None:
            return cached
        mp = units.STATS[me.unit_type(observation)].movement_points
        here = self._key(me.position(observation))

        def standable(key: Key) -> bool:
            return key == here or (self._cost_grid.get(key) is not None and key not in self._blocked)

        def can_stop(key: Key) -> bool:
            return key == here or (standable(key) and key not in self._marsh)

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
        self._turn_cache[goal_keys] = turns
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
                -option[0],
            ),
        )

    def _target_utility(self, observation: SkirmishObservation, paths: list[PathOption], enemy: VisibleUnit) -> float:
        """Weighted utility score: type value, damage dealt, kill bonus, threat removed, counter-attack risk."""
        own_type = me.unit_type(observation)
        own_range = units.STATS[own_type].attack_range
        attack_ends = [end for _, end in paths if self._distance(end, enemy["position"]) <= own_range]
        attackable = bool(attack_ends)
        here = me.position(observation)
        best_damage = max((self._expected_damage(observation, own_type, end, enemy, start=here) for end in attack_ends), default=0)
        killable = attackable and enemy["hit_points"] <= best_damage
        # Incoming charge needs the enemy's own start, which is unknown, so it is left to the threat model.
        incoming = self._expected_damage(observation, enemy["type"], enemy["position"], self._self_unit(observation)) if attackable else 0

        KILL_BONUS, ATTACKABLE_BONUS, THREATENS_ALLY_BONUS = 1000, 50, 30
        TYPE_VALUE = {"archer": 20, "cavalry": 12, "footman": 8}
        DAMAGE_WEIGHT, RISK_WEIGHT, DISTANCE_PENALTY = 2, 1.5, 1

        score = TYPE_VALUE[enemy["type"]] - DISTANCE_PENALTY * self._distance(me.position(observation), enemy["position"])
        if attackable:
            score += ATTACKABLE_BONUS + DAMAGE_WEIGHT * best_damage - RISK_WEIGHT * incoming
        if killable:
            score += KILL_BONUS
        if self._threatens_visible_ally(observation, enemy):
            score += THREATENS_ALLY_BONUS
        return score

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

    def _threatens_visible_ally(self, observation: SkirmishObservation, enemy: VisibleUnit) -> bool:
        attack_range = units.STATS[enemy["type"]].attack_range
        return any(
            self._distance(enemy["position"], ally["position"]) <= attack_range
            for ally in visible.allies(observation)
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
        """Fewest activations to goal first; terrain preference breaks ties, then nearness."""
        center = tile.at_center(observation)
        turns = self._route_turns(observation, [goal])
        return min(
            paths,
            key=lambda option: (
                self._arrival(turns, option[1], goal),
                -self._terrain_preference(observation, option[1]),
                self._distance(option[1], goal),
                self._distance(option[1], center),
                option[0],
            ),
        )[0]

    def _flee(
        self, observation: SkirmishObservation, paths: Iterable[PathOption], threats: Iterable[tuple[str, Position]]
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
                        option[0],
                    ),
                )[0]

        return max(
            safest_paths,
            key=lambda option: (
                center_progress(option),
                self._terrain_preference(observation, option[1]),
                sum(self._distance(option[1], position) for _, position in threats),
                -option[0],
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
                option[0],
            ),
        )[0]

    def _order(self, path_id: int, target_id: str | None, observation: SkirmishObservation) -> SkirmishAction:
        if path_id == 0:
            return action.stay(target_id, observation) if target_id else action.stay()
        return action.move(path_id, target_id, observation) if target_id else action.move(path_id)

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