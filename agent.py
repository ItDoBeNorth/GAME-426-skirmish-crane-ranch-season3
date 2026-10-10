"""A role-based Season 1 agent for Skirmish at Crane Reach.

Each unit runs a separate instance of this class. Read ``environment.md`` beside this file for
the rules, helpers, and later-season mechanics. Prepare episode state in ``reset``; the
constructor takes no arguments. Movement always comes from the action mask: this agent scores
complete legal paths rather than recreating movement rules.
"""

from __future__ import annotations

import random
from typing import Iterable

from sandbox.crane import action, me, paths, tile, units, visible
from sandbox.observation_types import AxialPosition, SkirmishAction, SkirmishObservation, VisibleUnit


Position = dict[str, int]
PathOption = tuple[int, AxialPosition]


class Agent:
    """Use archer kiting and local melee support to contest the center."""

    def reset(self, seed, observation) -> None:
        # Called once before each match. The opening observation is available here for
        # precomputation outside the decision clock. Each unit's memory remains private.
        self._activation = 0
        self._rng = random.Random(seed)
        self._last_seen_enemies: dict[str, tuple[str, Position]] = {}
        self._last_seen_allies: dict[str, Position] = {}
        self._enemy_seen_on: dict[str, int] = {}

    def act(self, observation: SkirmishObservation) -> SkirmishAction:
        self._activation += 1
        enemies = visible.enemies(observation)
        allies = visible.allies(observation)
        self._remember(enemies, allies)
        paths = self._paths(observation)
        if me.unit_type(observation) == "archer":
            return self._act_archer(observation, paths, enemies, allies)
        return self._act_melee(observation, paths, enemies, allies)

    def _remember(self, enemies: Iterable[VisibleUnit], allies: Iterable[VisibleUnit]) -> None:
        for enemy in enemies:
            self._last_seen_enemies[enemy["unit_id"]] = (enemy["type"], self._copy(enemy["position"]))
            self._enemy_seen_on[enemy["unit_id"]] = self._activation
        for ally in allies:
            self._last_seen_allies[ally["unit_id"]] = self._copy(ally["position"])

    def _act_archer(self, observation: SkirmishObservation, paths: list[PathOption], enemies: list[VisibleUnit], allies: list[VisibleUnit]) -> SkirmishAction:
        melee_threats = [enemy for enemy in enemies if enemy["type"] in {"cavalry", "footman"}]
        here = me.position(observation)
        under_threat = any(
            self._distance(here, enemy["position"]) <= self._threat_range(enemy["type"]) for enemy in melee_threats
        )
        target = self._select_target(observation, paths, enemies)
        firing_paths = self._attack_paths(paths, target, "archer") if target is not None else []
        safe_firing = self._safe_paths(firing_paths, melee_threats) if firing_paths else []

        if safe_firing:
            # A firing tile that's also safe from every melee threat exists: kite while
            # preserving the shot, preferring the option farthest from the target.
            path = max(safe_firing, key=lambda option: (self._distance(option[1], target["position"]), -option[0]))
            return self._order(path[0], target["unit_id"], observation)

        if under_threat:
            # No safe firing tile exists (or none is reachable this turn): escaping outranks
            # preserving the shot, so search every legal path, not only firing-range ones.
            threat_pairs = [(enemy["type"], enemy["position"]) for enemy in melee_threats]
            best_path = self._flee(observation, paths, threat_pairs)
            end = next(end for path_id, end in paths if path_id == best_path)
            name = (
                target["unit_id"]
                if target is not None and self._distance(end, target["position"]) <= units.STATS["archer"].attack_range
                else None
            )
            return self._order(best_path, name, observation)

        if firing_paths:
            # Not under threat (or no melee threats visible): pick the least-risky firing tile.
            path = max(
                firing_paths,
                key=lambda option: (
                    min(
                        (
                            self._distance(option[1], enemy["position"]) - self._threat_range(enemy["type"])
                            for enemy in melee_threats
                        ),
                        default=0,
                    ),
                    sum(self._distance(option[1], enemy["position"]) for enemy in melee_threats),
                    self._distance(option[1], target["position"]),
                    -option[0],
                ),
            )
            return self._order(path[0], target["unit_id"], observation)

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
        adjacent = [enemy for enemy in enemies if self._distance(me.position(observation), enemy["position"]) <= attack_range]
        if adjacent:
            target = self._select_target(observation, paths, adjacent)
            return self._order(0, target["unit_id"] if target else None, observation)

        archer_threats = self._archer_threats(observation, enemies)
        if archer_threats:
            opportunistic = self._select_target(observation, paths, [enemy for enemy in enemies if enemy["type"] in {"cavalry", "footman"}])
            if opportunistic is not None:
                attack_paths = self._attack_paths(paths, opportunistic, unit_type)
                if attack_paths:
                    return self._order(self._toward_option(attack_paths, opportunistic["position"])[0], opportunistic["unit_id"], observation)
            return self._archer_response(observation, paths, archer_threats)

        target = self._select_target(observation, paths, [enemy for enemy in enemies if enemy["type"] in {"cavalry", "footman"}])
        if target is not None:
            attack_paths = self._attack_paths(paths, target, unit_type)
            if attack_paths:
                return self._order(self._toward_option(attack_paths, target["position"])[0], target["unit_id"], observation)
            if self._threatens_visible_ally(observation, target):
                return self._order(self._toward(observation, paths, target["position"]), None, observation)
            safer_paths = self._safe_paths(paths, [target])
            return self._order(self._toward(observation, safer_paths or paths, target["position"]), None, observation)
        return self._melee_rendezvous(observation, paths, allies)

    def _cavalry_should_engage(self, observation: SkirmishObservation, archer_pos: Position) -> bool:
        """EV calc: does engaging the archer yield a favorable trade for cavalry's current HP?"""
        here = me.position(observation)
        movement = units.STATS["cavalry"].movement_points
        route_costs = self._route_costs(observation, archer_pos)
        cost_to_adjacent = min(
            (cost for (q, r), cost in route_costs.items() if self._distance({"q": q, "r": r}, archer_pos) <= 1),
            default=None,
        )
        if cost_to_adjacent is None:
            return False
        turns_to_close = max(1, -(-cost_to_adjacent // movement))
        dmg_to_archer = self._expected_damage(observation, "cavalry", archer_pos, {"position": archer_pos, "hit_points": 999})
        turns_to_kill = max(1, -(-units.STATS["archer"].hit_points // max(1, dmg_to_archer)))
        dmg_from_archer = self._expected_damage(observation, "archer", archer_pos, {"position": here})
        projected_loss = dmg_from_archer * max(0, turns_to_close + turns_to_kill - 1)
        SAFETY_FLOOR = 3
        return me.hit_points(observation) - projected_loss > SAFETY_FLOOR

    def _archer_response(self, observation: SkirmishObservation, paths: list[PathOption], threats: list[tuple[str, Position]]) -> SkirmishAction:
        safe_paths = self._safe_memory_paths(paths, threats)
        if me.unit_type(observation) == "footman":
            return self._order(self._flee(observation, safe_paths or paths, threats), None, observation)
        # Cavalry: if a favorable trade exists with a visible archer, engage it; else flee.
        archers = [enemy for enemy in visible.enemies(observation) if enemy["type"] == "archer"]
        if archers and self._cavalry_should_engage(observation, archers[0]["position"]):
            target = archers[0]
            attack_paths = self._attack_paths(paths, target, "cavalry")
            if attack_paths:
                return self._order(self._toward_option(attack_paths, target["position"])[0], target["unit_id"], observation)
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
        here = me.position(observation)
        return [(path_id, tile.at_path_end(here, path_id)) for path_id in action.legal_paths(observation)]

    def _route_costs(self, observation: SkirmishObservation, goal: Position) -> dict[tuple[int, int], int]:
        """Dijkstra: cost from every reachable tile to goal, using real terrain move costs."""
        import heapq
        start = (goal["q"], goal["r"])
        costs = {start: 0}
        frontier = [(0, start)]
        while frontier:
            cost, key = heapq.heappop(frontier)
            if cost > costs.get(key, float("inf")):
                continue
            pos = {"q": key[0], "r": key[1]}
            for neighbor in tile.neighbors(pos).values():
                terrain = tile.terrain_at(observation, neighbor)
                if terrain["terrain"] in ("water", "void"):
                    continue
                enter_cost = {"grass": 1, "hill": 2}.get(terrain["terrain"], 1) + {"none": 0, "forest": 1, "marsh": 2}.get(terrain["feature"], 0)
                key = (neighbor["q"], neighbor["r"])
                new_cost = cost + enter_cost
                if new_cost < costs.get(key, float("inf")):
                    costs[key] = new_cost
                    heapq.heappush(frontier, (new_cost, key))
        return costs

    def _target_utility(self, observation: SkirmishObservation, paths: list[PathOption], enemy: VisibleUnit) -> float:
        """Weighted utility score: type value, damage dealt, kill bonus, threat removed, counter-attack risk."""
        own_type = me.unit_type(observation)
        own_range = units.STATS[own_type].attack_range
        attack_ends = [end for _, end in paths if self._distance(end, enemy["position"]) <= own_range]
        attackable = bool(attack_ends)
        best_damage = max((self._expected_damage(observation, own_type, end, enemy) for end in attack_ends), default=0)
        killable = attackable and enemy["hit_points"] <= best_damage
        incoming = self._expected_damage(observation, enemy["type"], enemy["position"], {"position": me.position(observation)}) if attackable else 0

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

    def _expected_damage(self, observation: SkirmishObservation, attacker_type: str, attacker_position: AxialPosition, target: VisibleUnit) -> int:
        """Compute expected damage accounting for hill/forest terrain effects (abilities are off this season)."""
        attacker_terrain = tile.terrain_at(observation, attacker_position)["terrain"]
        defender_tile = tile.terrain_at(observation, target["position"])
        total = units.STATS[attacker_type].damage
        if attacker_terrain == "hill" and defender_tile["terrain"] != "hill":
            total += 1
        if attacker_terrain != "hill" and defender_tile["terrain"] == "hill":
            total -= 1
        if defender_tile["feature"] == "forest" and self._distance(attacker_position, target["position"]) > 1:
            total -= 1
        return max(1, total)

    def _can_kill_this_turn(self, observation: SkirmishObservation, paths: list[PathOption], enemy: VisibleUnit) -> bool:
        attack_range = units.STATS[me.unit_type(observation)].attack_range
        attacker_type = me.unit_type(observation)
        return any(
            self._distance(end, enemy["position"]) <= attack_range
            and enemy["hit_points"] <= self._expected_damage(observation, attacker_type, end, enemy)
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
        visible_ids = {enemy["unit_id"] for enemy in enemies}
        for unit_id, (kind, position) in self._last_seen_enemies.items():
            if (
                kind == "archer"
                and unit_id not in visible_ids
                and self._activation - self._enemy_seen_on[unit_id] == 1
                and self._distance(here, position) <= self._threat_range("archer")
            ):
                threats.append((kind, position))
        return threats

    def _ally_goal(self, observation: SkirmishObservation, allies: Iterable[VisibleUnit]) -> Position | None:
        choices = [(ally["type"], ally["position"]) for ally in allies] or [("", position) for position in self._last_seen_allies.values()]
        if not choices:
            return None
        return self._copy(min(choices, key=lambda choice: (self._distance(me.position(observation), choice[1]), choice[0] != "cavalry"))[1])

    def _last_seen_goal(self, unit_type: str) -> Position | None:
        positions = [position for kind, position in self._last_seen_enemies.values() if kind == unit_type]
        return self._copy(positions[0]) if positions else None

    def _last_seen_non_archer_goal(self) -> Position | None:
        positions = [position for kind, position in self._last_seen_enemies.values() if kind != "archer"]
        return self._copy(self._rng.choice(positions)) if positions else None

    def _toward(self, observation: SkirmishObservation, paths: Iterable[PathOption], goal: Position) -> int:
        center = tile.at_center(observation)
        route_costs = self._route_costs(observation, goal)
        def goal_cost(option):
            key = (option[1]["q"], option[1]["r"])
            return route_costs.get(key, self._distance(option[1], goal))
        return min(
            paths,
            key=lambda option: (
                goal_cost(option),
                self._distance(option[1], center),
                option[0],
            ),
        )[0]

    def _toward_option(self, paths: Iterable[PathOption], goal: Position) -> PathOption:
        return min(paths, key=lambda option: (self._distance(option[1], goal), option[0]))

    def _flee(
        self, observation: SkirmishObservation, paths: Iterable[PathOption], threats: Iterable[tuple[str, Position]]
    ) -> int:
        paths = list(paths)
        threats = list(threats)
        here = me.position(observation)
        center = tile.at_center(observation)
        center_costs = self._route_costs(observation, center)
        safety_margin = lambda option: min(
            self._distance(option[1], position) - self._threat_range(kind) for kind, position in threats
        )
        best_margin = max(safety_margin(option) for option in paths)
        safest_paths = [option for option in paths if safety_margin(option) == best_margin]
        def center_progress(option):
            key = (option[1]["q"], option[1]["r"])
            here_cost = center_costs.get((here["q"], here["r"]), self._distance(here, center))
            end_cost = center_costs.get(key, self._distance(option[1], center))
            return here_cost - end_cost
        best_center_progress = max(center_progress(option) for option in safest_paths)

        if best_center_progress <= 0:
            last_ally = self._ally_goal(observation, [])
            if last_ally is not None:
                return min(
                    safest_paths,
                    key=lambda option: (
                        self._distance(option[1], last_ally),
                        -sum(self._distance(option[1], position) for _, position in threats),
                        option[0],
                    ),
                )[0]

        return max(
            safest_paths,
            key=lambda option: (
                center_progress(option),
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