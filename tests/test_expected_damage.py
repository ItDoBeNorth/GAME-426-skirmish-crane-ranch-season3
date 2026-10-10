"""Parity: Agent._expected_damage must match the engine's combat.damage on Season 3 rules."""

from __future__ import annotations

import random

from agent import Agent
from sandbox.env import META, make_env
from sandbox.env.skirmish_crane import combat
from sandbox.env.skirmish_crane.engine import Unit
from sandbox.env.skirmish_crane.hexes import distance, neighbors
from sandbox.harness.environment import resolve_parameters

SEASON_3 = {"seat_plan": "army", "field_extent": 12, "terrain": True, "unit_abilities": True}
KINDS = ("footman", "archer", "cavalry")
RANGE = {"footman": 1, "archer": 6, "cavalry": 1}


def _as_dict(position):
    return {"q": position[0], "r": position[1]}


def _visible(unit: Unit) -> dict:
    return {
        "unit_id": unit.unit_id,
        "side": unit.side,
        "type": unit.kind,
        "position": _as_dict(unit.position),
        "hit_points": unit.hit_points,
        "has_acted": False,
    }


def _run_trials(seed: int, trials: int) -> int:
    env = make_env(resolve_parameters(META, SEASON_3))
    try:
        env.reset(seed=seed)
        battlefield = env.match.battlefield
        base = env.observe(env.agent_selection)
    finally:
        env.close()
    rng = random.Random(seed)
    passable = [
        (q, r) for r, row in enumerate(battlefield.tiles) for q, cell in enumerate(row) if cell.passable
    ]
    agent = Agent()
    checked = 0
    for _ in range(trials):
        attacker_kind, defender_kind = rng.choice(KINDS), rng.choice(KINDS)
        defender_pos = rng.choice(passable)
        ends = [p for p in passable if p != defender_pos and distance(p, defender_pos) <= RANGE[attacker_kind]]
        if not ends:
            continue
        end = rng.choice(ends)
        starts = [p for p in passable if distance(p, end) <= 4]
        start = rng.choice(starts)
        attacker = Unit("red_" + attacker_kind + "_0", "red", attacker_kind, end, 10)
        defender = Unit("blue_" + defender_kind + "_0", "blue", defender_kind, defender_pos, 10)
        others: list[Unit] = []
        taken = {end, defender_pos, start}
        for index, side in enumerate(("blue", "red", "blue")):
            spots = [p for p in neighbors(defender_pos, battlefield.extent) if p in passable and p not in taken]
            if spots and rng.random() < 0.6:
                spot = rng.choice(spots)
                taken.add(spot)
                others.append(Unit(f"{side}_footman_{index + 1}", side, "footman", spot, 10))
        engine_units = {unit.unit_id: unit for unit in (attacker, defender, *others)}
        expected = combat.damage(attacker, defender, battlefield, engine_units, abilities=True, start=start)

        state = dict(base["observation"])
        defender_is_self = rng.random() < 0.5
        if defender_is_self:
            me_unit, seen = defender, [attacker, *others]
        else:
            # The attacker's own observation is taken at its activation start, before it walks.
            me_unit, seen = Unit(attacker.unit_id, "red", attacker_kind, start, 10), [defender, *others]
        state["self"] = {
            "unit_id": me_unit.unit_id,
            "type": me_unit.kind,
            "position": _as_dict(me_unit.position),
            "hit_points": me_unit.hit_points,
            "movement_points": 2,
            "direction": 2,
        }
        state["visible_units"] = [_visible(unit) for unit in seen if unit.position != me_unit.position]
        observation = {"observation": state, "action_mask": base["action_mask"]}
        target = agent._self_unit(observation) if defender_is_self else _visible(defender)
        got = agent._expected_damage(observation, attacker_kind, _as_dict(end), target, start=_as_dict(start))
        assert got == expected, (seed, attacker_kind, defender_kind, start, end, defender_pos, others, got, expected)
        checked += 1
    return checked


def test_expected_damage_matches_engine():
    assert sum(_run_trials(seed, 800) for seed in range(3)) > 1000
