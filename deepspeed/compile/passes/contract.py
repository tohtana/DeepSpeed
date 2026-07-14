# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

# Capability tags produced and consumed by the built-in DeepCompile passes. Keeping the tags
# in one place lets passes declare dependencies on each other without hard-coding pass names.
CAP_Z3_GATHER_RELEASE = "z3_gather_release"


@dataclass(frozen=True)
class PassContract:
    """Lightweight metadata describing what an optimization pass expects and produces.

    Contracts let DeepCompile validate a pass schedule before it runs. A pass may only appear
    after every capability it ``requires`` has been ``provides``-d by an earlier pass. Two passes
    may not share a schedule step if either pass names the other in ``conflicts_with``. ``phase``
    is informational for now and records whether a pass rewrites the forward graph, the backward
    graph, or both.
    """

    name: str
    provides: FrozenSet[str] = frozenset()
    requires: FrozenSet[str] = frozenset()
    conflicts_with: FrozenSet[str] = frozenset()
    phase: str = "both"


class PassContractError(ValueError):
    """Raised when a pass schedule violates the registered pass contracts."""


_pass_contracts: Dict[str, PassContract] = {}


def _set_pass_contract(name: str, contract: Optional[PassContract]) -> None:
    if contract is not None and not isinstance(contract, PassContract):
        raise TypeError(f"contract must be a PassContract or None, but got {type(contract)}")
    if contract is not None and contract.name != name:
        raise ValueError(f"Pass contract name '{contract.name}' does not match registered pass name '{name}'")

    if contract is None:
        _pass_contracts.pop(name, None)
    else:
        _pass_contracts[name] = contract


def register_pass_contract(contract: PassContract) -> None:
    _set_pass_contract(contract.name, contract)


def get_pass_contract(name: str) -> Optional[PassContract]:
    return _pass_contracts.get(name)


def _resolve_pass_name(pass_ref, fn_to_name: Optional[Dict]) -> Optional[str]:
    # Schedules may reference a pass either by its registered name or by its callable. We only
    # know how to look up a contract by name, so translate callables back to their name here.
    if isinstance(pass_ref, str):
        return pass_ref
    if fn_to_name is not None:
        if hasattr(fn_to_name, "get"):
            name_or_names = fn_to_name.get(pass_ref)
        else:
            # Use object identity for callable registries. Callable instances may be unhashable,
            # and equality does not imply that two callable objects are the same registered pass.
            name_or_names = [name for fn, name in fn_to_name if fn is pass_ref]
        if name_or_names is None or isinstance(name_or_names, str):
            return name_or_names

        names = sorted(set(name_or_names))
        if len(names) == 1:
            return names[0]
        if names:
            raise PassContractError(f"Pass callable is registered under multiple names: {names}. "
                                    "Use a registered pass name to disambiguate it.")
    return None


def validate_schedule(schedule: List[Tuple[int, List]], fn_to_name: Optional[Dict] = None) -> None:
    """Validate that a DeepCompile pass schedule satisfies the registered pass contracts.

    ``schedule`` uses the ``[(step, passes), ...]`` format consumed by ``init_schedule``. Each
    entry in ``passes`` may be a registered pass name or a pass callable; pass ``fn_to_name`` to
    resolve callables. This may be a mapping from callables to one or more names, or an iterable of
    ``(callable, name)`` pairs for identity-based resolution of unhashable callables. Ambiguous
    callable aliases must be supplied by name in the schedule. Passes that have no registered
    contract are treated as unconstrained and skipped, so mixed schedules of contracted and ad-hoc
    passes remain valid. Raises :class:`PassContractError` on the first unmet requirement,
    ambiguity, or conflict.

    Each step is validated independently: DeepCompile resets Dynamo and recompiles from the
    original graph at every launched step (see ``launch_compile_passes``), so capabilities and
    conflicts in one step do not carry over to later steps. A pass must therefore find every
    capability it requires among the passes scheduled earlier within the same step.
    """
    for step, passes in schedule:
        provided: set = set()
        applied: List[str] = []

        for pass_ref in passes:
            name = _resolve_pass_name(pass_ref, fn_to_name)
            if name is None:
                continue

            contract = _pass_contracts.get(name)
            if contract is None:
                continue

            missing = contract.requires - provided
            if missing:
                raise PassContractError(f"Pass '{name}' (step {step}) requires {sorted(missing)}, which no earlier "
                                        f"pass provides. Passes scheduled so far: {applied}.")

            # Conflicts are treated symmetrically: either pass may declare the incompatibility.
            conflicts = set(contract.conflicts_with.intersection(applied))
            for prev_name in applied:
                prev_contract = _pass_contracts.get(prev_name)
                if prev_contract is not None and name in prev_contract.conflicts_with:
                    conflicts.add(prev_name)
            if conflicts:
                raise PassContractError(f"Pass '{name}' (step {step}) conflicts with already-scheduled pass(es) "
                                        f"{sorted(conflicts)}.")

            provided |= contract.provides
            applied.append(name)
