# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

from deepspeed.compile.passes import (contract as contract_mod, offload_adam_states, offload_parameters, prefetch,
                                      selective_gather, zero1_compile, zero3_compile)
from deepspeed.compile.passes.contract import (PassContract, PassContractError, register_pass_contract,
                                               get_pass_contract, validate_schedule)

# Mirrors the registration in DeepSpeedEngine.__init__ so the built-in contracts can be exercised
# without building an engine.
BUILTIN_PASSES = {
    zero1_compile.NAME: (zero1_compile.add_z1_reduce, zero1_compile.CONTRACT),
    zero1_compile.NAME_Z2: (zero1_compile.add_z2_reduce, zero1_compile.CONTRACT_Z2),
    zero3_compile.NAME: (zero3_compile.add_z3_gather_release, zero3_compile.CONTRACT),
    prefetch.NAME: (prefetch.schedule_prefetch, prefetch.CONTRACT),
    selective_gather.NAME: (selective_gather.selective_gather, selective_gather.CONTRACT),
    offload_parameters.NAME: (offload_parameters.offload_parameter_fwd, offload_parameters.CONTRACT),
    offload_adam_states.NAME: (offload_adam_states.move_opt_states, offload_adam_states.CONTRACT),
    offload_adam_states.NAME_SYNC: (offload_adam_states.move_opt_states_sync, offload_adam_states.CONTRACT_SYNC),
    offload_adam_states.NAME_FOR_INIT:
    (offload_adam_states.offload_adam_states_for_init, offload_adam_states.CONTRACT_FOR_INIT),
}


@pytest.fixture
def clean_registry():
    # validate_schedule reads a module-level registry, so isolate each test from the built-in
    # contracts and from one another by snapshotting and restoring it.
    saved = dict(contract_mod._pass_contracts)
    contract_mod._pass_contracts.clear()
    yield
    contract_mod._pass_contracts.clear()
    contract_mod._pass_contracts.update(saved)


@pytest.fixture
def builtin_registry(clean_registry):
    for name, (_, contract) in BUILTIN_PASSES.items():
        register_pass_contract(name, contract)
    return {name: fn for name, (fn, _) in BUILTIN_PASSES.items()}


def _register_zero3_and_prefetch():
    register_pass_contract("zero3", PassContract(provides=frozenset({"z3"})))
    register_pass_contract("prefetch", PassContract(requires=frozenset({"z3"})))


def test_register_and_get(clean_registry):
    contract = PassContract(provides=frozenset({"z3"}))
    register_pass_contract("zero3", contract)
    assert get_pass_contract("zero3") is contract
    assert get_pass_contract("missing") is None


def test_none_contract_clears_previous(clean_registry):
    # Re-registering a pass without a contract must drop the stale one, not keep it.
    register_pass_contract("prefetch", PassContract(requires=frozenset({"z3"})))
    assert get_pass_contract("prefetch") is not None
    register_pass_contract("prefetch", None)
    assert get_pass_contract("prefetch") is None
    # With the contract cleared, the pass is unconstrained again.
    validate_schedule([(0, ["prefetch"])])


def test_valid_order_passes(clean_registry):
    _register_zero3_and_prefetch()
    validate_schedule([(0, ["zero3"]), (10, ["zero3", "prefetch"])])


def test_missing_requirement_raises(clean_registry):
    _register_zero3_and_prefetch()
    with pytest.raises(PassContractError, match="requires"):
        validate_schedule([(0, ["prefetch"])])


def test_requirement_within_same_step(clean_registry):
    # Passes in the same step run left to right, so a provider earlier in the list satisfies a
    # later consumer.
    _register_zero3_and_prefetch()
    validate_schedule([(0, ["zero3", "prefetch"])])


def test_requirement_does_not_carry_across_steps(clean_registry):
    # DeepCompile recompiles from the original graph at each step, so a provider in an earlier
    # step does not satisfy a consumer in a later step; the requirement must be met per step.
    _register_zero3_and_prefetch()
    with pytest.raises(PassContractError, match="requires"):
        validate_schedule([(0, ["zero3"]), (10, ["prefetch"])])


def test_conflict_is_symmetric(clean_registry):
    register_pass_contract("a", PassContract(conflicts_with=frozenset({"b"})))
    register_pass_contract("b", PassContract())
    # "b" declares nothing, but "a" lists it as a conflict; either ordering must be rejected.
    with pytest.raises(PassContractError, match="conflicts"):
        validate_schedule([(0, ["a", "b"])])
    with pytest.raises(PassContractError, match="conflicts"):
        validate_schedule([(0, ["b", "a"])])


def test_conflict_with_uncontracted_pass(clean_registry):
    # "custom" is absent from the registry, but a conflict naming it must hold in both orderings.
    register_pass_contract("a", PassContract(conflicts_with=frozenset({"custom"})))
    with pytest.raises(PassContractError, match="conflicts"):
        validate_schedule([(0, ["a", "custom"])])
    with pytest.raises(PassContractError, match="conflicts"):
        validate_schedule([(0, ["custom", "a"])])
    # Steps are validated independently.
    validate_schedule([(0, ["a"]), (10, ["custom"])])


def test_uncontracted_passes_are_skipped(clean_registry):
    _register_zero3_and_prefetch()
    # An ad-hoc pass with no registered contract must not break validation of the rest.
    validate_schedule([(0, ["zero3", "ad_hoc_pass", "prefetch"])])


def test_callables_resolved_by_identity(clean_registry):
    _register_zero3_and_prefetch()

    def zero3_fn():
        pass

    def prefetch_fn():
        pass

    name_registry = {"zero3": zero3_fn, "prefetch": prefetch_fn}
    validate_schedule([(0, [zero3_fn]), (10, [zero3_fn, prefetch_fn])], name_registry)
    with pytest.raises(PassContractError, match="requires"):
        validate_schedule([(0, [prefetch_fn])], name_registry)


def test_ambiguous_callable_requires_explicit_name(clean_registry):
    register_pass_contract("zero3", PassContract(provides=frozenset({"z3"})))

    def shared_fn():
        pass

    # The same callable registered under two names cannot be resolved to a single contract.
    name_registry = {"zero3": shared_fn, "zero3_alias": shared_fn}
    with pytest.raises(PassContractError, match="multiple names"):
        validate_schedule([(0, [shared_fn])], name_registry)


def test_builtin_conflicts_name_registered_passes(builtin_registry):
    # conflicts_with holds pass names, so a rename must not silently disable the check.
    for name, (_, contract) in BUILTIN_PASSES.items():
        assert contract.conflicts_with <= set(BUILTIN_PASSES), name


def test_builtin_default_schedules_validate(builtin_registry):
    # The schedules init_z1 and init_z3 build when the user supplies none.
    validate_schedule([(0, [zero1_compile.NAME])], builtin_registry)
    validate_schedule([(0, [zero3_compile.NAME]), (5, [zero3_compile.NAME, prefetch.NAME, selective_gather.NAME])],
                      builtin_registry)
    validate_schedule([(0, [zero3_compile.NAME, offload_parameters.NAME])], builtin_registry)
    validate_schedule([(0, [zero3_compile.NAME]),
                       (1, [offload_adam_states.NAME_FOR_INIT, zero3_compile.NAME, offload_adam_states.NAME])],
                      builtin_registry)


def test_offload_targets_conflict(builtin_registry):
    # init_z3 rejects this combination only while building its own schedule, so a user-supplied
    # schedule reaches the passes unchecked without a contract.
    with pytest.raises(PassContractError, match="conflicts"):
        validate_schedule([(0, [zero3_compile.NAME, offload_parameters.NAME, offload_adam_states.NAME])],
                          builtin_registry)
    with pytest.raises(PassContractError, match="conflicts"):
        validate_schedule([(0, [zero3_compile.NAME, offload_adam_states.NAME_SYNC, offload_parameters.NAME])],
                          builtin_registry)


def test_offload_adam_states_variants_conflict(builtin_registry):
    with pytest.raises(PassContractError, match="conflicts"):
        validate_schedule([(0, [offload_adam_states.NAME, offload_adam_states.NAME_SYNC])], builtin_registry)


def test_offload_parameters_requires_zero3(builtin_registry):
    with pytest.raises(PassContractError, match="requires"):
        validate_schedule([(0, [offload_parameters.NAME])], builtin_registry)


def test_zero1_and_zero3_conflict(builtin_registry):
    with pytest.raises(PassContractError, match="conflicts"):
        validate_schedule([(0, [zero3_compile.NAME, zero1_compile.NAME])], builtin_registry)
    with pytest.raises(PassContractError, match="conflicts"):
        validate_schedule([(0, [zero1_compile.NAME_Z2, zero3_compile.NAME])], builtin_registry)


def test_schedule_written_with_callables_is_validated(builtin_registry):
    # Schedules are normally written with the pass callables, which resolve back to their names.
    schedule = [(0, [
        zero3_compile.add_z3_gather_release, offload_parameters.offload_parameter_fwd,
        offload_adam_states.move_opt_states
    ])]
    with pytest.raises(PassContractError, match="conflicts"):
        validate_schedule(schedule, builtin_registry)
