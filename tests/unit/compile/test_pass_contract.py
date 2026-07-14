# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

from deepspeed.compile.backend import opt_passes, register_compile_pass
from deepspeed.compile.passes import contract as contract_mod
from deepspeed.compile.passes.contract import (PassContract, PassContractError, register_pass_contract,
                                               get_pass_contract, validate_schedule)
from deepspeed.runtime.engine import DeepSpeedEngine


@pytest.fixture
def clean_registry():
    # Passes and contracts are module-global, so isolate each test from built-ins and other tests.
    saved_contracts = dict(contract_mod._pass_contracts)
    saved_passes = dict(opt_passes)
    contract_mod._pass_contracts.clear()
    opt_passes.clear()
    yield
    contract_mod._pass_contracts.clear()
    contract_mod._pass_contracts.update(saved_contracts)
    opt_passes.clear()
    opt_passes.update(saved_passes)


def _register_zero3_and_prefetch():
    register_pass_contract(PassContract(name="zero3", provides=frozenset({"z3"})))
    register_pass_contract(PassContract(name="prefetch", requires=frozenset({"z3"})))


def test_register_and_get(clean_registry):
    contract = PassContract(name="zero3", provides=frozenset({"z3"}))
    register_pass_contract(contract)
    assert get_pass_contract("zero3") is contract
    assert get_pass_contract("missing") is None


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
    register_pass_contract(PassContract(name="a", conflicts_with=frozenset({"b"})))
    register_pass_contract(PassContract(name="b"))
    # "b" declares nothing, but "a" lists it as a conflict; either ordering must be rejected.
    with pytest.raises(PassContractError, match="conflicts"):
        validate_schedule([(0, ["a", "b"])])
    with pytest.raises(PassContractError, match="conflicts"):
        validate_schedule([(0, ["b", "a"])])


def test_conflict_does_not_carry_across_steps(clean_registry):
    register_pass_contract(PassContract(name="a", conflicts_with=frozenset({"b"})))
    register_pass_contract(PassContract(name="b"))
    validate_schedule([(0, ["a"]), (10, ["b"])])


def test_uncontracted_passes_are_skipped(clean_registry):
    _register_zero3_and_prefetch()
    # An ad-hoc pass with no registered contract must not break validation of the rest.
    validate_schedule([(0, ["zero3", "ad_hoc_pass", "prefetch"])])


def test_callables_resolved_via_fn_to_name(clean_registry):
    _register_zero3_and_prefetch()

    def zero3_fn():
        pass

    def prefetch_fn():
        pass

    fn_to_name = {zero3_fn: "zero3", prefetch_fn: "prefetch"}
    validate_schedule([(0, [zero3_fn, prefetch_fn])], fn_to_name)
    with pytest.raises(PassContractError, match="requires"):
        validate_schedule([(0, [prefetch_fn])], fn_to_name)


def test_register_compile_pass_clears_stale_contract(clean_registry):

    def original_fn():
        pass

    def replacement_fn():
        pass

    register_compile_pass("custom", original_fn, PassContract(name="custom", requires=frozenset({"missing"})))
    register_compile_pass("custom", replacement_fn)

    assert opt_passes["custom"] is replacement_fn
    assert get_pass_contract("custom") is None
    validate_schedule([(0, ["custom"])])


def test_register_compile_pass_rejects_name_mismatch_before_mutation(clean_registry):

    def original_fn():
        pass

    def replacement_fn():
        pass

    original_contract = PassContract(name="custom")
    register_compile_pass("custom", original_fn, original_contract)

    with pytest.raises(ValueError, match="does not match"):
        register_compile_pass("custom", replacement_fn, PassContract(name="different"))

    assert opt_passes["custom"] is original_fn
    assert get_pass_contract("custom") is original_contract
    assert get_pass_contract("different") is None


def test_duplicate_callable_alias_preserves_named_schedule_identity(clean_registry):

    def shared_fn():
        pass

    register_compile_pass("provider", shared_fn, PassContract(name="provider", provides=frozenset({"cap"})))
    register_compile_pass("consumer", shared_fn, PassContract(name="consumer", requires=frozenset({"cap"})))

    class FakeEngine:

        def compile_autosp(self):
            return False

        def get_deepcompile_backend(self, backend, compile_kwargs, schedule):
            return "resolved_backend"

    resolved_backend, schedule = DeepSpeedEngine.get_deepspeed_compile_backend(FakeEngine(), "eager", {},
                                                                               [(0, ["provider"])])
    assert resolved_backend == "resolved_backend"
    assert schedule == [(0, [shared_fn])]


def test_duplicate_callable_alias_is_rejected_as_ambiguous(clean_registry):

    def shared_fn():
        pass

    register_compile_pass("provider", shared_fn, PassContract(name="provider", provides=frozenset({"cap"})))
    register_compile_pass("consumer", shared_fn, PassContract(name="consumer", requires=frozenset({"cap"})))

    fn_to_names = {shared_fn: ["provider", "consumer"]}
    with pytest.raises(PassContractError, match="multiple names"):
        validate_schedule([(0, [shared_fn])], fn_to_names)


def test_unhashable_callable_resolves_by_identity(clean_registry):

    class UnhashableCallable:
        __hash__ = None

        def __call__(self):
            pass

    pass_fn = UnhashableCallable()
    register_compile_pass("custom", pass_fn, PassContract(name="custom"))

    class FakeEngine:

        def compile_autosp(self):
            return False

        def get_deepcompile_backend(self, backend, compile_kwargs, schedule):
            return "resolved_backend"

    for pass_ref in ("custom", pass_fn):
        resolved_backend, schedule = DeepSpeedEngine.get_deepspeed_compile_backend(FakeEngine(), "eager", {},
                                                                                   [(0, [pass_ref])])
        assert resolved_backend == "resolved_backend"
        assert schedule[0][1][0] is pass_fn


def test_equality_colliding_callables_keep_identity(clean_registry):

    class EqualCallable:

        def __init__(self, value):
            self.value = value

        def __call__(self):
            pass

        def __eq__(self, other):
            return isinstance(other, EqualCallable)

        def __hash__(self):
            return 0

    provider = EqualCallable("provider")
    consumer = EqualCallable("consumer")
    register_compile_pass("provider", provider, PassContract(name="provider", provides=frozenset({"cap"})))
    register_compile_pass("consumer", consumer, PassContract(name="consumer", requires=frozenset({"cap"})))
    fn_to_names = [(fn, name) for name, fn in opt_passes.items()]

    validate_schedule([(0, [provider])], fn_to_names)
    with pytest.raises(PassContractError, match="requires"):
        validate_schedule([(0, [consumer])], fn_to_names)
