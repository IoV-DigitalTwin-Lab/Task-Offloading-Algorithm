"""
Unit tests for metrics_engine.computation_model and metrics_engine.energy_model.

Validates that physical computation and energy formulas match the values
used in MyRSUApp.cc (specifically the CMOS energy formula: E = κ × f² × N).

Run with: python -m pytest metrics_engine/tests/test_computation.py -v
"""

import pytest

from metrics_engine.computation_model import (
    node_cpu_hz,
    execution_time_s,
    queue_delay_mm1,
    total_compute_latency_s,
    propagation_delay_s,
)
from metrics_engine.energy_model import (
    computation_energy_j,
    transmission_energy_j,
    total_offload_energy_j,
    local_execution_energy_j,
)
from metrics_engine.config import (
    RSU_CPU_HZ_DEFAULT,
    SV_CPU_HZ_DEFAULT,
    MIN_CPU_HZ,
    ENERGY_KAPPA_RSU,
    ENERGY_KAPPA_SV,
)


class TestNodeCPUHz:
    def test_rsu_default_when_zero_ghz(self):
        hz = node_cpu_hz(0.0, "RSU")
        assert hz == RSU_CPU_HZ_DEFAULT

    def test_sv_default_when_zero_ghz(self):
        hz = node_cpu_hz(0.0, "SV")
        assert hz == SV_CPU_HZ_DEFAULT

    def test_rsu_uses_provided_ghz(self):
        hz = node_cpu_hz(4.0, "RSU")
        assert hz == pytest.approx(4.0e9, rel=1e-9)

    def test_sv_uses_provided_ghz(self):
        hz = node_cpu_hz(2.0, "SV")
        assert hz == pytest.approx(2.0e9, rel=1e-9)

    def test_min_clamp_applied(self):
        hz = node_cpu_hz(0.0001, "SV")  # effectively zero
        assert hz >= MIN_CPU_HZ

    def test_unknown_node_type_uses_min(self):
        # Unknown node type should not raise; returns MIN_CPU_HZ or provided value
        hz = node_cpu_hz(1.0, "UNKNOWN")
        assert hz >= MIN_CPU_HZ


class TestExecutionTime:
    def test_positive_result(self):
        assert execution_time_s(1e8, 4.0e9) > 0.0

    def test_more_cycles_takes_longer(self):
        t_light = execution_time_s(1e7, 4.0e9)
        t_heavy = execution_time_s(1e9, 4.0e9)
        assert t_heavy > t_light

    def test_faster_cpu_takes_less_time(self):
        t_slow = execution_time_s(1e8, 1.0e9)
        t_fast = execution_time_s(1e8, 8.0e9)
        assert t_fast < t_slow

    def test_1e8_cycles_at_4ghz_25ms(self):
        # 100M cycles / 4 GHz = 25 ms
        t = execution_time_s(1e8, 4.0e9)
        assert t == pytest.approx(0.025, rel=1e-6)

    def test_never_negative(self):
        assert execution_time_s(0, 4.0e9) >= 0.0


class TestQueueDelay:
    def test_zero_queue_near_zero_delay(self):
        # Queue length 0 should give negligible (near zero) delay
        delay = queue_delay_mm1(0, 4.0e9, 1e8)
        assert delay >= 0.0
        assert delay < 0.01  # < 10 ms

    def test_longer_queue_more_delay(self):
        d_short = queue_delay_mm1(2, 4.0e9, 1e8)
        d_long  = queue_delay_mm1(10, 4.0e9, 1e8)
        assert d_long > d_short

    def test_queue_delay_non_negative(self):
        for q in [0, 1, 5, 20]:
            assert queue_delay_mm1(q, 4.0e9, 1e8) >= 0.0

    def test_very_loaded_queue_significant_delay(self):
        # Queue of 20 tasks, each 1e8 cycles, at 4 GHz → ~500 ms total
        delay = queue_delay_mm1(20, 4.0e9, 1e8)
        assert delay > 0.1  # at least 100 ms


class TestTotalComputeLatency:
    def test_positive_result(self):
        result = total_compute_latency_s(1e8, 4.0, 0, "RSU")
        assert result > 0.0

    def test_queue_increases_total(self):
        no_queue = total_compute_latency_s(1e8, 4.0, 0, "RSU")
        with_queue = total_compute_latency_s(1e8, 4.0, 5, "RSU")
        assert with_queue > no_queue

    def test_sv_vs_rsu_different_cpu_defaults(self):
        # SV has lower default CPU than RSU; so SV should be slower when cpu_ghz=0
        t_rsu = total_compute_latency_s(1e8, 0.0, 0, "RSU")
        t_sv  = total_compute_latency_s(1e8, 0.0, 0, "SV")
        assert t_sv > t_rsu


class TestPropagationDelay:
    def test_propagation_at_100m(self):
        # Speed of light ~3e8 m/s; 100m / 3e8 ≈ 333 ns
        t = propagation_delay_s(100.0)
        assert t == pytest.approx(100.0 / 3e8, rel=0.01)

    def test_longer_distance_longer_delay(self):
        assert propagation_delay_s(500.0) > propagation_delay_s(100.0)

    def test_always_positive(self):
        assert propagation_delay_s(10.0) > 0.0


class TestComputationEnergy:
    def test_rsu_energy_formula_matches_myrsupapp(self):
        # MyRSUApp.cc: energy_j = 2e-27 * f_hz * f_hz * cpu_cycles
        # ENERGY_KAPPA_RSU = 2e-27
        f_hz = 4.0e9
        n_cycles = 1e8
        expected = ENERGY_KAPPA_RSU * f_hz * f_hz * n_cycles
        computed = computation_energy_j(n_cycles, f_hz, "RSU")
        assert computed == pytest.approx(expected, rel=1e-6)

    def test_sv_energy_uses_sv_kappa(self):
        f_hz = 2.0e9
        n_cycles = 5e7
        expected = ENERGY_KAPPA_SV * f_hz * f_hz * n_cycles
        computed = computation_energy_j(n_cycles, f_hz, "SV")
        assert computed == pytest.approx(expected, rel=1e-6)

    def test_more_cycles_more_energy(self):
        e_light = computation_energy_j(1e7, 4.0e9, "RSU")
        e_heavy = computation_energy_j(1e9, 4.0e9, "RSU")
        assert e_heavy > e_light

    def test_energy_always_positive(self):
        assert computation_energy_j(1e8, 4.0e9, "RSU") > 0.0

    def test_rsu_energy_at_4ghz_100m_cycles_plausible(self):
        # 2e-27 * (4e9)^2 * 1e8 = 2e-27 * 16e18 * 1e8 = 3.2e0 → 3.2 J  (RSU is high power)
        e = computation_energy_j(1e8, 4.0e9, "RSU")
        assert 0.1 < e < 100.0, f"RSU energy unexpected: {e:.3f} J"


class TestTransmissionEnergy:
    def test_positive(self):
        assert transmission_energy_j(0.01, "V2I") > 0.0

    def test_longer_transmission_more_energy(self):
        e_short = transmission_energy_j(0.001, "V2I")
        e_long  = transmission_energy_j(0.1, "V2I")
        assert e_long > e_short

    def test_v2v_vs_v2i_different(self):
        e_v2i = transmission_energy_j(0.01, "V2I")
        e_v2v = transmission_energy_j(0.01, "V2V")
        # Just verify both compute without error and are positive
        assert e_v2i > 0.0 and e_v2v > 0.0


class TestTotalOffloadEnergy:
    def test_positive(self):
        e = total_offload_energy_j(
            cpu_cycles=1e8,
            cpu_available_ghz=4.0,
            tx_time_s=0.01,
            compute_time_s=0.025,
            link_type="V2I",
            node_type="RSU",
        )
        assert e > 0.0

    def test_increases_with_more_cycles(self):
        kwargs = dict(cpu_available_ghz=4.0, tx_time_s=0.01,
                      compute_time_s=0.025, link_type="V2I", node_type="RSU")
        e_light = total_offload_energy_j(cpu_cycles=1e7, **kwargs)
        e_heavy = total_offload_energy_j(cpu_cycles=1e9, **kwargs)
        assert e_heavy > e_light

    def test_sv_offload_energy_positive(self):
        e = total_offload_energy_j(
            cpu_cycles=5e7,
            cpu_available_ghz=2.0,
            tx_time_s=0.02,
            compute_time_s=0.05,
            link_type="V2V",
            node_type="SV",
        )
        assert e > 0.0


class TestLocalExecutionEnergy:
    def test_positive(self):
        assert local_execution_energy_j(1e8, 1.0e9) > 0.0

    def test_more_cycles_more_energy(self):
        e_light = local_execution_energy_j(1e7, 1.0e9)
        e_heavy = local_execution_energy_j(1e9, 1.0e9)
        assert e_heavy > e_light

    def test_uses_sv_kappa(self):
        # Local execution uses the vehicle (SV) kappa coefficient
        f_hz = 1.0e9
        n    = 1e8
        expected = ENERGY_KAPPA_SV * f_hz * f_hz * n
        computed = local_execution_energy_j(n, f_hz)
        assert computed == pytest.approx(expected, rel=1e-6)
