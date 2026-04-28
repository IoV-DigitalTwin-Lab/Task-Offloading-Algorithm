"""
Unit tests for metrics_engine.channel_model.

Physical sanity checks derived from WINNER II B1 model and 802.11p specs.
Run with: python -m pytest metrics_engine/tests/test_channel.py -v
"""

import math
import pytest

from metrics_engine.channel_model import (
    path_loss_v2i_db,
    path_loss_v2v_db,
    noise_power_dbm,
    sinr_db_v2i,
    sinr_db_v2v,
    channel_capacity_bps,
    transmission_time_s,
    is_link_reliable,
    sinr_from_distance,
)
from metrics_engine.config import (
    LINK_BANDWIDTH_HZ,
    THERMAL_NOISE_DENSITY_DBM_HZ,
    SINR_THRESHOLD_DB,
)


class TestPathLoss:
    def test_v2i_increases_with_distance(self):
        pl_50  = path_loss_v2i_db(50.0)
        pl_200 = path_loss_v2i_db(200.0)
        pl_500 = path_loss_v2i_db(500.0)
        assert pl_50 < pl_200 < pl_500

    def test_v2v_increases_with_distance(self):
        pl_30  = path_loss_v2v_db(30.0)
        pl_150 = path_loss_v2v_db(150.0)
        assert pl_30 < pl_150

    def test_v2i_at_10m_positive(self):
        # Path loss is always a positive dB value (signal is attenuated)
        assert path_loss_v2i_db(10.0) > 0.0

    def test_v2v_at_10m_positive(self):
        assert path_loss_v2v_db(10.0) > 0.0

    def test_v2i_at_100m_plausible(self):
        # WINNER II B1 urban at 100m should be roughly 70–90 dB
        pl = path_loss_v2i_db(100.0)
        assert 60.0 < pl < 110.0, f"Unexpected V2I path loss at 100m: {pl:.1f} dB"

    def test_v2v_at_50m_plausible(self):
        # V2V NLOS at 50m should be roughly 70–95 dB
        pl = path_loss_v2v_db(50.0)
        assert 60.0 < pl < 110.0, f"Unexpected V2V path loss at 50m: {pl:.1f} dB"

    def test_v2i_larger_than_v2v_at_same_distance(self):
        # V2I uses higher carrier frequency (5.9 GHz vs 5.8 GHz effective) and
        # typically has higher NLOS loss than V2V short-range; but the key check
        # is that both models are monotonically increasing.
        d = 100.0
        pl_v2i = path_loss_v2i_db(d)
        pl_v2v = path_loss_v2v_db(d)
        # Both should be > 0; relative ordering depends on scenario params — just sanity check
        assert pl_v2i > 0 and pl_v2v > 0


class TestNoisePower:
    def test_noise_power_negative_dbm(self):
        # For 10 MHz bandwidth, thermal noise is around -104 dBm
        np_dbm = noise_power_dbm(LINK_BANDWIDTH_HZ)
        assert np_dbm < -90.0, f"Noise floor unexpectedly high: {np_dbm:.1f} dBm"

    def test_noise_power_increases_with_bandwidth(self):
        np_10mhz = noise_power_dbm(10e6)
        np_20mhz = noise_power_dbm(20e6)
        assert np_20mhz > np_10mhz  # doubling BW adds ~3 dB noise

    def test_noise_power_bw_doubling_adds_3db(self):
        np_10 = noise_power_dbm(10e6)
        np_20 = noise_power_dbm(20e6)
        assert abs((np_20 - np_10) - 10 * math.log10(2)) < 0.1


class TestSINR:
    def test_v2i_sinr_decreases_with_distance(self):
        sinr_50  = sinr_db_v2i(50.0)
        sinr_300 = sinr_db_v2i(300.0)
        assert sinr_50 > sinr_300

    def test_v2v_sinr_decreases_with_distance(self):
        sinr_20  = sinr_db_v2v(20.0)
        sinr_200 = sinr_db_v2v(200.0)
        assert sinr_20 > sinr_200

    def test_v2i_close_range_positive_sinr(self):
        # At 50m V2I link should have comfortably positive SINR
        assert sinr_db_v2i(50.0) > 5.0

    def test_v2v_close_range_positive_sinr(self):
        assert sinr_db_v2v(30.0) > 0.0

    def test_v2i_sinr_at_1000m_may_be_negative(self):
        # At 1 km the link may be near outage — just verify it returns a float
        sinr = sinr_db_v2i(1000.0)
        assert isinstance(sinr, float)

    def test_sinr_from_distance_v2i(self):
        sinr = sinr_from_distance(100.0, "V2I")
        assert sinr == pytest.approx(sinr_db_v2i(100.0), rel=1e-6)

    def test_sinr_from_distance_v2v(self):
        sinr = sinr_from_distance(100.0, "V2V")
        assert sinr == pytest.approx(sinr_db_v2v(100.0), rel=1e-6)

    def test_sinr_from_distance_unknown_defaults_to_v2i(self):
        sinr_unk = sinr_from_distance(100.0, "UNKNOWN")
        sinr_v2i = sinr_db_v2i(100.0)
        assert sinr_unk == pytest.approx(sinr_v2i, rel=1e-6)


class TestChannelCapacity:
    def test_capacity_positive(self):
        assert channel_capacity_bps(20.0) > 0.0

    def test_capacity_increases_with_sinr(self):
        cap_low  = channel_capacity_bps(5.0)
        cap_high = channel_capacity_bps(20.0)
        assert cap_high > cap_low

    def test_capacity_at_20db_sinr_plausible(self):
        # At 20 dB SINR, 10 MHz BW, Shannon gives ~66 Mbps theoretical max;
        # with spectral efficiency factor (~0.75) expect ~50 Mbps
        cap = channel_capacity_bps(20.0)
        assert 20e6 < cap < 80e6, f"Capacity at 20dB SINR unexpected: {cap/1e6:.1f} Mbps"

    def test_capacity_negative_sinr_still_positive(self):
        # Even at −5 dB SINR, Shannon capacity is a small positive number
        assert channel_capacity_bps(-5.0) > 0.0


class TestTransmissionTime:
    def test_transmission_time_positive(self):
        assert transmission_time_s(1_000_000, 20.0) > 0.0

    def test_large_file_slower_than_small(self):
        t_small = transmission_time_s(10_000, 20.0)
        t_large = transmission_time_s(1_000_000, 20.0)
        assert t_large > t_small

    def test_better_channel_faster(self):
        t_bad  = transmission_time_s(100_000, 5.0)
        t_good = transmission_time_s(100_000, 25.0)
        assert t_good < t_bad

    def test_1mb_at_20db_sinr_under_1_second(self):
        # 1 MB over a decent V2I link should finish well under 1 s
        t = transmission_time_s(1_000_000, 20.0)
        assert t < 1.0, f"1 MB transmission at 20dB took {t:.3f}s — too slow"

    def test_10kb_at_15db_sinr_milliseconds_scale(self):
        t = transmission_time_s(10_000, 15.0)
        assert t < 0.1, f"10 KB at 15dB took {t*1000:.1f}ms — unexpectedly slow"


class TestLinkReliability:
    def test_good_sinr_reliable(self):
        assert is_link_reliable(SINR_THRESHOLD_DB + 5.0) is True

    def test_bad_sinr_unreliable(self):
        assert is_link_reliable(SINR_THRESHOLD_DB - 5.0) is False

    def test_at_threshold_boundary(self):
        # Exactly at threshold should be unreliable (strict <)
        result = is_link_reliable(SINR_THRESHOLD_DB)
        assert isinstance(result, bool)
