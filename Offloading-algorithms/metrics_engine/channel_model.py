"""
V2X Channel Model: path loss, SINR, Shannon capacity, transmission delay.

Models used:
  - V2I path loss: WINNER II B1 LOS (street canyon)
  - V2V path loss: WINNER II B1 NLOS
  - Thermal noise: Johnson-Nyquist at 290K
  - Capacity: Shannon C = B * log2(1 + SINR)

References:
  Molisch et al., "WINNER II Channel Models", EC-IST-4-027756, 2007.
  Kenney, "Dedicated Short-Range Communications", Proc. IEEE, 2011.
  Rappaport, "Wireless Communications", 2nd ed., 2002.
"""

import math
from metrics_engine.config import (
    CARRIER_FREQUENCY_HZ, LINK_BANDWIDTH_HZ, LINK_RATE_EFFICIENCY,
    WINNER_II_B1_A, WINNER_II_B1_B, WINNER_II_B1_C,
    WINNER_II_B1_NLOS_A, WINNER_II_B1_NLOS_B, WINNER_II_B1_NLOS_C,
    TX_POWER_V2I_DBM, TX_POWER_V2V_DBM,
    THERMAL_NOISE_DENSITY_DBM_PER_HZ, NOISE_FIGURE_DB,
    SINR_TYPICAL_V2I_DB, SINR_TYPICAL_V2V_DB,
    SINR_MIN_RELIABLE_DB, FADING_MARGIN_DB,
)


def path_loss_v2i_db(distance_m: float, freq_hz: float = CARRIER_FREQUENCY_HZ) -> float:
    """
    V2I path loss (vehicle to RSU, LOS urban street canyon).
    WINNER II B1 LOS: PL = A·log10(d) + B + C·log10(f_GHz)   [dB]

    Args:
        distance_m: separation between vehicle and RSU, metres
        freq_hz: carrier frequency in Hz

    Returns:
        path loss in dB
    """
    d = max(distance_m, 1.0)  # avoid log(0)
    freq_ghz = freq_hz / 1e9
    return (WINNER_II_B1_A * math.log10(d)
            + WINNER_II_B1_B
            + WINNER_II_B1_C * math.log10(freq_ghz))


def path_loss_v2v_db(distance_m: float, freq_hz: float = CARRIER_FREQUENCY_HZ) -> float:
    """
    V2V path loss (vehicle to vehicle, NLOS urban).
    WINNER II B1 NLOS: PL = A·log10(d) + B + C·log10(f_GHz)  [dB]
    """
    d = max(distance_m, 1.0)
    freq_ghz = freq_hz / 1e9
    return (WINNER_II_B1_NLOS_A * math.log10(d)
            + WINNER_II_B1_NLOS_B
            + WINNER_II_B1_NLOS_C * math.log10(freq_ghz))


def noise_power_dbm(bandwidth_hz: float = LINK_BANDWIDTH_HZ,
                    noise_figure_db: float = NOISE_FIGURE_DB) -> float:
    """
    Receiver thermal noise power: N = kTB + NF.

    Returns:
        noise power in dBm
    """
    return (THERMAL_NOISE_DENSITY_DBM_PER_HZ
            + 10.0 * math.log10(bandwidth_hz)
            + noise_figure_db)


def sinr_db_v2i(distance_m: float,
                tx_power_dbm: float = TX_POWER_V2I_DBM,
                bandwidth_hz: float = LINK_BANDWIDTH_HZ) -> float:
    """
    SINR (dB) for a V2I link using LOS path loss model.
    SINR = P_tx - PL(d) - N (all in dB/dBm).
    Includes a 3 dB Rayleigh fading margin for urban multipath.

    Args:
        distance_m: vehicle-to-RSU distance
        tx_power_dbm: vehicle transmit power (dBm)
        bandwidth_hz: channel bandwidth (Hz)

    Returns:
        SINR in dB
    """
    pl = path_loss_v2i_db(distance_m)
    n  = noise_power_dbm(bandwidth_hz)
    return tx_power_dbm - pl - n - FADING_MARGIN_DB


def sinr_db_v2v(distance_m: float,
                tx_power_dbm: float = TX_POWER_V2V_DBM,
                bandwidth_hz: float = LINK_BANDWIDTH_HZ) -> float:
    """
    SINR (dB) for a V2V link using NLOS path loss model.
    Includes a 3 dB Rayleigh fading margin.
    """
    pl = path_loss_v2v_db(distance_m)
    n  = noise_power_dbm(bandwidth_hz)
    return tx_power_dbm - pl - n - FADING_MARGIN_DB


def channel_capacity_bps(sinr_db: float,
                          bandwidth_hz: float = LINK_BANDWIDTH_HZ,
                          efficiency: float = LINK_RATE_EFFICIENCY) -> float:
    """
    Shannon spectral efficiency: C = B · log2(1 + SINR) · η

    η accounts for practical modulation/coding/MAC overhead (0.7 typical).

    Args:
        sinr_db: signal-to-noise+interference ratio (dB)
        bandwidth_hz: channel bandwidth (Hz)
        efficiency: practical fraction of Shannon capacity achieved (0-1)

    Returns:
        achievable throughput in bits/second
    """
    sinr_linear = 10.0 ** (sinr_db / 10.0)
    sinr_linear = max(sinr_linear, 1e-9)  # guard against -inf dB
    raw_capacity = bandwidth_hz * math.log2(1.0 + sinr_linear)
    return max(raw_capacity * efficiency, 1e3)   # minimum 1 kbps (always transmit something)


def transmission_time_s(data_bytes: float,
                         sinr_db: float,
                         bandwidth_hz: float = LINK_BANDWIDTH_HZ,
                         efficiency: float = LINK_RATE_EFFICIENCY) -> float:
    """
    Transmission delay: t_trans = (data_bits) / C(SINR)

    Args:
        data_bytes: payload in bytes
        sinr_db: link SINR in dB
        bandwidth_hz: channel bandwidth
        efficiency: practical capacity factor

    Returns:
        transmission delay in seconds
    """
    capacity = channel_capacity_bps(sinr_db, bandwidth_hz, efficiency)
    data_bits = max(data_bytes * 8.0, 1.0)
    return data_bits / capacity


def is_link_reliable(sinr_db: float) -> bool:
    """
    Return True if the link SINR exceeds the minimum reliable threshold.
    Below SINR_MIN_RELIABLE_DB the BER exceeds 1e-5 for QPSK coding.
    """
    return sinr_db >= SINR_MIN_RELIABLE_DB


def sinr_from_distance(distance_m: float, link_type: str = "V2I") -> float:
    """
    Convenience: compute SINR for a given distance and link type.

    Args:
        distance_m: node separation in metres
        link_type: "V2I" (vehicle to RSU) or "V2V" (vehicle to vehicle)

    Returns:
        SINR in dB, clamped to realistic range [-5, 25] dB
    """
    if link_type == "V2I":
        sinr = sinr_db_v2i(distance_m)
        fallback = SINR_TYPICAL_V2I_DB
    else:
        sinr = sinr_db_v2v(distance_m)
        fallback = SINR_TYPICAL_V2V_DB

    # Clamp to realistic V2X SINR range (Rappaport 2002, urban V2X)
    sinr = max(-5.0, min(25.0, sinr))
    if not math.isfinite(sinr):
        sinr = fallback
    return sinr
