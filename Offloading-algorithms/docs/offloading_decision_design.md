# Offloading Decision Logic: Gate B & RSU Decision Timeout

**Target audience:** Simulation team (PayloadVehicleApp.cc / TaskOffloadingDecision.cc)
**Branch:** `feature/drl-redis-integration` (simulation repo)

---

## Background: Why the Current Logic Is Wrong

In the current code, `sendTaskMetadataToRSU()` is called **before** `makeDecision()`. This means the vehicle broadcasts the task over the wireless channel and pushes it to the DRL's Redis queue **even when it will execute the task locally**. The DRL then trains on tasks that the vehicle was never going to offload. This is the root cause of the integration being broken.

**Fix order (critical):** Gate A → Gate B → *if* offload decision → THEN `sendTaskMetadataToRSU()`.

---

## Term Glossary

| Term | Unit | Description |
|---|---|---|
| `cpu_cycles` | cycles | Number of CPU clock cycles required to complete the task |
| `cpu_available` (vehicle) | Hz | CPU frequency available to this task right now (`cpu_allocable` minus already-allocated share) |
| `mem_footprint_bytes` | bytes | Working memory the task needs to hold during execution |
| `deadline_seconds` | seconds | Time budget from task creation to when it must be finished |
| `remaining_deadline` | seconds | `task->deadline - simTime()` — how much time is left right now |
| `T_local` | seconds | Estimated time to finish the task using **this vehicle's** CPU |
| `T_offload` | seconds | Estimated total time if the task is sent to an RSU: transmission + edge processing + result return |
| `T_tx` | seconds | Time to transmit the task payload over the wireless channel |
| `T_edge` | seconds | Time for the RSU's edge server to process the task |
| `T_rx` | seconds | Time to receive the result back (usually small, often negligible) |
| `E_local` | joules | Energy consumed by the vehicle to process the task locally |
| `E_offload` | joules | Energy consumed by the vehicle to transmit the task (processing energy is at the RSU) |
| `P_compute` | watts | Vehicle CPU power draw during active computation |
| `P_tx` | watts | Vehicle radio transmitter power during uplink |
| `P_idle` | watts | Vehicle idle power while waiting for RSU result |
| `bandwidth_bps` | bits/s | Wireless channel capacity (V2I link) from vehicle to RSU |
| `rsu_cpu_available` | Hz | Edge server CPU frequency currently available (read from RSU status broadcast, stored in `rsuMetrics`) |
| `RSSI_threshold` | dBm | Minimum acceptable received signal strength for a reliable V2I link |
| `α`, `β` | dimensionless | Weights for latency vs energy in the BOTH_FEASIBLE cost function (`α + β = 1`) |
| `MUST_OFFLOAD` | — | Task classification: local execution cannot meet deadline; offloading is required |
| `MUST_LOCAL` | — | Task classification: offloading cannot meet deadline (RSU too slow or unreachable); local is required |
| `BOTH_FEASIBLE` | — | Task classification: both local and offload can meet the deadline; choose the cheaper option |
| `INFEASIBLE` | — | Task classification: neither path can meet the deadline; task fails immediately |
| `rsuDecisionTimeout` | seconds | How long the vehicle waits for a decision from the RSU before acting on a fallback |

---

## Problem 1: Proposed Gate B Logic

Gate B is the **local feasibility check** performed by the vehicle before deciding whether to offload. It runs **before any wireless transmission**.

### Step 1 — Compute T_local

```
T_local = cpu_cycles / cpu_available
```

`cpu_available` is the vehicle's CPU frequency available at the moment of task generation.
If other tasks are currently running, `cpu_available < cpu_allocable` (the vehicle's total allocable share).

Example: task needs 1.8 × 10⁹ cycles, vehicle has 5 × 10⁹ Hz available → `T_local = 0.36 s`

### Step 2 — Compute T_offload (requires recent RSU state from broadcast)

```
T_tx   = (mem_footprint_bytes × 8) / bandwidth_bps
T_edge = cpu_cycles / rsu_cpu_available
T_rx   ≈ 0   (result payload is small; can be included if needed)

T_offload = T_tx + T_edge + T_rx
```

`rsu_cpu_available` comes from the RSU's periodic status broadcast, stored in `rsuMetrics`.
`bandwidth_bps` is the estimated V2I channel capacity (can use the channel model already in the vehicle).

Example: task is 512 KB, RSU has 8 GHz available, channel is 20 Mbps
- `T_tx = (512 × 1024 × 8) / (20 × 10⁶) = 0.210 s`
- `T_edge = 1.8 × 10⁹ / (8 × 10⁹) = 0.225 s`
- `T_offload = 0.210 + 0.225 = 0.435 s`

### Step 3 — Classify and decide

```
can_do_locally = (T_local < remaining_deadline)
can_offload    = rsu_reachable AND (T_offload < remaining_deadline)
```

| Case | Condition | Action |
|---|---|---|
| **MUST_OFFLOAD** | `!can_do_locally && can_offload` | Offload — no local fallback, ever |
| **MUST_LOCAL** | `can_do_locally && !can_offload` | Execute locally |
| **BOTH_FEASIBLE** | `can_do_locally && can_offload` | Compare cost function (see below) |
| **INFEASIBLE** | `!can_do_locally && !can_offload` | Fail immediately — report to RSU, do NOT retry |

### Step 4 — BOTH_FEASIBLE: cost-function comparison

When both paths meet the deadline, choose the one with lower weighted cost:

```
E_local   = P_compute × T_local
E_offload = P_tx × T_tx + P_idle × (T_edge + T_rx)

cost_local   = α × T_local   + β × E_local
cost_offload = α × T_offload + β × E_offload

if (cost_offload < cost_local):
    → offload   (faster AND/OR more energy-efficient)
else:
    → execute locally   (local is good enough; saves RSU capacity for MUST_OFFLOAD tasks)
```

Suggested starting weights: `α = 0.6` (latency matters more in IoV), `β = 0.4`.
These can be tuned as parameters in `omnetpp.ini`.

**Why this matters:** Baseline agents (`random`, `least_queue`, etc.) always try to offload regardless of feasibility. Our algorithm only offloads when it is necessary or beneficial. This keeps the RSU less congested, so when a `MUST_OFFLOAD` task arrives later the RSU has capacity — a structural advantage over baselines.

### Code location

This logic replaces the current `HeuristicDecisionMaker::makeDecision()` call **and** the two OFFLOAD_SKIP gates inside `PayloadVehicleApp::generateTask()` (around lines 1107–1155 in the current file).

The `HeuristicDecisionMaker` class stays; its `makeDecision()` method should be rewritten to implement the four-case classification above. It is called by the vehicle only, not by the RSU.

---

## Problem 2: Proposed rsuDecisionTimeout Behaviour

The timeout value and the fallback action must now depend on the **Gate B classification**.

### MUST_OFFLOAD case

```
rsuDecisionTimeout = remaining_deadline
// (use the full remaining deadline as the wait budget)
```

If the timeout fires (RSU did not respond in time):
- Task **FAILS** — report `"OFFLOAD_TIMEOUT_FAIL"` to RSU
- **No local fallback** — local execution was already confirmed infeasible by Gate B
- Delete the task

```cpp
// On rsuDecisionTimeout fire for a MUST_OFFLOAD task:
task->state = FAILED;
tasks_failed++;
sendTaskFailureToRSU(task, "OFFLOAD_TIMEOUT_NO_LOCAL_FALLBACK");
delete task;
```

### BOTH_FEASIBLE case

The vehicle has chosen to offload because it is cheaper, but local is still viable.

```
// Leave enough time to execute locally if RSU is slow
rsuDecisionTimeout = min(
    remaining_deadline - T_local - SAFETY_MARGIN,   // must receive decision before local-only deadline
    remaining_deadline                               // hard cap
)
```

If the timeout fires:
- **Fall back to local execution** — local is still within deadline by definition of BOTH_FEASIBLE
- This is the existing fallback path; it is correct only for this case

### How to track which case a task is in

Add a field to the pending offloading decision tracking structure:

```cpp
struct PendingOffloadDecision {
    Task* task;
    bool must_offload;   // true = MUST_OFFLOAD; false = BOTH_FEASIBLE
    ...
};
```

The `rsuDecisionTimeout` handler checks `must_offload` to choose the right behaviour.

---

## Summary of Changes Required

| File | Change |
|---|---|
| `TaskOffloadingDecision.cc` | Rewrite `HeuristicDecisionMaker::makeDecision()` using T_local / T_offload / cost function |
| `PayloadVehicleApp.cc` | Move `sendTaskMetadataToRSU()` to **after** Gate B decision |
| `PayloadVehicleApp.cc` | Remove the two OFFLOAD_SKIP gates (lines ~1107–1155); replace with Gate B classification |
| `PayloadVehicleApp.cc` | Set `rsuDecisionTimeout` based on case (MUST_OFFLOAD vs BOTH_FEASIBLE) |
| `PayloadVehicleApp.cc` | On `rsuDecisionTimeout` fire: MUST_OFFLOAD → FAIL; BOTH_FEASIBLE → local fallback |
| `PayloadVehicleApp.h` | Add `bool must_offload` field to pending offloading decision struct |

---

## What Does NOT Change

- Gate A (RSSI check): the `-65 dBm` threshold is correct, keep it
- The `offloadedTaskTimeout` handler (for tasks the vehicle sent to RSU and awaits execution result): unchanged
- The RSU-side `checkDecisionMsg` handler (for writing decisions and dispatching sub-tasks): unchanged (separate fix)
- Baseline agents in Python: they never see Gate A/B; they receive only tasks that our algorithm committed to offloading
