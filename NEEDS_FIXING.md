# NEEDS FIXING — known gaps, limitations and recommended fixes

**Branch:** `feature/part1-wire-tracing`
**Status of this document:** current as of the flip-flop recognition work.
**Frozen reference:** `feature/pattern-engine` holds the last known-good state.

This is a deliberately blunt list of what does **not** work, why, and what would
fix it. Everything here has been reproduced and measured, not guessed. Where a
number appears it came from an actual run, and the command that produced it is
given so it can be re-checked.

---

## How to read the severity

| Severity | Meaning |
|---|---|
| **A — silently wrong** | Produces confident, plausible output that is incorrect. Worst kind. |
| **B — fails to recognise** | Declines to produce a result. Visible, not misleading. |
| **C — inherent limit** | Cannot be resolved from the information available; needs a different input signal. |
| **D — process gap** | Nothing is wrong today, but there is no way to *prove* it stays that way. |

---

## A — Silently wrong

### A1. Block-form external wiring is never traced *(open)*

A block symbol is recognised from its box and pin labels only. The wires
**running to** those pins are not followed, so:

* two pins driven by the same signal look independent;
* the outside net names are lost.

**Observed:** a JK symbol with `T` wired to *both* `J` and `K` — a T flip-flop
built from a JK — emits

```verilog
module images__9_ (input CLK, input J, input K, output Q, output Qbar);
```

`J` and `K` are exposed as separate ports. The device is right; the circuit
around it is wrong.

**Impact:** any schematic with more than one block, or with blocks wired to each
other (a shift register, a ripple counter), comes out as disconnected devices.

**Recommended fix:** give the block path its own connectivity stage. Locate each
pin stub where it meets the box edge, follow it into the existing skeleton/net
graph (`build_nets` already produces exactly this), and resolve shared nets and
outside labels. Reuses Part 1 machinery; no new CV.

**Effort:** days. **Risk:** low — additive, does not touch the gate path.

---

## B — Fails to recognise

### B1. Gate-drawn flip-flops from real images *(open — the big one)*

**12 of 12** gate-level flip-flop schematics in the test set fail. This is the
single largest gap.

The pattern engine is **not** at fault: every one of these circuits is
recognised correctly from a hand-written graph (see `production_v2/tests/`).
Part 1 fragments the wiring before the engine ever sees it.

**Measured on a JK schematic** — detection is perfect (4 NANDs, correct
classes), the binary wire mask is clean, all 14 pins assign, and yet:

```
net 155: [G3.in2, G4.in0]   8 edges   <- two INPUTS shorted, no driver
net 206: [G1.out]          16 edges   <- an OUTPUT driving nothing
```

Two concrete breaks were located:

* the cross-coupling wire breaks across a **31 px gap** where it passes the
  corner of a gate's bounding box;
* the outer Q/Q̄ feedback rectangle breaks across a **343 px** span.

**Root causes (both plausible, neither yet fixed):**

1. **Gate erasure uses the bounding rectangle.** A NAND's body is rounded, so a
   wire routed past the box corner is erased along with the gate.
   *Fix:* erase the gate's actual ink — flood-fill the body from inside the box
   — instead of filling the whole rectangle.
2. **`DIR_WEIGHT = 35` rewards wires "approaching from the correct side".**
   Feedback runs right-to-left and is therefore actively penalised. Sequential
   circuits violate the left-to-right assumption the tracer was tuned on.
   *Fix:* relax directional scoring for wires that terminate on two gate pins,
   which a genuine feedback wire always does.

**Effort:** weeks. **Risk:** high. `context.md` documents plausible-looking
changes in this area costing real accuracy (propagation 322→314 and 322→315,
both reverted). Do not attempt without D1 below.

### B2. Missing gate-level patterns *(partly closed)*

Present: SR (NOR), SR (NAND), gated SR (NAND **and** NOR), D latch (NAND **and**
NOR), JK, master-slave DFF, XOR.

Still missing:

* **T flip-flop** (gate-realised)
* **Edge-triggered D** — the 6-NAND textbook circuit, very common
* **JK master-slave**
* **Every async set/reset variant**

Reset circuits currently **refuse to match** rather than matching wrongly — the
unclaimed-input guard (see C3) makes that failure safe rather than silent.

**Recommended fix:** write the topology JSONs. The matcher already supports
variadic ports, so this is data entry, and validators are reused across drawing
styles. Each new file needs a synthetic-graph test.

**Effort:** days. **Risk:** none — additive, and the corpus guards against false
positives.

### B3. Only four block device types

`SR`, `JK`, `D`, `T` (plus the two latch variants). No counters, registers,
shift registers, multiplexers or decoders.

**Recommended fix:** extend `DEVICE_TABLE` / `LATCH_TABLE` in
`pattern_engine/block_form.py` plus an emitter and a BOM part each.

**Effort:** hours per device. **Risk:** none.

---

## C — Inherent limits

### C1. Q/Q̄ and J/K can come out mirrored

A cross-coupled pair is structurally symmetric, so topology alone cannot say
which gate is the "Q side" or the "J side". The recovered logic is functionally
correct; only the labelling may be swapped.

**Recommended fix:** the information *is* on the page — these schematics label
`Q` and `Q̄` explicitly. Read those labels and bind them to the macro's output
ports. Requires C2's label plumbing.

### C2. No concept of a clock anywhere in Part 1

Nothing distinguishes a clock from a data input geometrically, so
level-sensitive versus edge-triggered cannot always be decided from structure.

**Recommended fix:** OCR-driven. Textbook schematics write `CLK` / `Ck` /
`Clock`; treat a net so labelled as the clock and bind it to the pattern's CLK
port. Infrastructure exists (`ocr_net_names`, Stage 7b).

### C3. Async reset is refused, not handled *(deliberate)*

A latch with an undeclared extra input does **not** match. This is intentional:
matching would emit HDL that silently drops the reset. Recorded here so the
behaviour is not mistaken for a bug.

Closing it properly means adding explicit reset-variant patterns (B2).

### C4. Greedy, non-overlapping matching

A gate consumed by one match is unavailable to others, so a small match can
block a larger, better one. Harmless at nine patterns; will bite as the library
grows.

**Recommended fix:** score candidates and prefer the largest/most specific, or
backtrack. Revisit when the library exceeds ~20 patterns.

---

## D — Process gaps

### D1. No sequential regression corpus *(blocks B1)*

All 177 corpus images are **combinational**. The repeated "0 regressions" result
is real, but it only proves combinational behaviour was preserved. Every
sequential guarantee currently rests on synthetic graphs plus a handful of
ad-hoc images.

**This is the prerequisite for B1.** Attempting the wire-tracing fixes without a
labelled sequential set means tuning blind — precisely how the regressions in
`context.md` happened.

**Recommended fix:** assemble 30–50 sequential schematics (latches, flip-flops,
gate-drawn and block-form, clean and photographed) with expected netlists, and
extend `corpus_run.py` / `corpus_diff.py` to cover them.

**Effort:** days, mostly labelling. **Risk:** none.

### D2. No functional equivalence checking

The truth-table validator only handles combinational subgraphs; sequential
patterns rely on structural predicates. There is no proof that a recovered
flip-flop behaves like the gates it replaced.

**Recommended fix:** extend `pattern_engine/evaluator.py` to simulate over input
*sequences* and compare state trajectories before and after compression.

---

## Recommended order of work

1. **D1** — build the sequential corpus. Everything sequential is unmeasured
   until this exists.
2. **B1** — the wire-tracing fixes, validated against D1. Biggest payoff.
3. **A1** — block pin tracing. Unlocks multi-block schematics.
4. **B2 / B3** — fill out the pattern and device libraries. Cheap, parallelisable.
5. **C1 / C2** — OCR-driven clock and Q/Q̄ binding.
6. **D2 / C4** — verification and matching strategy, once the library is large.

---

## Reproducing the measurements

```bash
# unit tests (no models, no corpus needed)
python -m pytest production_v2/tests -q

# full corpus regression against a recorded baseline
python corpus_run.py after.json
python corpus_diff.py baseline.json after.json
```

A change is acceptable only when the diff shows **0 regressions**. Compare
per-image gate classes, not just the propagation total — that total is a coarse
health metric and has hidden real damage before: a change that flipped gate
classes on 9 images left the total unchanged at 749.
