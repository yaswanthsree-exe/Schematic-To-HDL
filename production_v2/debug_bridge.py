"""
Debug script: directly patch build_nets internals to trace the EP-to-interior
bridge for isolated pin endpoints.
"""
import logging, sys, numpy as np, inspect, textwrap
logging.basicConfig(level=logging.WARNING)

import predict as P

# ── Grab the source of build_nets to understand variable names ──────────────
# We will call build_nets but inject print statements around the dense_lbl
# building and bridge loop by running a modified copy.

# Instead, let's just run the pipeline and then do a post-hoc inspection
# by re-running the skel building ourselves.

# The easiest approach: monkey-patch cv2.ximgproc.thinning to capture
# intermediate state, then run predict_circuit.

# Actually, let's take the simplest approach: call predict_circuit with debug
# logging and parse the log.

import io
buf = io.StringIO()
hdl = logging.StreamHandler(buf)
hdl.setLevel(logging.DEBUG)
logging.getLogger('predict').setLevel(logging.DEBUG)
logging.getLogger('predict').addHandler(hdl)

result = P.predict_circuit(
    r'C:\Yaswanth\Yash\schematic_to_netlist_backupexp\final_prototype\kjabsd.jpg'
)

log_text = buf.getvalue()

# Print lines relevant to bridging
print("=== BRIDGE LOG LINES ===")
for line in log_text.splitlines():
    if any(k in line for k in ['bridge', 'Bridge', 'gap', 'Gap', 'endpoint',
                                 'Endpoint', 'dense', 'Dense', 'ep_', 'EP']):
        print(" ", line)

print()
print("=== PIN-NET FINAL ===")
pin_nets = {}
# re-derive from netlist text
print(result.netlist)
