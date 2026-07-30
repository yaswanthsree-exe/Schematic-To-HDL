def test_networkx_available():
    import networkx as nx
    from networkx.algorithms.isomorphism import DiGraphMatcher
    assert nx.__version__.startswith("3.")
    assert DiGraphMatcher is not None


def test_can_import_hdl_gen():
    from hdl_gen.graph_to_hdl import _OP
    assert _OP["XOR"] == ("^", False, False)
