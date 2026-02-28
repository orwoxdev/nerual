"""
Computational graph utilities for Orwox Neural.
"""

def topological_sort(root):
    """Performs topological sort on the computational graph."""
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(root)
    return topo
