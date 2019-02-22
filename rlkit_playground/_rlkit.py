def _import():
    import os
    import sys
    rlkit_path = os.path.join(os.path.dirname(__file__), "../../misc/rlkit")
    print(rlkit_path)
    if rlkit_path not in sys.path:
        sys.path.insert(0, rlkit_path)
_import()
