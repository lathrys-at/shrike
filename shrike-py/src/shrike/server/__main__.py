"""``python -m shrike.server`` entry point.

Preserves the documented foreground-run invocation now that ``server`` is a
package rather than a single module.
"""

from shrike.server.server import main

if __name__ == "__main__":
    main()
