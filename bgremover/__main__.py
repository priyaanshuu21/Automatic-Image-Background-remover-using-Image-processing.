"""Package execution entry point for ``python -m bgremover``.

``python -m bgremover gui`` opens the desktop front-end; any other
invocation runs the command-line interface (``--help`` lists the
``remove``, ``batch``, ``stages``, ``analyze`` and ``info`` commands).
"""

from __future__ import annotations

import sys

from bgremover.cli import main

if "gui" in sys.argv:
    from bgremover.gui import launch

    launch()
else:
    sys.exit(main())
