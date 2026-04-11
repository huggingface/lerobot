"""
Cross-cutting modules that bridge multiple lerobot packages.

Unlike ``lerobot.utils`` (which must remain dependency-free), modules here
are allowed to import from ``lerobot.policies``, ``lerobot.processor``,
``lerobot.configs``, etc.  They are deliberately NOT re-exported from the
top-level ``lerobot`` package.
"""
