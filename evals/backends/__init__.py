"""Model backends: one module per model family, all implementing ``LanguageModel``.

Evaluators receive a ``LanguageModel`` and never import a model class. A
backend is named on the command line as ``--backend <alias>`` or
``--backend <module>:<Class>``; its construction options travel as
``--backend-arg key=value`` pairs that the backend's ``from_args`` parses.
"""

from .base import BACKEND_ALIASES, LanguageModel, load_backend, parse_backend_args, resolve_backend_class

__all__ = ["BACKEND_ALIASES", "LanguageModel", "load_backend", "parse_backend_args", "resolve_backend_class"]
