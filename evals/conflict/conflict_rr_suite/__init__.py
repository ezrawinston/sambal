"""Grammar-vs-plausibility conflict suite generator (role-reversal templates).

Core idea: generate swap-only minimal pairs where the ungrammatical sentence is semantically
more plausible, then use a baseline LM as a plausibility oracle and a control minimal pair
to ensure the grammatical phenomenon is easy.
"""
