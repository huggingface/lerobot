"""Online LIBERO-Safety evaluation requires the official benchmark environment.

No object pose registry or simulator-derived training labels are defined here.
"""


class LiberoSafetyOnlineBackend:
    def __init__(self, official_evaluator=None):
        self.official_evaluator = official_evaluator

    def evaluate(self, policy):
        if self.official_evaluator is None:
            raise RuntimeError("Install and provide the official LIBERO-Safety evaluator")
        return self.official_evaluator(policy)
