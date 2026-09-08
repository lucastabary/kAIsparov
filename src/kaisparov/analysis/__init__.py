"""Post-hoc analysis of *played* moves — the "game review" side of the project.

Where :mod:`kaisparov.agents` decides what to play and :mod:`kaisparov.insights`
defines the vocabulary, this package grades a move that was played: how much
winning chance it gave away, and which chess.com-style label that earns
(``Brillant``, ``Erreur``, ``Gaffe``, ...).

Two pieces, deliberately split so the UI never has to import torch:

* an :class:`~kaisparov.analysis.evaluators.Evaluator` scores a position for the
  side to move — :class:`~kaisparov.analysis.evaluators.MaterialEvaluator` is
  torch-free and works on a fresh clone, :class:`~kaisparov.analysis.critic.CriticEvaluator`
  asks a trained model's value head instead;
* :class:`~kaisparov.analysis.judge.MoveJudge` turns "score every legal move,
  compare the one played to the best one" into a
  :class:`~kaisparov.insights.MoveVerdict`.
"""

from kaisparov.analysis.evaluators import Evaluator, HeuristicEvaluator, MaterialEvaluator
from kaisparov.analysis.judge import (
    GameReview,
    MoveJudge,
    ReviewThresholds,
    accuracy_from_loss,
    win_probability,
)

__all__ = [
    "Evaluator",
    "HeuristicEvaluator",
    "GameReview",
    "MaterialEvaluator",
    "MoveJudge",
    "ReviewThresholds",
    "accuracy_from_loss",
    "win_probability",
]
