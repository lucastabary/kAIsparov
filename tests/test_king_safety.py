"""Tests for the optional king-safety move filter (kaisparov.agents.safety) and the
``avoid_king_suicide`` flag on the baseline agents.

The recurring scenario is the "troll pawn": a pawn is the only thing shielding its
own king from an enemy rook down the file. Moving (here: capturing with) that pawn
opens the file and hangs the king — exactly the blunder the guard must prevent.
"""

from __future__ import annotations

from kaisparov.agents.material_agent import MaterialAgent
from kaisparov.agents.random_agent import RandomAgent
from kaisparov.agents.safety import hangs_own_king, safe_moves
from kaisparov.core.board import ChessGame
from kaisparov.core.coords import BOARD_SIZE
from kaisparov.core.movegen import all_moves
from kaisparov.core.pieces import Piece, PieceType, Player


def empty_game(turn: Player = Player.WHITE) -> ChessGame:
    grid = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    return ChessGame(initial_board=grid, turn=turn)


def place(game, coord, player, piece_type):
    game.grid[coord[0]][coord[1]] = Piece(player, piece_type)


def troll_pawn_game() -> ChessGame:
    """White to move. The white pawn on file 4 shields the white king from the black
    rook; capturing the bait pawn with it (the only capture) opens the file and hangs
    the king. The bait is a pawn on (3,2), which attacks (4,1)/(2,1) — not the king's
    square (4,0) — so the king is genuinely safe until the shield moves."""
    game = empty_game(turn=Player.WHITE)
    place(game, (4, 0), Player.WHITE, PieceType.KING)  # king behind the pawn
    place(game, (4, 1), Player.WHITE, PieceType.PAWN)  # the shield
    place(game, (4, 7), Player.BLACK, PieceType.ROOK)  # aims down file 4
    place(game, (3, 2), Player.BLACK, PieceType.PAWN)  # bait for the pawn capture
    return game


SUICIDE_MOVE = ((4, 1), (3, 2))  # pawn takes pawn -> opens the file -> king hangs


def test_hangs_own_king_detects_opening_the_file():
    game = troll_pawn_game()
    assert hangs_own_king(game, SUICIDE_MOVE) is True


def test_hangs_own_king_false_for_a_safe_move():
    game = troll_pawn_game()
    # Pushing the pawn straight keeps it on the file, still shielding the king.
    assert hangs_own_king(game, ((4, 1), (4, 2))) is False


def test_capturing_the_enemy_king_is_never_a_hang():
    game = empty_game(turn=Player.WHITE)
    place(game, (0, 0), Player.WHITE, PieceType.ROOK)
    place(game, (3, 0), Player.BLACK, PieceType.KING)
    # The winning move ends the game -> no reply -> not a hang, even though our own
    # (here absent) king safety is irrelevant once the opponent king is captured.
    assert hangs_own_king(game, ((0, 0), (3, 0))) is False


def test_safe_moves_drops_the_suicide_but_keeps_the_rest():
    game = troll_pawn_game()
    moves = all_moves(game.grid, game.turn, game.en_passant_target)
    filtered = safe_moves(game, moves)
    assert SUICIDE_MOVE in moves  # it is a legal candidate...
    assert SUICIDE_MOVE not in filtered  # ...but the guard removes it
    assert filtered  # and safe alternatives remain


def test_safe_moves_falls_back_when_every_move_hangs():
    game = empty_game(turn=Player.WHITE)
    place(game, (0, 0), Player.WHITE, PieceType.KING)  # lone king, cornered
    place(game, (5, 0), Player.BLACK, PieceType.ROOK)  # covers rank 0 -> (1,0)
    place(game, (0, 5), Player.BLACK, PieceType.ROOK)  # covers file 0 -> (0,1)
    place(game, (3, 3), Player.BLACK, PieceType.BISHOP)  # covers the diagonal -> (1,1)
    moves = all_moves(game.grid, game.turn, game.en_passant_target)
    assert moves  # the king has escape squares...
    assert all(hangs_own_king(game, m) for m in moves)  # ...but all of them hang
    assert safe_moves(game, moves) == moves  # so we fall back to the full list


def test_material_agent_grabs_the_bait_without_the_guard():
    game = troll_pawn_game()
    # The knight is the only capture available -> greedy material grabs it (blunder).
    assert MaterialAgent(seed=0).select_move(game) == SUICIDE_MOVE


def test_material_agent_refuses_the_bait_with_the_guard():
    game = troll_pawn_game()
    move = MaterialAgent(seed=0, avoid_king_suicide=True).select_move(game)
    assert move != SUICIDE_MOVE
    assert not hangs_own_king(game, move)


def test_random_agent_never_plays_a_suicide_with_the_guard():
    agent = RandomAgent(seed=0, avoid_king_suicide=True)
    # Draw many moves; the guarded agent must never open the file.
    assert all(agent.select_move(troll_pawn_game()) != SUICIDE_MOVE for _ in range(50))
