from chess.pieces import PieceType, Player, Piece, BOARD_SIZE, PIECES_MOVES

class ChessGame():
    def __init__(self):
        self.grid: list[list[Piece]] = self._build_grid()
        self.player_pieces = self._attribute_pieces()
        self.turn: Player = Player.WHITE
        self.count = 0
    
    def _build_grid(self) -> list[list[Piece]]:
        """builds and returns the grid. grid[0][0] is botdestm left

        Returns:
            list[list[PieceType]]: _description_
        """
        grid = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

        for i in range(BOARD_SIZE):
            grid[i][1] = Piece(Player.WHITE, PieceType.PAWN)
            grid[i][BOARD_SIZE - 2] = Piece(Player.BLACK, PieceType.PAWN)
        
        for i in [0, BOARD_SIZE-1]: 
            grid[i][0] = Piece(Player.WHITE, PieceType.ROOK)
            grid[i][BOARD_SIZE - 1] = Piece(Player.BLACK, PieceType.ROOK)
        
        for i in [1, BOARD_SIZE-2]: 
            grid[i][0] = Piece(Player.WHITE, PieceType.HORSE)
            grid[i][BOARD_SIZE - 1] = Piece(Player.BLACK, PieceType.HORSE)
        
        for i in [2, BOARD_SIZE-3]: 
            grid[i][0] = Piece(Player.WHITE, PieceType.BISHOP)
            grid[i][BOARD_SIZE - 1] = Piece(Player.BLACK, PieceType.BISHOP)
        
        grid[3][0] = Piece(Player.WHITE, PieceType.QUEEN)
        grid[3][BOARD_SIZE - 1] = Piece(Player.BLACK, PieceType.KING)

        grid[4][0] = Piece(Player.WHITE, PieceType.KING)
        grid[4][BOARD_SIZE - 1] = Piece(Player.BLACK, PieceType.QUEEN)

        return grid

    def _attribute_pieces(self):
        player_pieces = {Player.WHITE: [], Player.BLACK: []}
        for j in [1,0]:
            for i in range(BOARD_SIZE):
                player_pieces[Player.WHITE].append(self.grid[i][j])
        for j in [BOARD_SIZE-2, BOARD_SIZE-1]:
            for i in range(BOARD_SIZE):
                player_pieces[Player.BLACK].append(self.grid[i][j])
        return player_pieces

    def print_grid(self):
        print(f"turn: {self.turn.name} | count: {self.count}")
        for j in range(BOARD_SIZE-1, -1, -1):
            print(f"{j} ", end="")
            for i in range(BOARD_SIZE):
                if self.grid[i][j] is None:
                    print("  ", end=" ")
                else:
                    print(self.grid[i][j], end=" ")
            print()
        print("  ", end="")
        for i in range(BOARD_SIZE):
            print(f"{i}  ", end="")
    
    def _is_coord_in_grid(self, coord: tuple[int, int]) -> bool:
        return not(coord[0] < 0 or coord[0] >= BOARD_SIZE or coord[1] < 0 or coord[1] >= BOARD_SIZE)
    
    def possibleMoves(self, source: tuple[int, int]) -> list[tuple[int, int]]:
        """returns a list of possible moves for the piece at the given position.

        Args:
            source (tuple[int, int]): _description_
        """
        if (piece:=self.grid[source[0]][source[1]]) is None:
            print(f"Pas de pièce en {source}")
            return []
        r = []
        if (piece.type in [PieceType.KING, PieceType.HORSE]):
            # Horse (knight) simple moves
            if piece.type == PieceType.HORSE:
                return [move for move in PIECES_MOVES[piece.type] if self._is_coord_in_grid((source[0] + move[0], source[1] + move[1]))]

            # King: normal one-step moves + castling when allowed
            moves = [move for move in PIECES_MOVES[PieceType.KING] if self._is_coord_in_grid((source[0] + move[0], source[1] + move[1]))]

            # Castling: king moves two squares horizontally if neither king nor rook moved
            if not piece.hasMoved:
                y = source[1]

                # Kingside castling (rook on the right)
                right_rook_x = BOARD_SIZE - 1
                right_rook = self.grid[right_rook_x][y]
                if right_rook is not None and right_rook.type == PieceType.ROOK and right_rook.player == piece.player and not right_rook.hasMoved:
                    between_clear = all(self.grid[x][y] is None for x in range(source[0] + 1, right_rook_x))
                    if between_clear and self._is_coord_in_grid((source[0] + 2, y)):
                        moves.append((2, 0))

                # Queenside castling (rook on the left)
                left_rook_x = 0
                left_rook = self.grid[left_rook_x][y]
                if left_rook is not None and left_rook.type == PieceType.ROOK and left_rook.player == piece.player and not left_rook.hasMoved:
                    between_clear = all(self.grid[x][y] is None for x in range(left_rook_x + 1, source[0]))
                    if between_clear and self._is_coord_in_grid((source[0] - 2, y)):
                        moves.append((-2, 0))

            return moves
        
        if (piece.type in [PieceType.QUEEN, PieceType.BISHOP, PieceType.ROOK]):
            for move_list in PIECES_MOVES[piece.type]:
                for move in move_list:
                    dest = (source[0] + move[0], source[1] + move[1])
                    print(f"^dest: {dest}, {type(dest)}")
                    if not self._is_coord_in_grid(dest): break
                    r.append(move)
                    if self.grid[dest[0]][dest[1]] is not None: break
        
        if (piece.type == PieceType.PAWN):
            base_move = PIECES_MOVES[piece.type][0]
            # base_move is defined for white (0,1). Flip direction for black.
            dir_y = base_move[1] if piece.player == Player.WHITE else -base_move[1]
            dx, dy = base_move[0], dir_y

            # one step forward
            dest = (source[0] + dx, source[1] + dy)
            if self._is_coord_in_grid(dest) and self.grid[dest[0]][dest[1]] is None:
                r.append((dx, dy))

            # two steps forward from starting position (ensure both squares empty)
            if (not piece.hasMoved):
                dest2 = (source[0] + dx*2, source[1] + dy*2)
                if self._is_coord_in_grid(dest2) and self._is_coord_in_grid(dest) and self.grid[dest[0]][dest[1]] is None and self.grid[dest2[0]][dest2[1]] is None:
                    r.append((dx*2, dy*2))

            # captures (diagonals) — flip vertical sign for black
            for diag in [(1,1), (-1,1)]:
                diag_dx = diag[0]
                diag_dy = diag[1] if piece.player == Player.WHITE else -diag[1]
                dest = (source[0] + diag_dx, source[1] + diag_dy)
                if self._is_coord_in_grid(dest) and self.grid[dest[0]][dest[1]] is not None and self.grid[dest[0]][dest[1]].player != piece.player:
                     r.append((diag_dx, diag_dy))
        return r

    def isMoveValid(self, source:tuple[int, int], dest:tuple[int, int]) -> bool:
        if self.grid[source[0]][source[1]] is None:
            print(f"Pas de pièce en {source}") 
            return False
        if self.grid[source[0]][source[1]].player != self.turn:
            print(f"its not your piece {source}")
            return False
        
        piece = self.grid[source[0]][source[1]]
        move = (dest[0] - source[0], dest[1] - source[1])
        if move not in self.possibleMoves(source):
            print(f"invalid move for {piece.type.name} from {source} to {dest}")
            return False
        
        return True

    def play(self, source:tuple[int, int], dest:tuple[int, int]) -> Piece | None:
        """plays a move from the given position to the given position.

        Args:
            source (tuple[int, int]): _description_
            dest (tuple[int, int]): _description_
        """
        if not self.isMoveValid(source, dest):
            print(f"invalid move from {source} to {dest}")
            return None
        
        piece = self.grid[source[0]][source[1]]
        dest_piece = self.grid[dest[0]][dest[1]]
        self.grid[dest[0]][dest[1]] = piece
        self.grid[source[0]][source[1]] = None

        piece.hasMoved = True
        # Handle castling: if king moved two squares horizontally, move the corresponding rook
        if piece.type == PieceType.KING:
            dx = dest[0] - source[0]
            if abs(dx) == 2:
                y = dest[1]
                if dx > 0:
                    # kingside: rook from right end to king's left side
                    rook_src_x = BOARD_SIZE - 1
                    rook_dest_x = source[0] + 1
                else:
                    # queenside: rook from left end to king's right side
                    rook_src_x = 0
                    rook_dest_x = source[0] - 1

                rook = self.grid[rook_src_x][y]
                if rook is not None and rook.type == PieceType.ROOK and rook.player == piece.player:
                    self.grid[rook_dest_x][y] = rook
                    self.grid[rook_src_x][y] = None
                    rook.hasMoved = True
        self.count += 1

        other_player = Player.BLACK if self.turn == Player.WHITE else Player.WHITE

        if dest_piece is not None: self.player_pieces[other_player].remove(dest_piece)

        if dest_piece is not None and dest_piece.type == PieceType.KING:
            print(f"Game over! {self.turn.name} wins by capturing the king.")
        else:
            self.turn = other_player
            self.print_grid()
        return dest_piece
