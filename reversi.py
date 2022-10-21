#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Ilya Ryzhkov / Katarzyna Węsierska
# Created Date: oct 2022
# version ='0.1'
# https://pl.wikipedia.org/wiki/Reversi
# Required: easyAI
# ---------------------------------------------------------------------------

import numpy as np
from easyAI import TwoPlayerGame

to_string = lambda a: "ABCDEFGH"[a[0]] + str(a[1] + 1)
to_array = lambda s: np.array(["ABCDEFGH".index(s[0]), int(s[1]) - 1])


class Reversi(TwoPlayerGame):
    """
    Każdy z dwóch graczy ma do dyspozycji pionki: jeden koloru białego,
    drugi czarnego. Początkowo na planszy znajdują się po dwa pionki każdego
    z graczy. Gracze układają na przemian pionki własnego koloru na wolnych
    polach planszy do momentu, aż plansza zostanie całkowicie zapełniona lub
    żaden z graczy nie będzie mógł wykonać dozwolonego ruchu. Dozwolony ruch
    to taki, w którym pionek jest ułożony na polu, które znajduje się w
    linii (poziomej, pionowej lub ukośnej) z innym pionkiem gracza
    wykonującego ruch, i na dokładnie wszystkich polach pomiędzy wybranym
    polem a tym pionkiem znajdują się pionki przeciwnika. Te pionki zostają
    po wykonaniu ruchu przejęte i zmieniają kolor na przeciwny. Wygrywa ten
    z graczy, którego większa liczba pionków znajduje się na planszy po
    zakończeniu gry; jeśli liczba pionków graczy jest jednakowa, następuje
    remis. :param players: players from easyAI framework.
    :param board: array passed to game

    """

    def __init__(self, players, board=None):
        self.players = players
        self.board = np.zeros((8, 8), dtype=int)
        self.board[3, [3, 4]] = [1, 2]
        self.board[4, [3, 4]] = [2, 1]
        self.current_player = 1

    def possible_moves(self):
        """ Dozwolone są tylko takie ruchy które prowadza do odwrócenia
        kolorów.
        :return: all moves allowed in the game
        """
        return [
            to_string((i, j))
            for i in range(8)
            for j in range(8)
            if (self.board[i, j] == 0)
               and (pieces_flipped(self.board, (i, j),
                                   self.current_player) != [])
        ]

    def make_move(self, pos):
        """Umieszcza pionek i odwraca kolory
        :param pos: """
        pos = to_array(pos)
        flipped = pieces_flipped(self.board, pos, self.current_player)
        for i, j in flipped:
            self.board[i, j] = self.current_player
        self.board[pos[0], pos[1]] = self.current_player

    def show(self):
        """ Drukuje na ekranie wirtualna plansze"""
        print(
            "\n"
            + "\n".join(
                ["  1 2 3 4 5 6 7 8"]
                + [
                    "ABCDEFGH"[k]
                    + " "
                    + " ".join(
                        [[".", "1", "2", "X"][self.board[k][i]] for i in
                         range(8)]
                    )
                    for k in range(8)
                ]
                + [""]
            )
        )

    def is_over(self):
        """Gra jest zakonczona gdy gracze nie moga wykonac juz zadnych ruchow
        :return : game is over
        """
        return self.possible_moves() == []

    def scoring(self):
        """ Początkowo zwraca się uwage na rozmieszczenie pionkow,
         nastepnie tylko na to jaka liczbe pionkow na planszy ma dany gracz.
        """

        if np.sum(self.board == 0) > 32:
            # mniej niz polowa tablicy jest pelna
            player = (self.board == self.current_player).astype(int)
            opponent = (self.board == self.opponent_index).astype(int)
            return ((player - opponent) * BOARD_SCORE).sum()
        else:
            npieces_player = np.sum(self.board == self.current_player)
            npieces_opponent = np.sum(self.board == self.opponent_index)
            return npieces_player - npieces_opponent


# Ta tablica uzywana jest przez AI.
BOARD_SCORE = np.array(
    [
        [9, 3, 3, 3, 3, 3, 3, 9],
        [3, 1, 1, 1, 1, 1, 1, 3],
        [3, 1, 1, 1, 1, 1, 1, 3],
        [3, 1, 1, 1, 1, 1, 1, 3],
        [3, 1, 1, 1, 1, 1, 1, 3],
        [3, 1, 1, 1, 1, 1, 1, 3],
        [3, 1, 1, 1, 1, 1, 1, 3],
        [9, 3, 3, 3, 3, 3, 3, 9],
    ]
)

DIRECTIONS = [
    np.array([i, j]) for i in [-1, 0, 1] for j in [-1, 0, 1] if
    (i != 0 or j != 0)
]


def pieces_flipped(board, pos, current_player):
    """
    Funkcja zwraca liste elementow do
    odwrocenia jesli gracz wykona wlasciwy ruch.
    :return : list of element to flip
    """

    flipped = []

    for d in DIRECTIONS:
        ppos = pos + d
        streak = []
        while (0 <= ppos[0] <= 7) and (0 <= ppos[1] <= 7):
            if board[ppos[0], ppos[1]] == 3 - current_player:
                streak.append(+ppos)
            elif board[ppos[0], ppos[1]] == current_player:
                flipped += streak
                break
            else:
                break
            ppos += d

    return flipped


if __name__ == "__main__":
    from easyAI import Human_Player, AI_Player, Negamax

    game = Reversi([Human_Player(Negamax(4)), AI_Player(Negamax(4))])
    game.play()
    if game.scoring() > 0:
        print("player %d wins." % game.current_player)
    elif game.scoring() < 0:
        print("player %d wins." % game.opponent_index)
    else:
        print("Draw.")
