"""This module contains agents that play reversi."""

import abc
import random
import asyncio

import numpy as np
import boardgame2 as bg2


class ReversiAgent(abc.ABC):
    """Reversi Agent."""

    def __init__(self, color):
        """
        Create an agent.
        
        Parameters
        -------------
        color : int
            BLACK is 1 and WHITE is -1. We can get these constants
            from bg2.BLACK and bg2.WHITE.

        """
        super().__init__()
        self._move = None
        self._color = color

    @property
    def player(self):
        """Return the color of this agent."""
        return self._color

    @property
    def pass_move(self):
        """Return move that skips the turn."""
        return np.array([-1, 0])

    @property
    def best_move(self):
        """Return move after the thinking.
        
        Returns
        ------------
        move : np.array
            The array contains an index x, y.

        """
        if self._move is not None:
            return self._move
        else:
            return self.pass_move

    async def move(self, board, valid_actions):
        """Return a move. The returned is also available at self._move."""
        self._move = None
        await self.search(board, valid_actions)
        return self.best_move

    @abc.abstractmethod
    async def search(self, board, valid_actions):
        """
        Set the intended move to self._move.
        
        The intended move is a np.array([r, c]) where r is the row index
        and c is the column index on the board. [r, c] must be one of the
        valid_actions, otherwise the game will skip your turn.

        Parameters
        -------------------
        board : np.array
            An 8x8 array that contains 
        valid_actions : np.array
            An array of shape (n, 2) where n is the number of valid move.

        Returns
        -------------------
        None
            This method should set self._move as a way to return.

        """
        raise NotImplementedError('You will have to implement this.')


class RandomAgent(ReversiAgent):
    """An agent that move randomly."""

    async def search(self, board, valid_actions):
        """Set the intended move to self._move."""
        await asyncio.sleep(0.5)
        randidx = random.randint(0, len(valid_actions) - 1)
        self._move = valid_actions[randidx]
        print('okay')

