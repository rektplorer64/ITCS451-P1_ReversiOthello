"""This module contains agents that play reversi."""

import abc
import asyncio
import copy
import random
import sys
import traceback

import gym
import numpy as np

_ENV = gym.make('Reversi-v0')
_ENV.reset()


def transition(board, player, action):
    """Return a new board if the action is valid, otherwise None."""
    if _ENV.is_valid((board, player), action):
        new_board, __ = _ENV.get_next_state((board, player), action)
        return new_board
    return None


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
        try:
            await self.search(board, valid_actions)
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            print(e)
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
        # If you want to "simulate a move", you can call the following function:
        # transition(board, self.player, valid_actions[0])
        # print("\nRandom Agent Possible Actions: " + str(valid_actions))
        await asyncio.sleep(2.0)
        randidx = random.randint(0, len(valid_actions) - 1)
        self._move = valid_actions[randidx]


class MyAgent(ReversiAgent):

    def __index__(self):
        super(MyAgent, self)
        # self.transpositionTable = set()

    async def search(self, board, valid_actions: np.array):
        evaluation, bestAction = self.minimax(board, valid_actions, 5, 0, - sys.maxsize - 1, sys.maxsize, True)

        # self.createState(board, valid_actions, self._color)

        print("Me Selected: " + str(bestAction))
        self._move = bestAction

    def minimax(self, board: np.array, validActions: np.array, depth: int, levelCount: int, alpha: int, beta: int,
                maximizingPlayer: bool):
        if depth == 0:
            return self.evaluateStatistically(board)

        bestAction = None
        if maximizingPlayer:
            mAlpha = alpha
            maxEval: int = - sys.maxsize - 1
            player = self._color

            for action in validActions:
                # TODO: transition function does not return new actions! lol.
                newState, newValidActions = self.createState(board, action, player)
                evaluation = self.minimax(newState, newValidActions, depth - 1, levelCount + 1, mAlpha, beta, not maximizingPlayer)

                if maxEval < evaluation:
                    maxEval = evaluation

                    if levelCount == 0:
                        bestAction = action

                mAlpha = max(mAlpha, evaluation)
                if beta <= mAlpha:
                    break
            if levelCount != 0:
                return maxEval
            else:
                return maxEval, bestAction
        else:
            mBeta = beta
            minEval: int = sys.maxsize
            player = self.getOpponent(self._color)

            for action in validActions:
                # TODO: transition function does not return new actions! lol.
                newState, newValidActions = self.createState(board, action, player)
                evaluation = self.minimax(newState, newValidActions, depth - 1, levelCount + 1, alpha, mBeta, not maximizingPlayer)

                if minEval > evaluation:
                    minEval = evaluation

                    if levelCount == 0:
                        bestAction = action

                mBeta = min(mBeta, evaluation)
                if mBeta <= alpha:
                    break
            if levelCount != 0:
                return minEval
            else:
                return minEval, bestAction

    def evaluateStatistically(self, board: np.array) -> int:
        countA = 0
        countB = 0
        evalBoard = np.array(list(zip(*board.nonzero())))

        # print("Print Board: " + str(evalBoard))
        for row in evalBoard:
            if board[row[0]][row[1]] == self._color:
                countA += 1
            else:
                countB += 1
        return countA - countB

    @staticmethod
    def getOpponent(player: int):
        if player == 1:
            return -1
        else:
            return 1

    def createState(self, board, action, player: int) -> (np.array, np.array):
        newState = transition(board, player, action)

        validMoves = _ENV.get_valid((newState, self.getOpponent(player)))
        # print("Valid Moves: " + str(validMoves))

        validMoves = np.array(list(zip(*validMoves.nonzero())))

        return newState, validMoves
