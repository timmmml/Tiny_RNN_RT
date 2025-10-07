import numpy as np
from random import random
import torch

# -------------------------------------------------------------------------------------
# Reduced two step
# -------------------------------------------------------------------------------------


class Two_step:
    """Basic two-step task without choice at second step."""

    def __init__(
        self,
        com_prob=0.8,
        rew_gen="blocks",
        block_length=50,
        probs=[0.2, 0.8],
        step_SD=0.1,
    ):
        assert rew_gen in [
            "walks",
            "fixed",
            "blocks",
            "trans_rev",
        ], "Reward generator type not recognised."
        self.com_prob = com_prob  # Probability of common transition.
        self.rew_gen = rew_gen  # Which reward generator to use.
        if rew_gen in ("blocks", "trans_rev"):
            self.block_length = block_length  # Length of each block.
            self.probs = probs  # Reward probabilities within block.
        elif rew_gen == "walks":
            self.step_SD = step_SD  # Standard deviation of random walk step sizes.
        elif rew_gen == "fixed":
            self.probs = probs

    def reset(self, n_trials=1000):
        "Generate a fresh set of reward probabilities."
        if self.rew_gen == "walks":
            self.reward_probs = _gauss_rand_walks(n_trials, self.step_SD)
        elif self.rew_gen == "fixed":
            self.reward_probs = _const_reward_probs(n_trials, self.probs)
        elif self.rew_gen == "blocks":
            self.reward_probs = _fixed_length_blocks(
                n_trials, self.probs, self.block_length
            )
        elif (
            self.rew_gen == "trans_rev"
        ):  # Version with reversals in transition matrix.
            # self.reward_probs  = _fixed_length_blocks(n_trials, self.probs, self.block_length * 2)
            # self.trans_probs = _fixed_length_blocks(n_trials + self.block_length,
            #                   self.probs, self.block_length * 2)[self.block_length:,0]
            self.reward_probs = _fixed_length_blocks(
                n_trials, self.probs, self.block_length
            )
            self.trans_probs = _fixed_length_blocks(
                n_trials, self.probs, self.block_length * 5
            )[:, 0]
            self.trans_prob_iter = iter(self.trans_probs)
        self.rew_prob_iter = iter(self.reward_probs)

    def trial(self, choice):
        "Given first step choice generate second step and outcome."
        if self.rew_gen == "trans_rev":
            self.com_prob = next(self.trans_prob_iter)
        transition = int(random() < self.com_prob)  # 1 if common, 0 if rare.
        second_step = int(
            choice == transition
        )  # Choice 1 (0) commonly leads to second_step 1 (0).
        outcome = int(random() < next(self.rew_prob_iter)[second_step])
        return (second_step, outcome)


# -------------------------------------------------------------------------------------
# Two step TORCH
# -------------------------------------------------------------------------------------


class Two_step_torch:
    """Basic two-step task without choice at second step."""

    def __init__(
        self,
        com_prob=0.8,
        rew_gen="blocks",
        block_length=50,
        probs=[0.2, 0.8],
        step_SD=0.1,
    ):
        assert rew_gen in [
            "walks",
            "fixed",
            "blocks",
            "trans_rev",
        ], "Reward generator type not recognised."
        self.com_prob = com_prob  # Probability of common transition.
        self.rew_gen = rew_gen  # Which reward generator to use.
        if rew_gen in ("blocks", "trans_rev"):

            self.block_length = block_length  # Length of each block.
            self.probs = probs  # Reward probabilities within block.
        elif rew_gen == "walks":
            self.step_SD = step_SD  # Standard deviation of random walk step sizes.
        elif rew_gen == "fixed":
            self.probs = probs

    def reset(self, n_blocks=100, n_trials=1000):
        "Generate a fresh set of reward probabilities."
        if self.rew_gen == "walks":
            self.reward_probs = _gauss_rand_walks_torch(
                n_blocks, n_trials, self.step_SD
            )
        elif self.rew_gen == "fixed":
            self.reward_probs = _const_reward_probs_torch(
                n_blocks, n_trials, self.probs
            )
        elif self.rew_gen == "blocks":
            self.reward_probs = _fixed_length_blocks_torch(
                n_blocks, n_trials, self.probs, self.block_length
            )
        elif (
            self.rew_gen == "trans_rev"
        ):  # Version with reversals in transition matrix.
            # self.reward_probs  = _fixed_length_blocks(n_trials, self.probs, self.block_length * 2)
            # self.trans_probs = _fixed_length_blocks(n_trials + self.block_length,
            #                   self.probs, self.block_length * 2)[self.block_length:,0]
            self.reward_probs = _fixed_length_blocks_torch(
                n_blocks, n_trials, self.probs, self.block_length
            )
            self.trans_probs = _fixed_length_blocks_torch(
                n_blocks, n_trials, self.probs, self.block_length * 5
            )[:, :, 0]
            self.trans_prob_iter = iter(self.trans_probs.permute(1, 0))

        self.rew_prob_iter = iter(self.reward_probs.permute(1, 0, 2))

    def reset_without_regen(self):
        self.rew_prob_iter = iter(self.reward_probs.permute(1, 0, 2))
        if hasattr(self, "trans_probs"):
            self.trans_prob_iter = iter(self.trans_probs)

    def trial(self, choice):
        "Given first step choice generate second step and outcome."
        if self.rew_gen == "trans_rev":
            self.com_prob = next(self.trans_prob_iter).to(choice.device)
        if isinstance(self.com_prob, float):
            self.com_prob = torch.tensor(self.com_prob, device=choice.device).repeat(
                choice.shape[0]
            )
        if self.com_prob.shape[0] != choice.shape[0]:
            self.com_prob = torch.tensor(self.com_prob[0], device=choice.device).repeat(
                choice.shape[0]
            )

        # print(self.com_prob.shape)
        transition = torch.tensor(
            (torch.rand_like(self.com_prob) < self.com_prob),
            dtype=torch.int,
            device=choice.device,
        )  # 1 if common, 0 if rare.
        second_step = torch.tensor(
            (choice == transition), dtype=torch.int, device=choice.device
        )  # Choice 1 (0) commonly leads to second_step 1 (0).
        rew_prob_iter = next(self.rew_prob_iter).to(choice.device)
        # print(self.com_prob.shape)
        # print(rew_prob_iter[:,second_step].shape)
        outcome = torch.tensor(
            (
                torch.rand_like(self.com_prob)
                < torch.tensor(
                    [rew_prob_iter[i, s] for i, s in enumerate(second_step)],
                    device=choice.device,
                )
            ),
            dtype=torch.int,
            device=choice.device,
        )
        correct_choice = torch.tensor(
            [
                0 if rew_prob_iter[i, 0] > rew_prob_iter[i, 1] else 1
                for i, s in enumerate(second_step)
            ],
            device=choice.device,
        )
        return (second_step, outcome, correct_choice)


# -------------------------------------------------------------------------------------
# Original two step
# -------------------------------------------------------------------------------------


class Orig_two_step:
    """Orignal version of the two step task used in Daw et al. 2011."""

    def __init__(self, com_prob=0.7, rew_gen="walks", step_SD=0.025):
        self.com_prob = com_prob  # Probability of common transition.
        self.step_SD = step_SD  # Standard deviation of random walk step sizes.
        self.rew_gen = rew_gen

    def reset(self, n_trials=1000):
        "Generate a fresh set of reward probabilities."
        if self.rew_gen == "walks":
            self.reward_probs = _gauss_rand_walks(
                n_trials, self.step_SD, p_range=[0.25, 0.75], n_walks=4
            ).reshape(n_trials, 2, 2)
        elif self.rew_gen == "blocks":
            self.reward_probs = np.tile(
                _fixed_length_blocks(n_trials, [0.2, 0.8], 50).reshape(n_trials, 2, 1),
                [1, 1, 2],
            )
        self.rew_prob_iter = iter(self.reward_probs)

    def first_step(self, choice):
        "Given first step choice generate second step."
        transition = int(random() < self.com_prob)  # 1 if common, 0 if rare.
        second_step = int(
            choice == transition
        )  # Choice 1 (0) commonly leads to second_step 1 (0).
        return second_step

    def second_step(self, second_step, choice_2):
        outcome = int(random() < next(self.rew_prob_iter)[second_step, choice_2])
        return outcome


# -------------------------------------------------------------------------------------
# Reward generators.
# -------------------------------------------------------------------------------------


def _gauss_rand_walks(n_trials, step_SD, p_range=[0, 1], n_walks=2):
    "Generate a set of reflecting Gaussian random walks."
    walks = np.random.normal(scale=step_SD, size=(n_trials, n_walks))
    walks[0, :] = np.random.rand(n_walks)
    walks = np.cumsum(walks, 0)
    walks = np.mod(walks, 2.0)
    walks[walks > 1.0] = 2.0 - walks[walks > 1.0]
    if p_range != [0, 1]:
        walks = walks * (p_range[1] - p_range[0]) + p_range[0]
    return walks


def _gauss_rand_walks_torch(n_blocks, n_trials, step_SD, p_range=[0, 1], n_walks=2):
    """Generate Gaussian random walk for n_block blocks"""
    walks = torch.randn(n_blocks, n_trials, n_walks) * step_SD
    walks[:, 0, :] = torch.rand(n_blocks, n_walks)
    walks = walks.cumsum(1)
    walks = walks.remainder(2.0)
    walks[walks > 1.0] = 2.0 - walks[walks > 1.0]
    if p_range != [0, 1]:
        walks = walks * (p_range[1] - p_range[0]) + p_range[0]
    return walks


def _const_reward_probs(n_trials, probs):
    "Constant reward probabilities on each arm."
    return np.tile(probs, (n_trials, 1))


def _const_reward_probs_torch(n_blocks, n_trials, probs):
    "Constant reward probabilities on each arm."
    return torch.tensor(probs).repeat(n_blocks, n_trials, 1)


def _fixed_length_blocks(n_trials, probs, block_length):
    "Reversals in reward probabilities every block_length_trials."
    block_1 = np.tile(probs, (block_length, 1))
    block_2 = np.tile(probs[::-1], (block_length, 1))
    return np.tile(
        np.vstack([block_1, block_2]),
        (
            int(np.ceil(n_trials / (2.0 * block_length))),
            1,
        ),
    )[:n_trials, :]


def _fixed_length_blocks_torch(n_blocks, n_trials, probs, block_length):
    "Reversals in reward probabilities every block_length_trials."
    block_1 = torch.tensor(probs).repeat(block_length, 1)
    block_2 = torch.tensor(probs[::-1]).repeat(block_length, 1)
    return (
        torch.cat((block_1, block_2))
        .repeat(int(np.ceil(n_trials / (2.0 * block_length))), 1, 1)[:n_trials, :]
        .repeat(n_blocks, 1, 1)
    )
