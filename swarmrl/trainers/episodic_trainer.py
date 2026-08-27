"""
Module for the EpisodicTrainer
"""

import warnings
from typing import TYPE_CHECKING

import numpy as np
from rich.progress import BarColumn, Progress, TimeRemainingColumn

from swarmrl.trainers.trainer import Trainer

if TYPE_CHECKING:
    from espressomd import System

from loguru import logger


class EpisodicTrainer(Trainer):
    """
    Class for the simple MLP RL implementation.
    Deprecated: Use UniversalTrainer instead.

    Attributes
    ----------
    rl_protocols : list(protocol)
            A list of RL protocols to use in the simulation.
    """

    def __init__(self, *args, **kwargs):
        # Discussable. If we want to go for the universal trainer,
        # we don't need this class anymore. Then, we can also rename the
        # UniversalTrainer to Trainer, removing the Parent class?
        warnings.warn(
            "EpisodicTrainer is deprecated and might be removed in future versions."
            "Please use the UniversalTrainer instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

    def perform_rl_training(
        self,
        get_engine: callable,
        system: "System",
        n_episodes: int,
        episode_length: int,
        reset_frequency: int = 1,
        load_bar: bool = True,
        save_episodic_data: bool = True,
        episode_offset: int = 0,
    ):
        """
        Perform the RL training.

        Parameters
        ----------
        get_engine : callable
                Function to get the engine for the simulation.
        system : espressomd.System
                Engine used to perform steps for each agent.
        n_episodes : int
                Absolute episode target for this training run. When resuming
                (``episode_offset > 0``), this stays the *original* total, so the
                loop runs only the remaining episodes rather than n_episodes more.
        episode_length : int
                Number of time steps in one episode.
        reset_frequency : int (default=1)
                After how many episodes is the simulation reset.
        load_bar : bool (default=True)
                If true, show a progress bar.
        save_episodic_data : bool (default=False)
                If true, save the episode data. If false, the data is of the
                last episode is overwritten by the new data. Make sure that the
                system runner supports episodic data saving. The get_engine function
                should take a system and a str(cycle_index) as arguments. The
                cycle_index is passed to the EsperessoMD engine as 'h5_group_tag'. See
                the implementation in the test_semi_episodic_data_writing function in
                CI/espresso_tests/integration_tests/test_rl_trainers.py
        episode_offset : int (default=0)
                Absolute episode number to resume from (e.g. parsed from a restored
                checkpoint's directory name). Episode numbering, checkpoint naming,
                and trajectory cycle numbering all become absolute so a resumed run
                continues the same sequence instead of restarting at 0 in a fresh
                directory.

        Notes
        -----
        If you are using semi-episodic training but your task kills the
        simulation, the system will be reset.
        """
        killed = False
        rewards = np.zeros(n_episodes)
        current_reward = 0.0
        force_fn = self.initialize_training()
        progress = Progress(
            "Episode: {task.fields[Episode]}",
            BarColumn(),
            "Episode reward: {task.fields[current_reward]} Running Reward:"
            " {task.fields[running_reward]}",
            TimeRemainingColumn(),
        )

        with progress:
            task = progress.add_task(
                "Episodic Training",
                total=n_episodes,
                completed=episode_offset,
                Episode=episode_offset + 1,
                current_reward=current_reward,
                running_reward=np.mean(rewards),
                visible=load_bar,
            )

            break_training = False
            stop_after_episode = -1
            for episode in range(episode_offset, n_episodes):
                # Check if the system should be reset.
                if episode % reset_frequency == 0 or killed:
                    if self.engine is not None:
                        self.engine.finalize()

                    logger.info(f"Resetting the system at episode {episode}")
                    self.engine = None
                    if save_episodic_data:
                        try:
                            # Derived from the absolute episode rather than a
                            # separately-tracked counter, so a resumed run
                            # continues cycle numbering instead of restarting
                            # at cycle_0 (reproduces the same sequence of
                            # values as a plain incrementing counter would,
                            # since resets only ever land on episode ==
                            # k * reset_frequency).
                            cycle_index = episode // reset_frequency
                            self.engine = get_engine(system, f"{cycle_index}")
                        except TypeError:
                            raise ValueError(
                                "The system runner does not support episodic data"
                                " saving. Your get_engine function should take a system"
                                " and a str(cycle_index) as arguments. The cycle_index"
                                " is passed to the EspressoMD engine as"
                                " 'h5_group_tag'."
                            )
                    else:
                        self.engine = get_engine(system)

                    # Initialize the tasks and observables.
                    for agent in self.agents.values():
                        agent.reset_agent(self.engine.colloids)

                self.engine.integrate(episode_length, force_fn)

                force_fn, current_reward, killed = self.update_rl()

                # rewards[0:episode_offset] is never filled in this process (a
                # resumed run allocates a fresh zero array but only starts
                # writing at episode_offset), so these windows are anchored
                # there instead of index 0 -- otherwise a resume would
                # silently dilute running/total reward with phantom zeros.
                display_episode = episode + 1
                local_filled = episode - episode_offset + 1
                if local_filled < 10:
                    running_reward = np.round(
                        np.mean(rewards[episode_offset:display_episode]), 2
                    )
                else:
                    running_reward = np.round(
                        np.mean(rewards[display_episode - 10 : display_episode]), 2
                    )

                rewards[episode] = current_reward
                self._log_episode_metrics(
                    episode=episode + 1,
                    current_reward=current_reward,
                    running_reward=running_reward,
                    total_reward=float(
                        np.mean(rewards[episode_offset:display_episode])
                    ),
                    killed=killed,
                )
                self.maybe_save_checkpoint(rewards, episode, current_reward)

                logger.debug(f"{episode=}")
                logger.debug(f"{current_reward=}")

                progress.update(
                    task,
                    advance=1,
                    Episode=episode + 1,
                    current_reward=np.round(current_reward, 2),
                    running_reward=running_reward,
                )

                if not break_training:
                    break_training, stop_after_episode = self.check_for_stop_criterion()

                if break_training:
                    if episode < stop_after_episode:
                        logger.info(
                            "Stopping criterion reached, but running out training"
                            f" until {stop_after_episode}"
                        )
                    else:
                        logger.info(
                            f"Stopping training after episode {stop_after_episode}"
                        )
                        break

        if self.engine is not None:
            self.engine.finalize()
        self.finalize_agents()
        return np.array(rewards)
