import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from copy import deepcopy

import torch
from torch import Tensor

from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.policies.utils import get_device_from_parameters


class PolicyRolloutWrapper:
    """Use this wrapper around a policy that you plan to roll out in an environment.

    This wrapper bridges the gap between the world of policies where we use the RL formalism, and real world
    environments where time may be a factor. It may still be used on synchronous simulation environments.

    The policy is designed to take in a sequence of s observations ending at some time-step n and return a
    sequence of h actions starting from that timestep n. Formally:

    a_{n:n+h-1} = π(o_{n-s+1:n})

    This wrapper manages observations in a cache according to their timestamps in order to give the policy the
    correct inputs for the desired action timestamps.

    When running in synchronous simulation environments, the logic reduces to simply:

    1. Run inference synchronously to get a sequence of actions. Return the first action.
    2. Keep returning actions from the previously generated sequence.
    3. When the sequence is depleted, go back to 1.

    When simulating real-time or running in the real world, the logic does something more like:

    1. Run inference synchronously to get a sequence of actions. Return the first action.
    2. Keep returning actions from the previously generated sequence.
    3. When the sequence has <= n_action_buffer actions in it, run asynchronous inference. In the meantime go
       back to 2.

    The implementation details are more involved and are documented inline. The main point though is that we
    can set n_action_buffer to be just large enough such that before we run out of actions, we already have a
    new sequence ready.
    """

    def __init__(self, policy: Policy, fps: float, n_action_buffer: int = 0):
        """
        Args:
            policy: The policy to wrap.
            fps: The observation/action clock frequency. It is assumed that these are the same, but it is
                possible for the phases of the clocks to be offset from one another.
            n_action_buffer: As soon as the action buffer has <= n_action_buffer actions left, start an
                inference run.
        """
        self.policy = policy
        self.period_ms = int(round(1000 * (1 / fps)))
        # We'll allow half a clock cycle of tolerance on timestamp retrieval.
        self.timestamp_tolerance_ms = int(round(1000 * (1 / fps / 2)))
        self.n_action_buffer = n_action_buffer

        # Set up async related logic.
        self._threadpool_executor = ThreadPoolExecutor(max_workers=1)
        self._thread_lock = threading.Lock()

        self.reset()

    def __del__(self):
        """TODO(now): This isn't really working. Runtime exits with an exception."""
        self._threadpool_executor.shutdown(wait=True, cancel_futures=True)

    def reset(self):
        """Reset observation and action caches.

        NOTE: Ensure that any access to these caches is within a thread-locked context as their state is
        managed in different threads.
        """
        with self._thread_lock:
            # Store a mapping from observation timestamp (the moment the observation was captured) to
            # observation batch.
            self._observation_cache: dict[float, dict[str, Tensor]] = {}
            # Store a mapping from action timestamp (the moment the policy intends for the action to be)
            # executed.
            self._action_cache: dict[float, Tensor] = {}

    def _invalidate_obsolete_observations(self):
        """TODO(now)"""

    def run_inference(
        self,
        observation_timestamp_ms: int,
        action_timestamp_ms: int,
        strict_observation_timestamps: bool = False,
    ):
        """
        Construct an observation sequence from the observation cache, use that as input for running inference,
        and update the action cache with the result.
        """
        # Stack relevant observations into a sequence.
        observation_timestamps_ms = torch.tensor(
            [action_timestamp_ms + i * self.period_ms for i in range(1 - self.policy.n_obs_steps, 1)]
        )
        with self._thread_lock:
            observation_cache_timestamps_ms = torch.tensor(sorted(self._observation_cache.keys()))
        dist = torch.cdist(
            observation_timestamps_ms.unsqueeze(-1).float(),
            observation_cache_timestamps_ms.unsqueeze(-1).float(),
            p=1,
        ).int()
        min_, argmin_ = dist.min(dim=1)
        if torch.any(min_ > self.timestamp_tolerance_ms):
            msg = "Couldn't find observations within the required timestamp tolerance."
            if strict_observation_timestamps:
                raise RuntimeError(msg)
            else:
                logging.warning(msg)
        with self._thread_lock:
            observation_sequence_batch = {
                k: torch.stack(
                    [
                        self._observation_cache[ts.item()][k]
                        for ts in observation_cache_timestamps_ms[argmin_]
                    ],
                    dim=1,
                )
                for k in next(iter(self._observation_cache.values()))
                if k.startswith("observation")
            }

            # Forget any observations we won't be needing any more.
            self._invalidate_obsolete_observations()  # TODO(now)

        # Run inference.
        device = get_device_from_parameters(self.policy)
        observation_sequence_batch = {
            key: observation_sequence_batch[key].to(device, non_blocking=True)
            for key in observation_sequence_batch
        }
        actions = self.policy.run_inference(observation_sequence_batch).cpu()  # (batch, seq, action_dim)

        # Update action cache.
        with self._thread_lock:
            self._action_cache.update(
                {
                    observation_timestamp_ms + i * self.period_ms: action
                    for i, action in enumerate(actions.transpose(1, 0))
                }
            )

    def _get_contiguous_action_sequence_from_cache(self, first_action_timestamp_ms: float) -> Tensor | None:
        with self._thread_lock:
            action_cache = deepcopy(self._action_cache)
        if len(action_cache) == 0:
            return None
        action_cache_timestamps_ms = torch.tensor(sorted(action_cache))
        action_timestamps_ms = (
            torch.arange(0, action_cache_timestamps_ms.max() + self.period_ms, self.period_ms)
            + first_action_timestamp_ms
        )
        dist = torch.cdist(
            action_timestamps_ms.unsqueeze(-1).float(),
            action_cache_timestamps_ms.unsqueeze(-1).float(),
            p=1,
        ).int()
        min_, argmin_ = dist.min(dim=1)
        if min_[0] > self.timestamp_tolerance_ms:
            return None
        # Get contiguous sequence of argmins_ starting from 0.
        where_jump = torch.where(argmin_.diff() != 1)[0]
        if len(where_jump) > 0:
            argmin_ = argmin_[: where_jump[0] + 1]
        # Retrieve and stack the actions.
        action_sequence = torch.stack(
            [action_cache[ts.item()] for ts in action_cache_timestamps_ms[argmin_]],
            dim=0,
        )
        return action_sequence

    def provide_observation_get_actions(
        self,
        observation_batch: dict[str, Tensor],
        observation_timestamp: float,
        first_action_timestamp: float,
        strict_observation_timestamps: bool = False,
        timeout: float | None = None,
    ) -> Tensor | None:
        """Provide an observation and get an action sequence back.

        This method does several things:
        1. Accepts an observation with a timestamp. This is added to the observation cache.
        2. Runs inference either synchronously or asynchronously.
        3. Returns a sequence of actions starting from the requested timestamp (from a cache which is
           populated by inference outputs).

        If `timeout` is not provided, inference is run synchronously. If `timeout` is provided, inference runs
        asynchronously, and the function aims to return either an action sequence or None within the timeout
        period. It is guaranteed that if the timeout is not honored, a RuntimeError is raised (TODO(now)).

        All time related arguments are to be provided in units of seconds, relative to an arbitrary reference
        point which is fixed throughout the duration of the rollout.

        Args:
            observation_batch: Mapping of observation type key to observation batch tensor for a single time
                step.
            observation_timestamp: The timestamp associated with observation_batch. It should be as faithful
                as possible to the time at which the observation was captured.
            first_action_timestamp: The timestamp of the first action in the requested action sequence.
            strict_observation_timestamps: Whether to raise a RuntimeError if there are no observations in the
                cache with the timestamps needed to construct the inference inputs (ie there are no
                observations within `self.timestamp_tolerance_ms`).
        Returns:
            A (sequence, batch, action_dime) tensor for a sequence of actions starting from the requested
            `first_action_timestamp` and spaced by `1/fps` or None if the `timeout` is reached and there is no
            first action available.
        """
        start = time.perf_counter()
        # Immediately convert timestamps to integer milliseconds (so that hashing them for the cache keys
        # isn't susceptible to floating point issues).
        observation_timestamp_ms = int(round(observation_timestamp * 1000))
        del observation_timestamp  # defensive against accidentally using the seconds version
        first_action_timestamp_ms = int(round(first_action_timestamp * 1000))
        del first_action_timestamp  # defensive against accidentally using the seconds version

        # Update observation cache.
        if not set(observation_batch).issubset(self.policy.input_keys):
            raise ValueError(
                f"Missing observation_keys: {set(self.policy.input_keys).difference(set(observation_batch))}"
            )
        with self._thread_lock:
            self._observation_cache[observation_timestamp_ms] = observation_batch

        ret = None  # placeholder for this function's return value

        # Try retrieving an action sequence from the cache starting from `first_action_timestamp` and spaced
        # by `1 / fps`. While doing so remove stale actions (those which are older and outside tolerance).
        with self._thread_lock:
            action_cache_timestamps_ms = torch.tensor(sorted(self._action_cache))
        if len(action_cache_timestamps_ms) > 0:
            diff = action_cache_timestamps_ms - first_action_timestamp_ms
            to_delete = torch.where(torch.bitwise_and(diff < 0, diff.abs() > self.timestamp_tolerance_ms))[0]
            for ix in to_delete:
                with self._thread_lock:
                    del self._action_cache[action_cache_timestamps_ms[ix.item()].item()]
            # If the first action is in the cache, construct the action sequence.
            if diff.abs().argmin() <= self.timestamp_tolerance_ms:
                ret = self._get_contiguous_action_sequence_from_cache(first_action_timestamp_ms)

        if first_action_timestamp_ms < observation_timestamp_ms:
            raise RuntimeError("No action could be found in the cache, and we can't generate a past action.")

        # We would like to run inference if we don't have many actions left in the cache.
        want_to_run_inference = ret is None or (ret is not None and ret.shape[0] - 1 <= self.n_action_buffer)
        # Return an action right away if we know we don't want to run inference.
        if not want_to_run_inference:
            print("CACHE")
            return ret

        # We can't run inference if a previous inference is already running.
        if hasattr(self, "_future") and self._future.running():
            # Try to give the previous inference a chance to finish (within the allowable time limit).
            try:
                # TODO(now): the 1e-3 needs explaining
                timeout_ = None if timeout is None else timeout - (time.perf_counter() - start) - 1e-3
                self._future.result(timeout=timeout_)
            except TimeoutError:
                if ret is None:
                    # TODO(now): Decide if error or return
                    # raise RuntimeError(
                    #     "Inference is too slow to keep up with the rollout! There were no cached actions "
                    #     "matching the requested timestamps and a previous inference run was still in progress."
                    # ) from e
                    logging.warning("Your inference is begining to fall behind your rollout loop!")
                    return None
                else:
                    logging.warning("Your inference is begining to fall behind your rollout loop!")
                    print("CACHE")
                    return ret

        print("INFERENCE")
        # Start the inference job.
        self._future = self._threadpool_executor.submit(
            self.run_inference,
            observation_timestamp_ms,
            first_action_timestamp_ms,
            strict_observation_timestamps,
        )

        # Attempt to wait for inference to complete, within the bounds of the `timeout` parameter.
        try:
            if timeout is None:
                self._future.result()
            else:
                elapsed = time.perf_counter() - start
                buffer = 1e-3  # ample buffer for the rest of the function to complete
                timeout_ = timeout - elapsed - buffer
                if timeout_ > 0:
                    self._future.result(timeout=timeout_)
        except TimeoutError:
            pass

        # Return the actions we had extracted from the cache before starting inference (only if inference
        # is still running, otherwise we can get actions from the fresher cache).
        if self._future.running() and ret is not None:
            return ret

        return self._get_contiguous_action_sequence_from_cache(first_action_timestamp_ms)
