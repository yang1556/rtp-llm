import bisect
from dataclasses import dataclass
from random import randint, random

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2


@dataclass
class EstimatorSample:
    decoded_tokens: int  # The number of tokens already decoded for this request.
    prefill_length: int  # The length of the prefill phase. In multi-turn conversations, this is the total concatenated length of all prefill phases.


class Estimator:
    def __init__(
        self, max_period: int = 32, period_size: int = 32, token_per_block: int = 64
    ):
        """
        Initializes the KV cache usage estimator.

        This estimator uses Monte Carlo sampling to estimate the probability of satisfying KV cache
        capacity limits over a future time horizon. It's designed to handle scenarios where
        calculating the exact probability is computationally infeasible (2^n complexity).
        The estimation error is O(1 / (sample_time * len(running_reqs))^0.5).

        Args:
            max_period (int): The maximum possible lifespan (in periods) for a request.
                              This defines the prediction horizon.
            period_size (int): The size of a single period in decode steps (tokens).
                               This discretizes the time axis to accelerate computation
                               and prevent overly sparse probability density functions.
                               For example, period_size=32 means one period contains 32 decode steps.
            token_per_block (int): The number of tokens that can be stored in a single KV cache block.
        """
        self._estimations: list[np.ndarray] = (
            []
        )  # Stores Monte Carlo samples of future block usage. Each element is an array representing block usage over 'max_period'.
        self._peroid_size: int = period_size  # Size of a period in tokens.
        self._max_period: int = (
            max_period  # Maximum number of periods to consider for prediction.
        )
        # Initial probability mass. A smaller value allows for faster initial fitting of the
        # probability distribution but can also lead to more fluctuations.
        self._init_mass: int = 32
        self.init_pmf()  # Initializes the probability mass function (PMF).

        self._update_interval: int = (
            300  # How often (in terms of completed requests) to decay the PMF.
        )
        self._decay: float = 0.9  # Decay factor applied to the PMF during updates.
        self._counter: int = 0  # Counter for tracking update intervals.

        # Dynamic programming cache for CDF calculations. Stores pre-computed CDFs for different ages.
        self._cdf_dp: list[np.ndarray] = [None for _ in range(self._max_period)]
        self._sample_times: int = (
            100  # Number of Monte Carlo samples to generate during 'build'.
        )
        self._token_per_block = token_per_block  # Tokens per KV cache block.
        # Pre-computed array of cumulative period sizes, used to calculate total tokens over time.
        self._ONE: np.ndarray = np.cumsum(
            np.ones(shape=[self._max_period], dtype=np.float32) * self._peroid_size
        )

        self._discount_factor: float = 1  # kv cache discount.
        # self._p_discount_factor: float = 0.0 # prefill kv cache usage discount factor
        # self._d_discount_factor: float = 0.0 # decode discount factor

    def _refresh(self) -> None:
        """
        Clears all cached information, including CDF dynamic programming cache and estimations.
        This method should be called before rebuilding estimations or re-initializing the PMF.

        Returns:
            None
        """
        self._cdf_dp = [None for _ in range(self._max_period)]
        self._estimations.clear()

    def _predict_future_period(self, num_of_decoded_tokens: int) -> int:
        """
        Predicts the future lifespan (in periods) of a request based on its current age
        and the learned probability distribution of request lifespans.

        Args:
            num_of_decoded_tokens (int): The number of tokens already decoded for the request
                                         (its current "age").

        Returns:
            int: The predicted future lifespan of the request in periods. This value will be
                 between 0 and `_max_period`.

        Note:
            Uses `bisect_left` on the CDF to simulate sampling from the lifespan distribution.
        """
        _age = round(num_of_decoded_tokens / self._peroid_size)
        if _age >= self._max_period:
            # If the request is already older than or equal to the max period, it's considered to
            # have a future lifespan of 1 period (or clamped to the max period for safety).
            return min(
                1, self._max_period - 1
            )  # Clamped to max_period - 1 to avoid index out of bounds in _pmf_to_cdf for _pmf[age:]
        else:
            _cdf = self._pmf_to_cdf(age=_age)
            # Sample a random value and find where it would be inserted in the CDF to get a predicted period.
            # bisect_left returns an index, which corresponds to the predicted period.
            predicted_period = bisect.bisect_left(_cdf, random())
            return min(
                predicted_period, self._max_period - 1
            )  # Ensure the predicted period does not exceed max_period - 1

    def _pmf_to_cdf(self, age: int) -> np.ndarray:
        """
        Converts the probability mass function (PMF) to a cumulative distribution function (CDF)
        for a given request age. The PMF is truncated to consider only future periods
        relative to the given `age`, effectively calculating a posterior probability.

        Args:
            age (int): The current age of the request in periods. This is used to
                       truncate the PMF, considering only remaining lifespan.

        Returns:
            np.ndarray: A NumPy array representing the normalized cumulative distribution function
                        for the remaining lifespan of a request of the given age.

        Raises:
            ValueError: If `age` is out of bounds for the `_pmf` array. (Implicitly handled by array slicing).
        """
        if self._cdf_dp[age] is not None:
            return self._cdf_dp[age]

        # Truncate the PMF to consider only periods from the current 'age' onwards.
        _pmf_truncated = self._pmf[age:]
        if np.sum(_pmf_truncated) == 0:
            # If the truncated PMF has no mass, it means there's no observed data for this age group
            # or beyond. Return a zero probability array, indicating no further lifespan.
            return np.zeros_like(_pmf_truncated)

        normalized_pmf = _pmf_truncated / np.sum(_pmf_truncated)
        self._cdf_dp[age] = np.cumsum(normalized_pmf)
        return self._cdf_dp[age]

    def init_pmf(self) -> None:
        """
        Initializes the probability mass function (PMF) for request lifespans.
        The PMF is initially based on a Chi-squared distribution to provide a reasonable starting shape.

        Returns:
            None
        """
        self._refresh()  # Clear any existing estimations and CDF cache.
        dist = chi2(
            df=256
        )  # Initialize with a Chi-squared distribution. The 'df' (degrees of freedom) can be tuned.
        self._pmf: np.ndarray = np.array(
            [dist.pdf(_ * self._peroid_size) for _ in range(self._max_period)]
        )
        # Normalize the initial PMF so it sums to 1 (or close to 1, as it's a PDF scaled by period_size).
        # This normalization ensures that it represents probabilities.
        if np.sum(self._pmf) > 0:
            self._pmf /= np.sum(self._pmf)
        self._pmf *= self._init_mass

    def update(self, response_length: int) -> None:
        """
        Updates the probability mass function (PMF) when a request completes.
        This method records the actual lifespan of completed requests, increasing
        the probability mass for the observed period, and periodically decays the PMF.

        Args:
            response_length (int): The total length (in tokens) of the completed request.

        Returns:
            None
        """
        self._refresh()  # Clear caches as the PMF has changed.

        selected_period = round(response_length / self._peroid_size)
        if selected_period >= self._max_period:
            # If the response length exceeds our max prediction period, we don't update
            # its specific period, as it's outside our considered range.
            # Optionally, one might update the last bin `_pmf[self._max_period - 1]` to
            # account for very long requests, but the current implementation ignores them.
            return
        self._pmf[selected_period] += 1  # Increment the count for the observed period.

        self._counter += 1
        if self._counter >= self._update_interval:
            # Periodically decay the PMF to give more weight to recent observations
            # and prevent old data from dominating.
            self._pmf *= self._decay
            self._counter = 0

    def _estimate_occupied_blocks(
        self, req: EstimatorSample, future_period: int
    ) -> np.ndarray:
        """
        Estimates the number of KV cache blocks occupied by a single request
        over the entire prediction horizon, given its prefill length, decoded tokens,
        and predicted future lifespan.

        Args:
            req (RequestLengthInfo): An object containing information about the request's
                                     `decoded_tokens` and `prefill_length`.
            future_period (int): The predicted future lifespan (in periods) of this request.
                                 Blocks will only be considered occupied up to this future period.

        Returns:
            np.ndarray: A NumPy array of shape `(self._max_period,)` where each element
                        represents the estimated number of blocks occupied by this specific
                        request at that corresponding period in the future. Periods beyond
                        `future_period` will have 0 occupied blocks.
        """
        # Calculate total tokens for each period: prefill + already decoded + future decoded.
        # _ONE already accounts for future decoded tokens by period.
        _tokens = self._ONE + req.prefill_length + req.decoded_tokens
        # Convert tokens to blocks, rounding up since partial blocks still occupy a full block.
        _blocks = _tokens / self._token_per_block * self._discount_factor

        # Zero out blocks for periods beyond the predicted future lifespan.
        for i in range(self._max_period):
            if i > future_period:
                _blocks[i] = 0
            else:
                _blocks[i] += (i * self._peroid_size) / self._token_per_block
        return _blocks

    def accept(self, limit: int, length: int) -> float:
        """
        Evaluates the "fail rate" if a new request is accepted, given the current
        estimations of resource usage by running requests.

        Args:
            limit (int): The maximum allowed KV cache capacity in blocks.
            length (int): The new request being considered, with its `decoded_tokens` length

        Returns:
            float: The estimated probability (fail rate) that accepting this new request
                   will cause the total KV cache usage to exceed the `limit` at any point
                   in the future, based on the Monte Carlo samples.

        Raises:
            Exception: If `build` has not been called yet and `_estimations` is empty.
        """
        if len(self._estimations) == 0:
            raise Exception(
                "Estimator not built yet. Please call `build()` method first with `running_reqs`."
            )

        req = EstimatorSample(decoded_tokens=0, prefill_length=length)
        fail_tries: int = 0
        # Iterate through each Monte Carlo sample of future resource usage.
        for estimation in self._estimations:
            # For each sample, add the estimated blocks of the new request.
            # The new request's future period is predicted based on its current state.
            estimation_with_new_req = estimation + self._estimate_occupied_blocks(
                req, self._predict_future_period(req.decoded_tokens)
            )
            # Check if this combined usage exceeds the limit at any future period.
            if np.any(estimation_with_new_req > limit):
                fail_tries += 1
        return fail_tries / len(
            self._estimations
        )  # Calculate the proportion of failed samples.

    def build(self, running_reqs: list[EstimatorSample], used_blocks: int) -> None:
        """
        Builds the Monte Carlo estimations of future KV cache block usage.
        This method simulates the future lifespan of each currently running request
        for `_sample_times` iterations, accumulating the projected block usage for
        each period.

        Args:
            running_reqs (list[RequestLengthInfo]): A list of `RequestLengthInfo` objects
                                                    representing all currently active requests.
            used_blocks (int): kv block usage, this parameter is used for estimate discount factor.

        Returns:
            None
        """
        self._refresh()  # Clear previous estimations before building new ones.

        # estimate discount factor:
        total_prefill_length, total_decode_length = 0, 0
        for req in running_reqs:
            total_prefill_length += req.prefill_length
            total_decode_length += req.decoded_tokens

        for _ in range(self._sample_times):
            future_block_usage: np.ndarray = np.zeros(shape=[self._max_period])
            for r in running_reqs:
                # For each running request, predict its future lifespan.
                future_period_expect: int = self._predict_future_period(
                    r.decoded_tokens
                )
                # Add the estimated blocks for this request to the current sample's total usage.
                future_block_usage += self._estimate_occupied_blocks(
                    r, future_period_expect
                )
            self._estimations.append(future_block_usage)

    def plot_estimations(self, fname: str) -> None:
        """
        Plots the estimated future resource occupation and the normalized PMF.
        This function visualizes:
        - On the left: The normalized probability mass function (PMF) of request lifespans.
        - On the right: The mean, 75th, 90th, and 99th percentile curves of the
          `_estimations` (Monte Carlo samples) for future KV cache block usage.
        The resulting plot is saved as '1.png'.

        Returns:
            None
        """
        if not self._estimations:
            print(
                "No estimation data available to plot. Please run the `build` method first."
            )
            return

        fig, ax = plt.subplots(1, 2, figsize=(18, 7))  # Create a figure with 2 subplots

        # --- Left Subplot: Normalized PMF ---
        periods = np.arange(self._max_period)
        normalized_pmf = self._pmf / np.sum(
            self._pmf
        )  # Ensure normalization for plotting
        ax[0].bar(periods, normalized_pmf, color="skyblue", width=0.8)
        ax[0].set_title("Normalized PMF of Request Lifespans", fontsize=16)
        ax[0].set_xlabel("Period", fontsize=12)
        ax[0].set_ylabel("Probability", fontsize=12)
        ax[0].grid(True, linestyle="--", alpha=0.7)
        ax[0].set_xticks(periods[::4])  # Show fewer x-ticks for clarity

        # --- Right Subplot: Resource Allocation Estimation ---
        # Convert list of arrays to a 2D NumPy array for easier percentile calculations.
        estimations_np = np.array(self._estimations)

        # Calculate statistics across all samples for each period (axis=0).
        mean_estimation = np.mean(estimations_np, axis=0)
        percentile_75 = np.percentile(estimations_np, 75, axis=0)
        percentile_90 = np.percentile(estimations_np, 90, axis=0)
        percentile_99 = np.percentile(estimations_np, 99, axis=0)

        ax[1].plot(periods, mean_estimation, label="Mean", color="blue")
        ax[1].plot(periods, percentile_75, label="75%", linestyle="--", color="green")
        ax[1].plot(periods, percentile_90, label="90%", linestyle="-.", color="orange")
        ax[1].plot(periods, percentile_99, label="99%", linestyle=":", color="red")

        ax[1].set_title("Estimated Future KV Cache Block Usage", fontsize=16)
        ax[1].set_xlabel("Period", fontsize=12)
        ax[1].set_ylabel("Estimated Occupied Blocks", fontsize=12)
        ax[1].legend(fontsize=10)
        ax[1].grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()  # Adjust layout to prevent labels from overlapping.
        plt.savefig(f"{fname}.png")  # Save the plot.
        plt.clf()  # Clear the current figure to prevent plots from overlapping on subsequent calls.

    def update_discount_factor(self, prefill_lengths, decode_lengths, block_usages):
        if block_usages > 0:
            _factor = block_usages / (
                (prefill_lengths + decode_lengths) / self._token_per_block
            )
            self._discount_factor = self._discount_factor * 0.99 + 0.01 * _factor


if __name__ == "__main__":
    print("Initializing Estimator...")
    # Initialize the estimator
    estimator = Estimator(max_period=32, period_size=16, token_per_block=64)

    # --- Test Case 1: Initial State and PMF Update ---
    print("\n--- Test Case 1: Initial State and PMF Update ---")
    print(f"Initial PMF sum: {np.sum(estimator._pmf):.2f}")

    # Simulate some requests completing with varying lengths
    print("Updating PMF with simulated request completions...")
    for i in range(500):
        # Simulate request lengths, e.g., biased towards shorter lengths initially
        if random() < 1.7:
            length = np.random.randint(100, 500)  # Shorter requests
        else:
            length = np.random.randint(500, 1500)  # Longer requests
        estimator.update(length)

    # --- Test Case 2: Building Estimations with Running Requests ---
    print("\n--- Test Case 2: Building Estimations with Running Requests ---")
    running_requests: list[EstimatorSample] = []
    # Add some dummy running requests
    for i in range(250):
        d = randint(0, 250)
        p = randint(100, 200)
        running_requests.append(EstimatorSample(decoded_tokens=d, prefill_length=p))

    print(f"Building estimations with {len(running_requests)} running requests...")
    estimator.build(running_requests)
    print(f"Number of estimations built: {len(estimator._estimations)}")
    if estimator._estimations:
        print(f"Shape of first estimation: {estimator._estimations[0].shape}")
        print(
            f"Sample of first estimation (first 5 periods): {estimator._estimations[0][:5]}"
        )

    # --- Test Case 3: Accepting New Requests and Calculating Fail Rate ---
    print("\n--- Test Case 3: Accepting New Requests and Calculating Fail Rate ---")
    capacity_limit_blocks = 2540  # Example capacity limit in blocks

    # Define some waiting requests
    waiting_requests_scenario_1: list[EstimatorSample] = EstimatorSample(
        decoded_tokens=0, prefill_length=2000
    )
    waiting_requests_scenario_2: list[EstimatorSample] = EstimatorSample(
        decoded_tokens=30, prefill_length=1000
    )

    print(estimator.accept(limit=250, length=2000))
    print(estimator.accept(limit=220, length=1000))
    estimator.plot_estimations("1")

    print("\nSimulation complete.")
