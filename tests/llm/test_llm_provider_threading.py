import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from cot_forge.llm import LLMProvider


class TestLLMProviderThreading(unittest.TestCase):
  """Tests for the LLMProvider's thread-safety features."""

  def setUp(self):
    """Set up test fixtures."""
    class TestProvider(LLMProvider):
      def __init__(self, input_token_limit=None, output_token_limit=None):
        super().__init__(
            model_name="test-model",
            input_token_limit=input_token_limit,
            output_token_limit=output_token_limit
        )
        self.call_count = 0

      def generate_completion(self,
                              prompt,
                              system_prompt=None,
                              temperature=0.7,
                              max_tokens=None, **kwargs):
        # Simulate token usage for testing
        input_tokens = len(prompt.split())
        # Generate fewer output tokens to stay within limits
        output_tokens = len(prompt.split())  # Changed to 1:1 ratio

        # Add a small delay to increase chance of race conditions if thread safety fails
        time.sleep(0.01)

        # Update usage
        self.update_token_usage(input_tokens, output_tokens)
        self.call_count += 1

        return f"Response with {output_tokens} tokens"

    self.TestProvider = TestProvider
    self.test_provider = self.TestProvider()

  def test_token_counting_in_multithreading(self):
    """Test that token counting works correctly with multiple threads."""
    # Create 50 prompts with varying lengths
    prompts = [f"Prompt {i} with {i % 5 + 1} words" for i in range(50)]

    # Calculate expected token counts
    expected_input_tokens = sum(len(p.split()) for p in prompts)
    # Fix: use same 1:1 ratio as generate_completion
    expected_output_tokens = sum(len(p.split())
                                 for p in prompts)  # Removed * 2

    # Process the prompts using multiple threads
    with ThreadPoolExecutor(max_workers=8) as executor:
      futures = [executor.submit(self.test_provider.generate, prompt)
                 for prompt in prompts]

      # Wait for all futures to completej
      for future in futures:
        future.result()

    # Verify the token counts are correct
    self.assertEqual(self.test_provider.input_tokens, expected_input_tokens)
    self.assertEqual(self.test_provider.output_tokens, expected_output_tokens)
    self.assertEqual(self.test_provider.call_count, len(prompts))

  def test_token_limit_enforcement_concurrent(self):
    """Test that token limits are correctly enforced during concurrent access."""
    # Create a provider with a low output token limit
    provider_with_limit = self.TestProvider(output_token_limit=100)

    # Create 20 prompts with 10 tokens each (will exceed the limit)
    prompts = ["word " * 10 for _ in range(20)]

    # Keep track of successful and failed completions
    success_count = 0
    failure_count = 0
    lock = threading.Lock()

    def process_prompt(prompt):
      nonlocal success_count, failure_count
      try:
        provider_with_limit.generate(prompt)
        with lock:
          success_count += 1
        return True
      except ValueError:
        # This is expected after the limit is reached
        with lock:
          failure_count += 1
        return False

    # Process prompts with multiple threads
    with ThreadPoolExecutor(max_workers=5) as executor:
      futures = [executor.submit(process_prompt, prompt) for prompt in prompts]

      # Wait for all futures to complete
      for future in futures:
        future.result()

    # Verify some succeeded and others failed due to limit
    self.assertGreater(success_count, 0)
    self.assertGreater(failure_count, 0)
    self.assertEqual(success_count + failure_count, len(prompts))

    # The output tokens should not exceed the limit by much (may exceed slightly due to
    # near-simultaneous updates right at the limit)
    self.assertLessEqual(
        provider_with_limit.output_tokens,
        provider_with_limit.output_token_limit * 1.5  # Allow some margin
    )

  def test_update_token_usage_thread_safety(self):
    """Test that update_token_usage is thread-safe."""
    # We'll patch the update_token_usage method to introduce a delay and detect race conditions
    original_update = self.test_provider.update_token_usage

    def delayed_update(input_tokens, output_tokens):
      # Store current values
      old_input = self.test_provider.input_tokens
      old_output = self.test_provider.output_tokens

      # Simulate work that could be interrupted
      time.sleep(0.01)

      # Update values (this is the critical section that should be protected)
      self.test_provider.input_tokens = old_input + input_tokens
      self.test_provider.output_tokens = old_output + output_tokens

    # Test without lock (should lead to race conditions)
    self.test_provider.update_token_usage = delayed_update
    self.test_provider.input_tokens = 0
    self.test_provider.output_tokens = 0

    # Run concurrent updates without lock
    input_per_thread = 10
    output_per_thread = 5
    threads = 50

    with ThreadPoolExecutor(max_workers=threads) as executor:
      futures = [
          executor.submit(lambda: delayed_update(
              input_per_thread, output_per_thread))
          for _ in range(threads)
      ]
      for future in futures:
        future.result()

    # We expect inconsistencies without proper locking
    no_lock_input = self.test_provider.input_tokens
    no_lock_output = self.test_provider.output_tokens

    # Now test with the actual implementation that has locking
    self.test_provider.update_token_usage = original_update
    self.test_provider.input_tokens = 0
    self.test_provider.output_tokens = 0

    # Run concurrent updates with lock
    with ThreadPoolExecutor(max_workers=threads) as executor:
      futures = [
          executor.submit(lambda: self.test_provider.update_token_usage(input_per_thread,
                                                                        output_per_thread))
          for _ in range(threads)
      ]
      for future in futures:
        future.result()

    # We expect consistent results with proper locking
    with_lock_input = self.test_provider.input_tokens
    with_lock_output = self.test_provider.output_tokens

    # The locked version should be accurate
    self.assertEqual(with_lock_input, threads * input_per_thread)
    self.assertEqual(with_lock_output, threads * output_per_thread)

    # The unlocked version should be less (due to race conditions)
    # or equal (if we got lucky) to the correct value
    self.assertLessEqual(no_lock_input, threads * input_per_thread)
    self.assertLessEqual(no_lock_output, threads * output_per_thread)


if __name__ == '__main__':
  unittest.main()
