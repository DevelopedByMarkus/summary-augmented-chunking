from collections import defaultdict
from datetime import datetime, timedelta


class StatsTracker:
    """A singleton class to track statistics across the application."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all statistics to their initial state."""
        self.counters = defaultdict(int)
        self._timings = {}
        self._durations = {}

    def increment(self, key: str, value: int = 1):
        """Increments a counter."""
        self.counters[key] += value

    def set(self, key: str, value: int):
        """Sets a counter to a specific value."""
        self.counters[key] = value

    def start_timer(self, key: str):
        """Starts a timer for a given operation."""
        self._timings[key] = datetime.now()

    def stop_timer(self, key: str):
        """Stops a timer and records the duration."""
        if key in self._timings:
            self._durations[key] = datetime.now() - self._timings[key]
        else:
            self._durations[key] = timedelta(0)

    def get_duration(self, key: str) -> timedelta:
        """Gets the duration for a completed timer."""
        return self._durations.get(key, timedelta(0))

    def report(self) -> str:
        """Formats all collected statistics into a human-readable string."""
        report_lines = [
            "--- Run Statistics ---",
            "\n--- Document & Chunk Stats ---",
            f"Documents Processed:         {self.counters['documents_processed']}",
            f"Queries Processed:           {self.counters['queries_processed']}",
            f"Total Chunks Created:        {self.counters['chunks_created']}",

            "\n--- Caching Stats ---",
            f"Summaries from Cache:        {self.counters['summaries_from_cache']}",
            f"Summaries not from Cache:    {self.counters['summaries_not_from_cache']}",
            f"Embeddings from Cache:       {self.counters['embeddings_from_cache']}",
            f"Embeddings not from Cache:   {self.counters['embeddings_not_from_cache']}",

            "\n--- Summarization Quality ---",
            f"Summaries Truncated (Post-Retry): {self.counters['summaries_truncated_after_retries']}",
            f"Summaries Using Fallback:    {self.counters['summaries_incorrect_fallback']}",
            f"Documents Truncated (Too Long):   {self.counters['documents_truncated']}",

            "\n--- Performance Timings ---",
            f"Data Loading & Setup:        {str(self.get_duration('data_setup'))}",
            f"Chunking & Summarization:    {str(self.get_duration('chunking_and_summarization'))}",
            f"Embedding Generation:        {str(self.get_duration('embedding_generation'))}",
            f"Query Processing:            {str(self.get_duration('query_processing'))}",
            "----------------------------------",
            f"Overall Run Time:            {str(self.get_duration('overall_run'))}",
        ]
        return "\n".join(report_lines)


# Create the single, global instance of the tracker that will be imported elsewhere
stats_tracker = StatsTracker()
