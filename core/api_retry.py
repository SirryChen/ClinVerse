"""Retry transient inference requests without restarting a clinical episode."""

import logging
import time

from openai import APIConnectionError, APIStatusError

log = logging.getLogger(__name__)


def call_with_retry(call, *, max_attempts=5, **kwargs):
    """Retry the same request; authentication, bad input and code errors escape.

    These attempts are in addition to any retries configured on the SDK client.
    No clinical action or conversation update occurs inside this retry loop.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    for attempt in range(1, max_attempts + 1):
        try:
            return call(**kwargs)
        except (APIConnectionError, APIStatusError) as exc:
            transient = isinstance(exc, APIConnectionError) or (
                exc.status_code in {408, 409, 429} or exc.status_code >= 500
            )
            if not transient or attempt == max_attempts:
                raise
            delay = min(2 ** attempt, 30)
            log.warning(
                "Transient inference error %s; retrying the same request "
                "in %ss (attempt %s/%s)",
                type(exc).__name__, delay, attempt + 1, max_attempts,
            )
            time.sleep(delay)
