from typing import NamedTuple
from enum import Enum


class ParseRecurrenceError(Exception):
    """Something went wrong parsing a recurrence rule"""


class RecurrenceKind(Enum):
    LINEAR = 0
    LOG = 1


class Recurrence(NamedTuple):
    """
    Rule for how an event can recur throughout a simulation.

    The available rules are:

    - linear: at evenly spaced time intervals
    - log: at multiplicative time intervals; `next = time * (1 + delta)`
    """

    kind: RecurrenceKind = None
    interval: float = None

    @classmethod
    def from_str(cls, string=None):
        try:
            return Recurrence(kind=RecurrenceKind.LINEAR, interval=float(string))
        except ValueError:  # parse float failed
            pass

        try:
            kind, interval = string.split(":")
            if kind == "linear":
                return Recurrence(kind=RecurrenceKind.LINEAR, interval=float(interval))
            if kind == "log":
                return Recurrence(kind=RecurrenceKind.LOG, interval=float(interval))
        except ValueError:  # parse float failed or wrong number of arguments
            pass

        raise ParseRecurrenceError(f"badly formed recurrence rule {string}")

    def __str__(self):
        kind_str = str(self.kind).split(".")[1].lower()

        if self.kind == RecurrenceKind.LINEAR:
            return f"{kind_str} with interval {self.interval}"
        if self.kind == RecurrenceKind.LOG:
            return f"{kind_str} with multiplier {self.interval}"


class RecurringEvent(NamedTuple):
    """
    State of a recurring event that occurs during a simulation.

    Events signal side effects to a loop which monitors simulation progress,
    include writing checkpoints, collecting time series data, and other
    post-processing tasks.
    """

    last_time: float = None
    number: int = 0

    def next_time(self, time, recurrence):
        if recurrence.kind == RecurrenceKind.LINEAR:
            if self.number == 0:
                return time
            else:
                return self.last_time + recurrence.interval
        if recurrence.kind == RecurrenceKind.LOG:
            if self.number == 0:
                return time
            else:
                return self.last_time * (1.0 + recurrence.interval)

    def is_due(self, time, recurrence):
        next_time = self.next_time(time, recurrence)
        if next_time is None:
            return False
        else:
            return time >= next_time

    def next(self, time, recurrence):
        return self._replace(
            last_time=self.next_time(time, recurrence), number=self.number + 1
        )
