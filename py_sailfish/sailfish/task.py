from typing import NamedTuple
from enum import Enum


class ParseRecurrenceError(Exception):
    """Something went wrong parsing a recurrence rule"""


class RecurrenceKind(Enum):
    NEVER = 0
    ONCE = 1
    TWICE = 2
    LINEAR = 3
    LOG = 4


class Recurrence(NamedTuple):
    """
    Rule for how a task can recur throughout a simulation.

    The avilable rules are:

    - never: the task must never happen
    - once: it happens once, at the start of the simulation
    - twice: once at the start, and again at the end
    - linear: at evenly spaced time intervals
    - log: at multiplicative time intervals; `next = time * (1 + delta)`
    """

    kind: RecurrenceKind = RecurrenceKind.NEVER
    interval: float = None

    @classmethod
    def from_str(cls, string=None):
        if string is None or string == "never":
            return Recurrence()
        if string == "once":
            return Recurrence(kind=RecurrenceKind.ONCE)
        if string == "twice":
            return Recurrence(kind=RecurrenceKind.TWICE)

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

        if self.kind == RecurrenceKind.NEVER:
            return kind_str
        if self.kind == RecurrenceKind.ONCE:
            return kind_str
        if self.kind == RecurrenceKind.TWICE:
            return kind_str
        if self.kind == RecurrenceKind.LINEAR:
            return f"{kind_str} with interval {self.interval}"
        if self.kind == RecurrenceKind.LOG:
            return f"{kind_str} with multiplier {self.interval}"


class RecurringTask(NamedTuple):
    """
    State of a recurring side effect that occurs during a simulation.

    Side effects include writing checkpoints, collecting time series data, and
    other post-processing tasks.
    """

    name: str
    last_time: float = None
    number: int = 0

    def next_time(self, time, recurrence):
        if recurrence.kind == RecurrenceKind.NEVER:
            return None
        if recurrence.kind == RecurrenceKind.ONCE:
            if self.number == 0:
                return time
            return None
        if recurrence.kind == RecurrenceKind.TWICE:
            if self.number == 0:
                return time
            if self.number == 1:
                return float("inf")
            return None
        if recurrence.kind == RecurrenceKind.LINEAR:
            if self.number == 0:
                return time
            else:
                return self.last_time + recurrence.interval
        if recurrence.kind == RecurrenceKind.LOG:
            if self.number == 0:
                return time
            else:
                return self.last_time * (1.0 + self.recurrence.interval)

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
