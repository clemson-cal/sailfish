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
    name: str
    recurrence: Recurrence = None
    last_time: float = None
    number: int = 0

    def next_time(self, time):
        if self.recurrence.kind == RecurrenceKind.NEVER:
            return None
        if self.recurrence.kind == RecurrenceKind.ONCE:
            if self.number == 0:
                return time
            return None
        if self.recurrence.kind == RecurrenceKind.TWICE:
            if self.number == 0:
                return time
            if self.number == 1:
                return float("inf")
            return None
        if self.recurrence.kind == RecurrenceKind.LINEAR:
            if self.number == 0:
                return time
            else:
                return self.last_time + self.recurrence.interval
        if self.recurrence.kind == RecurrenceKind.LOG:
            if self.number == 0:
                return time
            else:
                return self.last_time * (1.0 + self.recurrence.interval)

    def is_due(self, time):
        next_time = self.next_time(time)
        if next_time is None:
            return False
        else:
            return time >= next_time

    def next(self, time):
        return RecurringTask(
            name=self.name,
            recurrence=self.recurrence,
            last_time=self.next_time(time),
            number=self.number + 1,
        )
