from collections import Counter
from datetime import datetime
from typing import Optional


class InspectionAggregator:
    def __init__(self, window_sec: int = 4):
        self.window_sec = window_sec
        self.current_window = None
        # count the number off each state
        self.state_counter = Counter()
        self.current_pid = None
        self.last_result = None

    # devide second
    def _window_floor(self, ts: datetime):
        sec = (ts.second // self.window_sec) * self.window_sec
        return ts.replace(
            second=sec,
            microsecond=0
        )
   
    # Reset if not in valid zone
    def reset(self):
        self.current_window = None
        self.state_counter.clear()
        self.current_pid = None

    def ingest(self, track):
        """
        return:
        None
        {
            "personID": 1,
            "state": "OK"
        }
        """

        ts = track["time"]
        if not isinstance(ts, datetime):
            ts = datetime.fromisoformat(str(ts))

        pid = track["personID"]
        state = str(track["state"])
        window = self._window_floor(ts)
        # take the second name

        # New peson appeared
        if (self.current_pid is not None and pid != self.current_pid):
            self.reset()
        self.current_pid = pid

        # First window?
        if self.current_window is None:
            self.current_window = window

        # If the next window
        if window != self.current_window:
            result = self._flush()
            self.current_window = window
            self.state_counter.clear()
            self.state_counter[state] += 1
            return result

        # if just old second, only count
        self.state_counter[state] += 1
        return None


    # flush the previous second?
    def _flush(self):
        if not self.state_counter:
            return None
        
        # counter is tuple => take [0][0]=state
        maj_state = self.state_counter.most_common(1)[0][0]
        result = {
            "personID": self.current_pid,
            "state": maj_state,
            "window_start": self.current_window,
            "frames": sum(self.state_counter.values()),
            "counter": dict(self.state_counter)
        }
        self.last_result = result
        return result