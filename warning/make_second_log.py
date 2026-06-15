import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Any, Dict, List, Tuple, Optional
import tkinter as tk
import sys
import requests

def show_simple_warning(
    title="WARNING!",
    message="Alert.",
    duration_ms=5000,
):
    root = tk.Tk()
    root.overrideredirect(True)      # không viền
    root.attributes("-topmost", True)

    # Màu cảnh báo
    bg = "#FFF3CD"      # nền vàng nhạt
    border = "#FFA500"  # viền cam
    fg = "#000000"

    frame = tk.Frame(root, bg=border)
    frame.pack()

    inner = tk.Frame(frame, bg=bg)
    inner.pack(padx=2, pady=2, fill="both", expand=True)

    # Title
    tk.Label(
        inner, text=title,
        font=("Segoe UI", 16, "bold"),
        bg=bg, fg=fg
    ).pack(pady=(10, 4), padx=20, anchor="w")

    message = message + f"\nNote: This warning will disappear after {duration_ms/1000}s!"
    # Content
    tk.Label(
        inner,
        text=message,
        font=("Segoe UI", 13),
        bg=bg, fg=fg,
        wraplength=560,     # tăng chút để dễ đọc
        justify="left"
    ).pack(pady=(0, 14), padx=20, anchor="w")

    # >>> Quan trọng: dùng reqwidth/reqheight để lấy kích thước thực tế
    root.update_idletasks()
    w = max(600, inner.winfo_reqwidth() + 4)   # đặt tối thiểu 600px để khỏi quá bé
    h = inner.winfo_reqheight() + 4
    sw = root.winfo_screenwidth()
    x = (sw - w) // 2
    y = 30  # cách mép trên 30px

    root.geometry(f"{w}x{h}+{x}+{y}")

    # Auto close
    root.after(duration_ms, root.destroy)

    # Touch to alert to exit 
    inner.bind("<Button-1>", lambda e: root.destroy())
    frame.bind("<Button-1>", lambda e: root.destroy())

    try:
        root.mainloop()
    except Exception:
        sys.exit(0)


class Second_Aggregator:
    def __init__(
        self,
        report_dir: str = "summary/Second_log",
        tie_break: str = "top",
        encoding: str = "utf-8",
        tzinfo: Optional[timezone] = None,  # ví dụ: zoneinfo.ZoneInfo("Asia/Tokyo")
        ng_threshold_seconds: int = 10,
        do_alert: bool = False,
        alert: str = "Teams",
        url: str = "",
        save_log: bool = True
        #in_valid_zone: bool = True
    ):
        self.report_dir = Path(report_dir)
        self.tie_break = tie_break
        self.encoding = encoding
        self.tzinfo = tzinfo
        self.save_log = save_log
        #self.row: List[Dict[str, Any]] = [] 
        self.url = url
        #self.in_valid_zone = in_valid_zone
        self.min_y2 = 424
        self.max_y2 = 550

        # csv information
        #self.csv_header = ["start_time", "person_id", "frames_total", "state", "xyxy"]
        self.csv_header = [
            "start_time", "person_id", "frames_total",
            "state", "xyxy",
            "valid_ratio", "in_valid_zone"
        ]
        self.first_xyxy: Dict[int, Tuple[float,float,float,float]] = {}  # pid, xyxy
        self.state_counter = defaultdict(Counter)             # (pid, second): Counter(state)
        self.last_seen: Dict[Tuple[int, datetime], str] = {}  # ((pid, second), last state)
        self.last_second_of_person: Dict[int, datetime] = {}  # (pid, last second)
        self.latest_hits: Dict[int, int] = {}                 # (pid, last frs)

        # Save log igore person outside
        self.valid_counter = defaultdict(int)  # (pid, second) -> số frame valid
        self.total_counter = defaultdict(int)  # tổng frame (optional, nhưng tốt)
        self.full_dir = self.report_dir / "full"
        self.valid_dir = self.report_dir / "valid"

        # Write alarm
        self.do_alert = do_alert
        self.alert = alert
        self.ng_streak: Dict[int, int] = {}            # (pid, streak - number of NG case )
        self.ng_threshold_seconds = ng_threshold_seconds                  # K continuous seconds

    # Convert to your time
    def _to_tz(self, ts: datetime) -> datetime:
        if self.tzinfo is None:
            return ts
        if ts.tzinfo is None:
            return ts.replace(tzinfo=self.tzinfo)
        return ts.astimezone(self.tzinfo)


    # Remove second/microsecond of time
    @staticmethod
    def _second_floor(ts: datetime) -> datetime:
        return ts.replace(microsecond=0)


    # Read each line of outputs per-frame from the tracker:
    # outputs = tracker.update(frame_dict["time"], frame_dict["frame_id"], frame_dict)
    def ingest_rows(self, rows):
        flushed_rows = []
        for r in rows:
            ts = r["time"]
            pid = int(r["personID"])
            hits = int(r["frames_total"])
            state = str(r["state"])
            xyxy = r["xyxy"]
            y2 = xyxy[3]
            in_valid = (self.min_y2 <= y2 <= self.max_y2)


            # 0) Standardize timestamps
            if not isinstance(ts, datetime):
                ts = datetime.fromisoformat(str(ts)) # str => datetime
            #ts = self._to_tz(ts)
            second = self._second_floor(ts)
            
            # 1) Update first bbox/hits for track
            if pid not in self.first_xyxy:
                self.first_xyxy[pid] = xyxy

            # 2) If start a new second -> flush the old second.
            prev_second = self.last_second_of_person.get(pid) # return previous second of pid
            if prev_second is not None and prev_second != second: 
                row = self._flush(pid, prev_second)
                if row and str(row["state"]).startswith("NG"):
                    flushed_rows.append(row)
            
            # 3) Count state: {(pid, second): Counter({'OK': 2, 'NG_H': 1, 'NG_R': 1, 'NG_HR': 1})
            self.state_counter[(pid, second)][state] += 1
            self.last_seen[(pid, second)] = state
            self.last_second_of_person[pid] = second
            self.latest_hits[pid] = hits

            # ✅ update counter
            self.total_counter[(pid, second)] += 1
            if in_valid:
                self.valid_counter[(pid, second)] += 1

        return flushed_rows

    # flush the previous second.
    def _flush(self, pid: int, second: datetime):
        # take the values and delete this values
        cnt = self.state_counter.pop((pid, second), Counter())
        if not cnt:
            return

        # Majority vote window: 
        max_c = max(cnt.values())
        top_states = [s for s, c in cnt.items() if c == max_c]
        if len(top_states) == 1:
            maj_state = top_states[0]
        else:
            if self.tie_break == "top":
                maj_state = sorted(top_states)[-1]            
            else:  ## choose the last state if not
                maj_state = self.last_seen.get((pid, second), top_states[-1])
        self.last_seen.pop((pid, second), None) # release the variable

        # Take the information of trackID for second log
        frames_total = self.latest_hits.get(pid, 0) # if can not get, return 0
        main_xyxy = self.first_xyxy.get(pid, (None, None, None, None))
        
        # Calculate valid ratio
        valid_count = self.valid_counter.pop((pid, second), 0)
        total_count = self.total_counter.pop((pid, second), sum(cnt.values()))
        valid_ratio = valid_count / max(total_count, 1)
        in_valid_zone = valid_ratio >= 0.5   # threshold


        row = {
            "start_time": second,
            "person_id": pid,
            "frames_total": frames_total,
            "state": maj_state,
            "xyxy": main_xyxy,
            "valid_ratio": round(valid_ratio, 2),
            "in_valid_zone": in_valid_zone
        }

        payload = {
            "start_time": str(second),
            "person_id": pid,
            "frames_total": frames_total,
            "state": str(maj_state),
            "xyxy": main_xyxy
        }

        # save to file second log
        """if self.save_log:
            csv_path = self._build_path_for_second(second)
            self._write_row(csv_path, row)"""

        if self.save_log:
            # FULL LOG 
            full_path = self._build_path_for_second(second, subdir="full")
            self._write_row(full_path, row)

            # VALID LOG (chỉ ghi nếu hợp lệ)
            if in_valid_zone:
                valid_path = self._build_path_for_second(second, subdir="valid")
                self._write_row(valid_path, row)


        # Update streak by second
        is_ng = str(maj_state).startswith("NG") and in_valid_zone
        if is_ng:
            self.ng_streak[pid] = self.ng_streak.get(pid, 0) + 1
        else:
            self.ng_streak[pid] = 0

        # Alarm Sending Conditions
        streak = self.ng_streak[pid]
        handlers = {
            "teams": lambda: requests.post(self.url, json=payload),
            "terminal": lambda: threading.Thread(
                target=show_simple_warning,
                kwargs=dict(
                    message=f"PersonID{pid} violated the rules for {streak} continuous seconds: {maj_state}.",
                    duration_ms=5000
                ),
                daemon=True
            ).start(),
            "mail": lambda: print("Coming soon...")
        }
        if self.do_alert and in_valid_zone and streak % self.ng_threshold_seconds == 0:
            # Convert => correct format.
            handlers.get(self.alert.lower(), lambda: print("Invalid alert type. Please choose the type of alert to send."))()
        if in_valid_zone and streak % 20 == 0:
            teams_path = self._build_path_for_second(second, subdir="alert")
            self._write_row(teams_path, row)
        self.first_xyxy.pop(pid, None)
        self.latest_hits.pop(pid, None)

        # Check if NG   
        if is_ng:
            return row
        return None


    # Create the file path using the date name
    def _build_path_for_second(self, second: datetime, subdir: str="", **kwargs) -> Path:
        date_str = second.date().isoformat()  # YYYY-MM-DD
        if subdir:
            return self.report_dir / subdir / f"second_log_{date_str}.csv"
        else:
            return self.report_dir / f"second_log_{date_str}.csv"


    # write outputs to file csv
    def _write_row(self, csv_path: Path, row: Dict):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_empty = (not csv_path.exists()) or (csv_path.stat().st_size == 0)
        with open(csv_path, "a", newline="", encoding=self.encoding) as f:
            w = csv.DictWriter(f, fieldnames=self.csv_header, lineterminator="\n")
            if file_empty:
                w.writeheader()
            w.writerow(row)


    # Flush the remaining seconds when pausing the program
    def shutdown(self):
        for (pid, second) in list(self.state_counter.keys()):
            self._flush(pid, second)

    
