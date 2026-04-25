import subprocess
import sys
import threading
from pathlib import Path
from tkinter import Tk, StringVar, Text, END, filedialog, ttk, BooleanVar, IntVar, Scrollbar, RIGHT, Y


class TrainerGUI:
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("ML Trainer (Cardio)")

        self.csv_path = StringVar(value="")
        self.delimiter = StringVar(value=";")
        self.target = StringVar(value="cardio")
        self.drop_cols = StringVar(value="id")
        self.model_type = StringVar(value="xgboost")
        self.tune = BooleanVar(value=False)
        self.tune_iter = IntVar(value=30)

        self._build_ui()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, text="CSV File").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.csv_path, width=60).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(frame, text="Browse", command=self._browse).grid(row=0, column=2, sticky="e")

        ttk.Label(frame, text="Delimiter").grid(row=1, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.delimiter, width=6).grid(row=1, column=1, sticky="w", padx=6)

        ttk.Label(frame, text="Target").grid(row=2, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.target, width=16).grid(row=2, column=1, sticky="w", padx=6)

        ttk.Label(frame, text="Drop Columns").grid(row=3, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.drop_cols, width=40).grid(row=3, column=1, sticky="w", padx=6)

        ttk.Label(frame, text="Model").grid(row=4, column=0, sticky="w")
        model_box = ttk.Combobox(frame, textvariable=self.model_type, values=["xgboost"], state="readonly")
        model_box.grid(row=4, column=1, sticky="w", padx=6)

        ttk.Checkbutton(frame, text="Tune", variable=self.tune).grid(row=5, column=0, sticky="w")
        ttk.Label(frame, text="Tune Iter").grid(row=5, column=1, sticky="w")
        ttk.Entry(frame, textvariable=self.tune_iter, width=8).grid(row=5, column=1, sticky="e", padx=6)

        ttk.Button(frame, text="Run Training", command=self._run_training).grid(row=6, column=0, sticky="w", pady=8)

        self.output = Text(frame, height=18, width=80)
        self.output.grid(row=7, column=0, columnspan=3, sticky="nsew")
        scroll = Scrollbar(frame, command=self.output.yview)
        scroll.grid(row=7, column=3, sticky="ns")
        self.output["yscrollcommand"] = scroll.set

        frame.columnconfigure(1, weight=1)

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.csv_path.set(path)

    def _append(self, text: str) -> None:
        self.output.insert(END, text)
        self.output.see(END)

    def _run_training(self) -> None:
        csv_path = self.csv_path.get().strip()
        if not csv_path:
            self._append("Please select a CSV file.\n")
            return

        cmd = [
            sys.executable,
            str(Path("ml_system") / "train.py"),
            csv_path,
            "--delimiter",
            self.delimiter.get(),
            "--target",
            self.target.get(),
            "--drop-cols",
            self.drop_cols.get(),
        ]
        if self.tune.get():
            cmd += ["--tune", "--tune-iter", str(self.tune_iter.get())]

        self._append(f"Running: {' '.join(cmd)}\n")

        def worker() -> None:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                self._append(line)
            proc.wait()
            self._append(f"\nFinished (exit code {proc.returncode}).\n")

        threading.Thread(target=worker, daemon=True).start()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    TrainerGUI().run()
