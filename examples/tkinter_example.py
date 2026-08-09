"""Tkinter consumer; the reusable package itself never imports Tkinter."""

import queue
import threading
import tkinter as tk

from voice_to_text import SpeechToText


class VoiceApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.speech = SpeechToText()
        self.results = queue.Queue()
        self.worker = None

        root.title("Voice To Text Example")
        self.status = tk.StringVar(value="Status: Idle")
        tk.Label(root, textvariable=self.status).pack(padx=12, pady=8)
        self.text = tk.Text(root, width=56, height=14)
        self.text.pack(padx=12, pady=8)
        actions = tk.Frame(root)
        actions.pack(pady=8)
        tk.Button(actions, text="Start Listening", command=self.start).pack(side=tk.LEFT, padx=4)
        tk.Button(actions, text="Stop Listening", command=self.stop).pack(side=tk.LEFT, padx=4)
        tk.Button(actions, text="Clear", command=lambda: self.text.delete("1.0", tk.END)).pack(
            side=tk.LEFT, padx=4)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.process_results)

    def start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        self.status.set("Status: Listening...")
        self.worker = threading.Thread(target=self.listen_worker, daemon=True)
        self.worker.start()

    def listen_worker(self) -> None:
        try:
            for result in self.speech.listen_continuously():
                self.results.put(result)
        except Exception as exc:
            self.results.put(exc)

    def process_results(self) -> None:
        try:
            while True:
                result = self.results.get_nowait()
                if isinstance(result, Exception):
                    self.status.set(f"Error: {result}")
                else:
                    self.text.insert(tk.END, result.text + "\n")
                    self.text.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self.process_results)

    def stop(self) -> None:
        self.speech.stop()
        self.status.set("Status: Stopped")

    def close(self) -> None:
        self.speech.close()
        self.root.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    VoiceApp(window)
    window.mainloop()
