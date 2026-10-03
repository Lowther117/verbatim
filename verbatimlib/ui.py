"""The window: a queue of files on the left, options on the right, the transcript below."""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

from . import audio, export, models, paths, proc, theme

try:  # optional: native drag and drop. The window works without it.
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
    _HAVE_DND = True
except Exception:  # noqa: BLE001 - not installed, or no binary for this platform
    TkinterDnD = None  # type: ignore
    DND_FILES = "DND_Files"
    _HAVE_DND = False

MAC = sys.platform == "darwin"
MOD = "Command" if MAC else "Control"     # the modifier in key bindings
KEY = "Cmd" if MAC else "Ctrl"            # and how it is written in menus

DENOISE = [("auto", "Automatic - only when the recording is noisy"), ("on", "Always"), ("off", "Off")]
FILTER = [("balanced", "Balanced (recommended)"), ("strict", "Strict - leave out anything doubtful"),
          ("off", "Off - keep everything the model returns")]
SPEAKER_COUNTS = ["Work it out"] + [str(n) for n in range(1, 13)]
# One colour per speaker name, readable on the dark and on the light panel.
SPEAKER_DARK = ["#5b9cff", "#4ac26b", "#e0b44a", "#ff8a80", "#c792ea", "#4dd0e1", "#f78c6c", "#aed581",
                "#ff80ab", "#90a4ae"]
SPEAKER_LIGHT = ["#1f6feb", "#146c2e", "#8a5a00", "#b42318", "#7b3fb5", "#00707d", "#b3541e", "#4b7a12",
                 "#b0276f", "#44545c"]

HELP = f"""Verbatim turns recordings into text and labels who is speaking. Everything runs on this computer: \
no audio, text or file name is sent anywhere.

ADDING FILES
Add files, add a whole folder, or drop files on the window. Any audio or video format is fine. Press Start \
and the queue is worked through one file at a time; more files can be added while it runs. Each transcript is \
saved automatically to the save folder in every format ticked on the right.

SPEAKERS
With "Label who is speaking" on, each voice becomes Speaker 1, Speaker 2 and so on, in order of first speaking. \
If you know how many people are on the recording, say so: a fixed number is always more reliable than letting \
the program work it out. Afterwards, Rename speakers puts real names in, and giving two speakers the same name \
merges them. If one paragraph has gone to the wrong person, right-click it and pick who really said it.

KEEPING INVENTED TEXT OUT
Speech models can produce words that were never said when given silence, music or noise. Three things stop that. \
A speech detector means only stretches where someone is talking are ever transcribed. Noise reduction (automatic \
by default) cleans a noisy recording first, so that noise is not mistaken for speech or for an extra speaker. \
Then every piece of text is checked: known phantom phrases, looped phrases, and text the model itself was unsure \
of are left out. Nothing is hidden - "Removed" under the transcript lists everything that was left out and why. \
Strict leaves out more; Off keeps everything.

SPEED
"Test this computer" under Speed times the different ways of running on a minute of sample audio - one stage at \
a time, both stages at once, and the graphics processor where there is one - and keeps whichever is quickest \
here. It is worth doing once on each computer.

COMING BACK TO A TRANSCRIPT
Tick JSON under "Save each transcript as" and the saved .json file can be reopened later (File > Open saved \
transcript) to rename speakers or save other formats without transcribing again.

SHORTCUTS
{KEY}+O add files, {KEY}+Enter start, Esc stop, {KEY}+D light/dark, Delete remove from the queue."""


def open_path(path: str) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]  # noqa: S606
        elif MAC:
            subprocess.Popen(["open", path], stdin=subprocess.DEVNULL)
        else:
            subprocess.Popen(["xdg-open", path], stdin=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001
        pass


class Item:
    """One file in the queue."""
    _next = 1

    def __init__(self, path: str):
        self.id = f"q{Item._next}"
        Item._next += 1
        self.path = path
        self.state = "waiting"      # waiting | working | done | failed | stopped
        self.status = "Waiting"
        self.duration: Optional[float] = None
        self.result: Optional[dict] = None
        self.outputs: Dict[str, str] = {}
        self.folder = ""
        self.stem = ""
        self.error = ""
        self.fraction = 0.0


class App:
    def __init__(self, root: tk.Misc, files: Optional[List[str]] = None):
        self.root = root
        self.settings = paths.load_settings()
        self.dark = bool(self.settings.get("dark", True))
        self.c = theme.apply(root, self.dark)
        self.ui = theme.ui_family(root)
        self.mono = theme.mono_family(root)
        root.title(f"{paths.APP} - speech to text")
        root.geometry(self._sane_geometry(self.settings.get("geometry")))
        root.minsize(940, 620)

        self.items: Dict[str, Item] = {}
        self.order: List[str] = []
        self.running = False
        self.current: Optional[Item] = None
        self.proc: Optional[subprocess.Popen] = None
        self.jobdir = ""
        self.events_pos = 0
        self.started = 0.0
        self.poll_id: Optional[str] = None
        self.shown: Optional[str] = None
        self.inbox: "queue.Queue" = queue.Queue()   # callables posted by worker threads
        self.probe_q: "queue.Queue" = queue.Queue()
        self.downloading = False
        self.testing = False
        self.cancel_download = threading.Event()
        self._closing = False
        self._para_spans: List[dict] = []

        s = self.settings
        self.v_model = tk.StringVar()
        self.v_speakers = tk.BooleanVar(value=bool(s.get("speakers", True)))
        self.v_count = tk.StringVar(value=SPEAKER_COUNTS[0])
        self.v_denoise = tk.StringVar()
        self.v_filter = tk.StringVar()
        self.v_timestamps = tk.BooleanVar(value=bool(s.get("timestamps", True)))
        chosen = s.get("formats") or ["txt", "docx"]
        self.v_formats = {f: tk.BooleanVar(value=f in chosen) for f in export.FORMATS}
        self.model_id = s.get("model") if s.get("model") in models.ASR else models.DEFAULT_ASR
        self.denoise = s.get("denoise") if s.get("denoise") in dict(DENOISE) else "auto"
        self.filtering = s.get("filtering") if s.get("filtering") in dict(FILTER) else "balanced"

        self._build_menu()
        self._build()
        self._recolor()
        self._bind()
        self._refresh_models()
        self._refresh_save_label()
        self._refresh_speed_label()
        self._update_buttons()
        threading.Thread(target=self._probe_worker, daemon=True).start()
        threading.Thread(target=self._sweep_temp, daemon=True).start()
        self.root.after(120, self._pump)
        if files:
            self.add_paths(files)
        self.root.after(400, self._update_hint)

    def _sane_geometry(self, geometry) -> str:
        """The saved window position, unless it is off the screen (a second monitor since unplugged)."""
        default = "1180x800"
        if not isinstance(geometry, str):
            return default
        try:
            size, _, rest = geometry.partition("+")
            w, h = (int(v) for v in size.split("x"))
            x, y = (int(v) for v in rest.split("+")) if rest else (0, 0)
            sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            if w < 600 or h < 400:
                return default
            if x < -50 or y < -50 or x > sw - 100 or y > sh - 100:
                return f"{min(w, sw)}x{min(h, sh)}"
            return geometry
        except Exception:  # noqa: BLE001
            return default

    def _sweep_temp(self):
        """Remove job folders left in the temp folder by a crash or a forced quit."""
        try:
            base = tempfile.gettempdir()
            cutoff = time.time() - 12 * 3600
            for name in os.listdir(base):
                if name.startswith(("verbatim-job-", "verbatim-speedtest-")):
                    p = os.path.join(base, name)
                    if os.path.isdir(p) and os.path.getmtime(p) < cutoff:
                        shutil.rmtree(p, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass

    # --- construction ---------------------------------------------------------

    def _build_menu(self):
        m = tk.Menu(self.root)
        f = tk.Menu(m, tearoff=0)
        f.add_command(label="Add files…", accelerator=f"{KEY}+O", command=self.add_files)
        f.add_command(label="Add folder…", command=self.add_folder)
        f.add_command(label="Open saved transcript (.json)…", command=self.open_transcript)
        f.add_separator()
        f.add_command(label="Default save folder…", command=self.choose_save_dir)
        f.add_command(label="Reset save folder to Downloads", command=self.reset_save_dir)
        f.add_command(label="Open save folder", command=lambda: self._open_folder(self.save_dir()))
        if not MAC:  # a Mac has Quit in the application menu already
            f.add_separator()
            f.add_command(label="Quit", accelerator=f"{KEY}+Q", command=self.on_close)
        m.add_cascade(label="File", menu=f)
        q = tk.Menu(m, tearoff=0)
        q.add_command(label="Start", accelerator=f"{KEY}+Enter", command=self.start)
        q.add_command(label="Stop", accelerator="Esc", command=self.stop)
        q.add_separator()
        q.add_command(label="Run selected again", command=self.rerun_selected)
        q.add_command(label="Remove selected", accelerator="Delete", command=self.remove_selected)
        q.add_command(label="Clear finished", command=self.clear_finished)
        m.add_cascade(label="Queue", menu=q)
        mo = tk.Menu(m, tearoff=0)
        mo.add_command(label="Download and manage models…", command=self.models_dialog)
        mo.add_command(label="Open models folder", command=lambda: self._open_folder(paths.models_dirs()[0]))
        mo.add_separator()
        mo.add_command(label="Test this computer's speed…", command=self.speed_test)
        m.add_cascade(label="Models", menu=mo)
        v = tk.Menu(m, tearoff=0)
        v.add_command(label="Toggle light / dark", accelerator=f"{KEY}+D", command=self.toggle_theme)
        m.add_cascade(label="View", menu=v)
        h = tk.Menu(m, tearoff=0)
        h.add_command(label="How it works", command=self.show_help)
        h.add_command(label=f"About {paths.APP}", command=self.show_about)
        m.add_cascade(label="Help", menu=h)
        self.root.configure(menu=m)
        self.context = tk.Menu(self.root, tearoff=0)
        self.menus = [m, f, q, mo, v, h, self.context]

    def _build(self):
        root = self.root
        top = ttk.Frame(root, padding=(12, 10, 12, 6))
        top.pack(fill="x")
        self.top = top
        ttk.Label(top, text=paths.APP, style="Title.TLabel").pack(side="left")
        ttk.Label(top, text="   audio and video to text, with speakers - all on this computer",
                  style="Sub.TLabel").pack(side="left", pady=(5, 0))
        self.btn_theme = ttk.Button(top, text="Light", width=7, command=self.toggle_theme)
        self.btn_theme.pack(side="right")

        self.hint = ttk.Frame(root, style="Hint.TFrame")
        self.hint_label = ttk.Label(self.hint, text="", style="Hint.TLabel")
        self.hint_label.pack(side="left", fill="x", expand=True)
        self.hint_button = ttk.Button(self.hint, text="Download", style="Hint.TButton", command=self._hint_action)
        self.hint_button.pack(side="right", padx=8, pady=4)

        bar = ttk.Frame(root, padding=(12, 4, 12, 6))
        bar.pack(fill="x")
        ttk.Button(bar, text="Add files…", command=self.add_files).pack(side="left")
        ttk.Button(bar, text="Add folder…", command=self.add_folder).pack(side="left", padx=(6, 0))
        self.btn_remove = ttk.Button(bar, text="Remove", command=self.remove_selected)
        self.btn_remove.pack(side="left", padx=(6, 0))
        self.btn_clear = ttk.Button(bar, text="Clear finished", command=self.clear_finished)
        self.btn_clear.pack(side="left", padx=(6, 0))
        self.btn_stop = ttk.Button(bar, text="Stop", command=self.stop)
        self.btn_stop.pack(side="right")
        self.btn_start = ttk.Button(bar, text="Start", style="Accent.TButton", command=self.start)
        self.btn_start.pack(side="right", padx=(0, 6))

        status = ttk.Frame(root, padding=(12, 4, 12, 8))
        status.pack(side="bottom", fill="x")
        self.progress = ttk.Progressbar(status, mode="determinate", maximum=1000, length=220)
        self.progress.pack(side="right")
        self.status = ttk.Label(status, text="Add recordings, then press Start.", style="Dim.TLabel")
        self.status.pack(side="left", fill="x", expand=True)

        body = ttk.Frame(root, padding=(12, 0, 12, 0))
        body.pack(fill="both", expand=True)
        side = ttk.Frame(body, width=350)
        side.pack(side="right", fill="y", padx=(12, 0))
        side.pack_propagate(False)
        self._build_options(side)

        split = ttk.Panedwindow(body, orient="vertical")
        split.pack(side="left", fill="both", expand=True)

        qframe = ttk.Frame(split)
        cols = ("length", "status", "speakers", "words")
        self.tree = ttk.Treeview(qframe, columns=cols, show="tree headings", selectmode="extended", height=8)
        self.tree.heading("#0", text="Recording", anchor="w")
        self.tree.heading("length", text="Length", anchor="e")
        self.tree.heading("status", text="Status", anchor="w")
        self.tree.heading("speakers", text="Speakers", anchor="e")
        self.tree.heading("words", text="Words", anchor="e")
        self.tree.column("#0", width=280, minwidth=160, stretch=True)
        self.tree.column("length", width=70, anchor="e", stretch=False)
        self.tree.column("status", width=230, anchor="w", stretch=True)
        self.tree.column("speakers", width=72, anchor="e", stretch=False)
        self.tree.column("words", width=66, anchor="e", stretch=False)
        sb = ttk.Scrollbar(qframe, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.tree.pack(side="left", fill="both", expand=True)
        split.add(qframe, weight=2)

        tframe = ttk.Frame(split)
        tools = ttk.Frame(tframe, padding=(0, 8, 0, 6))
        tools.pack(fill="x")
        self.t_title = ttk.Label(tools, text="Transcript", style="Field.TLabel")
        self.t_title.pack(side="left")
        self.btn_folder = ttk.Button(tools, text="Open folder", command=self.open_output_folder)
        self.btn_folder.pack(side="right")
        self.btn_saveas = ttk.Button(tools, text="Save as…", command=self.save_as)
        self.btn_saveas.pack(side="right", padx=(0, 6))
        self.btn_copy = ttk.Button(tools, text="Copy", command=self.copy_text)
        self.btn_copy.pack(side="right", padx=(0, 6))
        self.btn_removed = ttk.Button(tools, text="Removed", command=self.show_removed)
        self.btn_removed.pack(side="right", padx=(0, 6))
        self.btn_rename = ttk.Button(tools, text="Rename speakers…", command=self.rename_speakers)
        self.btn_rename.pack(side="right", padx=(0, 6))
        wrap = ttk.Frame(tframe)
        wrap.pack(fill="both", expand=True)
        self.text = tk.Text(wrap, wrap="word", relief="flat", borderwidth=0, padx=14, pady=10, height=12,
                            font=(self.ui, 11), spacing1=1, spacing3=3, undo=False)
        tsb = ttk.Scrollbar(wrap, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=tsb.set, state="disabled")
        tsb.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        split.add(tframe, weight=3)
        self._drop_targets = [top, bar, status, body, side, qframe, tframe, tools, self.tree, self.text]
        self._placeholder()

    def _build_options(self, side: ttk.Frame):
        def label(text, pad=(10, 3)):
            ttk.Label(side, text=text, style="Field.TLabel").pack(anchor="w", pady=pad)

        label("Speech model", pad=(0, 3))
        self.cb_model = ttk.Combobox(side, textvariable=self.v_model, state="readonly")
        self.cb_model.pack(fill="x")
        self.cb_model.bind("<<ComboboxSelected>>", self._model_chosen)
        self.model_note = ttk.Label(side, text="", style="Dim.TLabel", wraplength=338, justify="left")
        self.model_note.pack(anchor="w", pady=(4, 0))

        label("Speakers")
        row = ttk.Frame(side)
        row.pack(fill="x")
        ttk.Checkbutton(row, text="Label who is speaking", variable=self.v_speakers,
                        command=self._option_changed).pack(side="left")
        row2 = ttk.Frame(side)
        row2.pack(fill="x", pady=(4, 0))
        ttk.Label(row2, text="How many people:").pack(side="left")
        self.cb_count = ttk.Combobox(row2, textvariable=self.v_count, state="readonly", values=SPEAKER_COUNTS, width=12)
        self.cb_count.pack(side="right")
        self.cb_count.bind("<<ComboboxSelected>>", self._option_changed)

        label("Noise reduction")
        self.cb_denoise = ttk.Combobox(side, textvariable=self.v_denoise, state="readonly",
                                       values=[t for _k, t in DENOISE])
        self.cb_denoise.pack(fill="x")
        self.cb_denoise.bind("<<ComboboxSelected>>", self._option_changed)

        label("Leave out invented text")
        self.cb_filter = ttk.Combobox(side, textvariable=self.v_filter, state="readonly",
                                      values=[t for _k, t in FILTER])
        self.cb_filter.pack(fill="x")
        self.cb_filter.bind("<<ComboboxSelected>>", self._option_changed)

        label("Save each transcript as")
        grid = ttk.Frame(side)
        grid.pack(fill="x")
        short = {"txt": "Text", "docx": "Word", "md": "Markdown", "srt": "SRT subtitles", "vtt": "VTT subtitles",
                 "csv": "CSV", "json": "JSON"}
        for i, fmt in enumerate(export.FORMATS):
            ttk.Checkbutton(grid, text=short[fmt], variable=self.v_formats[fmt],
                            command=self._option_changed).grid(row=i // 2, column=i % 2, sticky="w", padx=(0, 14))
        ttk.Checkbutton(side, text="Include timestamps", variable=self.v_timestamps,
                        command=self._option_changed).pack(anchor="w", pady=(6, 0))

        label("Save folder")
        self.save_label = ttk.Label(side, text="", style="Dim.TLabel", wraplength=338, justify="left")
        self.save_label.pack(anchor="w")
        row3 = ttk.Frame(side)
        row3.pack(fill="x", pady=(5, 0))
        ttk.Button(row3, text="Change…", command=self.choose_save_dir).pack(side="left")
        ttk.Button(row3, text="Reset to Downloads", command=self.reset_save_dir).pack(side="left", padx=(6, 0))

        label("Speed")
        self.speed_label = ttk.Label(side, text="", style="Dim.TLabel", wraplength=338, justify="left")
        self.speed_label.pack(anchor="w")
        self.btn_speed = ttk.Button(side, text="Test this computer…", command=self.speed_test)
        self.btn_speed.pack(anchor="w", pady=(5, 0))

        self.v_denoise.set(dict(DENOISE)[self.denoise])
        self.v_filter.set(dict(FILTER)[self.filtering])
        n = self.settings.get("num_speakers", 0)
        self.v_count.set(str(n) if str(n) in SPEAKER_COUNTS else SPEAKER_COUNTS[0])

    def _bind(self):
        r = self.root
        r.protocol("WM_DELETE_WINDOW", self.on_close)
        r.bind(f"<{MOD}-o>", lambda _e: self.add_files())
        r.bind(f"<{MOD}-Return>", lambda _e: self.start())
        r.bind("<Escape>", lambda _e: self.stop())
        r.bind(f"<{MOD}-d>", lambda _e: (self.toggle_theme(), "break")[1])
        # Ctrl+D is also Tk's "delete character" in text fields. A class binding
        # that only returned "break" would swallow the key before the window
        # saw it, so the class binding does the toggling itself.
        for cls in ("Entry", "TEntry", "Text", "TCombobox"):
            r.bind_class(cls, f"<{MOD}-d>", lambda _e: (self.toggle_theme(), "break")[1])
        if MAC:
            # Cmd+Q and the application menu's Quit go straight to Tk's own
            # exit unless told otherwise, skipping every clean-up below.
            try:
                r.createcommand("::tk::mac::Quit", self.on_close)
                r.createcommand("::tk::mac::OpenDocument", lambda *files: self.add_paths(list(files)))
            except tk.TclError:
                pass
        else:
            r.bind("<Control-q>", lambda _e: self.on_close())
        self.tree.bind("<<TreeviewSelect>>", self._selected)
        self.tree.bind("<Delete>", lambda _e: self.remove_selected())
        self.tree.bind("<BackSpace>", lambda _e: self.remove_selected())
        self.tree.bind("<Double-1>", self._double_click)
        for seq in ("<Button-3>", "<Button-2>", "<Control-Button-1>"):
            self.text.bind(seq, self._paragraph_menu)
        if _HAVE_DND:
            for w in self._drop_targets:
                try:
                    w.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
                    w.dnd_bind("<<Drop>>", self._dropped)  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass

    def _recolor(self):
        """Classic tk widgets are not covered by ttk styles: colour them by hand."""
        c = self.c
        self.text.configure(bg=c["panel"], fg=c["text"], insertbackground=c["accent"],
                            selectbackground=c["sel"], selectforeground=c["text"],
                            highlightthickness=1, highlightbackground=c["border"], highlightcolor=c["accent"])
        self.text.tag_configure("time", foreground=c["dim"], font=(self.mono, 9))
        self.text.tag_configure("dim", foreground=c["dim"])
        self.text.tag_configure("head", font=(self.ui, 12, "bold"))
        self.text.tag_configure("bad", foreground=c["bad"])
        self.text.tag_configure("warn", foreground=c["warn"])
        palette = SPEAKER_DARK if self.dark else SPEAKER_LIGHT
        for i, colour in enumerate(palette):
            self.text.tag_configure(f"spk{i}", foreground=colour, font=(self.ui, 11, "bold"))
        for menu in self.menus:
            menu.configure(bg=c["panel"], fg=c["text"], activebackground=c["sel"], activeforeground=c["text"],
                           relief="flat", borderwidth=0)
        self.tree.tag_configure("done", foreground=c["good"])
        self.tree.tag_configure("failed", foreground=c["bad"])
        self.tree.tag_configure("working", foreground=c["accent"])
        self.tree.tag_configure("waiting", foreground=c["text"])
        self.tree.tag_configure("stopped", foreground=c["warn"])
        self.btn_theme.configure(text="Light" if self.dark else "Dark")

    def toggle_theme(self):
        self.dark = not self.dark
        self.c = theme.apply(self.root, self.dark)
        self._recolor()
        self.settings["dark"] = self.dark
        paths.save_settings(self.settings)
        if self.shown and self.shown in self.items:
            self._show(self.items[self.shown])

    # --- settings ---------------------------------------------------------------

    def save_dir(self) -> str:
        return paths.default_save_dir(self.settings)

    def _refresh_save_label(self):
        self.save_label.configure(text=self.save_dir())

    def choose_save_dir(self):
        start = self.save_dir()
        if not os.path.isdir(start):
            start = paths.downloads_dir()
        d = filedialog.askdirectory(parent=self.root, title="Save transcripts in", initialdir=start)
        if d:
            self.settings["save_dir"] = d
            paths.save_settings(self.settings)
            self._refresh_save_label()

    def reset_save_dir(self):
        self.settings.pop("save_dir", None)
        paths.save_settings(self.settings)
        self._refresh_save_label()

    def _options(self) -> dict:
        count = self.v_count.get()
        return {
            "model": self.model_id,
            "speakers": bool(self.v_speakers.get()),
            "num_speakers": int(count) if count.isdigit() else 0,
            "denoise": self.denoise,
            "filtering": self.filtering,
            "provider": self.settings.get("provider") or "cpu",
            "parallel": self.settings.get("parallel") or "auto",
        }

    def _formats(self) -> List[str]:
        return [f for f, v in self.v_formats.items() if v.get()]

    def _option_changed(self, _event=None):
        self.denoise = {t: k for k, t in DENOISE}.get(self.v_denoise.get(), "auto")
        self.filtering = {t: k for k, t in FILTER}.get(self.v_filter.get(), "balanced")
        self.cb_count.configure(state="readonly" if self.v_speakers.get() else "disabled")
        self.settings.update(self._options())
        self.settings["formats"] = self._formats()
        self.settings["timestamps"] = bool(self.v_timestamps.get())
        paths.save_settings(self.settings)
        self._update_hint()

    # --- speed ------------------------------------------------------------------------

    def _refresh_speed_label(self):
        s = self.settings
        provider = s.get("provider") or "cpu"
        parallel = s.get("parallel") or "auto"
        where = {"cpu": "the processor", "coreml": "the graphics processor (Core ML)",
                 "cuda": "the graphics processor (NVIDIA CUDA)",
                 "directml": "the graphics processor (DirectML)"}.get(provider, provider)
        how = {"on": "both stages at once", "off": "one stage at a time",
               "auto": "both stages at once when there are cores to spare"}[parallel if parallel in ("on", "off") else "auto"]
        tested = (s.get("speedtest") or {}).get("when")
        tail = f"Tested {tested}." if tested else "Not yet tested on this computer."
        self.speed_label.configure(text=f"Runs on {where}, {how}. {tail}")

    def speed_test(self):
        if self.running or self.downloading or self.testing:
            messagebox.showinfo("Busy", "Wait for the current work to finish, then run the speed test.",
                                parent=self.root)
            return
        need = models.missing([self.model_id, "vad", "segmentation", "embedding"])
        if need:
            self.download(need, then=self.speed_test)
            return
        if not messagebox.askokcancel(
                "Test this computer",
                "This transcribes about two minutes of sample audio several different ways - one stage at a time, both "
                "stages at once, and on the graphics processor if there is one - and keeps whichever is quickest "
                "on this computer.\n\nIt takes a few minutes and the computer will be busy while it runs.",
                parent=self.root):
            return
        from . import speedtest
        self.testing = True
        self._update_buttons()
        self.progress.configure(mode="indeterminate")
        self.progress.start(60)
        self.set_status("Speed test: starting…")

        def work():
            try:
                out = speedtest.run(self.model_id, lambda t: self.inbox.put(lambda: self.set_status("Speed test - " + t)))
                self.inbox.put(lambda: self._speed_done(out, None))
            except Exception as exc:  # noqa: BLE001
                self.inbox.put(lambda e=str(exc): self._speed_done(None, e))

        threading.Thread(target=work, daemon=True).start()

    def _speed_done(self, out: Optional[dict], error: Optional[str]):
        self.testing = False
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self._update_buttons()
        if error or not out:
            self.set_status("Speed test failed.", "bad")
            messagebox.showerror("Speed test", error or "The speed test did not finish.", parent=self.root)
            return
        self.settings.update(out["choice"])
        self.settings["speedtest"] = {"when": out["when"], "summary": out["summary"]}
        paths.save_settings(self.settings)
        self._refresh_speed_label()
        self.set_status("Speed test finished - the quickest way to run on this computer is now in use.", "good")
        messagebox.showinfo("Speed test", out["summary"], parent=self.root)

    # --- models -------------------------------------------------------------------

    def _model_labels(self) -> Dict[str, str]:
        out = {}
        for mid, spec in models.ASR.items():
            suffix = "" if models.installed(mid) else "  [download]"
            out[mid] = spec["label"] + suffix
        return out

    def _refresh_models(self):
        labels = self._model_labels()
        self.cb_model.configure(values=list(labels.values()))
        self.v_model.set(labels[self.model_id])
        spec = models.ASR[self.model_id]
        self.model_note.configure(text=f"{spec['blurb']}  Speed: {spec['speed']}.")
        self.cb_count.configure(state="readonly" if self.v_speakers.get() else "disabled")
        self._update_hint()

    def _model_chosen(self, _event=None):
        labels = self._model_labels()
        for mid, text in labels.items():
            if text == self.v_model.get():
                self.model_id = mid
                break
        self.cb_model.selection_clear()
        self._refresh_models()
        self._option_changed()

    def _needed(self) -> List[str]:
        o = self._options()
        return models.missing(models.needed_for(o["model"], o["speakers"], o["denoise"] != "off"))

    def _update_hint(self):
        need = self._needed() if not self.downloading else []
        if need:
            mb = sum(models.CATALOGUE[m]["mb"] for m in need)
            what = models.CATALOGUE[need[0]]["label"] if len(need) == 1 else f"{len(need)} models"
            self.hint_label.configure(text=f"Not on this computer yet: {what} (about {mb} MB, downloaded once).")
            self.hint_button.configure(text="Download")
            if not self.hint.winfo_ismapped():
                self.hint.pack(fill="x", padx=12, pady=(0, 4), after=self.top)
        elif not self.downloading and self.hint.winfo_ismapped():
            self.hint.pack_forget()

    def _hint_action(self):
        if self.downloading:
            self.cancel_download.set()
            return
        self.download(self._needed())

    def download(self, ids: List[str], then=None):
        """Fetch models on a worker thread, reporting in the hint bar."""
        ids = [i for i in ids if not models.installed(i)]
        if not ids:
            if then:
                then()
            return
        if self.downloading:
            return
        self.downloading = True
        self.cancel_download.clear()
        self.hint_button.configure(text="Cancel")
        if not self.hint.winfo_ismapped():
            self.hint.pack(fill="x", padx=12, pady=(0, 4), after=self.top)
        self.hint_label.configure(text="Starting download…")
        self._update_buttons()

        def work():
            error = None
            try:
                for n, mid in enumerate(ids, 1):
                    label = models.CATALOGUE[mid]["label"]
                    last = [0.0]

                    def progress(stage, done, total, label=label, n=n, last=last):
                        now = time.time()
                        if now - last[0] < 0.2 and done != total:
                            return
                        last[0] = now
                        if stage == "download":
                            pct = f"{done * 100 // total}%" if total else f"{done // (1 << 20)} MB"
                            size = f" of {total // (1 << 20)} MB" if total else ""
                            text = f"Downloading {label} ({n} of {len(ids)}): {pct}{size}"
                        else:
                            text = f"Unpacking {label}…"
                        self.inbox.put(lambda t=text: self.hint_label.configure(text=t))

                    models.install(mid, progress, self.cancel_download.is_set)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            self.inbox.put(lambda: self._download_finished(error, then))

        threading.Thread(target=work, daemon=True).start()

    def _download_finished(self, error, then):
        self.downloading = False
        self._refresh_models()
        self._update_buttons()
        if error:
            self.set_status(error, "bad")
            if "cancel" not in error.lower():
                messagebox.showerror("Download failed", error, parent=self.root)
            return
        self.set_status("Models ready.")
        if then:
            then()

    def models_dialog(self):
        win = tk.Toplevel(self.root)
        win.title("Models")
        win.configure(bg=self.c["bg"])
        win.transient(self.root)
        frame = ttk.Frame(win, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Speech models", style="Title.TLabel").grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(frame, text="Each is downloaded once from the sherpa-onnx project on GitHub and kept in the "
                              "models folder. Audio never leaves this computer.",
                  style="Dim.TLabel", wraplength=640, justify="left").grid(row=1, column=0, columnspan=4, sticky="w",
                                                                         pady=(2, 10))
        rows = {}
        helper_button: List[ttk.Button] = []

        def refresh():
            busy = self.running or self.testing
            for mid, (state, button) in rows.items():
                bundled = models.installed(mid) and not os.path.isdir(os.path.join(paths.models_dirs()[0], mid))
                if models.installed(mid):
                    state.configure(text="Built in" if bundled else "Installed", style="Good.TLabel")
                    # never pull a model out from under a job that is using it
                    button.configure(text="Remove", state="disabled" if (bundled or busy) else "normal")
                else:
                    state.configure(text="Not downloaded", style="Dim.TLabel")
                    button.configure(text="Download", state="disabled" if self.downloading else "normal")
            for b in helper_button:
                b.configure(state="disabled" if self.downloading else "normal")

        def act(mid):
            if models.installed(mid):
                if messagebox.askyesno("Remove model", f"Delete {models.CATALOGUE[mid]['label']} from this computer?",
                                       parent=win):
                    models.remove(mid)
                    self._refresh_models()
                    refresh()
            else:
                self.download([mid], then=lambda: win.winfo_exists() and refresh())
                refresh()

        r = 2
        for mid, spec in models.ASR.items():
            ttk.Label(frame, text=spec["label"], style="Field.TLabel").grid(row=r, column=0, sticky="w", pady=(6, 0))
            ttk.Label(frame, text=f"{spec['mb']} MB").grid(row=r, column=1, sticky="e", padx=12, pady=(6, 0))
            state = ttk.Label(frame, text="")
            state.grid(row=r, column=2, sticky="w", padx=(0, 12), pady=(6, 0))
            button = ttk.Button(frame, text="", width=10, command=lambda m=mid: act(m))
            button.grid(row=r, column=3, sticky="e", pady=(6, 0))
            ttk.Label(frame, text=f"{spec['blurb']} Speed: {spec['speed']}.", style="Dim.TLabel", wraplength=640,
                      justify="left").grid(row=r + 1, column=0, columnspan=4, sticky="w")
            rows[mid] = (state, button)
            r += 2
        helpers = [m for m in models.SUPPORT if not models.installed(m)]
        text = ("Helper models (speech detector, noise reduction, speaker models): all installed." if not helpers else
                f"Helper models missing: {', '.join(models.SUPPORT[m]['label'] for m in helpers)}.")
        ttk.Label(frame, text=text, style="Dim.TLabel", wraplength=640, justify="left").grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(14, 0))
        if helpers:
            b = ttk.Button(frame, text="Download", width=10,
                           command=lambda: self.download(helpers, then=lambda: win.winfo_exists() and win.destroy()))
            b.grid(row=r, column=3, sticky="e", pady=(14, 0))
            helper_button.append(b)
        ttk.Button(frame, text="Close", command=win.destroy).grid(row=r + 1, column=3, sticky="e", pady=(16, 0))
        frame.columnconfigure(0, weight=1)
        refresh()

        def tick():
            if win.winfo_exists():
                refresh()
                win.after(700, tick)
        tick()

    # --- queue --------------------------------------------------------------------

    def _start_dir(self) -> str:
        start = self.settings.get("last_dir")
        return start if start and os.path.isdir(start) else paths.downloads_dir()

    def add_files(self):
        names = filedialog.askopenfilenames(parent=self.root, title="Add recordings", initialdir=self._start_dir())
        if names:
            self.settings["last_dir"] = os.path.dirname(names[0])
            paths.save_settings(self.settings)
            self.add_paths(list(names))

    def add_folder(self):
        d = filedialog.askdirectory(parent=self.root, title="Add every recording in a folder",
                                    initialdir=self._start_dir())
        if d:
            self.settings["last_dir"] = d
            paths.save_settings(self.settings)
            self.add_paths([d])

    def add_paths(self, given: List[str]):
        """Files are added as they are; folders are searched for audio and video,
        on a worker thread because a network share can take a while to walk."""
        files = []
        folders = []
        opened = 0
        for p in given:
            p = os.path.abspath(os.path.expanduser(p))
            if os.path.isdir(p):
                folders.append(p)
            elif os.path.isfile(p):
                if p.lower().endswith(".json") and self._looks_like_transcript(p):
                    self._open_transcript_file(p)
                    opened += 1
                else:
                    files.append(p)
        if files or not (folders or opened):
            self._add_found(files, bool(given) and not folders)
        if not folders:
            return
        self.set_status("Looking through the folder…")

        def walk():
            found: List[str] = []
            for folder in folders:
                for base, dirs, names in os.walk(folder):
                    dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                    for n in sorted(names):
                        if os.path.splitext(n)[1].lower() in audio.EXTENSIONS and not n.startswith("."):
                            found.append(os.path.join(base, n))
            self.inbox.put(lambda: self._add_found(found, True))

        threading.Thread(target=walk, daemon=True).start()

    def _add_found(self, found: List[str], say_if_none: bool):
        have = {os.path.normcase(self.items[i].path) for i in self.order if self.items[i].state in ("waiting", "working")}
        added = 0
        for p in found:
            if os.path.normcase(p) in have:
                continue
            have.add(os.path.normcase(p))
            item = Item(p)
            self.items[item.id] = item
            self.order.append(item.id)
            self.tree.insert("", "end", iid=item.id, text=os.path.basename(p),
                             values=("", item.status, "", ""), tags=("waiting",))
            self.probe_q.put(item.id)
            added += 1
        if added:
            self.set_status(f"{added} recording{'s' if added != 1 else ''} added. "
                            + ("They join the queue." if self.running else "Press Start."))
        elif say_if_none:
            self.set_status("Nothing new to add - no audio or video files found there, or already queued.", "warn")
        self._update_buttons()

    def _dropped(self, event):
        try:
            dropped = list(self.root.tk.splitlist(event.data))
        except Exception:  # noqa: BLE001
            return None
        self.add_paths([p for p in dropped if p and os.path.exists(p)])
        return getattr(event, "action", "copy")

    def _probe_worker(self):
        """Read each new file's length off the main thread; adding 200 files must not freeze the window."""
        while True:
            iid = self.probe_q.get()
            item = self.items.get(iid)
            if item is None:
                continue
            try:
                info = audio.probe(item.path)
            except Exception:  # noqa: BLE001
                info = {"duration": None}
            self.inbox.put(lambda i=iid, d=info.get("duration"): self._probed(i, d))

    def _probed(self, iid: str, duration: Optional[float]):
        item = self.items.get(iid)
        if item is None or not self.tree.exists(iid):
            return
        if duration and not item.duration:
            item.duration = duration
            self.tree.set(iid, "length", export.clock(duration))

    def _row(self, item: Item):
        if not self.tree.exists(item.id):
            return
        self.tree.set(item.id, "status", item.status)
        self.tree.item(item.id, tags=(item.state,))
        if item.result:
            if item.result.get("duration"):
                self.tree.set(item.id, "length", export.clock(item.result["duration"]))
            n = len(item.result.get("speakers") or {})
            self.tree.set(item.id, "speakers", str(n) if n else "")
            words = item.result.get("stats", {}).get("words")
            if words is None:
                words = sum(len(s.get("text", "").split()) for s in item.result.get("segments", []))
            self.tree.set(item.id, "words", str(words))

    def remove_selected(self):
        for iid in self.tree.selection():
            item = self.items.get(iid)
            if item is None or item.state == "working":
                continue
            self.tree.delete(iid)
            self.order.remove(iid)
            del self.items[iid]
            if self.shown == iid:
                self.shown = None
                self._placeholder()
        self._update_buttons()

    def clear_finished(self):
        for iid in list(self.order):
            if self.items[iid].state in ("done", "failed"):
                self.tree.delete(iid)
                self.order.remove(iid)
                del self.items[iid]
                if self.shown == iid:
                    self.shown = None
                    self._placeholder()
        self._update_buttons()

    def rerun_selected(self):
        for iid in self.tree.selection():
            item = self.items.get(iid)
            if item and item.state in ("done", "failed", "stopped"):
                if not os.path.isfile(item.path):
                    self.set_status(f"{os.path.basename(item.path)} is no longer where it was - it cannot be run again.",
                                    "warn")
                    continue
                item.state, item.status, item.result, item.error, item.outputs = "waiting", "Waiting", None, "", {}
                self._row(item)
                self.tree.set(iid, "speakers", "")
                self.tree.set(iid, "words", "")
                if self.shown == iid:
                    self._show(item)
        self._update_buttons()

    def _update_buttons(self):
        waiting = any(self.items[i].state in ("waiting", "stopped") for i in self.order)
        idle = not self.running and not self.downloading and not self.testing
        self.btn_start.configure(state="normal" if (waiting and idle) else "disabled")
        self.btn_stop.configure(state="normal" if self.running else "disabled")
        finished = any(self.items[i].state in ("done", "failed") for i in self.order)
        self.btn_clear.configure(state="normal" if finished else "disabled")
        self.btn_remove.configure(state="normal" if self.order else "disabled")
        self.btn_speed.configure(state="normal" if idle else "disabled")

    # --- running ------------------------------------------------------------------

    def start(self):
        if self.running or self.downloading or self.testing:
            return
        if not any(self.items[i].state in ("waiting", "stopped") for i in self.order):
            return
        if not self._formats():
            messagebox.showinfo("Nothing to save", "Tick at least one format under \"Save each transcript as\".",
                                parent=self.root)
            return
        need = self._needed()
        if need:
            mb = sum(models.CATALOGUE[m]["mb"] for m in need)
            names = "\n".join("  " + models.CATALOGUE[m]["label"] for m in need)
            if not messagebox.askyesno("Download needed", f"These need downloading first (about {mb} MB, once):\n\n"
                                       f"{names}\n\nDownload now and then start?", parent=self.root):
                return
            self.download(need, then=self.start)
            return
        for iid in self.order:
            if self.items[iid].state == "stopped":
                self.items[iid].state, self.items[iid].status = "waiting", "Waiting"
                self._row(self.items[iid])
        self.running = True
        self._update_buttons()
        self._next()

    def _next(self):
        if not self.running:
            return
        item = next((self.items[i] for i in self.order if self.items[i].state == "waiting"), None)
        if item is None:
            self.running = False
            self.current = None
            self.progress.configure(value=0)
            done = sum(1 for i in self.order if self.items[i].state == "done")
            failed = sum(1 for i in self.order if self.items[i].state == "failed")
            msg = f"Finished: {done} transcript{'s' if done != 1 else ''} saved in {self.save_dir()}"
            if failed:
                msg += f" - {failed} failed"
            self.set_status(msg, "bad" if failed else "good")
            self._update_buttons()
            try:
                self.root.bell()
            except tk.TclError:
                pass
            return
        need = self._needed()
        if need:  # the model was switched mid-queue to one that is not downloaded
            self.running = False
            self._update_buttons()
            self.set_status("Stopped: the chosen model is not downloaded. Press Start to download it and carry on.",
                            "warn")
            return
        self.current = item
        item.state, item.status, item.fraction, item.error = "working", "Starting…", 0.0, ""
        self._row(item)
        self.tree.see(item.id)
        try:
            self.jobdir = tempfile.mkdtemp(prefix="verbatim-job-")
            job = {"file": item.path, "options": self._options(),
                   "events": os.path.join(self.jobdir, "events.jsonl"),
                   "result": os.path.join(self.jobdir, "result.json"),
                   "cancel": os.path.join(self.jobdir, "cancel"), "workdir": self.jobdir}
            job_path = os.path.join(self.jobdir, "job.json")
            with open(job_path, "w", encoding="utf-8") as fh:
                json.dump(job, fh)
            open(job["events"], "wb").close()
            self.events_pos = 0
            self.started = time.time()
            with open(os.path.join(self.jobdir, "engine.log"), "wb") as log:
                self.proc = proc.start_engine(job_path, log)
        except Exception as exc:  # noqa: BLE001
            self._finish(item, error=f"The engine could not be started: {exc}")
            return
        self.poll_id = self.root.after(150, self._poll)

    def _poll(self):
        self.poll_id = None
        item, engine = self.current, self.proc
        if item is None or engine is None:
            return
        try:
            finished = self._read_events(item, engine)
        except Exception as exc:  # noqa: BLE001 - a fault here must not strand the queue
            proc.kill_tree(engine)
            self._finish(item, error=f"The window lost track of the engine: {exc}")
            return
        if finished is None:
            self.poll_id = self.root.after(150, self._poll)
            return
        try:
            engine.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill_tree(engine)
        if finished["type"] == "done":
            try:
                with open(os.path.join(self.jobdir, "result.json"), encoding="utf-8") as fh:
                    result = json.load(fh)
            except Exception as exc:  # noqa: BLE001
                self._finish(item, error=f"The result could not be read: {exc}")
                return
            self._finish(item, result=result)
        else:
            self._finish(item, error=finished.get("message") or "Failed.", trace=finished.get("trace") or "")

    def _read_events(self, item: Item, engine: subprocess.Popen) -> Optional[dict]:
        finished = None
        try:
            # binary, so that byte offsets mean the same thing on Windows (where
            # a text-mode read would swallow the \r of every line ending)
            with open(os.path.join(self.jobdir, "events.jsonl"), "rb") as fh:
                fh.seek(self.events_pos)
                chunk = fh.read()
            # only whole lines: the engine may be half-way through writing one
            cut = chunk.rfind(b"\n")
            if cut >= 0:
                self.events_pos += cut + 1
                for line in chunk[:cut].splitlines():
                    try:
                        ev = json.loads(line.decode("utf-8", "replace"))
                    except ValueError:
                        continue
                    if ev.get("type") == "progress":
                        self._progress(item, ev)
                    elif ev.get("type") in ("done", "error"):
                        finished = ev
        except OSError:
            pass
        if finished is None and engine.poll() is not None:
            time.sleep(0.05)
            if os.path.exists(os.path.join(self.jobdir, "result.json")):
                finished = {"type": "done"}
            else:
                finished = {"type": "error", "message": f"The engine stopped unexpectedly (code {engine.returncode}).",
                            "trace": self._log_tail()}
        return finished

    def _log_tail(self) -> str:
        try:
            with open(os.path.join(self.jobdir, "engine.log"), encoding="utf-8", errors="replace") as fh:
                return fh.read()[-1500:]
        except OSError:
            return ""

    def _progress(self, item: Item, ev: dict):
        frac = float(ev.get("fraction") or 0.0)
        item.fraction = frac
        elapsed = time.time() - self.started
        eta = ""
        if frac > 0.08 and elapsed > 4:
            left = elapsed * (1 - frac) / frac
            eta = f" - about {max(1, round(left / 60))} min left" if left >= 50 else " - under a minute left"
        item.status = f"{ev.get('label', 'Working')} {frac * 100:.0f}%"
        self._row(item)
        pos = sum(1 for i in self.order if self.items[i].state in ("done", "failed")) + 1
        total = sum(1 for i in self.order if self.items[i].state in ("done", "failed", "working", "waiting"))
        self.progress.configure(value=int(frac * 1000))
        self.set_status(f"File {pos} of {total}: {os.path.basename(item.path)} - {ev.get('label', '')}{eta}")

    def _finish(self, item: Item, result: Optional[dict] = None, error: str = "", trace: str = ""):
        self.proc = None
        self.current = None
        shutil.rmtree(self.jobdir, ignore_errors=True)
        if result is not None:
            item.result = result
            item.error = ""
            took = result.get("stats", {}).get("seconds_taken", 0)
            try:
                item.folder = self.save_dir()
                # every box could have been unticked while this file was running
                item.outputs = export.write_all(result, item.folder, self._formats() or ["txt"],
                                                bool(self.v_timestamps.get()))
                first = next(iter(item.outputs.values()))
                item.stem = os.path.splitext(os.path.basename(first))[0]
                item.state = "done"
                item.status = (f"Done in {export.clock(took)}" if result.get("segments")
                               else "Done - no speech found")
            except Exception as exc:  # noqa: BLE001 - disk full, folder gone, no permission
                item.state = "failed"
                item.error = (f"The transcript below was made but could not be saved to {item.folder}: {exc}. "
                              f"Use Save as… to put it somewhere else.")
                item.status = "Transcribed, but could not be saved - see below"
        else:
            item.state, item.error = "failed", error + (("\n\n" + trace) if trace else "")
            item.status = "Failed - " + (error.splitlines()[0][:90] if error else "unknown reason")
        self._row(item)
        if self.shown in (None, item.id) or not self.tree.selection():
            self.tree.selection_set(item.id)
            self._show(item)
        self._update_buttons()
        self.root.after(50, self._next)

    def stop(self, ask: bool = True):
        if not self.running:
            return
        item, engine = self.current, self.proc
        if ask and item is not None and time.time() - self.started > 60:
            minutes = int((time.time() - self.started) // 60)
            if not messagebox.askyesno("Stop?", f"{os.path.basename(item.path)} has been running for {minutes} "
                                       f"minute{'s' if minutes != 1 else ''}. Stopping throws that work away.\n\n"
                                       "Stop anyway?", parent=self.root):
                return
        self.running = False
        if self.poll_id is not None:
            try:
                self.root.after_cancel(self.poll_id)
            except Exception:  # noqa: BLE001
                pass
            self.poll_id = None
        self.current, self.proc = None, None
        if engine is not None:
            # ask nicely first (the engine stops ffmpeg and lets go of its
            # files), then stop it and everything it started
            try:
                open(os.path.join(self.jobdir, "cancel"), "wb").close()
                engine.wait(timeout=1.5)
            except Exception:  # noqa: BLE001
                pass
            proc.kill_tree(engine)
        if self.jobdir:
            shutil.rmtree(self.jobdir, ignore_errors=True)
        if item is not None:
            item.state, item.status = "stopped", "Stopped - Start runs it again"
            self._row(item)
            if self.shown == item.id:
                self._show(item)
        self.progress.configure(value=0)
        self.set_status("Stopped.", "warn")
        self._update_buttons()

    # --- transcript pane ------------------------------------------------------------

    def _selected(self, _event=None):
        sel = self.tree.selection()
        if len(sel) == 1 and sel[0] in self.items:
            self._show(self.items[sel[0]])

    def _double_click(self, _event=None):
        sel = self.tree.selection()
        if len(sel) == 1 and sel[0] in self.items:
            item = self.items[sel[0]]
            if item.outputs:
                open_path(next(iter(item.outputs.values())))

    def _write(self, parts):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        for text, tag in parts:
            self.text.insert("end", text, tag)
        self.text.configure(state="disabled")
        self.text.yview_moveto(0)

    def _placeholder(self):
        self._para_spans = []
        self.t_title.configure(text="Transcript")
        self._write([("Nothing to show yet.\n\n", ("head",)),
                     ("Add recordings with the buttons above" + (", or drop them on this window" if _HAVE_DND else "")
                      + ". Any audio or video format works. Press Start and each transcript appears here and is "
                        "saved to the save folder.\n", ("dim",))])
        for b in (self.btn_rename, self.btn_copy, self.btn_saveas, self.btn_removed, self.btn_folder):
            b.configure(state="disabled")
        self.btn_removed.configure(text="Removed")

    def _show(self, item: Item):
        self.shown = item.id
        self._para_spans = []
        self.t_title.configure(text=os.path.basename(item.path))
        t = item.result
        have = t is not None
        self.btn_copy.configure(state="normal" if have else "disabled")
        self.btn_saveas.configure(state="normal" if have else "disabled")
        self.btn_folder.configure(state="normal" if (item.outputs or item.folder) else "disabled")
        self.btn_rename.configure(state="normal" if (have and t.get("speakers")) else "disabled")
        removed = [r for r in (t or {}).get("removed", []) if "kept_as" not in r]
        self.btn_removed.configure(text=f"Removed ({len(removed)})" if removed else "Removed",
                                   state="normal" if removed else "disabled")
        if not have:
            if item.state == "failed":
                self._write([("This recording could not be transcribed.\n\n", ("head", "bad")), (item.error + "\n", ())])
            elif item.state == "working":
                self._write([("Working on this one…\n", ("dim",))])
            else:
                self._write([(item.status + "\n", ("dim",))])
            return
        parts = []
        hours = (t.get("duration") or 0) >= 3600
        if item.error:
            parts.append((item.error + "\n", ("bad",)))
        for note in t.get("notes", []):
            parts.append((note + "\n", ("dim",)))
        if item.outputs:
            parts.append(("Saved: " + ", ".join(os.path.basename(p) for p in item.outputs.values())
                          + f"  in  {item.folder}\n", ("dim",)))
        if t.get("speakers") and t.get("segments"):
            parts.append(("Right-click a paragraph to change who said it.\n", ("dim",)))
        if parts:
            parts.append(("\n", ()))
        ids = list((t.get("speakers") or {}).keys())
        names: List[str] = []
        for sid in ids:
            n = t["speakers"][sid]
            if n not in names:
                names.append(n)
        for n, p in enumerate(export.paragraphs(t)):
            who = export.speaker_name(t, p["speaker"])
            stamp = export.clock(p["start"], hours)
            ptag = f"para{n}"
            self._para_spans.append(p)
            if who:
                tag = f"spk{names.index(who) % len(SPEAKER_DARK)}" if who in names else "head"
                parts.append((who, (tag, ptag)))
                parts.append((f"   {stamp}\n", ("time", ptag)))
                parts.append((p["text"] + "\n", (ptag,)))
                parts.append(("\n", ()))
            else:
                parts.append((f"{stamp}   ", ("time", ptag)))
                parts.append((p["text"] + "\n", (ptag,)))
                parts.append(("\n", ()))
        if not t.get("segments"):
            parts.append(("No speech was found in this recording.\n", ("warn",)))
        self._write(parts)

    def _paragraph_menu(self, event):
        """Right-click on a paragraph: say who really spoke it."""
        item = self._current_result()
        if not item or not item.result.get("speakers"):
            return None
        index = self.text.index(f"@{event.x},{event.y}")
        para = None
        for tag in self.text.tag_names(index):
            if tag.startswith("para") and tag[4:].isdigit():
                para = int(tag[4:])
        if para is None or para >= len(self._para_spans):
            return None
        t = item.result
        segs = list(self._para_spans[para]["segments"])
        current = self._para_spans[para]["speaker"]
        self.context.delete(0, "end")
        self.context.add_command(label="Who said this?", state="disabled")
        self.context.add_separator()
        for sid, name in t["speakers"].items():
            self.context.add_command(label=("• " if sid == current else "   ") + name,
                                     command=lambda s=sid: self._reassign(item, t, segs, s))
        self.context.add_separator()
        self.context.add_command(label="   Someone new", command=lambda: self._reassign(item, t, segs, None))
        try:
            self.context.tk_popup(event.x_root, event.y_root)
        finally:
            self.context.grab_release()
        return "break"

    def _reassign(self, item: Item, t: dict, segs: List[int], sid: Optional[str]):
        if item.result is not t:
            return
        if sid is None:
            n = 1
            while f"S{n}" in t["speakers"]:
                n += 1
            sid = f"S{n}"
            taken = set(t["speakers"].values())
            k = len(t["speakers"]) + 1
            while f"Speaker {k}" in taken:
                k += 1
            t["speakers"][sid] = f"Speaker {k}"
        for i in segs:
            if 0 <= i < len(t["segments"]):
                t["segments"][i]["speaker"] = sid
        used = {s.get("speaker") for s in t["segments"]}
        t["speakers"] = {k: v for k, v in t["speakers"].items() if k in used}
        self._resave(item, "Speaker changed and the files re-saved.")

    def _current_result(self) -> Optional[Item]:
        item = self.items.get(self.shown) if self.shown else None
        return item if item is not None and item.result is not None else None

    def copy_text(self):
        item = self._current_result()
        if not item:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(export.to_txt(item.result, bool(self.v_timestamps.get())))
        self.set_status("Transcript copied.")

    def save_as(self):
        item = self._current_result()
        if not item:
            return
        types = [(label, f"*.{fmt}") for fmt, label in export.FORMATS.items()]
        folder = item.folder if os.path.isdir(item.folder) else self.save_dir()
        if not os.path.isdir(folder):
            folder = paths.downloads_dir()
        path = filedialog.asksaveasfilename(parent=self.root, title="Save transcript as", initialdir=folder,
                                            initialfile=(item.stem or export.safe_stem(item.path)) + ".txt",
                                            defaultextension=".txt", filetypes=types)
        if not path:
            return
        fmt = os.path.splitext(path)[1].lstrip(".").lower()
        if fmt not in export.FORMATS:
            fmt, path = "txt", path + ".txt"
        try:
            with open(path, "wb") as fh:
                fh.write(export.render(item.result, fmt, bool(self.v_timestamps.get())))
            self.set_status(f"Saved {path}")
        except OSError as exc:
            messagebox.showerror("Could not save", str(exc), parent=self.root)

    def open_output_folder(self):
        item = self.items.get(self.shown) if self.shown else None
        self._open_folder(item.folder if item and item.folder else self.save_dir())

    def _open_folder(self, folder: str):
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError:
            pass
        open_path(folder)

    # --- reopening a saved transcript ---------------------------------------------------

    @staticmethod
    def _looks_like_transcript(path: str) -> bool:
        try:
            if os.path.getsize(path) > 200 * 1024 * 1024:
                return False
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            return isinstance(data, dict) and isinstance(data.get("segments"), list) and "speakers" in data
        except Exception:  # noqa: BLE001
            return False

    def open_transcript(self):
        start = self.save_dir() if os.path.isdir(self.save_dir()) else paths.downloads_dir()
        path = filedialog.askopenfilename(parent=self.root, title="Open a saved transcript", initialdir=start,
                                          filetypes=[("Verbatim transcript", "*.json")])
        if not path:
            return
        if not self._looks_like_transcript(path):
            messagebox.showerror("Not a transcript", "That file is not a transcript saved by Verbatim (the JSON format).",
                                 parent=self.root)
            return
        self._open_transcript_file(path)

    def _open_transcript_file(self, path: str):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            for s in data["segments"]:
                s["start"], s["end"], s["text"] = float(s["start"]), float(s["end"]), str(s["text"])
                s.setdefault("speaker", None)
            if not isinstance(data.get("speakers"), dict):
                data["speakers"] = {}
            data.setdefault("removed", [])
            data.setdefault("notes", [])
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Could not open", f"That transcript could not be read: {exc}", parent=self.root)
            return
        item = Item(str(data.get("file") or path))
        item.result = data
        item.state, item.status = "done", "Opened from a saved transcript"
        item.folder = os.path.dirname(os.path.abspath(path))
        item.stem = os.path.splitext(os.path.basename(path))[0]
        # the other formats saved alongside it are kept in step when names change
        for fmt in export.FORMATS:
            sibling = os.path.join(item.folder, f"{item.stem}.{fmt}")
            if os.path.isfile(sibling):
                item.outputs[fmt] = sibling
        self.items[item.id] = item
        self.order.append(item.id)
        self.tree.insert("", "end", iid=item.id, text=os.path.basename(item.path),
                         values=("", item.status, "", ""), tags=("done",))
        self._row(item)
        self.tree.selection_set(item.id)
        self._show(item)
        self._update_buttons()

    # --- dialogs ------------------------------------------------------------------------

    def _dialog(self, title: str, geometry: str = "") -> tk.Toplevel:
        win = tk.Toplevel(self.root)
        win.title(title)
        if geometry:
            win.geometry(geometry)
        win.configure(bg=self.c["bg"])
        win.transient(self.root)
        try:
            win.grab_set()   # modal: the item behind the dialog cannot be removed or re-run under it
        except tk.TclError:
            pass
        return win

    def show_removed(self):
        item = self._current_result()
        if not item:
            return
        removed = [r for r in item.result.get("removed", []) if "kept_as" not in r]
        win = self._dialog("Left out of the transcript", "760x420")
        ttk.Label(win, text="These pieces came back from the speech model but were judged not to be real speech. "
                            "If one of them was really said, set \"Leave out invented text\" to Off and run the file "
                            "again.", style="Dim.TLabel", wraplength=720, justify="left", padding=(14, 12, 14, 6)).pack(anchor="w")
        box = tk.Text(win, wrap="word", relief="flat", padx=14, pady=8, font=(self.ui, 10), bg=self.c["panel"],
                      fg=self.c["text"], highlightthickness=1, highlightbackground=self.c["border"])
        box.tag_configure("time", foreground=self.c["dim"], font=(self.mono, 9))
        box.tag_configure("why", foreground=self.c["warn"])
        for r in removed:
            box.insert("end", f"{export.clock(r['start'])}   ", ("time",))
            box.insert("end", r["text"].strip() + "\n")
            box.insert("end", f"        {r['reason']}\n\n", ("why",))
        box.configure(state="disabled")
        box.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        ttk.Button(win, text="Close", command=win.destroy).pack(anchor="e", padx=14, pady=(0, 12))
        win.bind("<Escape>", lambda _e: win.destroy())

    def rename_speakers(self):
        item = self._current_result()
        if not item or not item.result.get("speakers"):
            return
        t = item.result
        win = self._dialog("Rename speakers")
        frame = ttk.Frame(win, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Type the real names. Give two speakers the same name to merge them into one.",
                  style="Dim.TLabel", wraplength=560, justify="left").grid(row=0, column=0, columnspan=2, sticky="w",
                                                                         pady=(0, 10))
        entries = {}
        for r, (sid, name) in enumerate(t["speakers"].items(), 1):
            segs = [s for s in t["segments"] if s.get("speaker") == sid]
            total = sum(s["end"] - s["start"] for s in segs)
            sample = max(segs, key=lambda s: len(s["text"]))["text"] if segs else ""
            if len(sample) > 90:
                sample = sample[:88] + "…"
            var = tk.StringVar(value=name)
            e = ttk.Entry(frame, textvariable=var, width=24)
            e.grid(row=r * 2 - 1, column=0, sticky="w", pady=(6, 0))
            ttk.Label(frame, text=f"spoke for {export.clock(total)}", style="Dim.TLabel").grid(
                row=r * 2 - 1, column=1, sticky="w", padx=(10, 0), pady=(6, 0))
            ttk.Label(frame, text=f"“{sample}”", style="Dim.TLabel", wraplength=560, justify="left").grid(
                row=r * 2, column=0, columnspan=2, sticky="w")
            entries[sid] = var
            if r == 1:
                e.focus_set()
                e.selection_range(0, "end")

        def apply(_event=None):
            if item.result is not t or item.id not in self.items:
                win.destroy()
                return
            first: Dict[str, str] = {}
            merged: Dict[str, str] = {}
            speakers: Dict[str, str] = {}
            for sid, var in entries.items():
                name = var.get().strip() or t["speakers"][sid]
                key = name.casefold()
                if key in first:
                    merged[sid] = first[key]
                else:
                    first[key] = sid
                    speakers[sid] = name
            if merged:
                for s in t["segments"]:
                    if s.get("speaker") in merged:
                        s["speaker"] = merged[s["speaker"]]
            t["speakers"] = speakers
            win.destroy()
            self._resave(item, "Speaker names updated and the files re-saved.")

        buttons = ttk.Frame(frame)
        buttons.grid(row=len(entries) * 2 + 1, column=0, columnspan=2, sticky="e", pady=(16, 0))
        ttk.Button(buttons, text="Cancel", command=win.destroy).pack(side="right")
        ttk.Button(buttons, text="Apply and re-save", style="Accent.TButton", command=apply).pack(side="right", padx=(0, 6))
        win.bind("<Return>", apply)
        win.bind("<Escape>", lambda _e: win.destroy())

    def _resave(self, item: Item, message: str):
        """Write the item's files again in place, after its speakers were changed."""
        try:
            if item.outputs and os.path.isdir(item.folder):
                item.outputs = export.write_all(item.result, item.folder, list(item.outputs), bool(self.v_timestamps.get()),
                                                overwrite=True, stem=item.stem)
                self.set_status(message)
            else:
                self.set_status("Changed here. Use Save as… to save it.", "warn")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Could not re-save", str(exc), parent=self.root)
        self._row(item)
        if self.shown == item.id:
            top = self.text.yview()[0]
            self._show(item)
            self.text.yview_moveto(top)

    # --- odds and ends ----------------------------------------------------------------

    def set_status(self, text: str, kind: str = ""):
        style = {"good": "Good.TLabel", "bad": "Bad.TLabel", "warn": "Warn.TLabel"}.get(kind, "Dim.TLabel")
        self.status.configure(text=text, style=style)

    def _pump(self):
        """Run whatever worker threads posted. Worker threads never touch widgets themselves."""
        if self._closing:
            return
        for _ in range(200):
            try:
                job = self.inbox.get_nowait()
            except queue.Empty:
                break
            try:
                job()
            except Exception:  # noqa: BLE001 - one bad callback must not stop the pump
                pass
        self.root.after(120, self._pump)

    def show_help(self):
        win = self._dialog("How it works", "720x600")
        box = tk.Text(win, wrap="word", relief="flat", padx=16, pady=12, font=(self.ui, 10), bg=self.c["panel"],
                      fg=self.c["text"], spacing3=4, highlightthickness=0)
        box.tag_configure("h", font=(self.ui, 10, "bold"), foreground=self.c["accent"])
        for para in HELP.split("\n"):
            if para.isupper() and len(para) < 40:
                box.insert("end", para + "\n", ("h",))
            else:
                box.insert("end", para + "\n")
        box.configure(state="disabled")
        box.pack(fill="both", expand=True)
        win.bind("<Escape>", lambda _e: win.destroy())

    def show_about(self):
        messagebox.showinfo(f"About {paths.APP}",
                            f"{paths.APP} {paths.VERSION}\n\nSpeech to text with speaker labels, entirely offline.\n\n"
                            "Built on sherpa-onnx and ONNX Runtime. Models: NVIDIA Parakeet TDT, OpenAI Whisper, "
                            "Silero VAD, pyannote segmentation 3.0, NVIDIA TitaNet, GTCRN. Audio decoding by FFmpeg.\n\n"
                            f"Settings and models: {paths.data_dir()}",
                            parent=self.root)

    def on_close(self):
        if self._closing:
            return
        if self.running and not messagebox.askyesno("Still working", "A recording is being transcribed. Stop and quit?",
                                                    parent=self.root):
            return
        if self.downloading and not messagebox.askyesno(
                "Still downloading", "A model is still downloading. Quit anyway? It carries on from where it stopped "
                "next time.", parent=self.root):
            return
        self._closing = True
        if self.running:
            self.stop(ask=False)
        self.cancel_download.set()
        try:
            self.settings["geometry"] = self.root.winfo_geometry()
            paths.save_settings(self.settings)
        except Exception:  # noqa: BLE001
            pass
        self.root.destroy()


def make_root():
    """A Tk root, with native drag and drop loaded into it when that is possible."""
    global _HAVE_DND
    root = tk.Tk()
    if _HAVE_DND:
        # Loading tkdnd into an ordinary root, rather than using TkinterDnD.Tk(),
        # means a failure leaves one working window instead of a stray empty one.
        try:
            TkinterDnD._require(root)
        except Exception:  # noqa: BLE001 - no tkdnd build for this platform or this Tk
            _HAVE_DND = False
    return root


def launch(files: Optional[List[str]] = None) -> int:
    root = make_root()
    App(root, files or [])
    root.mainloop()
    return 0
