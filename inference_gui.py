from __future__ import annotations

import json
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk
from ultralytics import YOLO

WEIGHTS_PATH = Path("./weights/best.pt")
INITIAL_THRESHOLD = 0.4
OUTPUT_RELATIVE_DIR = Path("runs") / "inference_gui"
LEFT_COLUMN_WEIGHT = 8
RIGHT_COLUMN_WEIGHT = 2
EVENT_QUEUE_POLL_MS = 100
ZOOM_MIN_FACTOR = 0.75
ZOOM_MAX_FACTOR = 1.25
ZOOM_STEP = 0.05
PREVIEW_PLACEHOLDER_TEXT = "Click to select image(s)."
PREVIEW_READ_ERROR_TEMPLATE = "Failed to load image: {name}"
NO_IMAGE_STATUS_TEXT = "No image selected"
ALL_TAGS_ITEM_TEMPLATE = "All Price Tags ({count})"
TAG_ITEM_TEMPLATE = "Tag {tag_id:02d} | {details}"
ACTION_LABEL_TEMPLATE = "Action: {action}"
WINDOW_TITLE = "Shelf Price Tag Detection"
SELECT_DIALOG_TITLE = "Select image(s) for detection"
SAVE_DIALOG_TITLE = "Save preview as"
WARNING_NO_VALID_FILES_TITLE = "No valid files"
WARNING_NO_VALID_FILES_MESSAGE = "No valid image files were selected."
WARNING_NO_IMAGE_TITLE = "No image"
WARNING_NO_IMAGE_MESSAGE = "Please select image(s) first."
ERROR_SAVE_FAILED_TITLE = "Save failed"
ERROR_SAVE_FAILED_MESSAGE = (
    "Failed to save preview image. Choose a supported file extension."
)
ERROR_IMAGE_LOAD_MESSAGE = "Current image could not be loaded."
INFO_IMAGE_SAVED_TITLE = "Image Saved"
INFO_INFERENCE_RUNNING_TITLE = "Inference running"
ERROR_INFERENCE_TITLE = "Inference Error"
CLEAR_PROGRESS_TEXT = "Cleared. Click preview to select image(s)."
MODEL_LOADING_PROGRESS_TEXT = "Loading YOLO model..."
MODEL_READY_WAITING_TEXT = "Model ready. Waiting for image(s)..."
MODEL_READY_STARTING_TEXT = "Model ready. Starting inference..."
NO_INPUT_SELECTED_TEXT = "No input selected. Waiting for image(s)..."
IMAGES_SELECTED_WAIT_MODEL_TEXT = "Images selected. Starting when model is ready..."
INFERENCE_ERROR_PROGRESS_TEXT = "Encountered an error. Check message and retry."
INFERENCE_RUNNING_MESSAGE = (
    "Please wait for the current inference run to finish before clearing."
)
MODEL_LOAD_ERROR_TEMPLATE = "Failed to load model: {error}"
INFERENCE_ERROR_TEMPLATE = "Inference failed for {name}: {error}"
INFERENCE_DONE_TEMPLATE = "Inference complete. Saved summary: {summary_path}"
INFERENCE_DETECTING_TEMPLATE = "Detecting image {index}/{total}: {name}"
INFERENCE_OCR_PLACEHOLDER_TEMPLATE = "OCR stage placeholder for image {index}/{total}"
SAVE_FILENAME_TEMPLATE = "detections_{timestamp}.jpg"
SUMMARY_FILENAME_TEMPLATE = "detections_{timestamp}.json"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
IMAGE_FILETYPES = [
    ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]
SAVE_FILETYPES = [
    ("JPEG image", "*.jpg"),
    ("PNG image", "*.png"),
    ("Bitmap image", "*.bmp"),
    ("TIFF image", "*.tif"),
    ("All files", "*.*"),
]
PREVIEW_OVERLAY_ALPHA = 0.28
BOX_COLOR_BGR = (204, 102, 255)
BOX_LABEL_TEXT_COLOR_BGR = (255, 255, 255)
LABEL_FONT_SCALE = 0.72
LABEL_FONT_THICKNESS = 2
OCR_PLACEHOLDER_DELAY_S = 0.05
PREVIEW_MIN_DIMENSION = 100
PREVIEW_CANVAS_BG_VALUE = 248
ROOT_BG_COLOR = "#edf1f7"
PANEL_BG_COLOR = "#ffffff"
TEXT_PRIMARY_COLOR = "#111827"
TEXT_MUTED_COLOR = "#6b7280"
PREVIEW_BORDER_BG_COLOR = "#d7deea"
PREVIEW_BORDER_COLOR = "#c3ccdb"
PREVIEW_LABEL_TEXT_COLOR = "#374151"
OVERLAY_BUTTON_BG = "#ffffff"
OVERLAY_BUTTON_ACTIVE_BG = "#f3f4f6"
OVERLAY_BUTTON_PRESSED_BG = "#e5e7eb"
OVERLAY_BUTTON_DISABLED_BG = "#f3f4f6"
OVERLAY_BUTTON_FG = "#111827"
OVERLAY_BUTTON_DISABLED_FG = "#9ca3af"
OVERLAY_BUTTON_BORDER = "#cbd5e1"
OVERLAY_BUTTON_ACTIVE_BORDER = "#94a3b8"
OVERLAY_BUTTON_DISABLED_BORDER = "#d1d5db"
PROGRESS_TROUGH_COLOR = "#e5e7eb"
PROGRESS_FILL_COLOR = "#2563eb"
TAG_LIST_SELECT_BG = "#ede9fe"
TAG_LIST_SELECT_FG = "#5b21b6"
THRESHOLD_ENABLED_BG = "#ffffff"
THRESHOLD_ENABLED_FG = "#111827"
THRESHOLD_ENABLED_TROUGH = "#93c5fd"
THRESHOLD_ENABLED_ACTIVE = "#60a5fa"
THRESHOLD_DISABLED_BG = "#f3f4f6"
THRESHOLD_DISABLED_FG = "#9ca3af"
THRESHOLD_DISABLED_TROUGH = "#d1d5db"
ACTION_LOADING_MODEL = "loading model"
ACTION_WAITING_INPUT = "waiting for input frames"
ACTION_DETECTING = "detecting price tags"
ACTION_RUNNING_OCR = "running ocr"
ACTION_IDLE = "idle"
ACTION_ORDER = [
    ACTION_LOADING_MODEL,
    ACTION_WAITING_INPUT,
    ACTION_DETECTING,
    ACTION_RUNNING_OCR,
    ACTION_IDLE,
]


@dataclass
class PriceTagDetection:
    tag_id: int
    confidence: float
    bbox: list[int]
    ocr_value: str | None = None


class InferenceGUI:
    def __init__(self, root: tk.Tk, weights_path: Path):
        self.root = root
        self.weights_path = weights_path
        self.output_dir = self._resolve_output_dir()
        self.model: YOLO | None = None
        self.model_loaded = False
        self.inference_started = False
        self.is_inferencing = False
        self.pending_start = False
        self.stop_event = threading.Event()
        self.event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        self.image_paths: list[Path] = []
        self.current_index = 0
        self.selected_tag_id: int | None = None
        self.detections_by_image: dict[Path, list[PriceTagDetection]] = {}
        self.preview_cache: dict[Path, np.ndarray] = {}
        self.summary_path: Path | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.zoom_factor = 1.0

        self.threshold_var = tk.DoubleVar(value=INITIAL_THRESHOLD)
        self.action_var = tk.StringVar(
            value=ACTION_LABEL_TEMPLATE.format(action=ACTION_ORDER[0])
        )
        self.progress_label_var = tk.StringVar(value="Progress: model initialization")
        self.image_status_var = tk.StringVar(value=NO_IMAGE_STATUS_TEXT)

        self._build_ui()
        self._set_action(
            ACTION_LOADING_MODEL, progress_text=MODEL_LOADING_PROGRESS_TEXT
        )

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Escape>", lambda _: self._on_close())

        self._start_model_loading()
        self.root.after(EVENT_QUEUE_POLL_MS, self._process_event_queue)

    def _build_ui(self) -> None:
        self.root.title(WINDOW_TITLE)
        self.root.configure(bg=ROOT_BG_COLOR)
        try:
            self.root.state("zoomed")
        except tk.TclError:
            try:
                self.root.attributes("-zoomed", True)
            except tk.TclError:
                pass

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Root.TFrame", background=ROOT_BG_COLOR)
        style.configure("Panel.TFrame", background=PANEL_BG_COLOR)
        style.configure(
            "Root.TLabel", background=ROOT_BG_COLOR, foreground=TEXT_PRIMARY_COLOR
        )
        style.configure(
            "Panel.TLabel", background=PANEL_BG_COLOR, foreground=TEXT_PRIMARY_COLOR
        )
        style.configure(
            "Muted.TLabel", background=PANEL_BG_COLOR, foreground=TEXT_MUTED_COLOR
        )
        style.configure(
            "Overlay.TButton", padding=(9, 4), relief="solid", borderwidth=1
        )
        style.map(
            "Overlay.TButton",
            background=[
                ("!disabled", OVERLAY_BUTTON_BG),
                ("active", OVERLAY_BUTTON_ACTIVE_BG),
                ("pressed", OVERLAY_BUTTON_PRESSED_BG),
                ("disabled", OVERLAY_BUTTON_DISABLED_BG),
            ],
            foreground=[
                ("!disabled", OVERLAY_BUTTON_FG),
                ("disabled", OVERLAY_BUTTON_DISABLED_FG),
            ],
            bordercolor=[
                ("!disabled", OVERLAY_BUTTON_BORDER),
                ("active", OVERLAY_BUTTON_ACTIVE_BORDER),
                ("disabled", OVERLAY_BUTTON_DISABLED_BORDER),
            ],
        )
        style.configure(
            "Light.Horizontal.TProgressbar",
            troughcolor=PROGRESS_TROUGH_COLOR,
            bordercolor=PROGRESS_TROUGH_COLOR,
            background=PROGRESS_FILL_COLOR,
            lightcolor=PROGRESS_FILL_COLOR,
            darkcolor=PROGRESS_FILL_COLOR,
        )

        main_frame = ttk.Frame(self.root, style="Root.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=0)
        main_frame.columnconfigure(0, weight=LEFT_COLUMN_WEIGHT, uniform="main_cols")
        main_frame.columnconfigure(1, weight=RIGHT_COLUMN_WEIGHT, uniform="main_cols")

        left_frame = ttk.Frame(main_frame, style="Panel.TFrame")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=(8, 4))
        left_frame.rowconfigure(0, weight=1)
        left_frame.rowconfigure(1, weight=0)
        left_frame.columnconfigure(0, weight=1)

        self.preview_container = tk.Frame(
            left_frame,
            bg=PREVIEW_BORDER_BG_COLOR,
            highlightbackground=PREVIEW_BORDER_COLOR,
            highlightthickness=1,
            bd=0,
        )
        self.preview_container.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))
        self.preview_container.grid_propagate(False)
        self.preview_container.pack_propagate(False)

        self.preview_label = tk.Label(
            self.preview_container,
            bg=PANEL_BG_COLOR,
            fg=PREVIEW_LABEL_TEXT_COLOR,
            text=PREVIEW_PLACEHOLDER_TEXT,
            anchor="center",
            font=("Segoe UI", 14),
            cursor="hand2",
        )
        self.preview_label.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.preview_label.bind("<Button-1>", self._on_preview_click)
        self.preview_label.bind("<MouseWheel>", self._on_mouse_wheel_zoom)
        self.preview_label.bind("<Button-4>", self._on_mouse_wheel_zoom)
        self.preview_label.bind("<Button-5>", self._on_mouse_wheel_zoom)
        self.preview_label.bind("<Configure>", lambda _: self._render_current_preview())

        controls_row = ttk.Frame(left_frame, style="Panel.TFrame")
        controls_row.grid(row=1, column=0, sticky="ew", padx=10, pady=(4, 8))
        controls_row.columnconfigure(0, weight=1)
        controls_row.columnconfigure(1, weight=0)
        controls_row.columnconfigure(2, weight=1)

        self.clear_button = ttk.Button(
            controls_row,
            text="Clear",
            command=self._clear_images,
            style="Overlay.TButton",
            width=7,
        )
        self.clear_button.grid(row=0, column=0, sticky="w")

        nav_buttons = ttk.Frame(controls_row, style="Panel.TFrame")
        nav_buttons.grid(row=0, column=1, sticky="n")

        self.prev_button = ttk.Button(
            nav_buttons,
            text="Prev",
            command=self._show_prev_image,
            style="Overlay.TButton",
            width=6,
        )
        self.prev_button.grid(row=0, column=0, padx=(0, 12), pady=0)

        self.next_button = ttk.Button(
            nav_buttons,
            text="Next",
            command=self._show_next_image,
            style="Overlay.TButton",
            width=6,
        )
        self.next_button.grid(row=0, column=2, padx=(12, 0), pady=0)

        self.save_button = ttk.Button(
            nav_buttons,
            text="Save",
            command=self._save_current_preview,
            style="Overlay.TButton",
            width=7,
        )
        self.save_button.grid(row=0, column=1, padx=12, pady=0)

        self.image_status_label = ttk.Label(
            controls_row,
            textvariable=self.image_status_var,
            style="Muted.TLabel",
            font=("Segoe UI", 10),
        )
        self.image_status_label.grid(row=0, column=2, sticky="e")

        right_frame = ttk.Frame(main_frame, style="Panel.TFrame")
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=(8, 4))
        right_frame.rowconfigure(1, weight=1)
        right_frame.columnconfigure(0, weight=1)

        ttk.Label(
            right_frame,
            text="Select a tag to isolate one box.",
            style="Muted.TLabel",
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 6))

        list_frame = ttk.Frame(right_frame, style="Panel.TFrame")
        list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        self.tag_listbox = tk.Listbox(
            list_frame,
            exportselection=False,
            activestyle="none",
            bg=PANEL_BG_COLOR,
            fg=TEXT_PRIMARY_COLOR,
            selectbackground=TAG_LIST_SELECT_BG,
            selectforeground=TAG_LIST_SELECT_FG,
            font=("Segoe UI", 10),
            relief=tk.SOLID,
            bd=1,
        )
        self.tag_listbox.grid(row=0, column=0, sticky="nsew")
        self.tag_listbox.bind("<<ListboxSelect>>", self._on_tag_select)

        list_scroll = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self.tag_listbox.yview
        )
        list_scroll.grid(row=0, column=1, sticky="ns")
        self.tag_listbox.configure(yscrollcommand=list_scroll.set)

        status_frame = ttk.Frame(main_frame, style="Panel.TFrame")
        status_frame.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=(2, 4))
        status_frame.columnconfigure(0, weight=1)

        ttk.Label(
            status_frame, textvariable=self.action_var, style="Panel.TLabel"
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(4, 1))
        ttk.Label(
            status_frame, textvariable=self.progress_label_var, style="Panel.TLabel"
        ).grid(row=1, column=0, sticky="w", padx=10, pady=(0, 3))

        self.progress_bar = ttk.Progressbar(
            status_frame,
            style="Light.Horizontal.TProgressbar",
            mode="indeterminate",
        )
        self.progress_bar.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 4))
        self.progress_bar.start(14)

        threshold_frame = ttk.Frame(main_frame, style="Panel.TFrame")
        threshold_frame.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=(2, 4))
        threshold_frame.columnconfigure(0, weight=1)

        ttk.Label(
            threshold_frame, text="Confidence Threshold", style="Panel.TLabel"
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(4, 1))

        self.threshold_value_label = ttk.Label(
            threshold_frame,
            text=f"{self.threshold_var.get():.1f}",
            style="Panel.TLabel",
        )
        self.threshold_value_label.grid(
            row=1, column=0, sticky="w", padx=10, pady=(0, 1)
        )

        self.threshold_scale = tk.Scale(
            threshold_frame,
            from_=0.0,
            to=1.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            showvalue=False,
            command=self._on_threshold_change,
            bg=THRESHOLD_ENABLED_BG,
            fg=THRESHOLD_ENABLED_FG,
            troughcolor=THRESHOLD_DISABLED_TROUGH,
            highlightthickness=0,
            bd=0,
            activebackground=THRESHOLD_ENABLED_ACTIVE,
        )
        self.threshold_scale.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 4))
        self._update_threshold_visual_state(enabled=True)
        self._update_navigation_controls()

    def _start_model_loading(self) -> None:
        threading.Thread(target=self._load_model_worker, daemon=True).start()

    def _load_model_worker(self) -> None:
        try:
            model = YOLO(str(self.weights_path))
        except Exception as exc:  # noqa: BLE001
            self.event_queue.put(("error", MODEL_LOAD_ERROR_TEMPLATE.format(error=exc)))
            return
        self.event_queue.put(("model_loaded", model))

    def _set_action(
        self,
        action: str,
        *,
        progress_text: str = "",
        progress_value: float | None = None,
        indeterminate: bool = False,
    ) -> None:
        self.action_var.set(ACTION_LABEL_TEMPLATE.format(action=action))
        if progress_text:
            self.progress_label_var.set(progress_text)
        if indeterminate:
            if self.progress_bar["mode"] != "indeterminate":
                self.progress_bar.configure(mode="indeterminate")
                self.progress_bar.start(14)
        else:
            if self.progress_bar["mode"] != "determinate":
                self.progress_bar.stop()
                self.progress_bar.configure(mode="determinate")
            if progress_value is not None:
                self.progress_bar["value"] = max(0.0, min(100.0, progress_value))

    def _on_threshold_change(self, _value: str) -> None:
        value = round(self.threshold_var.get(), 1)
        self.threshold_var.set(value)
        self.threshold_value_label.configure(text=f"{value:.1f}")

    def _update_threshold_visual_state(self, *, enabled: bool) -> None:
        if enabled:
            self.threshold_scale.configure(
                state=tk.NORMAL,
                bg=THRESHOLD_ENABLED_BG,
                fg=THRESHOLD_ENABLED_FG,
                troughcolor=THRESHOLD_ENABLED_TROUGH,
                activebackground=THRESHOLD_ENABLED_ACTIVE,
            )
        else:
            self.threshold_scale.configure(
                state=tk.DISABLED,
                bg=THRESHOLD_DISABLED_BG,
                fg=THRESHOLD_DISABLED_FG,
                troughcolor=THRESHOLD_DISABLED_TROUGH,
                activebackground=THRESHOLD_DISABLED_TROUGH,
            )

    def _set_preview_placeholder(self, text: str) -> None:
        self.preview_photo = None
        self.preview_label.configure(image="", text=text)

    def _set_tag_list_items(
        self, detections: list[PriceTagDetection], *, include_all_option: bool
    ) -> None:
        self.tag_listbox.delete(0, tk.END)
        if include_all_option:
            self.tag_listbox.insert(
                tk.END, ALL_TAGS_ITEM_TEMPLATE.format(count=len(detections))
            )
        for det in detections:
            self.tag_listbox.insert(
                tk.END,
                TAG_ITEM_TEMPLATE.format(
                    tag_id=det.tag_id, details=self._format_detection_text(det)
                ),
            )

    def _set_images(self, images: list[Path]) -> None:
        self.image_paths = images
        self.current_index = 0
        self.selected_tag_id = None
        self.zoom_factor = 1.0
        self.detections_by_image.clear()
        self.preview_cache.clear()
        self._set_tag_list_items([], include_all_option=bool(images))
        self.tag_listbox.selection_clear(0, tk.END)
        if images:
            self.tag_listbox.selection_set(0)
        self._update_image_status()
        self._render_current_preview()

    def _resolve_selected_images(self, file_paths: tuple[str, ...]) -> list[Path]:
        selected: list[Path] = []
        for path_str in file_paths:
            path = Path(path_str).expanduser()
            if path.is_file():
                selected.append(path)
        return selected

    def _select_images(self) -> None:
        file_paths = filedialog.askopenfilenames(
            title=SELECT_DIALOG_TITLE,
            filetypes=IMAGE_FILETYPES,
        )
        if not file_paths:
            if not self.image_paths:
                self._set_action(
                    ACTION_WAITING_INPUT,
                    progress_text=NO_INPUT_SELECTED_TEXT,
                    progress_value=0,
                )
            return

        selected = self._resolve_selected_images(file_paths)

        if not selected:
            messagebox.showwarning(
                WARNING_NO_VALID_FILES_TITLE, WARNING_NO_VALID_FILES_MESSAGE
            )
            return

        self._set_images(selected)
        self._update_threshold_visual_state(enabled=not self.inference_started)

        if self.model_loaded and not self.is_inferencing:
            self._start_inference()
        elif not self.model_loaded:
            self.pending_start = True
            self._set_action(
                ACTION_LOADING_MODEL,
                progress_text=IMAGES_SELECTED_WAIT_MODEL_TEXT,
                indeterminate=True,
            )

    def _start_inference(self) -> None:
        if not self.image_paths or self.model is None or self.is_inferencing:
            return
        self.pending_start = False
        self.is_inferencing = True
        self.inference_started = True
        self._update_threshold_visual_state(enabled=False)
        self._update_navigation_controls()
        confidence = float(self.threshold_var.get())

        worker = threading.Thread(
            target=self._run_inference_worker,
            args=(list(self.image_paths), confidence),
            daemon=True,
        )
        worker.start()

    def _run_inference_worker(
        self, image_paths: list[Path], conf_threshold: float
    ) -> None:
        detections_payload: dict[str, list[dict[str, Any]]] = {}
        total = len(image_paths)

        for idx, image_path in enumerate(image_paths, start=1):
            if self.stop_event.is_set():
                return

            self.event_queue.put(
                (
                    "action",
                    {
                        "action": ACTION_DETECTING,
                        "text": INFERENCE_DETECTING_TEMPLATE.format(
                            index=idx, total=total, name=image_path.name
                        ),
                        "progress": ((idx - 1) / total) * 100,
                    },
                )
            )

            try:
                result = self.model.predict(
                    source=str(image_path),
                    conf=conf_threshold,
                    show=False,
                    save=False,
                    verbose=False,
                )[0]
            except Exception as exc:  # noqa: BLE001
                self.event_queue.put(
                    (
                        "error",
                        INFERENCE_ERROR_TEMPLATE.format(
                            name=image_path.name, error=exc
                        ),
                    )
                )
                continue

            detections: list[PriceTagDetection] = []
            for det_index, box in enumerate(result.boxes, start=1):
                coords = [int(c) for c in box.xyxy[0].tolist()]
                confidence = float(box.conf[0].item())
                detections.append(
                    PriceTagDetection(
                        tag_id=det_index,
                        confidence=confidence,
                        bbox=coords,
                        ocr_value=None,
                    )
                )

            detections_payload[str(image_path)] = [asdict(d) for d in detections]
            self.event_queue.put(
                ("result", {"path": image_path, "detections": detections})
            )

            self.event_queue.put(
                (
                    "action",
                    {
                        "action": ACTION_RUNNING_OCR,
                        "text": INFERENCE_OCR_PLACEHOLDER_TEMPLATE.format(
                            index=idx, total=total
                        ),
                        "progress": (idx / total) * 100,
                    },
                )
            )
            time.sleep(OCR_PLACEHOLDER_DELAY_S)

        summary_path = self._write_summary(detections_payload, conf_threshold)
        self.event_queue.put(("done", summary_path))

    def _write_summary(
        self, detections_payload: dict[str, list[dict[str, Any]]], threshold: float
    ) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        summary_path = self.output_dir / SUMMARY_FILENAME_TEMPLATE.format(
            timestamp=timestamp
        )
        summary_data = {
            "created_at": datetime.now().isoformat(),
            "threshold": round(threshold, 1),
            "images": detections_payload,
        }
        summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
        return summary_path

    def _process_event_queue(self) -> None:
        while not self.event_queue.empty():
            event, payload = self.event_queue.get()
            if event == "model_loaded":
                self.model = payload
                self.model_loaded = True
                if self.pending_start and self.image_paths:
                    self._set_action(
                        ACTION_WAITING_INPUT,
                        progress_text=MODEL_READY_STARTING_TEXT,
                        progress_value=0,
                    )
                    self._start_inference()
                elif self.image_paths and not self.is_inferencing:
                    self._start_inference()
                else:
                    self._set_action(
                        ACTION_WAITING_INPUT,
                        progress_text=MODEL_READY_WAITING_TEXT,
                        progress_value=0,
                    )
            elif event == "action":
                action_name = payload["action"]
                self._set_action(
                    action_name,
                    progress_text=payload["text"],
                    progress_value=payload["progress"],
                )
            elif event == "result":
                path = payload["path"]
                detections = payload["detections"]
                self.detections_by_image[path] = detections
                if self._current_image_path() == path:
                    self._refresh_tag_list_for_current_image()
                    self._render_current_preview()
            elif event == "done":
                self.summary_path = payload
                self.is_inferencing = False
                summary_name = (
                    self._to_relative_display_path(payload) if payload else "N/A"
                )
                self._set_action(
                    ACTION_IDLE,
                    progress_text=INFERENCE_DONE_TEMPLATE.format(
                        summary_path=summary_name
                    ),
                    progress_value=100,
                )
                self._update_navigation_controls()
                self._refresh_tag_list_for_current_image()
                self._render_current_preview()
            elif event == "error":
                self.is_inferencing = False
                self._set_action(
                    ACTION_IDLE,
                    progress_text=INFERENCE_ERROR_PROGRESS_TEXT,
                    progress_value=0,
                )
                self._update_navigation_controls()
                messagebox.showerror(ERROR_INFERENCE_TITLE, str(payload))

        self.root.after(EVENT_QUEUE_POLL_MS, self._process_event_queue)

    def _current_image_path(self) -> Path | None:
        if not self.image_paths:
            return None
        if self.current_index < 0 or self.current_index >= len(self.image_paths):
            return None
        return self.image_paths[self.current_index]

    def _read_image(self, image_path: Path) -> np.ndarray | None:
        if image_path in self.preview_cache:
            return self.preview_cache[image_path]

        buffer = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            return None
        self.preview_cache[image_path] = image
        return image

    def _get_active_detections(self, image_path: Path) -> list[PriceTagDetection]:
        detections = self.detections_by_image.get(image_path, [])
        if self.selected_tag_id is None:
            return detections
        return [d for d in detections if d.tag_id == self.selected_tag_id]

    def _format_detection_text(self, det: PriceTagDetection) -> str:
        if det.ocr_value is None or det.ocr_value == "":
            return f"Conf: {det.confidence:.2f}"
        return f"Conf: {det.confidence:.2f} | OCR: {det.ocr_value}"

    def _compose_annotated_image(self, image_path: Path) -> np.ndarray | None:
        image = self._read_image(image_path)
        if image is None:
            return None

        draw_image = image.copy()
        detections = self._get_active_detections(image_path)

        if detections:
            dimmed = cv2.convertScaleAbs(
                draw_image, alpha=PREVIEW_OVERLAY_ALPHA, beta=0
            )
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                x1 = max(0, min(x1, draw_image.shape[1] - 1))
                y1 = max(0, min(y1, draw_image.shape[0] - 1))
                x2 = max(x1 + 1, min(x2, draw_image.shape[1]))
                y2 = max(y1 + 1, min(y2, draw_image.shape[0]))
                dimmed[y1:y2, x1:x2] = draw_image[y1:y2, x1:x2]
            draw_image = dimmed

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(draw_image, (x1, y1), (x2, y2), BOX_COLOR_BGR, 2)

            label_text = self._format_detection_text(det)
            (text_w, text_h), _ = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                LABEL_FONT_SCALE,
                LABEL_FONT_THICKNESS,
            )
            center_x = x1 + ((x2 - x1) // 2)
            text_x = max(
                4, min(center_x - (text_w // 2), draw_image.shape[1] - text_w - 4)
            )
            text_y = max(text_h + 6, y1 - 10)
            cv2.putText(
                draw_image,
                label_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                LABEL_FONT_SCALE,
                BOX_LABEL_TEXT_COLOR_BGR,
                LABEL_FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return draw_image

    def _render_current_preview(self) -> None:
        current_path = self._current_image_path()
        if current_path is None:
            self._set_preview_placeholder(PREVIEW_PLACEHOLDER_TEXT)
            return

        composed = self._compose_annotated_image(current_path)
        if composed is None:
            self._set_preview_placeholder(
                PREVIEW_READ_ERROR_TEMPLATE.format(name=current_path.name)
            )
            return

        frame_width = max(PREVIEW_MIN_DIMENSION, self.preview_label.winfo_width())
        frame_height = max(PREVIEW_MIN_DIMENSION, self.preview_label.winfo_height())
        image_height, image_width = composed.shape[:2]
        fit_scale = min(frame_width / image_width, frame_height / image_height)
        draw_scale = fit_scale * self.zoom_factor
        scaled_w = max(1, int(image_width * draw_scale))
        scaled_h = max(1, int(image_height * draw_scale))

        resized = cv2.resize(
            composed, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC
        )
        canvas = np.full(
            (frame_height, frame_width, 3), PREVIEW_CANVAS_BG_VALUE, dtype=np.uint8
        )

        offset_x = (frame_width - scaled_w) // 2
        offset_y = (frame_height - scaled_h) // 2

        src_x1 = max(0, -offset_x)
        src_y1 = max(0, -offset_y)
        dst_x1 = max(0, offset_x)
        dst_y1 = max(0, offset_y)

        copy_w = min(scaled_w - src_x1, frame_width - dst_x1)
        copy_h = min(scaled_h - src_y1, frame_height - dst_y1)
        if copy_w > 0 and copy_h > 0:
            canvas[dst_y1 : dst_y1 + copy_h, dst_x1 : dst_x1 + copy_w] = resized[
                src_y1 : src_y1 + copy_h, src_x1 : src_x1 + copy_w
            ]

        rgb_image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        self.preview_photo = ImageTk.PhotoImage(pil_image, master=self.root)
        self.preview_label.configure(image=self.preview_photo, text="")
        self.preview_label.image = self.preview_photo

    def _refresh_tag_list_for_current_image(self) -> None:
        current_path = self._current_image_path()
        detections = (
            self.detections_by_image.get(current_path, []) if current_path else []
        )

        self._set_tag_list_items(
            detections, include_all_option=current_path is not None
        )

        if current_path is None:
            self.tag_listbox.selection_clear(0, tk.END)
            return

        if self.selected_tag_id is None:
            self.tag_listbox.selection_set(0)
        else:
            index = min(self.selected_tag_id, len(detections))
            self.tag_listbox.selection_set(index)

    def _on_tag_select(self, _event: tk.Event) -> None:
        if not self.tag_listbox.curselection():
            return
        selected_index = self.tag_listbox.curselection()[0]
        if selected_index == 0:
            self.selected_tag_id = None
        else:
            self.selected_tag_id = selected_index
        self._render_current_preview()

    def _on_preview_click(self, _event: tk.Event) -> None:
        if not self.image_paths:
            self._select_images()

    def _adjust_zoom(self, delta: float) -> None:
        if not self.image_paths:
            return
        new_zoom = max(ZOOM_MIN_FACTOR, min(ZOOM_MAX_FACTOR, self.zoom_factor + delta))
        self.zoom_factor = round(new_zoom, 2)
        self._render_current_preview()

    def _on_mouse_wheel_zoom(self, event: tk.Event) -> None:
        if not self.image_paths:
            return
        delta = 0.0
        if hasattr(event, "delta") and event.delta:
            delta = ZOOM_STEP if event.delta > 0 else -ZOOM_STEP
        elif getattr(event, "num", None) == 4:
            delta = ZOOM_STEP
        elif getattr(event, "num", None) == 5:
            delta = -ZOOM_STEP
        if delta != 0.0:
            self._adjust_zoom(delta)

    def _clear_images(self) -> None:
        if self.is_inferencing:
            messagebox.showinfo(INFO_INFERENCE_RUNNING_TITLE, INFERENCE_RUNNING_MESSAGE)
            return

        self.pending_start = False
        self.inference_started = False
        self._set_images([])
        self.image_status_var.set(NO_IMAGE_STATUS_TEXT)
        self._update_threshold_visual_state(enabled=True)
        if self.model_loaded:
            self._set_action(
                ACTION_WAITING_INPUT,
                progress_text=CLEAR_PROGRESS_TEXT,
                progress_value=0,
            )

    def _update_navigation_controls(self) -> None:
        has_images = bool(self.image_paths)
        browse_cursor = "arrow" if has_images else "hand2"
        try:
            self.preview_label.configure(cursor=browse_cursor)
        except tk.TclError:
            # Recover from a stale Tk image handle by clearing and redrawing.
            self._set_preview_placeholder(PREVIEW_PLACEHOLDER_TEXT)
            self._render_current_preview()

        nav_state = tk.NORMAL if has_images else tk.DISABLED
        self.prev_button.configure(state=nav_state)
        self.next_button.configure(state=nav_state)
        self.save_button.configure(state=nav_state)
        clear_state = (
            tk.NORMAL if has_images and not self.is_inferencing else tk.DISABLED
        )
        self.clear_button.configure(state=clear_state)

    def _update_image_status(self) -> None:
        if not self.image_paths:
            self.image_status_var.set(NO_IMAGE_STATUS_TEXT)
            self._update_navigation_controls()
            return
        current = self.current_index + 1
        total = len(self.image_paths)
        self.image_status_var.set(f"Image {current}/{total}")
        self._update_navigation_controls()

    def _show_prev_image(self) -> None:
        if not self.image_paths:
            return
        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self.selected_tag_id = None
        self._update_image_status()
        self._refresh_tag_list_for_current_image()
        self._render_current_preview()

    def _show_next_image(self) -> None:
        if not self.image_paths:
            return
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.selected_tag_id = None
        self._update_image_status()
        self._refresh_tag_list_for_current_image()
        self._render_current_preview()

    def _save_current_preview(self) -> None:
        current_path = self._current_image_path()
        if current_path is None:
            messagebox.showwarning(WARNING_NO_IMAGE_TITLE, WARNING_NO_IMAGE_MESSAGE)
            return

        composed = self._compose_annotated_image(current_path)
        if composed is None:
            messagebox.showerror(ERROR_SAVE_FAILED_TITLE, ERROR_IMAGE_LOAD_MESSAGE)
            return

        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        save_target = filedialog.asksaveasfilename(
            title=SAVE_DIALOG_TITLE,
            initialfile=SAVE_FILENAME_TEMPLATE.format(timestamp=timestamp),
            defaultextension=".jpg",
            filetypes=SAVE_FILETYPES,
        )
        if not save_target:
            return

        save_path = Path(save_target).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        success = self._write_image_file(save_path, composed)

        if success:
            messagebox.showinfo(
                INFO_IMAGE_SAVED_TITLE, f"Preview saved to:\n{save_path.resolve()}"
            )
        else:
            messagebox.showerror(ERROR_SAVE_FAILED_TITLE, ERROR_SAVE_FAILED_MESSAGE)

    def _write_image_file(self, target_path: Path, image: np.ndarray) -> bool:
        suffix = target_path.suffix.lower() or ".jpg"
        if suffix == ".jpeg":
            suffix = ".jpg"
        success, encoded = cv2.imencode(suffix, image)
        if not success:
            return False
        encoded.tofile(str(target_path))
        return True

    def _resolve_output_dir(self) -> Path:
        return (Path.cwd() / OUTPUT_RELATIVE_DIR).resolve()

    def _to_relative_display_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    def _on_close(self) -> None:
        self.stop_event.set()
        self.root.destroy()


def _enable_high_dpi_support() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:  # noqa: BLE001
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    if not WEIGHTS_PATH.is_file():
        raise SystemExit(f"Model weights not found: {WEIGHTS_PATH.resolve()}")

    _enable_high_dpi_support()
    root = tk.Tk()
    InferenceGUI(root, WEIGHTS_PATH)
    root.mainloop()


if __name__ == "__main__":
    main()
