"""Thin Tkinter front-end for background removal.

All application state lives in :class:`GuiController`; this module only
builds widgets and forwards button clicks, so importing it never creates
a window and the whole behaviour stays testable headlessly.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

from bgremover.gui_controller import GuiController
from bgremover.pipeline import STAGE_KEYS

__all__ = ["BackgroundRemoverApp", "launch"]

_PREVIEW_SIDE = 360


def _photo(rgb: np.ndarray) -> ImageTk.PhotoImage:
    """Convert an RGB preview into a Tk photo image."""
    picture = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    picture.thumbnail((_PREVIEW_SIDE, _PREVIEW_SIDE))
    return ImageTk.PhotoImage(picture)


class BackgroundRemoverApp:
    """The main window, bound to a :class:`GuiController`.

    Parameters
    ----------
    root:
        The Tk root window (created by :func:`launch`).
    controller:
        Application logic; a default controller is built when omitted.
    """

    def __init__(
        self,
        root: tk.Tk,
        controller: GuiController | None = None,
    ) -> None:
        """Build every widget of the main window."""
        self.root = root
        self.controller = (
            controller if controller is not None else GuiController()
        )
        self.root.title("Background Remover")
        self.status = tk.StringVar(value=self.controller.summary())

        top = tk.Frame(root)
        top.pack(side=tk.TOP, fill=tk.X)
        for label in ("Open", "Remove background", "Save RGBA", "Quit"):
            button = tk.Button(top, text=label)
            button.pack(side=tk.LEFT, padx=4, pady=4)
            if label == "Open":
                button.configure(command=self.open_image)
            elif label == "Remove background":
                button.configure(command=self.remove_background)
            elif label == "Save RGBA":
                button.configure(command=self.save_rgba)
            else:
                button.configure(command=root.destroy)

        views = tk.Frame(root)
        views.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.before = tk.Label(views, text="input")
        self.before.pack(side=tk.LEFT, padx=6, pady=6)
        self.after = tk.Label(views, text="result")
        self.after.pack(side=tk.LEFT, padx=6, pady=6)
        self.stages = tk.Listbox(views, height=12, exportselection=False)
        self.stages.pack(side=tk.LEFT, padx=6, pady=6, fill=tk.Y)
        self.stages.bind("<<ListboxSelect>>", self.show_stage)

        status = tk.Label(root, textvariable=self.status, anchor=tk.W)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        self._photos: list[ImageTk.PhotoImage] = []

    def _show(self, label: tk.Label, rgb: np.ndarray) -> None:
        """Display an RGB preview inside a label widget."""
        photo = _photo(rgb)
        self._photos.append(photo)
        label.configure(image=photo)

    def _refresh_status(self) -> None:
        """Push the controller summary into the status bar."""
        self.status.set(self.controller.summary())

    def open_image(self) -> None:
        """Ask for a file, load it and show the input preview."""
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff "
                        "*.bmp")],
        )
        if not path:
            return
        try:
            image = self.controller.load(path)
        except (FileNotFoundError, ValueError) as exc:
            messagebox.showerror("Open failed", str(exc))
            return
        self.stages.delete(0, tk.END)
        self._show(self.before, image)
        self.after.configure(image="", text="result")
        self._refresh_status()

    def remove_background(self) -> None:
        """Run the pipeline and show the composited result."""
        try:
            self.controller.run()
        except RuntimeError as exc:
            messagebox.showwarning("Nothing to do", str(exc))
            return
        self._show(self.after, self.controller.preview_rgba())
        self.stages.delete(0, tk.END)
        for key in STAGE_KEYS:
            self.stages.insert(tk.END, key)
        self._refresh_status()

    def show_stage(self, _event: object = None) -> None:
        """Show the listbox-selected intermediate stage."""
        if not self.controller.has_result():
            return
        picked = self.stages.curselection()
        if not picked:
            return
        name = self.stages.get(picked[0])
        self._show(self.after, self.controller.preview_stage(name))

    def save_rgba(self) -> None:
        """Ask for a file and save the RGBA result there."""
        if not self.controller.has_result():
            messagebox.showwarning(
                "Nothing to do", "remove the background first"
            )
            return
        path = filedialog.asksaveasfilename(
            title="Save RGBA",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tif *.tiff")],
        )
        if not path:
            return
        try:
            self.controller.save_rgba(path)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self._refresh_status()


def launch(controller: GuiController | None = None) -> None:
    """Create the main window and enter the Tk event loop.

    Parameters
    ----------
    controller:
        Application logic; a default controller is built when omitted.
    """
    root = tk.Tk()
    BackgroundRemoverApp(root, controller=controller)
    root.mainloop()
