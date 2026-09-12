"""Shared figure style for the manuscript.

Palette: the validated six-slot categorical default from the data-visualisation reference
(worst adjacent CVD dE 9.1, normal-vision dE 19.6; the three sub-3:1 slots are always shipped
with direct labels or a legend, which is the documented relief for the contrast WARN).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

FIGS = Path(__file__).resolve().parents[1] / "figures"
FIGS.mkdir(exist_ok=True)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
VIOLET, RED = "#4a3aa7", "#e34948"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS = dict(good="#0ca30c", warning="#fab219", serious="#ec835a", critical="#d03b3b")
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
SURFACE, GRID = "#fcfcfb", "#e6e5e1"

mpl.rcParams.update({
    "figure.dpi": 110, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 8.2,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlesize": 9.0, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.titlepad": 6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.7, "legend.frameon": False,
    "legend.fontsize": 7.4, "lines.linewidth": 2.0, "lines.markersize": 4.0,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, zorder=0)
    ax.set_axisbelow(True)


def panel(ax, letter, title=None, dx=0.0, dy=1.0):
    """Panel letter on the axes' own left edge, with the title set just to its right on the same
    baseline. Placing the letter outside the axes collided with y-axis labels of varying width."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9.6, fontweight="bold",
            color=INK, va="bottom", ha="left")
    if title:
        ax.set_title(title, loc="left", x=0.068)


def legend(ax, loc="best", ncol=1, **kw):
    """Legend with a surface-coloured backing, so it never sits illegibly on top of marks."""
    kw.setdefault("handlelength", 1.6)
    kw.setdefault("borderpad", 0.4)
    lg = ax.legend(loc=loc, ncol=ncol, frameon=True, framealpha=0.92, facecolor=SURFACE,
                   edgecolor="none", **kw)
    lg.set_zorder(20)
    return lg


def trim(labels, n=34):
    """Shorten long category labels without silently truncating mid-word."""
    out = []
    for s in labels:
        s = str(s)
        if len(s) <= n:
            out.append(s)
        else:
            cut = s[:n].rsplit(" ", 1)[0]
            out.append(cut + "\u2026")
    return out


def note(fig, text, y=-0.01, width=None):
    """Figure footnote, wrapped explicitly. Matplotlib's own wrap=True is measured against the
    figure width but the saved bounding box is computed from the UNwrapped extent, which silently
    widened every figure it was used on and pushed the right-hand panels past the page edge."""
    import textwrap
    w = width or int(fig.get_size_inches()[0] * 27)
    body = "\n".join(textwrap.fill(p, w) for p in text.split("\n"))
    fig.text(0.007, y, body, fontsize=6.6, color=MUTED, va="top", ha="left")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> figures/{name}.pdf + .png")
