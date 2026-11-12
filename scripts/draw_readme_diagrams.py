"""Usage: python scripts/draw_readme_diagrams.py [OUT_DIR]

Draws the README's architecture and continuous-training diagrams as SVG, light and dark.

Nodes sit on a grid chosen so no two edges cross. Color says where a part runs.
"""

# SVG markup and the diagrams' descriptions read better unwrapped.
# ruff: noqa: E501

import sys
from html import escape
from pathlib import Path

OUT = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(__file__).resolve().parents[1] / "docs" / "images" / "readme"
)
DATE = "2026-11-11"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"

THEMES = {
    "light": dict(
        halo="#ffffff",
        text="#1f2328",
        muted="#59636e",
        edge="#6e7781",
        ext=("#f6f8fa", "#d1d9e0", "#59636e"),
        k3s=("#ddf4ff", "#54aeff", "#0969da"),
        data=("#fff8c5", "#d4a72c", "#9a6700"),
        serve=("#fbefff", "#c297ff", "#8250df"),
        good=("#dafbe1", "#4ac26b", "#1a7f37"),
        bad=("#ffebe9", "#ff8182", "#cf222e"),
    ),
    "dark": dict(
        halo="#0d1117",
        text="#f0f6fc",
        muted="#9198a1",
        edge="#8b949e",
        ext=("#151b23", "#3d444d", "#9198a1"),
        k3s=("#0f2342", "#1f6feb", "#4493f8"),
        data=("#2b2210", "#9e6a03", "#d29922"),
        serve=("#21173b", "#8957e5", "#ab7df8"),
        good=("#0f2a1a", "#238636", "#3fb950"),
        bad=("#2d1215", "#da3633", "#f85149"),
    ),
}

W_NODE, H_NODE = 180, 64


class Canvas:
    def __init__(self, width, height, theme):
        self.w, self.h, self.t = width, height, THEMES[theme]
        self.parts = []
        self.nodes = {}

    # ---- nodes -------------------------------------------------------------------------------
    def node(self, key, x, y, kind, label, title, sub, shape="box", w=W_NODE, h=H_NODE):
        fill, stroke, accent = self.t[kind]
        self.nodes[key] = (x, y, w, h)
        if shape == "cylinder":
            ry = 7
            body = (
                f"M{x},{y + ry} V{y + h - ry} A{w / 2},{ry} 0 0 0 {x + w},{y + h - ry} "
                f"V{y + ry} A{w / 2},{ry} 0 0 0 {x},{y + ry} Z"
            )
            self.parts.append(
                f'<path data-node="{key}" d="{body}" fill="{fill}" stroke="{stroke}" stroke-width="1.25"/>'
            )
            self.parts.append(
                f'<ellipse cx="{x + w / 2}" cy="{y + ry}" rx="{w / 2}" ry="{ry}" fill="{fill}" stroke="{stroke}" stroke-width="1.25"/>'
            )
            dy = 5
        else:
            rx = h / 2 if shape == "pill" else 8
            width = 2.25 if shape == "decision" else 1.25
            self.parts.append(
                f'<rect data-node="{key}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
            )
            dy = 0
        cx = x + w / 2
        lines = []
        if label:
            lines.append((y + 19 + dy, "label", label, accent))
        lines.append((y + (37 if label else 30) + dy, "title", title, self.t["text"]))
        if sub:
            lines.append((y + (53 if label else 47) + dy, "sub", sub, self.t["muted"]))
        for ty, cls, text, color in lines:
            self.parts.append(
                f'<text x="{cx}" y="{ty}" class="{cls}" fill="{color}" data-box="{key}">{escape(text)}</text>'
            )

    def side(self, key, where, offset=0.0):
        x, y, w, h = self.nodes[key]
        return {
            "l": (x, y + h / 2 + offset),
            "r": (x + w, y + h / 2 + offset),
            "t": (x + w / 2 + offset, y),
            "b": (x + w / 2 + offset, y + h),
        }[where]

    # ---- edges -------------------------------------------------------------------------------
    def edge(self, points, label=None, at=None, anchor="middle", dashed=False, both=False):
        d = "M" + " L".join(f"{px},{py}" for px, py in points)
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        start = ' marker-start="url(#arrow-start)"' if both else ""
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{self.t["edge"]}" stroke-width="1.4"{dash}{start} marker-end="url(#arrow)"/>'
        )
        if label:
            lx, ly = at or (
                (points[0][0] + points[-1][0]) / 2,
                (points[0][1] + points[-1][1]) / 2 + 4,
            )
            self.parts.append(
                f'<text x="{lx}" y="{ly}" class="edge" text-anchor="{anchor}" fill="{self.t["muted"]}" '
                f'stroke="{self.t["halo"]}" stroke-width="4" paint-order="stroke">{escape(label)}</text>'
            )

    def legend(self, y, items, lines=()):
        x = 20
        for kind, text in items:
            fill, stroke, _ = self.t[kind]
            self.parts.append(
                f'<rect x="{x}" y="{y - 10}" width="14" height="14" rx="3" fill="{fill}" stroke="{stroke}"/>'
            )
            self.parts.append(
                f'<text x="{x + 20}" y="{y + 1}" class="legend" fill="{self.t["muted"]}">{escape(text)}</text>'
            )
            x += 26 + len(text) * 6.4 + 18
        for dashed, text in lines:
            dash = ' stroke-dasharray="5 4"' if dashed else ""
            self.parts.append(
                f'<path d="M{x},{y - 3} H{x + 26}" stroke="{self.t["edge"]}" stroke-width="1.4"{dash} marker-end="url(#arrow)"/>'
            )
            self.parts.append(
                f'<text x="{x + 34}" y="{y + 1}" class="legend" fill="{self.t["muted"]}">{escape(text)}</text>'
            )
            x += 40 + len(text) * 6.4 + 18

    def svg(self, title, desc):
        t = self.t
        return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" role="img" aria-labelledby="t d">
<title id="t">{escape(title)}</title>
<desc id="d">{escape(desc)}</desc>
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,1 L9,5 L0,9 z" fill="{t["edge"]}"/>
  </marker>
  <marker id="arrow-start" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,1 L9,5 L0,9 z" fill="{t["edge"]}"/>
  </marker>
</defs>
<style>
  text {{ font-family: {FONT}; text-anchor: middle; }}
  .label {{ font-size: 9.5px; font-weight: 600; letter-spacing: 0.08em; }}
  .title {{ font-size: 13.5px; font-weight: 600; }}
  .sub {{ font-size: 11.5px; }}
  .edge {{ font-size: 11px; }}
  .legend {{ font-size: 11.5px; text-anchor: start; }}
</style>
{chr(10).join(self.parts)}
</svg>
"""


# Grid for the architecture: four columns, five rows.
COL = {"a": 20, "b": 270, "c": 520, "d": 770}
ROW = {1: 24, 2: 144, 3: 264, 4: 384, 5: 504}


def architecture(theme):
    c = Canvas(970, 628, theme)

    def n(key, cell, *rest, **kw):
        return c.node(key, COL[cell[0]], ROW[int(cell[1])], *rest, **kw)

    n("es", "a1", "ext", "WIKIMEDIA", "EventStreams", "recentchange · SSE")
    n("mw", "a2", "ext", "WIKIMEDIA", "MediaWiki API", "diffs · revert tags")
    n("ing", "b1", "k3s", "K3S", "ingestor", "enwiki article edits")
    n("lab", "b2", "k3s", "K3S · HOURLY", "labeler", "48-hour revert labels")
    n("rp", "c1", "k3s", "K3S", "Redpanda", "Kafka API · 4 topics")
    n("sc", "c2", "k3s", "K3S", "scorer", "ONNX fallback chain")
    n("pg", "c3", "data", "SUPABASE", "Postgres", "hot store · 3 days", shape="cylinder")
    n("api", "c4", "serve", "RENDER", "FastAPI", "queue · reviews · Explain")
    n("llm", "c5", "ext", "OPENAI · GEMINI BACKUP", "LLM", "one-line summaries")
    n("ar", "d1", "k3s", "K3S", "archiver", "every topic, compressed")
    n(
        "os",
        "d2",
        "data",
        "ORACLE OBJECT STORAGE",
        "Parquet archive",
        "rolling 30 days",
        shape="cylinder",
    )
    n(
        "ct",
        "d3",
        "k3s",
        "K3S · DAILY AND ON REQUEST",
        "Continuous training",
        "monitor, poller, trainer",
    )
    n("dh", "d4", "data", "DAGSHUB", "Model registry", "MLflow · DVC snapshots", shape="cylinder")
    n("obs", "d5", "k3s", "K3S · PRIVATE", "Prometheus · Grafana", "Alertmanager → Discord")
    n("user", "a4", "ext", "BROWSER", "Patroller", "reviews the queue", shape="pill")
    n("edge", "b4", "serve", "CLOUDFLARE", "Edge Worker", "one site address")
    n("web", "b5", "serve", "VERCEL", "Next.js app", "console · landing page")

    s = c.side
    c.edge([s("es", "r"), s("ing", "l")])
    c.edge([s("mw", "r", -12), s("ing", "l", 18)], "diffs", at=(232, 112))
    c.edge([s("mw", "r", 8), s("lab", "l", 8)], "revert tags", at=(235, 172))
    c.edge([s("ing", "r"), s("rp", "l")])
    c.edge([s("lab", "r", -12), s("rp", "l", 18)], "labels.final", at=(486, 112))
    c.edge([s("lab", "r", 18), s("pg", "l", -14)], "labels", at=(484, 238))
    c.edge([s("rp", "b", -26), s("sc", "t", -26)], "edits.raw", at=(578, 124), anchor="end")
    c.edge([s("sc", "t", 26), s("rp", "b", 26)], "edits.scored", at=(642, 124), anchor="start")
    c.edge([s("sc", "b"), s("pg", "t")], "scores", at=(618, 244), anchor="start")
    c.edge([s("pg", "b"), s("api", "t")], both=True)
    c.edge([s("api", "b"), s("llm", "t")], "Explain", at=(618, 484), anchor="start")
    c.edge([s("rp", "r"), s("ar", "l")], "all topics", at=(735, 46))
    c.edge([s("ar", "b"), s("os", "t")])
    c.edge([s("os", "b"), s("ct", "t")], "archive", at=(868, 244), anchor="start")
    c.edge([s("ct", "b"), s("dh", "t")], "promotes", at=(868, 364), anchor="start", both=True)
    # Model files: one dashed trunk from the registry, branching to the scorer and the API.
    trunk_x = 735
    dh_l = s("dh", "l")
    c.edge([dh_l, s("api", "r")], "models", at=(735, 410), dashed=True)
    c.edge([(trunk_x, dh_l[1]), (trunk_x, s("sc", "r")[1]), s("sc", "r")], dashed=True)
    c.edge([s("user", "r"), s("edge", "l")])
    c.edge([s("edge", "r"), s("api", "l")], "/api", at=(485, 410))
    c.edge([s("edge", "b"), s("web", "t")], "pages", at=(368, 484), anchor="start")

    c.legend(
        612,
        [
            ("ext", "Wikimedia and outside services"),
            ("k3s", "k3s on Oracle Cloud (ARM)"),
            ("data", "Managed storage"),
            ("serve", "Serving"),
        ],
        [(False, "data"), (True, "model files")],
    )
    return c.svg(
        "WikiGuard architecture",
        "Wikimedia's stream feeds the ingestor, which publishes edits to Redpanda. The scorer scores them "
        "into Postgres on Supabase; the archiver writes every topic to a parquet archive; the labeler turns "
        "revert tags into labels. Continuous training reads the archive and promotes models in the DagsHub "
        "registry, which the scorer and the FastAPI service on Render load. A patroller reaches one Cloudflare "
        "Worker address, which sends /api to Render and pages to the Next.js app on Vercel. Explain calls an LLM.",
    )


def loop(theme):
    c = Canvas(970, 396, theme)
    x = [20, 270, 520, 770]
    y1, y2, y3 = 20, 148, 276
    c.node("edit", x[0], y1, "k3s", "SCORE", "Edit scored, queued", "a person reviews it")
    c.node("rev", x[1], y1, "ext", "WIKIPEDIA", "Reverted or kept", "by the community")
    c.node("lab", x[2], y1, "k3s", "LABEL", "Label is final", "at 48 hours")
    c.node("mon", x[3], y1, "k3s", "DAILY MONITOR", "Live metrics, drift", "calibration · slices")
    c.node("re", x[3], y2, "k3s", "RETRAIN", "Train on a new snapshot", "CPU models on the cluster")
    c.node(
        "gate",
        x[2],
        y2,
        "data",
        "PROMOTION GATE",
        "Paired bootstrap",
        "golden set · slices",
        shape="decision",
    )
    c.node("shadow", x[1], y2, "k3s", "SHADOW", "Score live edits", "beside the champion")
    c.node("promo", x[0], y2, "good", "PROMOTE", "New serving chain", "registry aliases move")
    c.node("stay", x[2], y3, "ext", "GATE FAILS", "Champion stays", "candidate recorded")
    c.node(
        "roll",
        x[0],
        y3,
        "bad",
        "POST-PROMOTION CHECK",
        "Automatic rollback",
        "if worse on live labels",
    )

    s = c.side
    c.edge([s("edit", "r"), s("rev", "l")])
    c.edge([s("rev", "r"), s("lab", "l")])
    c.edge([s("lab", "r"), s("mon", "l")])
    c.edge(
        [s("mon", "b"), s("re", "t")], "drift, decay, or schedule", at=(868, 120), anchor="start"
    )
    c.edge([s("re", "l"), s("gate", "r")])
    c.edge([s("gate", "l"), s("shadow", "r")], "passes", at=(485, 172))
    c.edge([s("shadow", "l"), s("promo", "r")])
    c.edge([s("promo", "t"), s("edit", "b")], "serves", at=(118, 120), anchor="start")
    c.edge([s("gate", "b"), s("stay", "t")], "fails", at=(618, 248), anchor="start")
    c.edge([s("promo", "b"), s("roll", "t")], "worse live", at=(118, 248), anchor="start")
    c.legend(
        384,
        [
            ("k3s", "Runs in the cluster"),
            ("ext", "Outside WikiGuard"),
            ("data", "Decision"),
            ("good", "Promotion"),
            ("bad", "Rollback"),
        ],
    )
    return c.svg(
        "WikiGuard continuous-training loop",
        "Each edit is scored and queued. The community reverts or keeps it, and its label is final at 48 hours. "
        "A daily monitor computes live metrics and drift. Drift, decaying accuracy, or the schedule start a retrain "
        "on a new snapshot. The promotion gate checks the candidate with a paired bootstrap, the golden set, and "
        "slices. If it fails, the champion stays. If it passes, the candidate scores live edits in shadow, then is "
        "promoted to a new serving chain. A post-promotion check rolls back automatically if it is worse on live labels.",
    )


for theme in THEMES:
    (OUT / f"architecture_{theme}_{DATE}.svg").write_text(architecture(theme), "utf-8")
    (OUT / f"training_loop_{theme}_{DATE}.svg").write_text(loop(theme), "utf-8")
print("written to", OUT)
