"""Shared UI primitives: the step rail, metric tiles, provenance badges, score
bars, and the feedback capture widget used on every screen.

Feedback is QUEUED here, never applied. The rep walks the whole plan leaving
comments, and one re-run at the end applies all of them together -- so a
walkthrough costs one solve and one router call instead of one of each per
comment.
"""
from __future__ import annotations

import html
import json
from typing import Any, Dict, List, Optional, Sequence

import streamlit as st

from app import config
from app.db import store
from app.services.metrics import compact, money, pct

CSS = """
<style>
  /* ------------------------------------------------------------------ *
   * Light, minimal, premium. Three rules hold the whole thing together:
   *   1. ONE accent (ink blue). Colour means something or is absent.
   *   2. Hairline separation, not boxes. Borders at 1px #E4E7EC, no shadows
   *      except a single soft lift on the one card that matters.
   *   3. Generous vertical rhythm, tight horizontal. Reads like a document.
   * ------------------------------------------------------------------ */
  :root {
    --ink:        #14181F;
    --ink-soft:   #4A5361;
    --ink-faint:  #79828F;
    --line:       #E4E7EC;
    --line-soft:  #F0F2F5;
    --accent:     #1B3A6B;
    --accent-bg:  #F2F5FA;
    --good:       #0F7A55;
    --good-bg:    #EFF8F3;
    --warn:       #A8620A;
    --warn-bg:    #FDF6EC;
    --bad:        #B3261E;
    --bad-bg:     #FDF2F1;
    --canvas:     #F6F7F9;
  }

  /* ---------- layout ---------- */
  .block-container { padding-top: 1.9rem; padding-bottom: 5rem; max-width: 1180px; }
  #MainMenu, footer { visibility: hidden; }

  html, body, [class*="css"] { -webkit-font-smoothing: antialiased; }

  h1, h2, h3, h4 { letter-spacing: -0.021em; color: var(--ink); font-weight: 600; }
  h4 { font-size: 1.13rem; margin-bottom: .1rem; }

  /* Streamlit's default h3/subheader is heavy for a document-like page. */
  [data-testid="stHeading"] h3 { font-size: 1.32rem; font-weight: 600; }

  /* Quieter captions -- there are a lot of them and they are supporting text. */
  [data-testid="stCaptionContainer"] p {
    color: var(--ink-faint); font-size: .795rem; line-height: 1.5;
  }

  /* ---------- page header ---------- */
  .pagehead { margin: 0 0 1.15rem; }
  .pagehead .eyebrow {
    font-size: .655rem; letter-spacing: .13em; text-transform: uppercase;
    color: var(--ink-faint); font-weight: 600; margin-bottom: .3rem;
  }
  .pagehead h2 {
    font-size: 1.58rem; font-weight: 600; margin: 0 0 .28rem;
    letter-spacing: -0.024em;
  }
  .pagehead .lede { color: var(--ink-soft); font-size: .93rem; max-width: 66ch; }
  .pagehead .ctx {
    font-size: .76rem; color: var(--ink-faint); text-align: right;
    line-height: 1.7;
  }
  .pagehead .ctx b { color: var(--ink-soft); font-weight: 600; }

  /* ---------- buttons ---------- */
  .stButton > button {
    border-radius: .5rem; font-weight: 550; letter-spacing: -.005em;
    border-color: var(--line); transition: none;
  }
  .stButton > button[kind="primary"] {
    background: var(--accent); border-color: var(--accent);
    box-shadow: 0 1px 2px rgba(20,24,31,.08);
  }
  .stButton > button[kind="primary"]:hover { background: #16305A; }

  /* ---------- tabs ---------- */
  [data-baseweb="tab-list"] { gap: .3rem; border-bottom: 1px solid var(--line); }
  [data-baseweb="tab"] {
    font-size: .87rem; font-weight: 550; letter-spacing: -.006em;
    padding-left: .8rem; padding-right: .8rem;
  }

  /* ---------- dataframes: flatten the grid ---------- */
  [data-testid="stDataFrame"], [data-testid="stDataEditor"] {
    border: 1px solid var(--line); border-radius: .55rem; overflow: hidden;
  }

  /* ---------- expanders: hairline, not a box ---------- */
  [data-testid="stExpander"] {
    border: 1px solid var(--line); border-radius: .55rem; background: #fff;
  }
  [data-testid="stExpander"] summary { font-size: .865rem; font-weight: 550; }

  /* ---------- sidebar ---------- */
  [data-testid="stSidebar"] { border-right: 1px solid var(--line); }
  [data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

  /* ---------- step rail ---------- */
  .rail { margin: 0 0 2px; }
  .rail-item {
    display: flex; align-items: center; gap: 10px; padding: 7px 10px;
    border-radius: .45rem; font-size: .825rem; margin-bottom: 2px;
    border: 1px solid transparent; color: var(--ink-soft);
  }
  .rail-item .num {
    flex: 0 0 19px; height: 19px; border-radius: 50%; display: grid;
    place-items: center; font-size: .625rem; font-weight: 650;
    border: 1px solid var(--line); background: #fff; color: var(--ink-faint);
  }
  .rail-item .eng {
    margin-left: auto; font-size: .58rem; letter-spacing: .08em;
    font-family: ui-monospace, Menlo, monospace; color: var(--ink-faint);
  }
  .rail-done  { color: var(--ink-soft); }
  .rail-done  .num { border-color: var(--good); color: #fff; background: var(--good); }
  .rail-now   { background: var(--accent-bg); color: var(--accent); font-weight: 650; }
  .rail-now   .num { border-color: var(--accent); color: #fff; background: var(--accent); }
  .rail-lock  { color: #B2B9C4; }
  .rail-lock  .num { color: #C4CAD3; }

  /* ---------- tiles ---------- */
  .tiles { display: flex; flex-wrap: wrap; gap: .55rem; margin: .2rem 0 1.05rem; }
  .tile {
    border: 1px solid var(--line); border-radius: .6rem; background: #fff;
    padding: .68rem .85rem; min-width: 138px; flex: 1 1 138px;
  }
  .tile .v {
    font-size: 1.42rem; font-weight: 600; letter-spacing: -.028em;
    display: block; line-height: 1.2; color: var(--ink);
    font-variant-numeric: tabular-nums;
  }
  .tile .k {
    font-size: .705rem; color: var(--ink-faint); display: block; margin-top: .18rem;
    line-height: 1.35;
  }
  .tile .s {
    font-size: .665rem; color: var(--ink-faint); display: block; margin-top: .2rem;
    opacity: .85;
  }
  /* Tone tints the VALUE, not the border -- a grid of coloured outlines reads
     as an alert panel. One tinted number reads as a fact. */
  .tile.good .v { color: var(--good); }
  .tile.warn .v { color: var(--warn); }
  .tile.bad  .v { color: var(--bad); }
  .tile.hero {
    border-color: #D5DCE7; background: linear-gradient(180deg,#FBFCFE 0%,#F5F8FC 100%);
    box-shadow: 0 1px 3px rgba(20,24,31,.05);
  }
  .tile.hero .v { font-size: 1.72rem; color: var(--accent); }

  /* ---------- badges ---------- */
  .badge {
    display: inline-block; font-size: .625rem; font-weight: 650;
    letter-spacing: .045em; text-transform: uppercase;
    padding: .12rem .42rem; border-radius: .25rem; white-space: nowrap;
    vertical-align: 1.5px;
  }
  .b-read { background: var(--good-bg); color: var(--good); }
  .b-inf  { background: var(--accent-bg); color: var(--accent); }
  .b-rep  { background: #F4F1FB; color: #5B3FA8; }
  .b-fb   { background: var(--warn-bg); color: var(--warn); }
  .b-need { background: var(--bad-bg);  color: var(--bad); }
  .b-drv  { background: var(--line-soft); color: var(--ink-faint); }

  /* ---------- price band ---------- */
  .band { position: relative; height: 52px; margin: .5rem 0 .2rem; }
  .band .track { position: absolute; left:0; right:0; top: 22px; height: 5px;
                 border-radius: 3px; background: var(--line); }
  .band .rng { position: absolute; top: 22px; height: 5px; border-radius: 3px;
               background: #C3D0E4; }
  .band .mk { position: absolute; top: 15px; width: 3px; height: 19px;
              background: var(--accent); border-radius: 2px; }
  .band .lbl { position: absolute; top: 0; font-size: .685rem;
               color: var(--ink-faint); transform: translateX(-50%);
               white-space: nowrap; font-variant-numeric: tabular-nums; }
  .band .lbl.hi { color: var(--accent); font-weight: 650; }

  /* ---------- misc ---------- */
  .bar { height: 5px; border-radius: 3px; background: var(--line-soft);
         overflow: hidden; }
  .bar > i { display: block; height: 100%; background: var(--accent); border-radius: 3px; }
  .muted { color: var(--ink-faint); font-size: .8rem; }
  .mono  { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .82rem; }

  /* The explanation callout. Used a lot, so it stays quiet: a hairline rule and
     a tint, no heavy left bar competing with the text. */
  .why {
    border: 1px solid #E1E7F0; border-left: 2.5px solid var(--accent);
    padding: .65rem .85rem; background: #FAFBFD;
    border-radius: 0 .5rem .5rem 0; font-size: .875rem; line-height: 1.58;
    margin: .35rem 0 .8rem; color: var(--ink-soft);
  }
  .why b, .why strong { color: var(--ink); font-weight: 600; }
  .why .tag {
    display: block; font-size: .625rem; letter-spacing: .1em; font-weight: 700;
    text-transform: uppercase; color: var(--accent); margin-bottom: .25rem;
  }

  .refuse {
    border: 1px solid #F2D9D7; border-left: 2.5px solid var(--bad);
    padding: .65rem .85rem; background: var(--bad-bg);
    border-radius: 0 .5rem .5rem 0; font-size: .875rem; margin: .35rem 0;
    color: var(--ink-soft);
  }
  .refuse b { color: var(--bad); }

  .fbitem {
    border: 1px solid var(--line); border-radius: .45rem; background: #fff;
    padding: .38rem .55rem; margin-bottom: .3rem; font-size: .765rem;
    color: var(--ink-soft);
  }

  /* Reasoning list used on the proposal page. */
  .rlist { margin: .2rem 0 .9rem; padding: 0; list-style: none; }
  .rlist li {
    position: relative; padding: .3rem 0 .3rem 1.05rem; font-size: .885rem;
    color: var(--ink-soft); line-height: 1.55;
  }
  .rlist li::before {
    content: ""; position: absolute; left: .1rem; top: .78rem;
    width: 4px; height: 4px; border-radius: 50%; background: #B9C3D2;
  }
  .rlist li b { color: var(--ink); font-weight: 600; }

  /* Section rule with a label -- structures the proposal page without headings
     every 40px. */
  .srule {
    display: flex; align-items: center; gap: .7rem; margin: 1.5rem 0 .7rem;
    font-size: .655rem; letter-spacing: .13em; text-transform: uppercase;
    color: var(--ink-faint); font-weight: 650;
  }
  .srule::after {
    content: ""; flex: 1; height: 1px; background: var(--line);
  }

  /* Email draft preview -- looks like a mail client, not a code block. */
  .maildraft {
    border: 1px solid var(--line); border-radius: .6rem; background: #fff;
    overflow: hidden;
  }
  .maildraft .mhead {
    padding: .7rem .95rem; border-bottom: 1px solid var(--line-soft);
    background: #FBFCFD; font-size: .82rem; color: var(--ink-soft);
  }
  .maildraft .mhead .row { display: flex; gap: .5rem; padding: .1rem 0; }
  .maildraft .mhead .row span:first-child {
    color: var(--ink-faint); min-width: 4.1rem; font-size: .755rem;
    text-transform: uppercase; letter-spacing: .06em; padding-top: .1rem;
  }
  .maildraft .mhead b { color: var(--ink); font-weight: 600; }
  .maildraft .mbody {
    padding: 1rem 1.1rem; font-size: .875rem; line-height: 1.62;
    color: var(--ink); white-space: pre-wrap;
  }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# step rail
# --------------------------------------------------------------------------- #

def step_rail(current: str, unlocked: Sequence[str], mode: str) -> None:
    steps = config.steps_for(mode)
    keys = [s[0] for s in steps]
    cur_i = keys.index(current) if current in keys else 0
    rows = []
    for i, (key, label, engine) in enumerate(steps):
        if i < cur_i:
            cls, num = "rail-done", "&#10003;"
        elif i == cur_i:
            cls, num = "rail-now", str(i + 1)
        else:
            cls, num = ("rail-item" if key in unlocked else "rail-lock"), str(i + 1)
        eng = f'<span class="eng">{engine}</span>' if engine else ""
        rows.append(f'<div class="rail-item {cls}"><span class="num">{num}</span>'
                    f'<span>{html.escape(label)}</span>{eng}</div>')
    st.markdown(f'<div class="rail">{"".join(rows)}</div>', unsafe_allow_html=True)


def page_header(*, eyebrow: str, title: str, lede: str = "",
                context: Sequence[tuple] = ()) -> None:
    """The consistent top of every screen: where you are, what this is for, and
    the campaign you are working on."""
    ctx = ""
    if context:
        ctx = "<br>".join(f"{html.escape(str(k))} <b>{html.escape(str(v))}</b>"
                          for k, v in context)
    left, right = st.columns([3, 1.15])
    with left:
        st.markdown(
            f'<div class="pagehead"><div class="eyebrow">{html.escape(eyebrow)}</div>'
            f'<h2>{html.escape(title)}</h2>'
            + (f'<div class="lede">{lede}</div>' if lede else "")
            + '</div>', unsafe_allow_html=True)
    with right:
        if ctx:
            st.markdown(f'<div class="pagehead"><div class="ctx">{ctx}</div></div>',
                        unsafe_allow_html=True)


def section(label: str) -> None:
    st.markdown(f'<div class="srule">{html.escape(label)}</div>',
                unsafe_allow_html=True)


def reason_list(items: Sequence[str]) -> None:
    if not items:
        return
    lis = "".join(f"<li>{i}</li>" for i in items)
    st.markdown(f'<ul class="rlist">{lis}</ul>', unsafe_allow_html=True)


def plain_block(block_id) -> str:
    try:
        return config.BLOCK_PLAIN.get(int(block_id), str(block_id))
    except (TypeError, ValueError):
        return str(block_id)


def plain_screen_type(t) -> str:
    return config.SCREEN_TYPE_PLAIN.get(str(t), str(t).replace("_", " "))


def plain_position(p) -> str:
    return config.POSITION_PLAIN.get(str(p), str(p).replace("_", " "))


# --------------------------------------------------------------------------- #
# tiles
# --------------------------------------------------------------------------- #

def tiles(items: Sequence[Dict[str, Any]]) -> None:
    """items: {value, label, sub?, tone? in good|warn|bad}"""
    parts = []
    for it in items:
        tone = it.get("tone", "")
        sub = (f'<span class="s">{html.escape(str(it["sub"]))}</span>'
               if it.get("sub") else "")
        parts.append(
            f'<div class="tile {tone}"><span class="v">'
            f'{html.escape(str(it["value"]))}</span>'
            f'<span class="k">{html.escape(str(it["label"]))}</span>{sub}</div>')
    st.markdown(f'<div class="tiles">{"".join(parts)}</div>', unsafe_allow_html=True)


BADGE_CLASS = {
    "read": ("b-read", "read from brief"),
    "inferred": ("b-inf", "inferred"),
    "rep_edited": ("b-rep", "you edited"),
    "feedback_applied": ("b-fb", "from feedback"),
    "needs_you": ("b-need", "needs you"),
    "derived": ("b-drv", "derived"),
}


def badge(kind: str, text: str = "") -> str:
    cls, default = BADGE_CLASS.get(kind, ("b-drv", kind))
    return f'<span class="badge {cls}">{html.escape(text or default)}</span>'


def why_box(text: str, tag: str = "Why") -> None:
    label = (f'<span class="tag">{html.escape(tag)}</span>' if tag else "")
    st.markdown(f'<div class="why">{label}{text}</div>', unsafe_allow_html=True)


def refusal_box(stage: str, reason: str, remedy: str) -> None:
    st.markdown(
        f'<div class="refuse"><b>{html.escape(stage)}</b> — {html.escape(reason)}'
        f'<br><span class="muted">Remedy: {html.escape(remedy)}</span></div>',
        unsafe_allow_html=True)


def price_band(floor: float, target: float, cap: float, marker: float,
               marker_label: str = "quoted") -> None:
    lo, hi = float(floor), float(cap)
    span = max(hi - lo, 1e-9)
    pad = 0.12

    def x(v: float) -> float:
        return (pad + (1 - 2 * pad) * (float(v) - lo) / span) * 100.0

    st.markdown(
        f'<div class="band">'
        f'<div class="track"></div>'
        f'<div class="rng" style="left:{x(lo):.1f}%;right:{100 - x(hi):.1f}%"></div>'
        f'<div class="mk" style="left:{x(marker):.1f}%"></div>'
        f'<span class="lbl" style="left:{x(lo):.1f}%">floor {money(lo, 2)}</span>'
        f'<span class="lbl" style="left:{x(target):.1f}%">target {money(target, 2)}</span>'
        f'<span class="lbl" style="left:{x(hi):.1f}%">cap {money(hi, 2)}</span>'
        f'<span class="lbl hi" style="left:{x(marker):.1f}%;top:32px">'
        f'{html.escape(marker_label)} {money(marker, 2)}</span>'
        f'</div>', unsafe_allow_html=True)


def bar(fraction: float, width_px: int = 120) -> str:
    f = max(0.0, min(1.0, float(fraction)))
    return (f'<span class="bar" style="display:inline-block;width:{width_px}px;'
            f'vertical-align:middle"><i style="width:{f * 100:.0f}%"></i></span>')


# --------------------------------------------------------------------------- #
# feedback capture -- present on every screen, applied on none
# --------------------------------------------------------------------------- #

def feedback_panel(*, screen: str, campaign_id: Optional[int],
                   run_id: Optional[int] = None,
                   param_widgets=None,
                   placeholder: str = "e.g. client wants students in the mix too, "
                                      "not just professionals",
                   key_prefix: str = "") -> None:
    """Queue feedback for this screen.

    `param_widgets(container)` may render structured controls and return a dict
    payload; that lands as a `param` item the router applies without needing to
    interpret anything.
    """
    kp = key_prefix or screen
    with st.expander("Give the agent feedback on this step", expanded=False):
        st.caption(
            "Queued, not applied. Comment freely as you review — everything you "
            "leave here is applied together in a single re-run from the "
            "Recommendation step, so the walkthrough costs one solve instead of one "
            "per comment.")

        # Clearing the box after queuing has to happen BEFORE the widget is
        # instantiated: assigning to a widget-backed key once st.text_area has
        # run in the same script pass raises StreamlitAPIException. So the queue
        # handler below sets a flag and the next run consumes it here.
        if st.session_state.pop(f"fb_clear_{kp}", False):
            st.session_state[f"fb_txt_{kp}"] = ""

        text = st.text_area("In your own words", key=f"fb_txt_{kp}",
                            placeholder=placeholder, height=80,
                            label_visibility="visible")

        c1, c2 = st.columns([1, 1])
        with c1:
            reason = st.selectbox(
                "Categorise it (optional)", ["--"] + list(config.LOSS_REASONS),
                key=f"fb_reason_{kp}",
                help="The same vocabulary the historical lost-lead data uses, so "
                     "your feedback is trainable rather than free text.")
        payload: Dict[str, Any] = {}
        with c2:
            if param_widgets is not None:
                payload = param_widgets(st) or {}

        if st.button("Queue this feedback", key=f"fb_btn_{kp}",
                     disabled=campaign_id is None):
            if not text.strip() and reason == "--" and not payload:
                st.warning("Nothing to queue — add a note, a category, or a tweak.")
            else:
                mode = "nl" if text.strip() else ("param" if payload else "loss_reason")
                if payload and text.strip():
                    # Two distinct items: one interpretable, one exact.
                    store.add_feedback(campaign_id=campaign_id, run_id=run_id,
                                       screen=screen, mode="param", payload=payload)
                    mode = "nl"
                store.add_feedback(
                    campaign_id=campaign_id, run_id=run_id, screen=screen,
                    mode=mode, raw_text=text.strip() or None,
                    payload=payload if mode == "param" else None,
                    loss_reason=None if reason == "--" else reason)
                store.log(campaign_id=campaign_id, run_id=run_id, actor="rep",
                          event="feedback_queued",
                          detail=f"{screen}: {text.strip()[:160] or reason}")
                st.session_state[f"fb_clear_{kp}"] = True
                # A toast rather than st.success: the rerun below discards
                # anything written into the page body, so the old success
                # banner was never actually seen.
                st.toast("Queued. Apply it from the Recommendation step.")
                st.rerun()


def feedback_tray(campaign_id: Optional[int], *, compact_mode: bool = True,
                  key_prefix: str = "") -> List[dict]:
    """The pending-feedback list. Shown in the sidebar and on screen 06 so the
    rep always knows what a re-run will actually change.

    Both of those render in the SAME script run, so the discard buttons need a
    per-call-site key namespace -- keyed on the feedback id alone they collide
    and Streamlit raises StreamlitDuplicateElementKey.
    """
    if campaign_id is None:
        return []
    items = store.queued_feedback(campaign_id)
    if not items:
        if not compact_mode:
            st.caption("No feedback queued.")
        return items

    st.markdown(f"**{len(items)} queued** ")
    for it in items:
        label = it.get("raw_text") or it.get("loss_reason") or it.get("mode")
        if it.get("mode") == "param" and it.get("payload_json"):
            try:
                label = "tweak: " + json.dumps(json.loads(it["payload_json"]))
            except (TypeError, ValueError):
                pass
        if it.get("mode") == "drop_screen" and it.get("payload_json"):
            try:
                ids = json.loads(it["payload_json"]).get("screen_ids", [])
                label = f"drop {len(ids)} screen(s)"
            except (TypeError, ValueError):
                pass
        c1, c2 = st.columns([6, 1])
        with c1:
            st.markdown(
                f'<div class="fbitem"><span class="badge b-drv">{it["screen"]}</span> '
                f'{html.escape(str(label))[:170]}</div>', unsafe_allow_html=True)
        with c2:
            if st.button("x", key=f"fbdel_{key_prefix}_{it['id']}",
                         help="Discard"):
                store.dismiss_feedback(it["id"])
                st.rerun()
    return items


def stat_row(pairs: Sequence[tuple]) -> None:
    """Compact label/value list."""
    md = "\n".join(f"| {k} | {v} |" for k, v in pairs)
    st.markdown("| | |\n|---|---|\n" + md)
