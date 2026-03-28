import os
import sys

import httpx
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

COORDINATOR_URL = os.getenv("COORDINATOR_URL", "http://localhost:8000")

COLOR_TRUSTED     = "#1D9E75"   # teal
COLOR_SUSPICIOUS  = "#EF9F27"   # amber
COLOR_QUARANTINED = "#E24B4A"   # red
COLOR_BACKGROUND  = "#0D1117"
COLOR_CARD        = "#161B22"
COLOR_TEXT        = "#C9D1D9"
COLOR_MUTED       = "#8B949E"
COLOR_BORDER      = "#30363D"

app = Dash(
    __name__,
    title="SecureHive Dashboard",
    update_title="Updating...",
)

app.layout = html.Div(
    style={
        "backgroundColor": COLOR_BACKGROUND,
        "minHeight": "100vh",
        "padding": "24px 32px",
        "fontFamily": "monospace",
    },
    children=[

        html.Div(
            style={"marginBottom": "24px", "borderBottom": f"1px solid {COLOR_BORDER}", "paddingBottom": "16px"},
            children=[
                html.H1(
                    "PrivAgent-TrustShield",
                    style={"color": "#7F77DD", "margin": "0", "fontSize": "24px", "fontWeight": "600"},
                ),
                html.P(
                    "Real-time trust monitoring  ·  AES-256-GCM  ·  Ed25519  ·  DP-protected telemetry",
                    style={"color": COLOR_MUTED, "margin": "4px 0 0", "fontSize": "13px"},
                ),
            ],
        ),

        html.Div(
            id="status-cards",
            style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "16px", "marginBottom": "24px"},
        ),

        html.Div(
            style={"backgroundColor": COLOR_CARD, "borderRadius": "8px", "padding": "20px", "marginBottom": "24px", "border": f"1px solid {COLOR_BORDER}"},
            children=[
                html.P("Live trust scores", style={"color": COLOR_MUTED, "fontSize": "12px", "margin": "0 0 12px", "textTransform": "uppercase", "letterSpacing": "1px"}),
                dcc.Graph(id="trust-bar", style={"height": "220px"}, config={"displayModeBar": False}),
            ],
        ),

        html.Div(
            style={"backgroundColor": COLOR_CARD, "borderRadius": "8px", "padding": "20px", "marginBottom": "24px", "border": f"1px solid {COLOR_BORDER}"},
            children=[
                html.P("Trust score history (last 30 beats per agent)", style={"color": COLOR_MUTED, "fontSize": "12px", "margin": "0 0 12px", "textTransform": "uppercase", "letterSpacing": "1px"}),
                dcc.Graph(id="trust-history", style={"height": "280px"}, config={"displayModeBar": False}),
            ],
        ),

        html.Div(
            style={"backgroundColor": COLOR_CARD, "borderRadius": "8px", "padding": "20px", "border": f"1px solid {COLOR_BORDER}"},
            children=[
                html.P("Event log", style={"color": COLOR_MUTED, "fontSize": "12px", "margin": "0 0 12px", "textTransform": "uppercase", "letterSpacing": "1px"}),
                html.Div(id="event-log", style={"fontSize": "12px", "lineHeight": "1.8", "color": COLOR_TEXT, "maxHeight": "200px", "overflowY": "auto"}),
            ],
        ),

        dcc.Interval(id="poll-interval", interval=3000, n_intervals=0),

        dcc.Store(id="history-store", data={}),
        dcc.Store(id="log-store", data=[]),
    ],
)



def _score_color(score: float) -> str:
    if score >= 0.7:
        return COLOR_TRUSTED
    if score >= 0.4:
        return COLOR_SUSPICIOUS
    return COLOR_QUARANTINED


def _score_status(score: float) -> str:
    if score >= 0.7:
        return "TRUSTED"
    if score >= 0.4:
        return "SUSPICIOUS"
    return "QUARANTINED"


def _fetch_trust() -> dict:
    try:
        resp = httpx.get(f"{COORDINATOR_URL}/trust", timeout=2.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def _fetch_agent_history(agent_id: str) -> list:
    try:
        resp = httpx.get(f"{COORDINATOR_URL}/trust/{agent_id}", timeout=2.0)
        resp.raise_for_status()
        return resp.json().get("history", [])[-30:]
    except Exception:
        return []



@app.callback(
    Output("history-store", "data"),
    Output("log-store", "data"),
    Input("poll-interval", "n_intervals"),
    prevent_initial_call=False,
)
def poll_coordinator(n_intervals):
    scores = _fetch_trust()

    history: dict = {}
    for agent_id in scores:
        history[agent_id] = _fetch_agent_history(agent_id)

    return history, []


@app.callback(
    Output("status-cards", "children"),
    Input("history-store", "data"),
)
def update_status_cards(history: dict):
    if not history:
        return [html.P("Waiting for coordinator...", style={"color": COLOR_MUTED, "fontSize": "13px"})]

    cards = []
    for agent_id, agent_history in sorted(history.items()):
        if not agent_history:
            continue
        latest = agent_history[-1]
        score = latest.get("score", 1.0)
        color = _score_color(score)
        status = _score_status(score)

        if len(agent_history) >= 2:
            prev = agent_history[-2].get("score", score)
            trend = "▲" if score > prev else "▼" if score < prev else "─"
            trend_color = COLOR_TRUSTED if score >= prev else COLOR_QUARANTINED
        else:
            trend, trend_color = "─", COLOR_MUTED

        cards.append(html.Div(
            style={
                "backgroundColor": COLOR_CARD,
                "border": f"2px solid {color}",
                "borderRadius": "8px",
                "padding": "16px",
            },
            children=[
                html.P(agent_id.upper(), style={"color": COLOR_MUTED, "fontSize": "11px", "margin": "0 0 8px", "letterSpacing": "1px"}),
                html.Div(
                    style={"display": "flex", "alignItems": "baseline", "gap": "8px"},
                    children=[
                        html.Span(f"{score:.4f}", style={"color": color, "fontSize": "28px", "fontWeight": "700"}),
                        html.Span(trend, style={"color": trend_color, "fontSize": "16px"}),
                    ],
                ),
                html.P(status, style={"color": color, "fontSize": "11px", "margin": "8px 0 0", "fontWeight": "600", "letterSpacing": "1px"}),
            ],
        ))
    return cards


@app.callback(
    Output("trust-bar", "figure"),
    Input("history-store", "data"),
)
def update_bar_chart(history: dict):
    agents = sorted(history.keys())
    scores = []
    colors = []

    for agent_id in agents:
        agent_history = history.get(agent_id, [])
        if agent_history:
            score = agent_history[-1].get("score", 1.0)
        else:
            score = 1.0
        scores.append(score)
        colors.append(_score_color(score))

    fig = go.Figure(go.Bar(
        x=agents,
        y=scores,
        marker_color=colors,
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
        textfont={"color": COLOR_TEXT, "size": 12},
        width=0.5,
    ))

    fig.add_hline(y=0.7, line_dash="dash", line_color=COLOR_TRUSTED,   line_width=1, annotation_text="trusted", annotation_font_color=COLOR_TRUSTED)
    fig.add_hline(y=0.4, line_dash="dash", line_color=COLOR_QUARANTINED, line_width=1, annotation_text="quarantine", annotation_font_color=COLOR_QUARANTINED)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=COLOR_TEXT,
        yaxis=dict(range=[0, 1.15], gridcolor=COLOR_BORDER, gridwidth=0.5, tickfont={"size": 11}),
        xaxis=dict(gridcolor=COLOR_BORDER, tickfont={"size": 12}),
        margin=dict(l=40, r=20, t=10, b=30),
        showlegend=False,
    )
    return fig


@app.callback(
    Output("trust-history", "figure"),
    Input("history-store", "data"),
)
def update_history_chart(history: dict):
    agent_colors = {
        "planner":        "#7F77DD",
        "retriever":      "#1D9E75",
        "policy_checker": "#EF9F27",
        "executor":       "#E24B4A",
    }

    fig = go.Figure()

    for agent_id in sorted(history.keys()):
        agent_history = history.get(agent_id, [])
        if len(agent_history) < 2:
            continue

        scores = [h.get("score", 1.0) for h in agent_history]
        x_vals = list(range(len(scores)))
        color  = agent_colors.get(agent_id, COLOR_MUTED)

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=scores,
            name=agent_id,
            line=dict(color=color, width=2),
            mode="lines+markers",
            marker=dict(size=4),
        ))

    fig.add_hline(y=0.7, line_dash="dash", line_color=COLOR_TRUSTED,    line_width=1)
    fig.add_hline(y=0.4, line_dash="dash", line_color=COLOR_QUARANTINED, line_width=1)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=COLOR_TEXT,
        yaxis=dict(range=[0, 1.05], gridcolor=COLOR_BORDER, gridwidth=0.5, tickfont={"size": 11}),
        xaxis=dict(title="heartbeat number", gridcolor=COLOR_BORDER, tickfont={"size": 11}),
        margin=dict(l=40, r=20, t=10, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLOR_TEXT, size=12)),
    )
    return fig


@app.callback(
    Output("event-log", "children"),
    Input("history-store", "data"),
)
def update_event_log(history: dict):
    """Show the latest status line for each agent as a log entry."""
    if not history:
        return [html.P("No data yet — is the coordinator running?", style={"color": COLOR_MUTED})]

    log_lines = []
    for agent_id in sorted(history.keys()):
        agent_history = history.get(agent_id, [])
        if not agent_history:
            continue
        latest = agent_history[-1]
        score  = latest.get("score", 1.0)
        color  = _score_color(score)
        status = _score_status(score)
        beat   = len(agent_history)

        log_lines.append(html.Div(
            children=[
                html.Span(f"[beat {beat:03d}] ", style={"color": COLOR_MUTED}),
                html.Span(f"{agent_id:20s}", style={"color": COLOR_TEXT}),
                html.Span(f"trust={score:.4f}  ", style={"color": color}),
                html.Span(status, style={"color": color, "fontWeight": "600"}),
            ],
        ))
    return log_lines


if __name__ == "__main__":
    print(f"[dashboard] Starting on http://localhost:8050")
    print(f"[dashboard] Polling coordinator at {COORDINATOR_URL}")
    app.run(debug=False, host="0.0.0.0", port=8050)