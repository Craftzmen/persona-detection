"""Reusable chart builders for dashboard rendering and visual regression tests.

All charts share a unified aesthetic with dark text for light mode readability
and transparent backgrounds that blend with the dashboard surface.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Shared palette ──────────────────────────────────────────────────────────
_PALETTE = {
    "primary": "#4a6df0",
    "teal": "#2f9e8f",
    "teal_light": "#5ec4b6",
    "cyan": "#22d3ee",
    "violet": "#8b5cf6",
    "amber": "#f59e0b",
    "rose": "#f43f5e",
    "muted": "#64748b",
    "grid": "rgba(100, 116, 139, 0.12)",
    "edge": "rgba(100, 130, 160, 0.5)",
}

# Dark text for light-mode readability
_TITLE_COLOR = "#1e293b"
_AXIS_TEXT = "#475569"
_CHART_FONT = dict(family="Inter, sans-serif", color=_AXIS_TEXT, size=12)

_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=_CHART_FONT,
    margin=dict(l=48, r=24, t=56, b=42),
    hoverlabel=dict(
        bgcolor="rgba(15,23,42,0.92)",
        font_size=13,
        font_family="Inter, sans-serif",
        font_color="#e2e8f0",
        bordercolor="rgba(255,255,255,0.08)",
    ),
)


def _apply_axes(fig: go.Figure) -> go.Figure:
    """Apply consistent axis styling across charts."""

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="rgba(100,116,139,0.18)",
        tickfont=dict(size=11, color=_AXIS_TEXT),
        title_font=dict(size=12, color=_AXIS_TEXT),
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=_PALETTE["grid"],
        zeroline=False,
        showline=False,
        tickfont=dict(size=11, color=_AXIS_TEXT),
        title_font=dict(size=12, color=_AXIS_TEXT),
    )
    return fig


# ── Network graph ───────────────────────────────────────────────────────────

def build_network_figure(
    graph_data: dict[str, Any],
    title: str = "Suspicious Persona Network",
) -> go.Figure:
    """Build an interactive Plotly network chart from graph JSON."""

    graph = nx.Graph()

    for node in graph_data.get("nodes", []):
        graph.add_node(str(node.get("id", "unknown")), **node)

    for edge in graph_data.get("links", []):
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source and target:
            graph.add_edge(source, target, weight=float(edge.get("weight", 0.0)))

    if graph.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(
            **_BASE_LAYOUT,
            height=410,
            title=dict(text=title, font=dict(size=15, color=_TITLE_COLOR, family="Space Grotesk, sans-serif")),
            annotations=[
                dict(
                    text="No suspicious network edges detected",
                    showarrow=False,
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    font=dict(size=14, color="#64748b"),
                )
            ],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    position = nx.spring_layout(graph, seed=42)

    edge_x: list[float] = []
    edge_y: list[float] = []
    for source, target in graph.edges():
        x0, y0 = position[source]
        x1, y1 = position[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1.5, color=_PALETTE["edge"]),
        hoverinfo="none",
    )

    node_x = [position[n][0] for n in graph.nodes()]
    node_y = [position[n][1] for n in graph.nodes()]
    node_text = [str(n) for n in graph.nodes()]
    node_degree = [graph.degree(n) for n in graph.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=11, color=_TITLE_COLOR),
        hoverinfo="text",
        hovertext=[f"{n} (degree {d})" for n, d in zip(node_text, node_degree)],
        marker=dict(
            size=[12 + 3 * d for d in node_degree],
            color=node_degree,
            colorscale=[[0, "#2f9e8f"], [0.5, "#4fd1c5"], [1, "#22d3ee"]],
            showscale=True,
            colorbar=dict(
                thickness=12,
                len=0.5,
                tickfont=dict(size=10, color=_AXIS_TEXT),
                title=dict(text="Degree", font=dict(size=11, color=_AXIS_TEXT)),
                bgcolor="rgba(0,0,0,0)",
                outlinewidth=0,
            ),
            line=dict(width=1.5, color="rgba(15,23,42,0.25)"),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        **_BASE_LAYOUT,
        height=410,
        title=dict(text=title, font=dict(size=15, color=_TITLE_COLOR, family="Space Grotesk, sans-serif")),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# ── Hourly activity ─────────────────────────────────────────────────────────

def build_hourly_activity_figure(hour_df: pd.DataFrame) -> go.Figure:
    """Create styled hourly activity bar chart."""

    fig = go.Figure(
        go.Bar(
            x=hour_df["hour"],
            y=hour_df["posts"],
            marker=dict(
                color=hour_df["posts"],
                colorscale=[[0, "#1a5c52"], [0.5, "#2f9e8f"], [1, "#5ec4b6"]],
                line=dict(width=0),
                cornerradius=4,
            ),
            hovertemplate="<b>Hour %{x}:00</b><br>Posts: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        **_BASE_LAYOUT,
        height=380,
        title=dict(
            text="Activity by Hour",
            font=dict(size=14, color=_TITLE_COLOR, family="Space Grotesk, sans-serif"),
        ),
        xaxis_title="Hour of Day",
        yaxis_title="Post Count",
        bargap=0.2,
    )
    return _apply_axes(fig)


# ── Daily frequency ─────────────────────────────────────────────────────────

def build_daily_frequency_figure(day_df: pd.DataFrame) -> go.Figure:
    """Create day-level posting frequency area chart."""

    fig = go.Figure(
        go.Scatter(
            x=day_df["day"],
            y=day_df["posts"],
            mode="lines+markers",
            fill="tozeroy",
            fillcolor="rgba(79, 209, 197, 0.18)",
            line=dict(color=_PALETTE["teal"], width=2.5, shape="spline", smoothing=1.3),
            marker=dict(size=7, color=_PALETTE["teal_light"], line=dict(width=1.5, color=_PALETTE["teal"])),
            hovertemplate="<b>%{x}</b><br>Posts: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        **_BASE_LAYOUT,
        height=380,
        title=dict(
            text="Posting Frequency by Day",
            font=dict(size=14, color=_TITLE_COLOR, family="Space Grotesk, sans-serif"),
        ),
        xaxis_title="Day",
        yaxis_title="Post Count",
    )
    return _apply_axes(fig)


# ── Word-count distribution ────────────────────────────────────────────────

def build_wordcount_distribution_figure(wc_df: pd.DataFrame) -> go.Figure:
    """Create styled word-count histogram."""

    fig = px.histogram(
        wc_df,
        x="word_count",
        nbins=22,
        color_discrete_sequence=[_PALETTE["teal"]],
    )

    fig.update_traces(
        marker_line_width=0,
        hovertemplate="Words: %{x}<br>Count: %{y}<extra></extra>",
    )

    fig.update_layout(
        **_BASE_LAYOUT,
        height=380,
        title=dict(
            text="Word Count Distribution",
            font=dict(size=14, color=_TITLE_COLOR, family="Space Grotesk, sans-serif"),
        ),
        xaxis_title="Word Count",
        yaxis_title="Frequency",
        bargap=0.15,
    )
    return _apply_axes(fig)
