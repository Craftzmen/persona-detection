"""Premium Streamlit dashboard for synthetic persona investigation.

Clean, professional design with proper visual hierarchy, consistent spacing,
well-structured sections, and a polished modern aesthetic.

IMPORTANT: Streamlit does NOT nest widgets inside HTML divs from st.markdown().
All card wrappers use st.container(border=True) instead of div wrappers.
"""

from __future__ import annotations

import json
import secrets
from typing import Any

import pandas as pd
import streamlit as st

from app.config import DASHBOARD_AUTH_PASSWORD, DASHBOARD_AUTH_USERNAME
from app.integration_service import analyze_user, generate_report, read_analysis_history
from app.ui.visuals import (
    build_daily_frequency_figure,
    build_hourly_activity_figure,
    build_network_figure,
    build_wordcount_distribution_figure,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CACHED ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@st.cache_data(ttl=600, show_spinner=False)
def _cached_analysis(username: str) -> dict[str, Any]:
    """Cache expensive analysis calls for responsiveness."""
    return analyze_user(username)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  THEME & DESIGN SYSTEM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _inject_theme(dark_mode: bool) -> None:
    """Inject the full design system as CSS."""

    if dark_mode:
        bg = "#0b0f19"
        surface = "rgba(15, 23, 42, 0.65)"
        text = "#e2e8f0"
        text_secondary = "#8896ab"
        border = "rgba(148, 163, 184, 0.10)"
        border_hover = "rgba(148, 163, 184, 0.20)"
        glow = "0 4px 24px rgba(0, 0, 0, 0.35)"
        sidebar_bg = "rgba(8, 12, 24, 0.96)"
        divider = "rgba(148, 163, 184, 0.07)"
        tab_bg = "rgba(15, 23, 42, 0.45)"
        tab_active_bg = "rgba(74, 109, 240, 0.12)"
        tab_active_text = "#6d8cfa"
        input_border = "rgba(148, 163, 184, 0.18)"
        input_bg = "rgba(15, 23, 42, 0.4)"
        container_border = "rgba(148, 163, 184, 0.10)"
        container_bg = "rgba(15, 23, 42, 0.5)"
        scrollbar_thumb = "rgba(148, 163, 184, 0.15)"
        hero_bg = "rgba(20, 29, 50, 0.85)"
    else:
        bg = "#f0f4f8"
        surface = "#ffffff"
        text = "#0f172a"
        text_secondary = "#566578"
        border = "rgba(15, 23, 42, 0.07)"
        border_hover = "rgba(15, 23, 42, 0.14)"
        glow = "0 4px 24px rgba(15, 23, 42, 0.06)"
        sidebar_bg = "rgba(255, 255, 255, 0.90)"
        divider = "rgba(15, 23, 42, 0.05)"
        tab_bg = "rgba(241, 245, 249, 0.7)"
        tab_active_bg = "rgba(74, 109, 240, 0.08)"
        tab_active_text = "#4a6df0"
        input_border = "rgba(15, 23, 42, 0.15)"
        input_bg = "#ffffff"
        container_border = "rgba(15, 23, 42, 0.08)"
        container_bg = "#ffffff"
        scrollbar_thumb = "rgba(15, 23, 42, 0.10)"
        hero_bg = "#ffffff"

    st.markdown(
        f"""
        <style>
        /* ── Fonts ─────────────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            -webkit-font-smoothing: antialiased;
        }}

        /* ── Page ──────────────────────────────────────────────── */
        .main, .stApp {{
            background: {bg};
        }}

        .block-container {{
            max-width: 1340px;
            padding: 1.25rem 2rem 3rem 2rem;
        }}

        /* ── Sidebar ───────────────────────────────────────────── */
        section[data-testid="stSidebar"] {{
            background: {sidebar_bg};
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-right: 1px solid {border};
        }}

        /* ── Typography ────────────────────────────────────────── */
        h1 {{
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 1.85rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.025em;
            color: {text} !important;
        }}
        h2 {{
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 1.4rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.015em;
            color: {text} !important;
        }}
        h3 {{
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 1.15rem !important;
            font-weight: 600 !important;
            color: {text} !important;
        }}
        h4 {{
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            color: {text} !important;
        }}
        p, li, label, [data-testid="stMarkdownContainer"] {{
            color: {text};
        }}

        /* ── Streamlit container(border=True) styling ──────────── */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: {container_bg};
            border: 1px solid {container_border} !important;
            border-radius: 12px !important;
            box-shadow: {glow};
            transition: border-color 0.2s ease;
        }}
        [data-testid="stVerticalBlockBorderWrapper"] > div:first-child {{
            padding: 24px !important;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:hover {{
            border-color: {border_hover} !important;
        }}

        /* ── Hero ──────────────────────────────────────────────── */
        .hero-section {{
            background: {hero_bg};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 32px 36px 28px 36px;
            margin: 56px 0 1.25rem 0;
            position: relative;
            overflow: hidden;
        }}
        .hero-eyebrow {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 0.68rem;
            text-transform: uppercase;
            font-weight: 700;
            color: #4a6df0;
            margin-bottom: 10px;
        }}
        .hero-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2rem;
            line-height: 1.15;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: {text};
            margin-bottom: 8px;
        }}
        .hero-sub {{
            font-size: 0.88rem;
            line-height: 1.65;
            color: {text_secondary};
            max-width: 640px;
        }}

        /* ── Command bar ───────────────────────────────────────── */
        .command-bar {{
            background: #ffffff;
            border: 1px solid {border};
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: {glow};
            margin-bottom: 1.25rem;
        }}
        .command-label {{
            font-size: 0.68rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-weight: 700;
            color: {text_secondary};
            margin-bottom: 8px;
        }}

        /* ── Text inputs — visible border ──────────────────────── */
        .stTextInput > div > div > input {{
            border: 1px solid {input_border} !important;
            border-radius: 8px !important;
            background: {input_bg} !important;
            color: {text} !important;
            padding: 10px 14px !important;
            font-size: 0.9rem !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        }}
        .stTextInput > div > div > input:focus {{
            border-color: #4a6df0 !important;
            box-shadow: 0 0 0 3px rgba(74, 109, 240, 0.12) !important;
        }}
        .stTextInput > div > div > input::placeholder {{
            color: {text_secondary} !important;
            opacity: 0.7;
        }}

        /* ── KPI card ──────────────────────────────────────────── */
        .kpi-card {{
            background: {surface};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 18px 20px 16px 20px;
            box-shadow: {glow};
            min-height: 100px;
            position: relative;
            overflow: hidden;
            transition: transform 0.25s ease, border-color 0.25s ease;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            border-color: {border_hover};
        }}
        .kpi-card::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            border-radius: 12px 12px 0 0;
        }}
        .kpi-label {{
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: {text_secondary};
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .kpi-value {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.75rem;
            font-weight: 700;
            line-height: 1.15;
            color: {text};
        }}

        /* ── Section header ────────────────────────────────────── */
        .section-header {{
            display: flex;
            align-items: center;
            margin-top: 0.5rem;
            margin-bottom: 0.875rem;
            padding-bottom: 8px;
            border-bottom: 1px solid {divider};
        }}
        .section-header .section-text {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.15rem;
            font-weight: 700;
            color: {text};
        }}

        /* ── Risk badge ────────────────────────────────────────── */
        .risk-badge {{
            display: inline-flex;
            align-items: center;
            gap: 7px;
            padding: 7px 16px;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            color: #fff;
            letter-spacing: 0.015em;
        }}
        .risk-badge .badge-dot {{
            width: 6px; height: 6px;
            border-radius: 50%;
            background: rgba(255,255,255,0.6);
        }}

        /* ── Score gauge ───────────────────────────────────────── */
        .score-gauge {{
            background: {surface};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 16px 20px;
            text-align: center;
        }}
        .secondary-stat {{
            margin-top: 12px;
        }}
        .stTabs {{
            margin-top: 40px;
        }}
        .gauge-label {{
            font-size: 0.68rem;
            text-transform: uppercase;
            color: {text_secondary};
            font-weight: 600;
            margin-bottom: 2px;
        }}
        .gauge-value {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.4rem;
            font-weight: 700;
            background: linear-gradient(90deg, #4a6df0, #2f9e8f);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.2;
        }}

        /* ── Persona chips ─────────────────────────────────────── */
        .persona-chip {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            border: 1px solid {border};
            background: {tab_bg};
            color: {text};
            margin: 3px 3px;
            transition: border-color 0.2s ease, background 0.2s ease;
        }}
        .persona-chip:hover {{
            border-color: #4a6df0;
            background: {tab_active_bg};
        }}

        /* ── Metric mini card ──────────────────────────────────── */
        .metric-mini {{
            background: {surface};
            border: 1px solid {border};
            border-radius: 10px;
            padding: 14px 16px;
            text-align: center;
            transition: transform 0.2s ease;
        }}
        .metric-mini:hover {{
            transform: translateY(-1px);
        }}
        .metric-mini .mini-label {{
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {text_secondary};
            font-weight: 600;
            margin-bottom: 4px;
        }}
        .metric-mini .mini-value {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.3rem;
            font-weight: 700;
            color: {text};
        }}

        /* ── Export card ───────────────────────────────────────── */
        .export-card {{
            background: {surface};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: transform 0.2s ease;
            margin-bottom: 8px;
        }}
        .export-card:hover {{
            transform: translateY(-1px);
        }}
        .export-card .export-title {{
            font-weight: 700;
            font-size: 0.9rem;
            color: {text};
            margin-bottom: 4px;
        }}
        .export-card .export-desc {{
            font-size: 0.75rem;
            color: {text_secondary};
        }}

        /* ── Findings list ─────────────────────────────────────── */
        .finding-item {{
            display: flex;
            align-items: flex-start;
            gap: 10px;
            padding: 7px 0;
            border-bottom: 1px solid {divider};
        }}
        .finding-item:last-child {{ border-bottom: none; }}
        .finding-dot {{
            width: 7px; height: 7px;
            border-radius: 50%;
            margin-top: 6px;
            flex-shrink: 0;
        }}
        .finding-text {{
            font-size: 0.85rem;
            color: {text};
            line-height: 1.5;
        }}

        /* ── Empty state ───────────────────────────────────────── */
        .empty-state {{
            text-align: center;
            padding: 40px 24px;
            color: {text_secondary};
        }}
        .empty-state .empty-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.05rem;
            font-weight: 600;
            color: {text};
            margin-bottom: 6px;
        }}
        .empty-state .empty-desc {{
            font-size: 0.85rem;
            max-width: 380px;
            margin: 0 auto;
            line-height: 1.55;
        }}

        /* ── Login card ────────────────────────────────────────── */
        .login-card {{
            background: {surface};
            border: 1px solid {border};
            border-radius: 16px;
            padding: 36px 32px;
            box-shadow: {glow};
            max-width: 400px;
            margin: 60px auto 0 auto;
            text-align: center;
        }}
        .login-card .login-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.3rem;
            font-weight: 700;
            color: {text};
            margin-bottom: 4px;
        }}
        .login-card .login-sub {{
            font-size: 0.82rem;
            color: {text_secondary};
            margin-bottom: 20px;
        }}

        /* ── Tabs ──────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 2px;
            background: {tab_bg};
            border-radius: 10px;
            padding: 3px;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 7px;
            font-weight: 600;
            font-size: 0.82rem;
            padding: 7px 14px;
            color: {text_secondary};
        }}
        .stTabs [aria-selected="true"] {{
            background: {tab_active_bg} !important;
            color: {tab_active_text} !important;
        }}

        /* ── Primary button — white text ───────────────────────── */
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="baseButton-primary"],
        [data-testid="stFormSubmitButton"] > button {{
            background: linear-gradient(135deg, #4a6df0, #3b5de7) !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            color: #ffffff !important;
            transition: all 0.2s ease;
        }}
        .stButton > button[kind="primary"]:hover,
        .stButton > button[data-testid="baseButton-primary"]:hover,
        [data-testid="stFormSubmitButton"] > button:hover {{
            background: linear-gradient(135deg, #5b7df5, #4a6df0) !important;
            box-shadow: 0 4px 16px rgba(74,109,240,0.25) !important;
            color: #ffffff !important;
        }}
        .stButton > button[kind="primary"] p,
        .stButton > button[data-testid="baseButton-primary"] p,
        [data-testid="stFormSubmitButton"] > button p {{
            color: #ffffff !important;
        }}

        /* ── Download buttons ──────────────────────────────────── */
        .stDownloadButton > button {{
            border: 1px solid {border} !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            color: {text} !important;
            background: {surface} !important;
        }}
        .stDownloadButton > button:hover {{
            border-color: #4a6df0 !important;
            color: #4a6df0 !important;
        }}

        /* ── Dataframe ─────────────────────────────────────────── */
        [data-testid="stDataFrame"] {{
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid {border};
        }}

        /* ── Scrollbar ─────────────────────────────────────────── */
        ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: {scrollbar_thumb}; border-radius: 999px; }}

        /* ── Sidebar branding ──────────────────────────────────── */
        .sidebar-brand {{
            text-align: center;
            padding: 4px 0 18px 0;
            border-bottom: 1px solid {divider};
            margin-bottom: 14px;
        }}
        .sidebar-brand .brand-name {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.2rem;
            font-weight: 700;
            color: {text};
            letter-spacing: 0.1em;
        }}
        .sidebar-brand .brand-version {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 0.62rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            background: rgba(74,109,240,0.10);
            color: #4a6df0;
            margin-top: 4px;
        }}
        .sidebar-stat {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 7px 0;
            border-bottom: 1px solid {divider};
            font-size: 0.78rem;
        }}
        .sidebar-stat .stat-label {{
            color: {text_secondary};
            font-weight: 500;
        }}
        .sidebar-stat .stat-value {{
            color: {text};
            font-weight: 700;
            font-family: 'Space Grotesk', sans-serif;
        }}

        /* ── Metrics override ──────────────────────────────────── */
        [data-testid="stMetric"] {{
            background: {surface};
            border: 1px solid {border};
            border-radius: 10px;
            padding: 14px 16px;
        }}
        [data-testid="stMetric"] label {{
            color: {text_secondary} !important;
            font-size: 0.72rem !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600 !important;
        }}
        [data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: {text} !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 700 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  COMPONENT HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _risk_badge(risk_level: str) -> str:
    """Return styled risk badge HTML."""
    colors = {"Low": "#059669", "Medium": "#d97706", "High": "#dc2626"}
    bg = colors.get(risk_level, "#475569")
    return (
        f'<span class="risk-badge" style="background:{bg};">'
        f'<span class="badge-dot"></span>{risk_level} Risk</span>'
    )


def _kpi_card(label: str, value: str, accent: str = "#4a6df0") -> str:
    """Return markup for a KPI card with gradient top accent."""
    return (
        f'<div class="kpi-card" style="--accent:{accent};">'
        f'<style>.kpi-card[style*="--accent:{accent}"]::before'
        f'{{ background: linear-gradient(90deg, {accent}, {accent}66); }}</style>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'</div>'
    )


def _section_header(text: str) -> str:
    """Return styled section header HTML."""
    return (
        f'<div class="section-header">'
        f'<span class="section-text">{text}</span>'
        f'</div>'
    )


def _empty_state(title: str, description: str) -> str:
    """Return styled empty state HTML."""
    return (
        f'<div class="empty-state">'
        f'<div class="empty-title">{title}</div>'
        f'<div class="empty-desc">{description}</div>'
        f'</div>'
    )


def _metric_mini(label: str, value: str) -> str:
    """Return markup for a compact metric card."""
    return (
        f'<div class="metric-mini">'
        f'<div class="mini-label">{label}</div>'
        f'<div class="mini-value">{value}</div>'
        f'</div>'
    )


def _persona_chips(names: list[str]) -> str:
    """Render persona names as styled chips."""
    if not names:
        return '<div style="font-size:0.82rem;color:#8896ab;padding:6px 0;">No linked personas detected</div>'
    return "".join(f'<span class="persona-chip">@{n}</span>' for n in names)


def _confidence_text(score: float) -> str:
    """Return human-readable description for synthetic score."""
    if score < 0.3:
        return "Minimal synthetic indicators. Activity patterns appear organic."
    if score < 0.5:
        return "Some synthetic markers present. Light monitoring recommended."
    if score < 0.7:
        return "Moderate synthetic behavior detected. Multiple indicators suggest coordinated activity."
    return "Strong synthetic indicators. This account exhibits multiple hallmarks of artificial generation."


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AUTH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _is_auth_enabled() -> bool:
    return bool(DASHBOARD_AUTH_USERNAME and DASHBOARD_AUTH_PASSWORD)


def _authenticate_dashboard() -> bool:
    """Render branded auth card and validate credentials."""
    if not _is_auth_enabled():
        return True
    if st.session_state.get("dashboard_authenticated", False):
        return True

    st.markdown(
        '<div class="login-card">'
        '<div class="login-title">Secure Access</div>'
        '<div class="login-sub">Enter your credentials to access the console.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    col_l, col_form, col_r = st.columns([1, 2, 1])
    with col_form:
        with st.form("login_form", border=False):
            username = st.text_input("Username", key="dashboard_auth_user", placeholder="Username")
            password = st.text_input("Password", type="password", key="dashboard_auth_password", placeholder="Password")
            login_clicked = st.form_submit_button("Sign In", type="primary", use_container_width=True)

    if login_clicked:
        if secrets.compare_digest(username, DASHBOARD_AUTH_USERNAME) and secrets.compare_digest(password, DASHBOARD_AUTH_PASSWORD):
            st.session_state["dashboard_authenticated"] = True
            st.success("Authentication successful.")
            st.rerun()
        else:
            st.error("Invalid credentials.")

    return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HISTORY & COMPARISON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _render_history_and_comparison() -> None:
    """Display analysis history and side-by-side comparison."""
    history = read_analysis_history(limit=250)
    if not history:
        return

    st.markdown("---")
    st.markdown(_section_header("Analysis History"), unsafe_allow_html=True)
    history_df = pd.DataFrame(history)
    st.dataframe(history_df, use_container_width=True, height=230)

    if "username" not in history_df.columns:
        return

    usernames = sorted(history_df["username"].dropna().astype(str).unique().tolist())
    if len(usernames) < 2:
        return

    st.markdown(_section_header("Side-by-Side Comparison"), unsafe_allow_html=True)

    sel_l, sel_r = st.columns(2)
    user_a = sel_l.selectbox("Persona A", options=usernames, index=0, key="compare_user_a")
    user_b = sel_r.selectbox("Persona B", options=usernames, index=1, key="compare_user_b")

    if user_a == user_b:
        st.info("Choose two distinct personas to compare.")
        return

    with st.spinner("Computing comparative analysis..."):
        result_a = _cached_analysis(user_a)
        result_b = _cached_analysis(user_b)

    left_panel, right_panel = st.columns(2)

    with left_panel:
        with st.container(border=True):
            st.markdown(f"#### @{result_a['username']}")
            a1, a2 = st.columns(2)
            a1.markdown(_metric_mini("Synth Score", f"{result_a['synthetic_score']:.3f}"), unsafe_allow_html=True)
            a2.markdown(_metric_mini("Risk", result_a["risk_level"]), unsafe_allow_html=True)
            st.plotly_chart(
                build_network_figure(result_a.get("network_graph", {}), title=f"Network: @{result_a['username']}"),
                use_container_width=True,
            )

    with right_panel:
        with st.container(border=True):
            st.markdown(f"#### @{result_b['username']}")
            b1, b2 = st.columns(2)
            b1.markdown(_metric_mini("Synth Score", f"{result_b['synthetic_score']:.3f}"), unsafe_allow_html=True)
            b2.markdown(_metric_mini("Risk", result_b["risk_level"]), unsafe_allow_html=True)
            st.plotly_chart(
                build_network_figure(result_b.get("network_graph", {}), title=f"Network: @{result_b['username']}"),
                use_container_width=True,
            )

    # Delta
    delta = result_b["synthetic_score"] - result_a["synthetic_score"]
    sign = "+" if delta >= 0 else ""
    color = "#ef4444" if delta > 0 else "#10b981"
    st.markdown(
        f'<div style="text-align:center;padding:10px 0;font-size:0.85rem;">'
        f'Score Delta: <span style="font-weight:700;color:{color};font-family:Space Grotesk,sans-serif;">'
        f'{sign}{delta:.3f}</span></div>',
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _render_sidebar() -> bool:
    """Render sidebar with branding. Returns dark_mode flag."""
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">'
            '<div class="brand-name">SYNAPSE</div>'
            '<span class="brand-version">v1.0 · OSINT</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        dark_mode = st.toggle("Dark mode", value=False)
        st.markdown("---")

        st.markdown(
            '<div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.12em;'
            'font-weight:700;color:#8896ab;margin-bottom:8px;">Quick Stats</div>',
            unsafe_allow_html=True,
        )

        history = read_analysis_history(limit=500)
        total = len(history)

        if total > 0:
            risk_counts: dict[str, int] = {}
            for entry in history:
                r = entry.get("risk_level", "Unknown")
                risk_counts[r] = risk_counts.get(r, 0) + 1
            top_risk = max(risk_counts, key=risk_counts.get)  # type: ignore[arg-type]
            last_ts = history[0].get("timestamp", "—")
            if isinstance(last_ts, str) and len(last_ts) > 16:
                last_ts = last_ts[:16].replace("T", " ")

            st.markdown(
                f'<div class="sidebar-stat">'
                f'<span class="stat-label">Total analyses</span>'
                f'<span class="stat-value">{total}</span></div>'
                f'<div class="sidebar-stat">'
                f'<span class="stat-label">Top risk level</span>'
                f'<span class="stat-value">{top_risk}</span></div>'
                f'<div class="sidebar-stat">'
                f'<span class="stat-label">Latest run</span>'
                f'<span class="stat-value">{last_ts}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="font-size:0.78rem;color:#64748b;padding:4px 0;">'
                'No analyses recorded yet.</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.caption("Streamlit · Dashboard")

    return dark_mode



def render_dashboard() -> None:
    """Render the full dashboard UI."""

    st.set_page_config(
        page_title="SYNAPSE — Persona Investigation",
        page_icon="S",
        layout="wide",
    )

    dark_mode = _render_sidebar()
    _inject_theme(dark_mode=dark_mode)

    # ── Hero ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-section">
            <div class="hero-eyebrow">
                OSINT Detection Console
            </div>
            <div class="hero-title">Synthetic Persona Investigation</div>
            <div class="hero-sub">
                Collect behavioral signals, score synthetic behavior, inspect attribution
                networks, and export investigation artifacts — all from a single command bar.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Auth ────────────────────────────────────────────────────────────────
    if not _authenticate_dashboard():
        return

    # ── Command bar ─────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown(
            '<div style="font-size:0.68rem;letter-spacing:0.14em;text-transform:uppercase;'
            'font-weight:700;color:#566578;margin-bottom:8px;">Analysis Command Bar</div>',
            unsafe_allow_html=True,
        )
        with st.form("analysis_form", border=False):
            in_col, action_col = st.columns([4, 1])
            username = in_col.text_input(
                "Target Username",
                value="",
                placeholder="Enter username to investigate (e.g., nasa)",
                label_visibility="collapsed",
            )
            analyze_clicked = action_col.form_submit_button("Analyze", type="primary", use_container_width=True)

    # ── No analysis state ───────────────────────────────────────────────────
    if not analyze_clicked:
        if not read_analysis_history(limit=1):
            st.markdown(
                _empty_state(
                    "Ready to investigate",
                    "Enter a username above and click Analyze to begin your first synthetic persona investigation.",
                ),
                unsafe_allow_html=True,
            )
        _render_history_and_comparison()
        return

    if not username.strip():
        st.warning("Please provide a username to analyze.")
        return

    # ── Run pipeline ────────────────────────────────────────────────────────
    with st.spinner("Running full OSINT pipeline..."):
        result = _cached_analysis(username.strip())

    # ── KPIs ────────────────────────────────────────────────────────────────
    st.markdown(_section_header("Detection Summary"), unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(_kpi_card("Prediction", result["prediction"], accent="#4a6df0"), unsafe_allow_html=True)
    k2.markdown(_kpi_card("Synthetic Score", f"{result['synthetic_score']:.3f}", accent="#2f9e8f"), unsafe_allow_html=True)
    k3.markdown(_kpi_card("Risk Level", result["risk_level"], accent="#f59e0b"), unsafe_allow_html=True)
    k4.markdown(_kpi_card("Cluster ID", str(int(result.get("cluster_id", -1))), accent="#8b5cf6"), unsafe_allow_html=True)

    # ── Score gauge + risk badge + description ──────────────────────────────
    g_col, b_col, d_col = st.columns([1, 1, 2])

    g_col.markdown(
        f'<div class="score-gauge secondary-stat">'
        f'<div class="gauge-label">Synthetic Probability</div>'
        f'<div class="gauge-value">{result["synthetic_score"] * 100:.1f}%</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    b_col.markdown(
        f'<div style="display:flex;align-items:center;height:100%;justify-content:center;margin-top:12px;">'
        f'{_risk_badge(result["risk_level"])}'
        f'</div>',
        unsafe_allow_html=True,
    )

    d_col.markdown(
        f'<div style="display:flex;align-items:center;height:100%;margin-top:12px;">'
        f'<div style="font-size:0.85rem;line-height:1.6;color:#8896ab;">'
        f'{_confidence_text(result["synthetic_score"])}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ── Tabs ────────────────────────────────────────────────────────────────
    timeline = result.get("timeline", {})
    linked = result.get("linked_personas", [])
    behavioral = result.get("behavioral_features", {})
    stylometric = result.get("stylometric_features", {})

    overview_tab, behavior_tab, style_tab, attribution_tab, export_tab = st.tabs(
        ["Overview", "Behavior", "Stylometry", "Attribution", "Exports"]
    )

    # ── Overview ────────────────────────────────────────────────────────────
    with overview_tab:
        ov1, ov2 = st.columns([3, 2])

        with ov1:
            with st.container(border=True):
                st.markdown("#### Investigation Snapshot")
                st.markdown(
                    "This persona was evaluated through behavioral cadence analysis, "
                    "stylometric consistency checks, and cross-persona linkage scoring."
                )

                findings = []
                if result["prediction"] == "AI":
                    findings.append(("#ef4444", "Classified as synthetic / AI-generated persona"))
                else:
                    findings.append(("#10b981", "Classified as organic / human-operated account"))

                if result["synthetic_score"] > 0.7:
                    findings.append(("#ef4444", f"High synthetic probability: {result['synthetic_score']:.1%}"))
                elif result["synthetic_score"] > 0.4:
                    findings.append(("#f59e0b", f"Moderate synthetic probability: {result['synthetic_score']:.1%}"))
                else:
                    findings.append(("#10b981", f"Low synthetic probability: {result['synthetic_score']:.1%}"))

                cluster_id = int(result.get("cluster_id", -1))
                if cluster_id >= 0:
                    findings.append(("#8b5cf6", f"Belongs to attribution cluster #{cluster_id}"))
                if linked:
                    findings.append(("#4a6df0", f"{len(linked)} linked persona(s) detected via similarity"))

                html = ""
                for color, text in findings:
                    html += (
                        f'<div class="finding-item">'
                        f'<div class="finding-dot" style="background:{color};"></div>'
                        f'<div class="finding-text">{text}</div></div>'
                    )
                st.markdown(html, unsafe_allow_html=True)

        with ov2:
            with st.container(border=True):
                st.markdown("#### Linked Personas")
                st.markdown(_persona_chips(linked[:10]), unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown("#### Quick Stats")
                qm1, qm2 = st.columns(2)
                qm1.markdown(_metric_mini("Posts/Day", f"{behavioral.get('posts_per_day', 0):.1f}"), unsafe_allow_html=True)
                qm2.markdown(_metric_mini("Night Ratio", f"{behavioral.get('night_activity_ratio', 0):.2f}"), unsafe_allow_html=True)

    # ── Behavior ────────────────────────────────────────────────────────────
    with behavior_tab:
        st.markdown(_section_header("Behavioral Insights"), unsafe_allow_html=True)

        hour_dist = timeline.get("hour_distribution", {})
        day_dist = timeline.get("day_distribution", {})
        peak_hour = max(hour_dist, key=lambda h: hour_dist.get(h, 0), default="—") if hour_dist else "—"
        peak_day = max(day_dist, key=lambda d: day_dist.get(d, 0), default="—") if day_dist else "—"

        bm1, bm2, bm3, bm4 = st.columns(4)
        bm1.markdown(_metric_mini("Peak Hour", f"{peak_hour}:00" if peak_hour != "—" else "—"), unsafe_allow_html=True)
        bm2.markdown(_metric_mini("Peak Day", str(peak_day)[:3] if peak_day != "—" else "—"), unsafe_allow_html=True)
        bm3.markdown(_metric_mini("Posts/Day", f"{behavioral.get('posts_per_day', 0):.1f}"), unsafe_allow_html=True)
        bm4.markdown(_metric_mini("Night Activity", f"{behavioral.get('night_activity_ratio', 0):.1%}"), unsafe_allow_html=True)

        st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)

        hour_df = pd.DataFrame({"hour": list(range(24)), "posts": [hour_dist.get(str(i), 0) for i in range(24)]})
        day_df = pd.DataFrame({"day": list(day_dist.keys()), "posts": list(day_dist.values())})

        c_l, c_r = st.columns(2)
        with c_l:
            with st.container(border=True):
                st.plotly_chart(build_hourly_activity_figure(hour_df), use_container_width=True)
        with c_r:
            with st.container(border=True):
                st.plotly_chart(build_daily_frequency_figure(day_df), use_container_width=True)

    # ── Stylometry ──────────────────────────────────────────────────────────
    with style_tab:
        st.markdown(_section_header("Stylometric Profile"), unsafe_allow_html=True)

        sm1, sm2, sm3 = st.columns(3)
        sm1.markdown(_metric_mini("Vocabulary Richness", f"{timeline.get('vocabulary_richness', 0):.3f}"), unsafe_allow_html=True)
        sm2.markdown(_metric_mini("Avg Word Length", f"{timeline.get('avg_word_length', 0):.2f}"), unsafe_allow_html=True)
        sm3.markdown(_metric_mini("Avg Sentence Len", f"{stylometric.get('avg_sentence_length', 0):.1f}"), unsafe_allow_html=True)

        st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)

        word_counts = timeline.get("word_count_series", [])
        if word_counts:
            with st.container(border=True):
                wc_df = pd.DataFrame({"word_count": word_counts})
                st.plotly_chart(build_wordcount_distribution_figure(wc_df), use_container_width=True)
        else:
            st.markdown(_empty_state("No word count data", "Insufficient text samples for distribution analysis."), unsafe_allow_html=True)

        # Writing profile
        with st.container(border=True):
            st.markdown("#### Writing Profile Summary")
            vocab = timeline.get("vocabulary_richness", 0)
            grammar = stylometric.get("grammar_consistency", 0)
            punct = stylometric.get("punctuation_usage", 0)

            points = []
            if vocab > 0.6:
                points.append("High vocabulary diversity — suggests varied or sophisticated language use")
            elif vocab > 0.3:
                points.append("Moderate vocabulary diversity — consistent with typical social media usage")
            else:
                points.append("Low vocabulary diversity — may indicate repetitive or templated content")
            if grammar > 0.7:
                points.append("High grammar consistency — uniform writing mechanics across posts")
            if punct > 0.05:
                points.append("Notable punctuation density — frequent use of special characters")
            if not points:
                points.append("Insufficient data for detailed stylometric profiling")
            for p in points:
                st.markdown(f"- {p}")

    # ── Attribution ─────────────────────────────────────────────────────────
    with attribution_tab:
        st.markdown(_section_header("Attribution & Clustering"), unsafe_allow_html=True)

        at1, at2 = st.columns([1, 2])
        with at1:
            cid = int(result.get("cluster_id", -1))
            c_label = f"#{cid}" if cid >= 0 else "Unassigned"
            c_color = "#8b5cf6" if cid >= 0 else "#64748b"
            st.markdown(
                f'<div class="score-gauge" >'
                f'<div class="gauge-label">Cluster Assignment</div>'
                f'<div style="font-family:Space Grotesk,sans-serif;font-size:2.2rem;font-weight:700;'
                f'color:{c_color};line-height:1.2;">{c_label}</div></div>',
                unsafe_allow_html=True,
            )

        with at2:
            with st.container(border=True):
                st.markdown("#### Linked Personas")
                st.markdown(_persona_chips(linked), unsafe_allow_html=True)

        with st.container(border=True):
            st.plotly_chart(build_network_figure(result.get("network_graph", {})), use_container_width=True)

    # ── Exports ─────────────────────────────────────────────────────────────
    with export_tab:
        st.markdown(_section_header("Export Investigation Data"), unsafe_allow_html=True)

        ex1, ex2 = st.columns(2)

        with ex1:
            st.markdown(
                '<div class="export-card">'
                '<div class="export-title">JSON Analysis Data</div>'
                '<div class="export-desc">Machine-readable structured analysis output</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                label="Download JSON",
                data=json.dumps(result, indent=2),
                file_name=f"analysis_{result['username']}.json",
                mime="application/json",
                use_container_width=True,
            )

        with ex2:
            st.markdown(
                '<div class="export-card">'
                '<div class="export-title">PDF Investigation Report</div>'
                '<div class="export-desc">Formatted report for stakeholder review</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            pdf_bytes = generate_report(result)
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=f"report_{result['username']}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        with st.expander("View raw analysis payload"):
            st.json(result)

    # ── History ─────────────────────────────────────────────────────────────
    _render_history_and_comparison()
