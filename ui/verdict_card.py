"""Compact verdict card — rendered inline inside an assistant chat bubble."""
from __future__ import annotations

import streamlit as st

from services.rule_engine import VerificationReport
from ui.components import safe_html


PALETTE: dict[str, tuple[str, str, str, str]] = {
    "REAL":       ("真实可信",   "#10B981", "✓", "多源权威记录交叉印证通过"),
    "SUSPICIOUS": ("高度可疑",   "#F59E0B", "!", "部分字段不一致，建议人工复核"),
    "FAKE":       ("极可能虚假", "#EF4444", "✕", "关键字段缺失或与权威源冲突"),
    "ERROR":      ("验证失败",   "#64748B", "?", "未能完成验证，请重试或检查输入"),
}


def render_verdict_card(report: VerificationReport) -> None:
    """Compact card sized to fit inside an assistant chat message."""
    label, color, icon, hint = PALETTE.get(report.verdict, PALETTE["ERROR"])
    score = report.overall_score
    pct = max(0, min(100, int(score)))

    st.markdown(
        safe_html(f"""
        <div class="dw-verdict" style="
            border-left: 4px solid {color};
            background:
                radial-gradient(700px 220px at 0% 0%,
                    color-mix(in srgb, {color} 12%, transparent) 0%, transparent 60%),
                var(--dw-surface);
        ">
          <div style="display:flex; align-items:center; gap:0.9rem;">
            <div style="
                flex: 0 0 auto;
                width: 42px; height: 42px;
                border-radius: 12px;
                background: {color};
                color: #fff;
                font-size: 20px; font-weight: 800;
                display:flex; align-items:center; justify-content:center;
            ">{icon}</div>
            <div style="flex: 1 1 auto; min-width: 0;">
              <div style="
                  font-size: 1.05rem; font-weight: 700;
                  color: var(--dw-text); letter-spacing: -0.01em;
              ">{label}
                <span style="
                    font-size:0.72rem; font-weight:600;
                    color:{color}; margin-left:0.4rem;
                    letter-spacing:0.08em; text-transform:uppercase;
                ">{report.verdict}</span>
              </div>
              <div style="
                  font-size: 0.84rem; color: var(--dw-muted);
                  margin-top: 0.15rem;
              ">{hint}</div>
            </div>
            <div style="
                flex: 0 0 auto;
                font-size: 1.7rem; font-weight: 800;
                color: {color}; line-height: 1;
                letter-spacing: -0.03em;
            ">{score}<span style="
                font-size: 0.75rem; color: var(--dw-muted);
                font-weight: 500; letter-spacing: 0;
            "> /100</span></div>
          </div>
          <div style="
              margin-top: 0.85rem; height: 6px; border-radius: 999px;
              background: color-mix(in srgb, {color} 12%, var(--dw-surface-alt));
              overflow: hidden;
          ">
            <div style="
                width: {pct}%; height: 100%;
                background: {color}; border-radius: 999px;
                transition: width 480ms cubic-bezier(0.22, 1, 0.36, 1);
            "></div>
          </div>
        </div>
        """),
        unsafe_allow_html=True,
    )
