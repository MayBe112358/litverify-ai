"""Settings dialog launched from the sidebar.

Bundles everything that used to live as a settings expander / separate page:
- Theme
- DeepSeek API key + model tier
- External source toggles (CrossRef / OpenAlex / arXiv)
- Verdict thresholds
- Rule weights with YAML import/export
"""
from __future__ import annotations

import streamlit as st

from config.settings import settings
from services.rule_engine import load_rule_config
from ui.theme import THEMES, DEFAULT_THEME

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


def _rule_dict(rule) -> dict:
    return {
        "id": rule.id,
        "name": rule.name,
        "enabled": rule.enabled,
        "weight": float(rule.weight),
        "params": dict(rule.params or {}),
    }


@st.dialog("设置", width="large")
def open_settings_dialog() -> None:
    tabs = st.tabs(["外观与模型", "数据源与阈值", "验证规则"])

    # ----- Tab 1: appearance + model -----
    with tabs[0]:
        st.caption("外观")
        theme_names = list(THEMES.keys())
        current_theme = st.session_state.get("theme_name", DEFAULT_THEME)
        if current_theme not in theme_names:
            current_theme = DEFAULT_THEME
        choice = st.radio(
            "主题",
            theme_names,
            index=theme_names.index(current_theme),
            horizontal=True,
            key="settings_theme_radio",
        )
        if choice != current_theme:
            st.session_state["theme_name"] = choice
            st.rerun()

        st.divider()
        st.caption("DeepSeek")
        api_key = st.text_input(
            "API Key",
            value=st.session_state.get("deepseek_api_key", ""),
            type="password",
            placeholder="sk-...",
            key="settings_api_key_input",
        )
        if api_key != st.session_state.get("deepseek_api_key", ""):
            st.session_state["deepseek_api_key"] = api_key

        current_model = st.session_state.get("deepseek_chat_model", settings.chat_model)
        is_flash = current_model == settings.deepseek_flash_model
        tier = st.radio(
            "模型",
            ["Pro", "Flash"],
            index=1 if is_flash else 0,
            horizontal=True,
            key="settings_model_tier",
        )
        chosen_model = settings.deepseek_flash_model if tier == "Flash" else settings.deepseek_pro_model
        st.session_state["deepseek_chat_model"] = chosen_model
        st.session_state["deepseek_vl_model"] = chosen_model
        st.caption(f"当前调用：`{chosen_model}`")

    # ----- Tab 2: data sources + thresholds -----
    with tabs[1]:
        st.caption("数据源")
        cols = st.columns(3)
        cols[0].toggle("CrossRef", key="crossref_enabled")
        cols[1].toggle("OpenAlex", key="openalex_enabled")
        cols[2].toggle("arXiv", key="arxiv_enabled")

        st.divider()
        st.caption("判定阈值")
        st.slider("真实阈值", min_value=50, max_value=100, key="real_threshold")
        st.slider("可疑阈值", min_value=0, max_value=90, key="suspicious_threshold")
        r = int(st.session_state.get("real_threshold", 80))
        s = int(st.session_state.get("suspicious_threshold", 40))
        if s >= r:
            st.caption(
                f"≥ {r} 为真实；可疑阈值 {s} 高于真实阈值，"
                f"验证时会自动按 {max(0, r - 10)} 钳制。"
            )
        else:
            st.caption(f"≥ {r} 为真实；≥ {s} 为可疑。")

        st.divider()
        st.caption("学术 API（只读，配置在 .env）")
        cols = st.columns(2)
        cols[0].text_input("CrossRef Email", value=settings.crossref_email or "—", disabled=True)
        cols[1].text_input("OpenAlex Email", value=settings.openalex_email or "—", disabled=True)

    # ----- Tab 3: rule editor -----
    with tabs[2]:
        default_rules, default_thresholds = load_rule_config()
        default_overrides = [_rule_dict(r) for r in default_rules]
        current = st.session_state.get("rule_overrides_v2") or default_overrides

        top_l, top_r = st.columns([3, 1])
        with top_l:
            uploaded = st.file_uploader(
                "导入 YAML 规则",
                type=["yaml", "yml"],
                key="settings_rule_yaml",
                label_visibility="collapsed",
            )
        with top_r:
            if st.button("恢复默认", use_container_width=True, key="settings_rule_reset"):
                st.session_state.pop("rule_overrides_v2", None)
                for key in list(st.session_state.keys()):
                    if key.startswith("rule_enabled_") or key.startswith("rule_weight_"):
                        try:
                            del st.session_state[key]
                        except Exception:  # noqa: BLE001
                            pass
                st.toast("已恢复默认规则。", icon="↩️")
                st.rerun()

        if uploaded is not None and yaml is not None:
            try:
                payload = yaml.safe_load(uploaded.read().decode("utf-8")) or {}
                imported = [
                    {
                        "id": item["id"],
                        "name": item.get("name", item["id"]),
                        "enabled": bool(item.get("enabled", True)),
                        "weight": float(item.get("weight", 1)),
                        "params": dict(item.get("params") or {}),
                    }
                    for item in payload.get("rules") or []
                ]
                if imported:
                    st.session_state["rule_overrides_v2"] = imported
                    st.success(f"已载入 {len(imported)} 条规则。")
                    st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"YAML 解析失败：{exc}")

        edited: list[dict] = []
        for item in current:
            rid = item["id"]
            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 1, 2])
                enabled = c1.toggle(item["name"], value=item["enabled"], key=f"rule_enabled_{rid}")
                weight = c2.slider("权重", 0, 30, int(item["weight"]), key=f"rule_weight_{rid}")
                c3.code(rid, language=None)
                edited.append({
                    "id": rid,
                    "name": item["name"],
                    "enabled": enabled,
                    "weight": weight,
                    "params": dict(item.get("params") or {}),
                })
        st.session_state["rule_overrides_v2"] = edited

        weight_sum = sum(it["weight"] for it in edited if it["enabled"])
        m1, m2 = st.columns(2)
        m1.metric("启用权重之和", weight_sum)
        m2.metric("生效规则数", sum(1 for it in edited if it["enabled"]))

        if yaml is not None:
            payload = {
                "rules": edited,
                "thresholds": {
                    "real": int(st.session_state.get("real_threshold", 80)),
                    "suspicious": int(st.session_state.get("suspicious_threshold", 40)),
                },
            }
            yaml_text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
            st.download_button(
                "⬇️ 导出当前规则 YAML",
                yaml_text.encode("utf-8"),
                "rules_default.yaml",
                "text/yaml",
                use_container_width=True,
            )
