"""
report_generator.py — 多模态可视化报告生成器 v2

新增功能：
1. Chart.js 统计图表（置信度对比、面积分布、证据距离、标签频次）
2. 农民友好的灾情预警通知（通俗语言）
3. 双视角补救建议（农户指南 + 研究分析）
4. 更美观的 HTML 模板
"""

from __future__ import annotations

import base64
import json
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..types import (
    AgentReport,
    RetrievalHit,
    SeverityLevel,
    UrgencyLevel,
)
from .visual_annotator import (
    SEVERITY_LABELS_CN,
    VisualAnnotator,
)


def _img_to_base64(path: str) -> str:
    """将图片文件转为 base64 data URI"""
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(path)[1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext.lstrip("."), "image/png")
    return f"data:{mime};base64,{data}"


def _read_template() -> str:
    """读取 HTML 模板"""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "planting_report.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


# ================================================================
# 农民友好的语言映射
# ================================================================

_SEVERITY_FARMER = {
    SeverityLevel.NONE: ("没有发现问题", "🟢"),
    SeverityLevel.MINOR: ("问题较轻，影响不大", "🟡"),
    SeverityLevel.MODERATE: ("有一定影响，需要注意", "🟠"),
    SeverityLevel.SEVERE: ("问题比较严重，请尽快处理", "🔴"),
    SeverityLevel.CRITICAL: ("问题非常严重，请立即处理！", "🔴"),
}

_ANOMALY_FARMER = {
    "double_plant": {
        "name": "种重了（重复播种）",
        "what": "同一个位置种了两遍或者多遍，苗太密了",
        "why": "可能是播种机走了重复的路线，或者GPS定位出了偏差",
        "harm": "苗太密会互相抢水抢肥，都长不好，最后产量反而降低",
    },
    "planter_skip": {
        "name": "漏种了（跳播）",
        "what": "有些地方没有种上种子，出现了空行或者空段",
        "why": "可能是播种机排种器堵了、种子用完了没及时补、或者机器跳过了一段",
        "harm": "空出来的地方不长庄稼，白白浪费了这块地的产量",
    },
}

_URGENCY_FARMER = {
    UrgencyLevel.LOW: "不着急，等方便的时候处理就行",
    UrgencyLevel.MEDIUM: "最近找个时间处理一下",
    UrgencyLevel.HIGH: "这两天就要处理，别耽误了",
    UrgencyLevel.CRITICAL: "现在就要处理！越快越好！",
}

_REC_FARMER = {
    "double_plant": {
        SeverityLevel.MINOR: [
            ("先观察几天", "现在先不用动，等苗长到3-4片叶子的时候再看看，如果苗确实太密了再间苗。", "播种后2-3周"),
        ],
        SeverityLevel.MODERATE: [
            ("间苗（拔掉多余的苗）", "苗长到3-5片叶子的时候，把种重了的地方多余的苗拔掉，一穴留一株健壮的。", "苗期，越早越好"),
            ("多施点肥", "种密了的地方庄稼抢肥严重，适当多追一次肥，帮助剩下的苗长好。", "间苗后一周"),
        ],
        SeverityLevel.SEVERE: [
            ("赶紧间苗", "苗太密了会严重影响产量，必须尽快把多余的苗拔掉。", "发现后3天内"),
            ("追肥补水", "间苗后马上追一次肥、浇一次水，帮助剩下的苗恢复。", "间苗后立即"),
        ],
        SeverityLevel.CRITICAL: [
            ("立即间苗", "密度太大已经严重影响生长了，必须马上间苗。", "今天就开始"),
            ("考虑是否毁种重播", "如果间苗后还是太密，可能需要考虑翻掉重新种。请咨询当地农技站。", "尽快决定"),
        ],
    },
    "planter_skip": {
        SeverityLevel.MINOR: [
            ("补种", "把漏种的地方补种上，现在还来得及。用同品种的种子手工补种就行。", "发现后7天内"),
        ],
        SeverityLevel.MODERATE: [
            ("尽快补种", "漏种面积有点大了，要抓紧时间补上。手工补种或者小机器补种都行。", "发现后5天内"),
            ("补种后多关注", "补种的地方长得会比旁边晚，收获前多看看，可能需要分开收。", "整个生长期"),
        ],
        SeverityLevel.SEVERE: [
            ("赶紧补种，再晚就来不及了", "漏了这么多，必须马上行动补种，错过时间窗口就真的没收成了。", "发现后3天内"),
            ("联系种子经销商", "可能需要额外买种子，赶紧联系供应商。", "今天就联系"),
        ],
        SeverityLevel.CRITICAL: [
            ("评估是否需要毁种重播", "漏种面积太大了，补种可能已经来不及。请联系农技站的技术员来看看，评估是重新播种还是改种其他作物。", "今天就联系"),
            ("联系保险公司", "如果买了农业保险，赶紧拍照留证据，联系保险公司报案理赔。", "发现后立即"),
        ],
    },
}


class MultimodalReportGenerator:
    """
    多模态可视化报告生成器 v2

    新增：
    - Chart.js 图表数据注入
    - 农民友好的灾情预警
    - 双视角补救建议（农户 / 研究人员）
    """

    def __init__(self):
        self.annotator = VisualAnnotator()

    def generate(
        self,
        report: AgentReport,
        hits: List[RetrievalHit],
        output_dir: str,
        title: str = "播种异常分析报告",
        embed_images: bool = True,
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)

        # 1. 生成可视化素材
        visual_paths = self.annotator.save_all_visuals(report, hits, output_dir)

        # 2. 构建各部分 HTML
        detection_html = self._build_detection_html(report)
        visual_html = self._build_visual_html(visual_paths, embed_images)
        spatial_html = self._build_spatial_html(report)
        recommendations_html = self._build_recommendations_html(report)
        farmer_alert_html = self._build_farmer_alert_html(report)
        farmer_recommendations_html = self._build_farmer_recommendations_html(report)
        evidence_html = self._build_evidence_html(hits, embed_images)
        chart_data_json = self._build_chart_data(report, hits)

        # 3. 填充模板
        template = _read_template()
        html = template
        replacements = {
            "{{ title }}": title,
            "{{ image_path }}": os.path.basename(report.image_path),
            "{{ tile_id }}": report.tile_id or "N/A",
            "{{ timestamp }}": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "{{ overall_severity }}": report.overall_severity.value,
            "{{ overall_severity_cn }}": SEVERITY_LABELS_CN.get(report.overall_severity, "未知"),
            "{{ overall_confidence }}": f"{report.overall_confidence:.0%}",
            "{{ summary }}": report.summary,
            "{{ detection_results_html }}": detection_html,
            "{{ visual_html }}": visual_html,
            "{{ spatial_html }}": spatial_html,
            "{{ recommendations_html }}": recommendations_html,
            "{{ farmer_alert_html }}": farmer_alert_html,
            "{{ farmer_recommendations_html }}": farmer_recommendations_html,
            "{{ evidence_html }}": evidence_html,
            "{{ chart_data_json }}": chart_data_json,
        }
        for key, value in replacements.items():
            html = html.replace(key, value)

        # 4. 写入文件
        report_path = os.path.join(output_dir, "planting_report.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        # 5. 同时保存JSON格式的报告数据
        json_path = os.path.join(output_dir, "planting_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        # 6. 生成Markdown报告
        md_path = self._generate_markdown(report, hits, visual_paths, output_dir, title)

        return report_path

    # ================================================================
    # Chart.js 数据构建
    # ================================================================

    def _build_chart_data(self, report: AgentReport, hits: List[RetrievalHit]) -> str:
        """构建 Chart.js 图表所需的 JSON 数据"""

        type_names_cn = {
            "double_plant": "重复播种",
            "planter_skip": "跳播/漏播",
        }
        palette = {
            "double_plant": "rgba(242, 92, 92, 0.8)",
            "planter_skip": "rgba(91, 127, 255, 0.8)",
            "none": "rgba(100, 100, 130, 0.4)",
        }
        palette_light = {
            "double_plant": "rgba(242, 92, 92, 0.5)",
            "planter_skip": "rgba(91, 127, 255, 0.5)",
        }

        # 1) Confidence bar chart
        conf_labels = []
        conf_values = []
        conf_colors = []
        for d in report.detected_anomalies:
            conf_labels.append(type_names_cn.get(d.label, d.label))
            conf_values.append(round(d.confidence, 4))
            conf_colors.append(palette.get(d.label, palette["none"]))

        # 2) Area doughnut chart
        area_labels = []
        area_values = []
        area_colors = []
        total_affected = 0.0
        for d in report.detected_anomalies:
            if d.detected and d.affected_area_ratio > 0:
                pct = round(d.affected_area_ratio * 100, 2)
                area_labels.append(type_names_cn.get(d.label, d.label))
                area_values.append(pct)
                area_colors.append(palette.get(d.label, palette["none"]))
                total_affected += pct
        unaffected = max(0, round(100.0 - total_affected, 2))
        if unaffected > 0:
            area_labels.append("正常区域")
            area_values.append(unaffected)
            area_colors.append("rgba(61, 214, 140, 0.6)")

        # 3) Evidence distance chart
        dist_labels = []
        dist_values = []
        dist_colors = []
        for i, h in enumerate(hits[:10], 1):
            dist_labels.append(f"E{i}")
            dist_values.append(round(h.distance, 4))
            m = h.metadata or {}
            lp = m.get("labels_present", "")
            if "planter_skip" in lp:
                dist_colors.append(palette["planter_skip"])
            elif "double_plant" in lp:
                dist_colors.append(palette["double_plant"])
            else:
                dist_colors.append("rgba(166, 123, 255, 0.6)")

        # 4) Label frequency chart
        label_counter: Counter = Counter()
        for h in hits:
            m = h.metadata or {}
            try:
                labels = json.loads(m.get("labels_present", "[]"))
                for lb in labels:
                    label_counter[lb] += 1
            except Exception:
                pass

        label_name_map = {
            "double_plant": "重复播种",
            "planter_skip": "跳播/漏播",
            "nutrient_deficiency": "营养缺乏",
            "weed_cluster": "杂草丛",
            "drydown": "干枯",
            "storm_damage": "风暴损伤",
            "water": "积水",
            "waterway": "水渠",
            "endrow": "地头",
        }
        label_color_map = {
            "double_plant": "rgba(242, 92, 92, 0.7)",
            "planter_skip": "rgba(91, 127, 255, 0.7)",
            "nutrient_deficiency": "rgba(255, 255, 100, 0.7)",
            "weed_cluster": "rgba(61, 214, 140, 0.7)",
            "drydown": "rgba(245, 166, 35, 0.7)",
            "storm_damage": "rgba(166, 123, 255, 0.7)",
            "water": "rgba(0, 200, 255, 0.7)",
            "waterway": "rgba(0, 150, 200, 0.7)",
            "endrow": "rgba(180, 180, 180, 0.7)",
        }

        freq_labels = []
        freq_values = []
        freq_colors = []
        for lb, cnt in label_counter.most_common(8):
            freq_labels.append(label_name_map.get(lb, lb))
            freq_values.append(cnt)
            freq_colors.append(label_color_map.get(lb, "rgba(150,150,180,0.6)"))

        data = {
            "confidence": {"labels": conf_labels, "values": conf_values, "colors": conf_colors},
            "area": {"labels": area_labels, "values": area_values, "colors": area_colors},
            "distance": {"labels": dist_labels, "values": dist_values, "colors": dist_colors},
            "labelFreq": {"labels": freq_labels, "values": freq_values, "colors": freq_colors},
        }
        return json.dumps(data, ensure_ascii=False)

    # ================================================================
    # 农民友好的灾情预警
    # ================================================================

    def _build_farmer_alert_html(self, report: AgentReport) -> str:
        """构建通俗易懂的灾情预警通知"""
        detected = [d for d in report.detected_anomalies if d.detected]

        if not detected:
            return '''
            <div class="farmer-alert" style="border-color: rgba(61,214,140,0.3); background: rgba(61,214,140,0.06);">
              <div class="alert-title" style="color: #3dd68c;">🟢 好消息！您的田地没有发现播种问题</div>
              <div class="alert-body">
                系统对您的田地进行了智能检测，没有发现重复播种或漏播的情况。<br>
                请继续保持正常的田间管理。如有疑问，可以联系当地农技站。
              </div>
            </div>'''

        sev_text, sev_icon = _SEVERITY_FARMER.get(
            report.overall_severity, ("需要关注", "⚠️"))

        parts = [f'<div class="farmer-alert">']
        parts.append(f'<div class="alert-title">{sev_icon} {sev_text}</div>')
        parts.append('<div class="alert-body">')

        for d in detected:
            info = _ANOMALY_FARMER.get(d.label, {})
            name = info.get("name", d.label)
            what = info.get("what", "")
            harm = info.get("harm", "")
            area_pct = f"{d.affected_area_ratio:.1%}"

            parts.append(f'<p style="margin-bottom:12px;">')
            parts.append(f'<strong>发现问题：</strong><span class="highlight">{name}</span><br>')
            if what:
                parts.append(f'<strong>什么意思：</strong>{what}<br>')
            parts.append(f'<strong>影响范围：</strong>大约有 <span class="highlight">{area_pct}</span> 的地受到了影响<br>')
            if harm:
                parts.append(f'<strong>如果不处理：</strong>{harm}')
            parts.append('</p>')

        parts.append('</div></div>')
        return "\n".join(parts)

    def _build_farmer_recommendations_html(self, report: AgentReport) -> str:
        """构建农户友好的补救建议"""
        detected = [d for d in report.detected_anomalies if d.detected]

        if not detected:
            return '<p style="color: #3dd68c; font-size: 15px;">✅ 暂时不需要做什么，保持正常管理就好。</p>'

        parts = []
        step_num = 0

        for d in detected:
            sev = d.severity
            # 如果 severity 不在映射中，用 MODERATE 作为兜底
            if sev not in _REC_FARMER.get(d.label, {}):
                sev = SeverityLevel.MODERATE if sev.value not in ("none",) else SeverityLevel.MINOR

            recs = _REC_FARMER.get(d.label, {}).get(sev, [])
            if not recs:
                # 兜底：至少给一条通用建议
                recs = [("咨询农技专家", "建议联系当地农技站，请技术人员现场查看后给出具体指导。", "尽快")]

            info = _ANOMALY_FARMER.get(d.label, {})
            name = info.get("name", d.label)
            urg_text = _URGENCY_FARMER.get(report.urgency, "请尽快处理")

            parts.append(f'<h3 style="color: var(--accent-orange); margin: 16px 0 8px; font-size: 16px;">针对"{name}"的处理办法 — {urg_text}</h3>')
            parts.append('<ul class="farmer-action-list">')

            for title, detail, timing in recs:
                step_num += 1
                parts.append(f'''<li>
                    <span class="step-num">{step_num}</span>
                    <strong>{title}</strong><br>
                    {detail}
                    <span class="step-timing">⏰ 什么时候做：{timing}</span>
                </li>''')

            parts.append('</ul>')

        # 通用提醒
        parts.append('''
        <div style="margin-top:20px; padding:16px 20px; background:rgba(91,127,255,0.08);
                    border-radius:10px; border:1px solid rgba(91,127,255,0.2); font-size:14px; line-height:1.8;">
          <strong>💡 温馨提示：</strong><br>
          • 处理前建议先拍照留档，特别是买了农业保险的<br>
          • 如果拿不准，请联系当地农技站或者有经验的老农户<br>
          • 以上建议是基于卫星/无人机图像的初步判断，实际情况请以现场查看为准
        </div>''')

        return "\n".join(parts)

    # ================================================================
    # 检测结果 HTML
    # ================================================================

    def _build_detection_html(self, report: AgentReport) -> str:
        parts = []

        type_names = {
            "double_plant": "重复播种 (Double Plant)",
            "planter_skip": "跳播/漏播 (Planter Skip)",
        }

        for detail in report.detected_anomalies:
            name = type_names.get(detail.label, detail.label)
            detected_class = "detected" if detail.detected else "not-detected"
            status = "✓ 检出" if detail.detected else "✗ 未检出"

            conf_pct = f"{detail.confidence:.0%}"
            conf_color = self._confidence_color(detail.confidence)

            info_lines = []
            if detail.detected:
                sev_cn = SEVERITY_LABELS_CN.get(detail.severity, "未知")
                info_lines.append(f"严重程度: {sev_cn}")
                info_lines.append(f"受影响面积: {detail.affected_area_ratio:.1%}")
                info_lines.append(f"分布模式: {detail.distribution.value}")
                if detail.visual_evidence:
                    info_lines.append(f"视觉描述: {detail.visual_evidence}")
                if detail.evidence_refs:
                    info_lines.append(f"证据引用: {', '.join(detail.evidence_refs)}")

            info_html = "<br>".join(info_lines) if info_lines else "无异常"

            parts.append(f"""
            <div class="detection-item">
              <div class="label {detected_class}">{name}: {status}</div>
              <div class="info">{info_html}</div>
              <div class="confidence-bar-container">
                <div class="confidence-bar">
                  <div class="fill" style="width:{detail.confidence*100:.0f}%; background:{conf_color};"></div>
                </div>
                <div class="confidence-value" style="color:{conf_color};">{conf_pct}</div>
              </div>
            </div>
            """)

        return "\n".join(parts)

    # ================================================================
    # 可视化图像 HTML
    # ================================================================

    def _build_visual_html(self, visual_paths: Dict[str, str], embed: bool) -> str:
        parts = []

        annotated = visual_paths.get("annotated", "")
        heatmap = visual_paths.get("heatmap", "")

        if annotated or heatmap:
            parts.append('<div class="image-row">')
            if annotated:
                src = _img_to_base64(annotated) if embed else os.path.basename(annotated)
                parts.append(f"""
                <div class="image-container">
                  <img src="{src}" alt="异常区域标注">
                  <div class="image-caption">异常区域标注（红色=重播，蓝色=漏播）</div>
                </div>""")
            if heatmap:
                src = _img_to_base64(heatmap) if embed else os.path.basename(heatmap)
                parts.append(f"""
                <div class="image-container">
                  <img src="{src}" alt="严重度热力图">
                  <div class="image-caption">严重度热力图（绿→黄→红 = 低→高置信度）</div>
                </div>""")
            parts.append('</div>')

        panel = visual_paths.get("evidence_panel", "")
        if panel:
            src = _img_to_base64(panel) if embed else os.path.basename(panel)
            parts.append(f"""
            <div class="image-container">
              <img src="{src}" alt="证据对比面板">
              <div class="image-caption">查询图像 vs 检索证据对比</div>
            </div>""")

        if not parts:
            parts.append('<p style="color:var(--text-muted);">无可用的可视化素材（需要图像路径和patch数据）</p>')

        return "\n".join(parts)

    # ================================================================
    # 空间分析 HTML
    # ================================================================

    def _build_spatial_html(self, report: AgentReport) -> str:
        sp = report.spatial_summary
        if not sp:
            return '<p style="color:var(--text-muted);">空间分析不可用</p>'

        return f"""
        <div class="spatial-grid">
          <div class="spatial-stat">
            <div class="value">{sp.total_anomaly_regions}</div>
            <div class="label">异常区域数</div>
          </div>
          <div class="spatial-stat">
            <div class="value">{sp.total_affected_ratio:.1%}</div>
            <div class="label">受影响面积</div>
          </div>
          <div class="spatial-stat">
            <div class="value">{sp.primary_location}</div>
            <div class="label">主要位置</div>
          </div>
          <div class="spatial-stat">
            <div class="value">{sp.distribution_pattern.value}</div>
            <div class="label">分布模式</div>
          </div>
        </div>
        <p style="margin-top:14px; color:var(--text-secondary); font-size:14px; line-height:1.8;">
          {sp.pattern_description}
        </p>
        """

    # ================================================================
    # 研究人员版补救建议
    # ================================================================

    def _build_recommendations_html(self, report: AgentReport) -> str:
        if not report.recommendations:
            return '<p style="color:var(--text-muted);">暂无补救建议（未检测到需要干预的异常）</p>'

        parts = []
        urgency_cn = {
            UrgencyLevel.LOW: "低",
            UrgencyLevel.MEDIUM: "中",
            UrgencyLevel.HIGH: "高",
            UrgencyLevel.CRITICAL: "紧急",
        }
        target_cn = {
            "double_plant": "重复播种",
            "planter_skip": "跳播/漏播",
        }

        for i, rec in enumerate(report.recommendations, 1):
            urg_class = f"urgency-{rec.urgency.value}"
            urg_label = urgency_cn.get(rec.urgency, "中")
            target = target_cn.get(rec.target_anomaly, rec.target_anomaly)

            parts.append(f"""
            <div class="rec-item {urg_class}">
              <div class="action">
                {i}. {rec.action}
                <span class="urgency-tag {urg_class}">紧急度: {urg_label}</span>
              </div>
              <div class="detail">
                针对: {target} | 目标区域: {rec.target_area}<br>
                预期效果: {rec.expected_effect}
              </div>
              {f'<div class="detail" style="margin-top:4px;">备注: {rec.notes}</div>' if rec.notes else ''}
            </div>
            """)

        return "\n".join(parts)

    # ================================================================
    # 证据详情 HTML
    # ================================================================

    def _build_evidence_html(
        self, hits: List[RetrievalHit], embed: bool, max_show: int = 6
    ) -> str:
        if not hits:
            return '<p style="color:var(--text-muted);">无检索证据</p>'

        parts = ['<div class="evidence-grid">']
        for i, h in enumerate(hits[:max_show], 1):
            m = h.metadata or {}
            tile_id = m.get("tile_id", "N/A")
            labels = m.get("labels_present", "[]")
            rgb_path = m.get("path", "")

            thumb_html = ""
            if embed and rgb_path and os.path.exists(rgb_path):
                b64 = _img_to_base64(rgb_path)
                if b64:
                    thumb_html = f'<img src="{b64}" alt="evidence {i}">'

            parts.append(f"""
            <div class="evidence-card">
              <div class="ev-header">[E{i}] {tile_id}</div>
              <div class="ev-meta">
                余弦距离: {h.distance:.4f}<br>
                标签: {labels}
              </div>
              {thumb_html}
            </div>
            """)

        parts.append('</div>')
        return "\n".join(parts)

    @staticmethod
    def _confidence_color(conf: float) -> str:
        if conf >= 0.8:
            return "var(--accent-red)"
        if conf >= 0.6:
            return "var(--accent-orange)"
        if conf >= 0.4:
            return "var(--accent-blue)"
        return "var(--text-muted)"

    # ================================================================
    # Markdown 报告生成
    # ================================================================

    def _generate_markdown(
        self,
        report: AgentReport,
        hits: List[RetrievalHit],
        visual_paths: Dict[str, str],
        output_dir: str,
        title: str,
    ) -> str:
        """
        生成结构化 Markdown 报告。

        包含：基本信息、检测结果、统计数据表格、
        空间分析、可视化图片引用、农户建议、研究建议、证据详情。
        """
        lines: List[str] = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sev_cn = SEVERITY_LABELS_CN.get(report.overall_severity, "未知")

        # ---- 标题与概览 ----
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"> **生成时间**: {timestamp}  ")
        lines.append(f"> **图像**: `{os.path.basename(report.image_path)}`  ")
        lines.append(f"> **Tile ID**: `{report.tile_id or 'N/A'}`  ")
        lines.append(f"> **整体严重度**: {sev_cn} ({report.overall_severity.value})  ")
        lines.append(f"> **整体置信度**: {report.overall_confidence:.0%}  ")
        lines.append(f"> **紧急程度**: {report.urgency.value}")
        lines.append("")
        lines.append(f"**摘要**: {report.summary}")
        lines.append("")

        # ---- 检测结果 ----
        lines.append("## 1. 检测结果")
        lines.append("")

        type_names = {
            "double_plant": "重复播种 (Double Plant)",
            "planter_skip": "跳播/漏播 (Planter Skip)",
        }

        lines.append("| 异常类型 | 状态 | 置信度 | 严重度 | 受影响面积 | 分布模式 |")
        lines.append("|:---------|:----:|:------:|:------:|:----------:|:--------:|")
        for d in report.detected_anomalies:
            name = type_names.get(d.label, d.label)
            status = "✅ 检出" if d.detected else "❌ 未检出"
            conf = f"{d.confidence:.0%}"
            sev = SEVERITY_LABELS_CN.get(d.severity, "-") if d.detected else "-"
            area = f"{d.affected_area_ratio:.1%}" if d.detected else "-"
            dist = d.distribution.value if d.detected else "-"
            lines.append(f"| {name} | {status} | {conf} | {sev} | {area} | {dist} |")
        lines.append("")

        # ---- 统计数据 ----
        lines.append("## 2. 统计数据")
        lines.append("")

        # 置信度统计
        lines.append("### 2.1 各类异常置信度")
        lines.append("")
        for d in report.detected_anomalies:
            name = type_names.get(d.label, d.label)
            bar_len = int(d.confidence * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"- **{name}**: `{bar}` {d.confidence:.1%}")
        lines.append("")

        # 面积统计
        detected_anomalies = [d for d in report.detected_anomalies if d.detected and d.affected_area_ratio > 0]
        if detected_anomalies:
            lines.append("### 2.2 受影响面积分布")
            lines.append("")
            total_affected = sum(d.affected_area_ratio for d in detected_anomalies)
            for d in detected_anomalies:
                name = type_names.get(d.label, d.label)
                lines.append(f"- **{name}**: {d.affected_area_ratio:.1%}")
            lines.append(f"- **正常区域**: {max(0, 1.0 - total_affected):.1%}")
            lines.append("")

        # 证据距离统计
        if hits:
            lines.append("### 2.3 检索证据距离统计")
            lines.append("")
            distances = [h.distance for h in hits]
            avg_dist = sum(distances) / len(distances)
            min_dist = min(distances)
            max_dist = max(distances)
            lines.append(f"| 指标 | 值 |")
            lines.append(f"|:-----|:---|")
            lines.append(f"| 证据数量 | {len(hits)} |")
            lines.append(f"| 平均距离 | {avg_dist:.4f} |")
            lines.append(f"| 最小距离 | {min_dist:.4f} |")
            lines.append(f"| 最大距离 | {max_dist:.4f} |")
            lines.append("")

            # 标签频次
            label_counter: Counter = Counter()
            for h in hits:
                m = h.metadata or {}
                try:
                    labels = json.loads(m.get("labels_present", "[]"))
                    for lb in labels:
                        label_counter[lb] += 1
                except Exception:
                    pass
            if label_counter:
                label_name_map = {
                    "double_plant": "重复播种", "planter_skip": "跳播/漏播",
                    "nutrient_deficiency": "营养缺乏", "weed_cluster": "杂草丛",
                    "drydown": "干枯", "storm_damage": "风暴损伤",
                    "water": "积水", "waterway": "水渠", "endrow": "地头",
                }
                lines.append("### 2.4 证据标签频次")
                lines.append("")
                lines.append("| 标签 | 出现次数 | 占比 |")
                lines.append("|:-----|:--------:|:----:|")
                total_labels = sum(label_counter.values())
                for lb, cnt in label_counter.most_common():
                    lb_cn = label_name_map.get(lb, lb)
                    pct = cnt / total_labels * 100
                    lines.append(f"| {lb_cn} | {cnt} | {pct:.0f}% |")
                lines.append("")

        # ---- 空间分析 ----
        lines.append("## 3. 空间分布分析")
        lines.append("")
        sp = report.spatial_summary
        if sp:
            lines.append(f"| 指标 | 值 |")
            lines.append(f"|:-----|:---|")
            lines.append(f"| 异常区域数 | {sp.total_anomaly_regions} |")
            lines.append(f"| 受影响面积 | {sp.total_affected_ratio:.1%} |")
            lines.append(f"| 主要位置 | {sp.primary_location} |")
            lines.append(f"| 分布模式 | {sp.distribution_pattern.value} |")
            lines.append("")
            lines.append(f"**分析说明**: {sp.pattern_description}")
        else:
            lines.append("*空间分析不可用*")
        lines.append("")

        # ---- 可视化图片 ----
        lines.append("## 4. 可视化分析")
        lines.append("")
        img_names = {
            "annotated": ("异常区域标注", "红色=重播，蓝色=漏播"),
            "heatmap": ("严重度热力图", "绿→黄→红 = 低→高置信度"),
            "evidence_panel": ("证据对比面板", "查询图像 vs 检索证据"),
            "summary_card": ("分析摘要卡片", ""),
        }
        has_visual = False
        for key, (caption, note) in img_names.items():
            path = visual_paths.get(key, "")
            if path and os.path.exists(path):
                fname = os.path.basename(path)
                desc = f"{caption}（{note}）" if note else caption
                lines.append(f"### {caption}")
                lines.append(f"![{desc}](./{fname})")
                lines.append("")
                has_visual = True
        if not has_visual:
            lines.append("*无可用的可视化素材*")
            lines.append("")

        # ---- 灾情预警（农民版）----
        lines.append("## 5. 灾情预警通知（通俗版）")
        lines.append("")
        detected = [d for d in report.detected_anomalies if d.detected]
        if not detected:
            lines.append("🟢 **好消息！** 您的田地没有发现播种问题，请继续保持正常管理。")
        else:
            sev_text, sev_icon = _SEVERITY_FARMER.get(report.overall_severity, ("需要关注", "⚠️"))
            lines.append(f"{sev_icon} **{sev_text}**")
            lines.append("")
            for d in detected:
                info = _ANOMALY_FARMER.get(d.label, {})
                name = info.get("name", d.label)
                what = info.get("what", "")
                harm = info.get("harm", "")
                area_pct = f"{d.affected_area_ratio:.1%}"
                lines.append(f"**发现问题：{name}**")
                if what:
                    lines.append(f"- 什么意思：{what}")
                lines.append(f"- 影响范围：大约有 **{area_pct}** 的地受到了影响")
                if harm:
                    lines.append(f"- 如果不处理：{harm}")
                lines.append("")
        lines.append("")

        # ---- 补救建议（农户版）----
        lines.append("## 6. 补救建议")
        lines.append("")
        lines.append("### 6.1 农户操作指南")
        lines.append("")
        if not detected:
            lines.append("✅ 暂时不需要做什么，保持正常管理就好。")
        else:
            step_num = 0
            for d in detected:
                sev = d.severity
                if sev not in _REC_FARMER.get(d.label, {}):
                    sev = SeverityLevel.MODERATE if sev.value not in ("none",) else SeverityLevel.MINOR
                recs = _REC_FARMER.get(d.label, {}).get(sev, [])
                if not recs:
                    recs = [("咨询农技专家", "建议联系当地农技站，请技术人员现场查看后给出具体指导。", "尽快")]
                info = _ANOMALY_FARMER.get(d.label, {})
                name = info.get("name", d.label)
                urg_text = _URGENCY_FARMER.get(report.urgency, "请尽快处理")
                lines.append(f"**针对「{name}」— {urg_text}：**")
                lines.append("")
                for rec_title, detail, timing in recs:
                    step_num += 1
                    lines.append(f"**{step_num}. {rec_title}**")
                    lines.append(f"   {detail}")
                    lines.append(f"   ⏰ 什么时候做：{timing}")
                    lines.append("")

            lines.append("---")
            lines.append("")
            lines.append("💡 **温馨提示**：")
            lines.append("- 处理前建议先拍照留档，特别是买了农业保险的")
            lines.append("- 如果拿不准，请联系当地农技站或者有经验的老农户")
            lines.append("- 以上建议是基于卫星/无人机图像的初步判断，实际情况请以现场查看为准")
        lines.append("")

        # ---- 补救建议（研究版）----
        lines.append("### 6.2 研究分析建议")
        lines.append("")
        if not report.recommendations:
            lines.append("*暂无补救建议（未检测到需要干预的异常）*")
        else:
            urgency_cn = {
                UrgencyLevel.LOW: "低", UrgencyLevel.MEDIUM: "中",
                UrgencyLevel.HIGH: "高", UrgencyLevel.CRITICAL: "紧急",
            }
            target_cn = {"double_plant": "重复播种", "planter_skip": "跳播/漏播"}
            lines.append("| # | 措施 | 紧急度 | 针对 | 目标区域 | 预期效果 |")
            lines.append("|:-:|:-----|:------:|:----:|:---------|:---------|")
            for i, rec in enumerate(report.recommendations, 1):
                urg = urgency_cn.get(rec.urgency, "中")
                target = target_cn.get(rec.target_anomaly, rec.target_anomaly)
                lines.append(f"| {i} | {rec.action} | {urg} | {target} | {rec.target_area} | {rec.expected_effect} |")
        lines.append("")

        # ---- 证据详情 ----
        lines.append("## 7. 检索证据详情")
        lines.append("")
        if hits:
            lines.append("| # | Tile ID | 距离 | 标签 |")
            lines.append("|:-:|:--------|:----:|:-----|")
            for i, h in enumerate(hits[:10], 1):
                m = h.metadata or {}
                tid = m.get("tile_id", "N/A")
                labels = m.get("labels_present", "[]")
                lines.append(f"| E{i} | `{tid}` | {h.distance:.4f} | {labels} |")
        else:
            lines.append("*无检索证据*")
        lines.append("")

        # ---- 页脚 ----
        lines.append("---")
        lines.append("")
        lines.append("*AGRI_IMAGE_RAG_CHROMA — 播种异常分析Agent | 基于 CLIP + ChromaDB + Qwen2-VL 的 RAG 架构*")

        # 写入文件
        md_content = "\n".join(lines)
        md_path = os.path.join(output_dir, "planting_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return md_path