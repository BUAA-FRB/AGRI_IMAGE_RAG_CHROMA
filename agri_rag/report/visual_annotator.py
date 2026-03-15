"""
visual_annotator.py — 图像标注与可视化工具

功能：
1. 在图像上绘制异常区域bbox + 标签
2. 生成query vs evidence的对比面板
3. 生成严重度热力图
4. 生成分析摘要卡片
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from ..types import AgentReport, AnomalyRegion, RetrievalHit, SeverityLevel


# ============================================================
# 颜色配置
# ============================================================

# 异常类型 -> (R, G, B, alpha)
ANOMALY_COLORS = {
    "double_plant": (255, 60, 60),      # 红色
    "planter_skip": (60, 120, 255),     # 蓝色
    "drydown": (255, 180, 0),           # 橙色
    "nutrient_deficiency": (255, 255, 0), # 黄色
    "storm_damage": (160, 0, 200),      # 紫色
    "water": (0, 200, 255),             # 青色
    "waterway": (0, 150, 200),          # 深青
    "weed_cluster": (0, 200, 80),       # 绿色
    "endrow": (180, 180, 180),          # 灰色
}

SEVERITY_COLORS = {
    SeverityLevel.NONE: (200, 200, 200),
    SeverityLevel.MINOR: (255, 255, 100),
    SeverityLevel.MODERATE: (255, 180, 0),
    SeverityLevel.SEVERE: (255, 80, 0),
    SeverityLevel.CRITICAL: (255, 0, 0),
}

SEVERITY_LABELS_CN = {
    SeverityLevel.NONE: "正常",
    SeverityLevel.MINOR: "轻度",
    SeverityLevel.MODERATE: "中度",
    SeverityLevel.SEVERE: "重度",
    SeverityLevel.CRITICAL: "极重",
}


def _get_font(size: int = 14) -> ImageFont.FreeTypeFont:
    """尝试加载可用字体，回退到默认字体"""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


class VisualAnnotator:
    """图像标注与可视化工具"""

    def __init__(self, font_size: int = 14):
        self.font = _get_font(font_size)
        self.font_small = _get_font(max(10, font_size - 2))

    def draw_anomaly_regions(
        self,
        image_path: str,
        regions: List[AnomalyRegion],
        line_width: int = 3,
        show_label: bool = True,
        show_confidence: bool = True,
    ) -> Image.Image:
        """
        在图像上绘制异常区域的bbox标注。

        Args:
            image_path: 原始图像路径
            regions: 异常区域列表
            line_width: 边框线宽
            show_label: 是否显示标签名
            show_confidence: 是否显示置信度

        Returns:
            标注后的PIL Image
        """
        img = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_main = ImageDraw.Draw(img)

        for region in regions:
            color = ANOMALY_COLORS.get(region.anomaly_type, (200, 200, 200))
            x1, y1, x2, y2 = region.bbox

            # 半透明填充
            fill_color = color + (50,)  # alpha=50
            draw_overlay.rectangle([x1, y1, x2, y2], fill=fill_color)

            # 实线边框
            draw_main.rectangle(
                [x1, y1, x2, y2],
                outline=color + (255,),
                width=line_width,
            )

            # 标签文字
            if show_label or show_confidence:
                label_parts = []
                if show_label:
                    type_name = {
                        "double_plant": "重播",
                        "planter_skip": "漏播",
                    }.get(region.anomaly_type, region.anomaly_type)
                    label_parts.append(type_name)
                if show_confidence and region.confidence > 0:
                    label_parts.append(f"{region.confidence:.0%}")

                label_text = " ".join(label_parts)
                if label_text:
                    # 文字背景
                    text_bbox = self.font_small.getbbox(label_text)
                    tw = text_bbox[2] - text_bbox[0] + 6
                    th = text_bbox[3] - text_bbox[1] + 4
                    ty = max(0, y1 - th - 2)
                    draw_main.rectangle(
                        [x1, ty, x1 + tw, ty + th],
                        fill=color + (200,),
                    )
                    draw_main.text(
                        (x1 + 3, ty + 2),
                        label_text,
                        fill=(255, 255, 255, 255),
                        font=self.font_small,
                    )

        # 合成
        result = Image.alpha_composite(img, overlay)
        return result.convert("RGB")

    def create_evidence_panel(
        self,
        query_image_path: str,
        evidence_hits: List[RetrievalHit],
        max_evidence: int = 3,
        panel_height: int = 256,
    ) -> Image.Image:
        """
        创建 query image vs evidence images 的对比面板。

        布局：[Query Image] | [Evidence 1] | [Evidence 2] | [Evidence 3]

        Args:
            query_image_path: 查询图像路径
            evidence_hits: 证据检索结果
            max_evidence: 最多展示几个证据
            panel_height: 面板高度

        Returns:
            对比面板 PIL Image
        """
        images = []
        labels = []

        # Query image
        try:
            q_img = Image.open(query_image_path).convert("RGB")
            q_img = q_img.resize((panel_height, panel_height), Image.LANCZOS)
            images.append(q_img)
            labels.append("Query")
        except Exception:
            # placeholder
            q_img = Image.new("RGB", (panel_height, panel_height), (128, 128, 128))
            images.append(q_img)
            labels.append("Query (N/A)")

        # Evidence images
        for i, hit in enumerate(evidence_hits[:max_evidence]):
            m = hit.metadata or {}
            rgb_path = m.get("path", "")
            label_str = m.get("labels_present", "")

            try:
                e_img = Image.open(rgb_path).convert("RGB")
                e_img = e_img.resize((panel_height, panel_height), Image.LANCZOS)
                images.append(e_img)
                labels.append(f"E{i+1} d={hit.distance:.3f}")
            except Exception:
                e_img = Image.new("RGB", (panel_height, panel_height), (80, 80, 80))
                images.append(e_img)
                labels.append(f"E{i+1} (N/A)")

        # 拼接
        n = len(images)
        gap = 4
        label_h = 24
        total_w = n * panel_height + (n - 1) * gap
        total_h = panel_height + label_h

        panel = Image.new("RGB", (total_w, total_h), (40, 40, 40))
        draw = ImageDraw.Draw(panel)

        for idx, (img, label) in enumerate(zip(images, labels)):
            x_offset = idx * (panel_height + gap)
            panel.paste(img, (x_offset, 0))

            # 标签
            draw.text(
                (x_offset + 4, panel_height + 2),
                label,
                fill=(220, 220, 220),
                font=self.font_small,
            )

        return panel

    def create_severity_heatmap(
        self,
        image_path: str,
        regions: List[AnomalyRegion],
        opacity: int = 100,
    ) -> Image.Image:
        """
        基于异常区域生成严重度热力图叠加。

        颜色映射：置信度越高颜色越暖（绿→黄→红）

        Args:
            image_path: 原始图像
            regions: 异常区域列表
            opacity: 热力图透明度 (0-255)

        Returns:
            热力图叠加后的 PIL Image
        """
        img = Image.open(image_path).convert("RGBA")
        heatmap = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(heatmap)

        for region in regions:
            conf = max(0.0, min(1.0, region.confidence))

            # 绿(0) -> 黄(0.5) -> 红(1.0)
            if conf < 0.5:
                r = int(255 * (conf / 0.5))
                g = 255
            else:
                r = 255
                g = int(255 * (1 - (conf - 0.5) / 0.5))
            b = 0

            x1, y1, x2, y2 = region.bbox
            draw.rectangle(
                [x1, y1, x2, y2],
                fill=(r, g, b, opacity),
            )

        result = Image.alpha_composite(img, heatmap)
        return result.convert("RGB")

    def create_summary_card(
        self,
        report: AgentReport,
        width: int = 600,
    ) -> Image.Image:
        """
        生成分析摘要卡片（纯图像，可嵌入报告）。

        包含：标题、严重度指示、检测结果摘要、置信度。
        """
        padding = 16
        line_height = 22
        lines_content = []

        # 标题
        lines_content.append(("TITLE", f"播种异常分析报告"))
        lines_content.append(("DIVIDER", ""))

        # 整体状态
        sev = SEVERITY_LABELS_CN.get(report.overall_severity, "未知")
        sev_color = SEVERITY_COLORS.get(report.overall_severity, (200, 200, 200))
        lines_content.append(("STATUS", f"整体严重度: {sev} | 置信度: {report.overall_confidence:.0%}"))

        # 检测结果
        lines_content.append(("DIVIDER", ""))
        for detail in report.detected_anomalies:
            type_cn = {"double_plant": "重复播种", "planter_skip": "跳播/漏播"}.get(detail.label, detail.label)
            status = "✓ 检出" if detail.detected else "✗ 未检出"
            if detail.detected:
                sev_cn = SEVERITY_LABELS_CN.get(detail.severity, "")
                lines_content.append(("DETAIL", f"{type_cn}: {status} ({sev_cn}, {detail.confidence:.0%})"))
            else:
                lines_content.append(("DETAIL", f"{type_cn}: {status}"))

        # 空间信息
        if report.spatial_summary:
            lines_content.append(("DIVIDER", ""))
            lines_content.append(("INFO", f"异常区域: {report.spatial_summary.total_anomaly_regions}处"))
            lines_content.append(("INFO", f"受影响面积: {report.spatial_summary.total_affected_ratio:.1%}"))
            lines_content.append(("INFO", f"分布模式: {report.spatial_summary.pattern_description}"))

        # 建议数量
        if report.recommendations:
            lines_content.append(("DIVIDER", ""))
            lines_content.append(("INFO", f"补救建议: {len(report.recommendations)}条"))

        # 计算卡片高度
        height = padding * 2
        for tag, _ in lines_content:
            if tag == "TITLE":
                height += 30
            elif tag == "DIVIDER":
                height += 8
            else:
                height += line_height
        height = max(height, 100)

        # 绘制
        card = Image.new("RGB", (width, height), (30, 30, 45))
        draw = ImageDraw.Draw(card)

        y = padding
        for tag, text in lines_content:
            if tag == "TITLE":
                draw.text((padding, y), text, fill=(255, 255, 255), font=self.font)
                y += 30
            elif tag == "DIVIDER":
                draw.line([(padding, y + 3), (width - padding, y + 3)], fill=(80, 80, 100), width=1)
                y += 8
            elif tag == "STATUS":
                draw.text((padding, y), text, fill=sev_color, font=self.font)
                y += line_height
            elif tag == "DETAIL":
                draw.text((padding + 8, y), text, fill=(220, 220, 220), font=self.font_small)
                y += line_height
            else:
                draw.text((padding + 8, y), text, fill=(180, 180, 200), font=self.font_small)
                y += line_height

        return card

    def save_all_visuals(
        self,
        report: AgentReport,
        hits: List[RetrievalHit],
        output_dir: str,
    ) -> Dict[str, str]:
        """
        一次性生成所有可视化素材并保存。

        Returns:
            生成的文件路径字典: {"annotated": "...", "heatmap": "...", "evidence_panel": "...", "summary_card": "..."}
        """
        os.makedirs(output_dir, exist_ok=True)
        paths: Dict[str, str] = {}

        image_path = report.image_path
        all_regions = []
        for detail in report.detected_anomalies:
            if detail.detected:
                all_regions.extend(detail.regions)

        # 1. 异常区域标注图
        if all_regions and os.path.exists(image_path):
            annotated = self.draw_anomaly_regions(image_path, all_regions)
            p = os.path.join(output_dir, "annotated.png")
            annotated.save(p)
            paths["annotated"] = p

        # 2. 热力图
        if all_regions and os.path.exists(image_path):
            heatmap = self.create_severity_heatmap(image_path, all_regions)
            p = os.path.join(output_dir, "heatmap.png")
            heatmap.save(p)
            paths["heatmap"] = p

        # 3. 证据对比面板
        if hits and os.path.exists(image_path):
            panel = self.create_evidence_panel(image_path, hits)
            p = os.path.join(output_dir, "evidence_panel.png")
            panel.save(p)
            paths["evidence_panel"] = p

        # 4. 摘要卡片
        card = self.create_summary_card(report)
        p = os.path.join(output_dir, "summary_card.png")
        card.save(p)
        paths["summary_card"] = p

        return paths