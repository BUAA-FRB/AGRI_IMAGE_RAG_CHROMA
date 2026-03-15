"""
planting_agent.py — 播种异常专项分析Agent

针对 double_plant（重复播种）和 planter_skip（跳播/漏播）的深度分析。

分析流程（5个阶段）：
  1. 证据筛选与聚焦 — 从通用检索结果中过滤播种相关证据
  2. 空间分析       — 利用patch级bbox信息进行区域聚类
  3. 严重程度量化   — 基于面积比和分布模式分级
  4. VLM二次推理    — 专项prompt调用Qwen2-VL（Phase 2）
  5. 补救方案生成   — 基于决策矩阵生成行动建议
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import BaseAnomalyAgent
from .planting_prompts import build_planting_vlm_messages
from ..types import (
    AgentReport,
    AnomalyDetail,
    AnomalyRegion,
    DistributionPattern,
    Recommendation,
    RetrievalHit,
    SeverityLevel,
    SpatialAnalysis,
    UrgencyLevel,
)

logger = logging.getLogger(__name__)


# ============================================================
# 严重度分级阈值
# ============================================================

_SEVERITY_THRESHOLDS = {
    # (min_ratio, max_ratio) -> SeverityLevel
    (0.00, 0.05): SeverityLevel.MINOR,
    (0.05, 0.15): SeverityLevel.MODERATE,
    (0.15, 0.30): SeverityLevel.SEVERE,
    (0.30, float('inf')): SeverityLevel.CRITICAL,
}


# ============================================================
# 补救决策矩阵
# ============================================================

_RECOMMENDATION_MATRIX: Dict[Tuple[str, str], Dict[str, Any]] = {
    # (anomaly_type, severity) -> recommendation info
    ("double_plant", "minor"): {
        "action": "持续监测，暂不干预",
        "urgency": UrgencyLevel.LOW,
        "expected_effect": "轻度重播对产量影响有限（预计减产<3%），植株可自然调节密度",
        "notes": "关注重播区域的长势变化，若出现明显黄化或倒伏再采取措施",
    },
    ("double_plant", "moderate"): {
        "action": "在V2-V4生长期进行人工或机械间苗，优先处理密度最高的区域",
        "urgency": UrgencyLevel.MEDIUM,
        "expected_effect": "及时间苗可将产量损失控制在5%以内，改善通风透光条件",
        "notes": "间苗后适当补施氮肥帮助恢复，注意保留长势较好的植株",
    },
    ("double_plant", "severe"): {
        "action": "尽快进行大面积间苗处理，配合补施肥料；评估是否需要局部重播",
        "urgency": UrgencyLevel.HIGH,
        "expected_effect": "可挽回部分产量损失，但预计仍有10-15%减产",
        "notes": "检查播种机校准状态，防止后续作业重复出现；严重区域考虑改种短季作物",
    },
    ("double_plant", "critical"): {
        "action": "紧急评估经济损失，大面积间苗+补肥；考虑部分地块毁种重播或改种",
        "urgency": UrgencyLevel.CRITICAL,
        "expected_effect": "即使干预，预计减产超过20%；需要综合经济评估",
        "notes": "立即检修播种设备，记录问题地块GPS坐标；联系农技人员现场评估",
    },

    ("planter_skip", "minor"): {
        "action": "在播种窗口期内尽快补种缺失行",
        "urgency": UrgencyLevel.MEDIUM,
        "expected_effect": "及时补播可基本恢复正常产量，预计减产<2%",
        "notes": "补播时注意品种一致性和播种深度；检查播种机排种器是否堵塞",
    },
    ("planter_skip", "moderate"): {
        "action": "紧急补播缺失区域，同时检查并校准播种机",
        "urgency": UrgencyLevel.HIGH,
        "expected_effect": "补播越早效果越好；超过最佳补播窗口则预计减产8-12%",
        "notes": "优先补播面积最大的连续缺失区域；考虑适当增加补播密度以补偿损失",
    },
    ("planter_skip", "severe"): {
        "action": "全面评估补播可行性，对无法补救的区域考虑替代作物或覆盖作物",
        "urgency": UrgencyLevel.HIGH,
        "expected_effect": "大面积漏播补救困难，预计减产15-25%",
        "notes": "记录漏播模式（是否为固定行），排查播种机故障原因；向保险公司报告",
    },
    ("planter_skip", "critical"): {
        "action": "启动农业保险理赔流程；评估毁种重播或改种替代作物的经济性",
        "urgency": UrgencyLevel.CRITICAL,
        "expected_effect": "严重漏播难以完全补救，需要经济决策而非农艺决策",
        "notes": "保留现场影像作为保险证据；联系设备供应商检查播种机故障",
    },
}


class PlantingAnomalyAgent(BaseAnomalyAgent):
    """
    播种异常专项分析Agent

    处理类别：double_plant（重复播种）、planter_skip（跳播/漏播）
    """

    HANDLED_LABELS = ["double_plant", "planter_skip"]
    AGENT_NAME = "planting_anomaly_agent"

    def __init__(self, distance_threshold: float = 1.5):
        """
        Args:
            distance_threshold: 过滤检索结果的距离阈值
        """
        self.distance_threshold = distance_threshold

    def analyze(
        self,
        image_path: str,
        general_prediction: Dict[str, Any],
        hits: List[RetrievalHit],
        patch_hits: Optional[List[RetrievalHit]] = None,
        qwen_client: Optional[Any] = None,
    ) -> AgentReport:
        """
        执行播种异常专项分析（5个阶段）。

        Args:
            image_path: 待分析图像路径
            general_prediction: 通用RAG流程输出的JSON dict
            hits: 全局级检索结果
            patch_hits: patch级检索结果（用于空间定位）
            qwen_client: Qwen3VLClient实例（用于Phase 2二次推理）

        Returns:
            AgentReport
        """
        logger.info(f"[PlantingAgent] 开始播种异常分析: {image_path}")

        # ---- 阶段1：证据筛选与聚焦 ----
        filtered_global, filtered_patch = self._stage1_filter_evidence(hits, patch_hits)
        logger.info(
            f"[PlantingAgent] 阶段1完成: 全局证据 {len(filtered_global)}/{len(hits)}, "
            f"patch证据 {len(filtered_patch)}/{len(patch_hits) if patch_hits else 0}"
        )

        # ---- 阶段2：空间分析 ----
        tile_id = self._extract_tile_id(hits)
        spatial = self._stage2_spatial_analysis(
            filtered_patch, filtered_global, image_size=(512, 512), target_tile_id=tile_id
        )
        logger.info(
            f"[PlantingAgent] 阶段2完成: 区域={spatial.total_anomaly_regions}, "
            f"面积比={spatial.total_affected_ratio:.1%}, "
            f"分布={spatial.distribution_pattern.value}"
        )

        # ---- 阶段3：严重程度量化 ----
        anomaly_details = self._stage3_quantify_severity(
            general_prediction, filtered_global, spatial
        )
        overall_severity = self._compute_overall_severity(anomaly_details)
        logger.info(f"[PlantingAgent] 阶段3完成: 整体严重度={overall_severity.value}")

        # ---- 阶段4：VLM二次推理（Phase 2，可选） ----
        vlm_output = ""
        if qwen_client is not None:
            vlm_output = self._stage4_vlm_reasoning(
                image_path, filtered_global, anomaly_details, spatial, qwen_client
            )
            # 用VLM输出更新分析细节
            self._update_from_vlm_output(vlm_output, anomaly_details)
            logger.info(f"[PlantingAgent] 阶段4完成: VLM输出 {len(vlm_output)} chars")
        else:
            logger.info("[PlantingAgent] 阶段4跳过: 未提供qwen_client")

        # ---- 阶段5：补救方案生成 ----
        recommendations = self._stage5_generate_recommendations(anomaly_details)
        logger.info(f"[PlantingAgent] 阶段5完成: 生成 {len(recommendations)} 条建议")

        # ---- 组装报告 ----
        urgency = self._compute_overall_urgency(recommendations)
        evidence_refs = self._collect_evidence_refs(filtered_global)
        summary = self._generate_summary(anomaly_details, overall_severity, spatial)

        report = AgentReport(
            agent_name=self.AGENT_NAME,
            image_path=image_path,
            tile_id=self._extract_tile_id(hits),
            detected_anomalies=anomaly_details,
            overall_severity=overall_severity,
            spatial_summary=spatial,
            recommendations=recommendations,
            overall_confidence=self._compute_overall_confidence(anomaly_details),
            evidence_refs=evidence_refs,
            raw_vlm_output=vlm_output,
            raw_general_prediction=general_prediction,
            summary=summary,
            urgency=urgency,
        )

        logger.info(f"[PlantingAgent] 分析完成: {summary}")
        return report

    # ================================================================
    # 阶段1：证据筛选与聚焦
    # ================================================================

    def _stage1_filter_evidence(
        self,
        hits: List[RetrievalHit],
        patch_hits: Optional[List[RetrievalHit]],
    ) -> Tuple[List[RetrievalHit], List[RetrievalHit]]:
        """
        从通用检索结果中筛选播种异常相关的证据。

        筛选依据：
        1. metadata.labels_present 包含 double_plant 或 planter_skip
        2. 距离在阈值范围内
        """
        filtered_global = self._filter_hits_by_labels(
            hits, self.HANDLED_LABELS, self.distance_threshold
        )

        filtered_patch = []
        if patch_hits:
            filtered_patch = self._filter_hits_by_labels(
                patch_hits, self.HANDLED_LABELS, self.distance_threshold
            )

        return filtered_global, filtered_patch

    # ================================================================
    # 阶段2：空间分析
    # ================================================================

    def _stage2_spatial_analysis(
        self,
        patch_hits: List[RetrievalHit],
        global_hits: List[RetrievalHit],
        image_size: Tuple[int, int] = (512, 512),
        target_tile_id: str = "",
    ) -> SpatialAnalysis:
        """
        空间分析：区分自身patch和外部patch。

        - 自身patch（tile_id匹配）→ 用bbox精确定位
        - 只有外部patch → 用统计推理模式（基于检索证据的标签分布+label_areas元数据）

        Args:
            patch_hits: 筛选后的patch级检索结果
            global_hits: 筛选后的全局检索结果
            image_size: 原始图像尺寸 (width, height)
            target_tile_id: 目标图像的tile_id

        Returns:
            SpatialAnalysis
        """
        if not patch_hits and not global_hits:
            return SpatialAnalysis(
                total_anomaly_regions=0,
                total_affected_ratio=0.0,
                primary_location="无检索证据",
                distribution_pattern=DistributionPattern.NONE,
                pattern_description="未检索到播种异常相关证据",
            )

        w, h = image_size
        total_area = w * h

        # 区分自身patch和外部patch
        self_patches = []
        external_patches = []
        for hit in patch_hits:
            m = hit.metadata or {}
            tid = m.get("tile_id", "")
            if target_tile_id and target_tile_id in tid:
                self_patches.append(hit)
            else:
                external_patches.append(hit)

        logger.info(
            f"[PlantingAgent] Patch分类: 自身={len(self_patches)}, 外部={len(external_patches)}"
        )

        # ---- 模式A：有自身patch → 精确空间分析 ----
        if self_patches:
            return self._spatial_from_self_patches(self_patches, w, h, total_area)

        # ---- 模式B：只有外部patch → 统计推理 ----
        return self._spatial_from_external_evidence(
            external_patches, global_hits, w, h, total_area
        )

    def _spatial_from_self_patches(
        self,
        patches: List[RetrievalHit],
        w: int, h: int, total_area: int,
    ) -> SpatialAnalysis:
        """基于目标图像自身的patch做精确空间分析"""
        regions: List[AnomalyRegion] = []
        covered_pixels = set()

        for hit in patches:
            bbox = self._parse_bbox_from_metadata(hit)
            if not bbox:
                continue

            x1, y1, x2, y2 = bbox
            patch_area = (x2 - x1) * (y2 - y1)

            for py in range(y1, min(y2, h)):
                for px in range(x1, min(x2, w)):
                    covered_pixels.add((px, py))

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            position = self._describe_position(cx, cy, w, h)

            labels = self._parse_labels_from_metadata(hit)
            anomaly_type = ""
            for lb in self.HANDLED_LABELS:
                if lb in labels:
                    anomaly_type = lb
                    break

            regions.append(AnomalyRegion(
                bbox=bbox,
                area_pixels=patch_area,
                area_ratio=patch_area / total_area if total_area > 0 else 0,
                position_desc=position,
                anomaly_type=anomaly_type,
                confidence=max(0, 1.0 - hit.distance) if hit.distance < 1.0 else 0.1,
            ))

        total_ratio = min(1.0, len(covered_pixels) / total_area) if total_area > 0 else 0
        pattern = self._classify_distribution(regions, w, h)
        primary_location = self._find_primary_location(regions)
        pattern_desc = self._describe_pattern(pattern, len(regions), total_ratio)

        return SpatialAnalysis(
            total_anomaly_regions=len(regions),
            total_affected_ratio=total_ratio,
            primary_location=primary_location,
            distribution_pattern=pattern,
            pattern_description=pattern_desc,
            regions=regions,
        )

    def _spatial_from_external_evidence(
        self,
        external_patches: List[RetrievalHit],
        global_hits: List[RetrievalHit],
        w: int, h: int, total_area: int,
    ) -> SpatialAnalysis:
        """
        基于外部检索证据做统计推理（无法精确空间定位）。

        使用以下信息估计面积：
        1. 检索到的相似图像的 label_areas 元数据（如果有）
        2. 匹配的证据数量和距离做概率加权
        """
        import json as _json

        all_evidence = external_patches + global_hits

        # 从相似图像的label_areas元数据中估计面积
        area_estimates = {lb: [] for lb in self.HANDLED_LABELS}

        for hit in all_evidence:
            m = hit.metadata or {}
            labels = self._parse_labels_from_metadata(hit)

            # 尝试从label_areas获取面积信息
            label_areas_raw = m.get("label_areas", "")
            label_areas = {}
            if label_areas_raw:
                try:
                    label_areas = _json.loads(label_areas_raw)
                except:
                    pass

            for lb in self.HANDLED_LABELS:
                if lb in labels:
                    # 用距离做权重：越近的证据权重越高
                    weight = max(0, 1.0 - hit.distance) if hit.distance < 1.0 else 0.1
                    area_val = label_areas.get(lb, 0)
                    if isinstance(area_val, list):
                        area_val = sum(area_val)
                    area_ratio = float(area_val) / total_area if area_val and total_area > 0 else 0.05
                    area_estimates[lb].append((min(1.0, area_ratio), weight))

        # 加权平均估计面积
        total_estimated_ratio = 0.0
        label_ratios = {}
        n_evidence = 0

        for lb in self.HANDLED_LABELS:
            estimates = area_estimates[lb]
            if not estimates:
                label_ratios[lb] = 0.0
                continue
            n_evidence += len(estimates)
            weighted_sum = sum(ratio * w for ratio, w in estimates)
            weight_sum = sum(w for _, w in estimates)
            avg_ratio = weighted_sum / weight_sum if weight_sum > 0 else 0.0
            avg_ratio = min(1.0, avg_ratio)
            label_ratios[lb] = avg_ratio
            total_estimated_ratio += avg_ratio

        total_estimated_ratio = min(1.0, total_estimated_ratio)

        # 确定分布模式（基于证据数量粗估）
        if n_evidence == 0:
            pattern = DistributionPattern.NONE
        elif n_evidence <= 3:
            pattern = DistributionPattern.SCATTERED
        elif total_estimated_ratio > 0.3:
            pattern = DistributionPattern.WIDESPREAD
        elif n_evidence <= 6:
            pattern = DistributionPattern.CLUSTERED
        else:
            pattern = DistributionPattern.STRIP

        # 构建虚拟regions（用于后续严重度计算，但不做空间定位）
        regions = []
        for lb in self.HANDLED_LABELS:
            ratio = label_ratios.get(lb, 0.0)
            if ratio > 0:
                regions.append(AnomalyRegion(
                    bbox=(0, 0, w, h),  # 整张图（表示无法精确定位）
                    area_pixels=int(ratio * total_area),
                    area_ratio=ratio,
                    position_desc="基于相似图像统计推理（无法精确定位）",
                    anomaly_type=lb,
                    confidence=min(1.0, sum(
                        max(0, 1.0 - h.distance) for h in external_patches
                        if lb in self._parse_labels_from_metadata(h)
                    ) / max(1, len([
                        h for h in external_patches
                        if lb in self._parse_labels_from_metadata(h)
                    ]))),
                ))

        desc_parts = []
        for lb in self.HANDLED_LABELS:
            r = label_ratios.get(lb, 0.0)
            if r > 0:
                lb_cn = {"double_plant": "重复播种", "planter_skip": "跳播/漏播"}.get(lb, lb)
                desc_parts.append(f"{lb_cn}约{r:.1%}")

        if desc_parts:
            pattern_desc = (
                f"基于{n_evidence}条相似图像证据统计推理，"
                f"估计受影响面积：{'、'.join(desc_parts)}。"
                f"注意：精确空间位置需要目标图像自身的patch级检索才能确定。"
            )
        else:
            pattern_desc = "未检索到足够的播种异常证据进行空间分析"

        return SpatialAnalysis(
            total_anomaly_regions=len(regions),
            total_affected_ratio=total_estimated_ratio,
            primary_location="基于统计推理（无精确定位）",
            distribution_pattern=pattern,
            pattern_description=pattern_desc,
            regions=regions,
        )

    @staticmethod
    def _describe_position(cx: float, cy: float, w: int, h: int) -> str:
        """根据中心点坐标描述位置（九宫格）"""
        # 水平
        if cx < w / 3:
            h_pos = "左"
        elif cx < 2 * w / 3:
            h_pos = "中"
        else:
            h_pos = "右"

        # 垂直（图像坐标系：上方y小）
        if cy < h / 3:
            v_pos = "上"
        elif cy < 2 * h / 3:
            v_pos = "中"
        else:
            v_pos = "下"

        if h_pos == "中" and v_pos == "中":
            return "中央"
        if v_pos == "中":
            return f"中部偏{h_pos}"
        if h_pos == "中":
            return f"{v_pos}方中部"
        return f"{v_pos}{h_pos}"

    @staticmethod
    def _classify_distribution(
        regions: List[AnomalyRegion], w: int, h: int
    ) -> DistributionPattern:
        """
        根据异常区域的空间分布模式分类。

        判断逻辑：
        - 0个区域: NONE
        - 1-2个且分散: SCATTERED
        - 多个且沿某方向排列: STRIP（条带状）
        - 多个且集中在某区域: CLUSTERED
        - 覆盖大部分区域: WIDESPREAD
        """
        if not regions:
            return DistributionPattern.NONE

        n = len(regions)
        if n <= 2:
            return DistributionPattern.SCATTERED

        # 计算所有区域中心点
        centers = []
        for r in regions:
            cx = (r.bbox[0] + r.bbox[2]) / 2
            cy = (r.bbox[1] + r.bbox[3]) / 2
            centers.append((cx, cy))

        # 检查是否覆盖大部分区域
        total_covered = min(1.0, sum(r.area_ratio for r in regions))
        if total_covered > 0.3:
            return DistributionPattern.WIDESPREAD

        # 检查是否沿某方向排列（条带状）
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]

        x_spread = (max(xs) - min(xs)) / w if w > 0 else 0
        y_spread = (max(ys) - min(ys)) / h if h > 0 else 0

        # 如果一个方向的扩展远大于另一个，认为是条带状
        if x_spread > 0.5 and y_spread < 0.3:
            return DistributionPattern.STRIP
        if y_spread > 0.5 and x_spread < 0.3:
            return DistributionPattern.STRIP

        # 检查是否聚集
        if x_spread < 0.4 and y_spread < 0.4:
            return DistributionPattern.CLUSTERED

        return DistributionPattern.SCATTERED

    @staticmethod
    def _find_primary_location(regions: List[AnomalyRegion]) -> str:
        """找到异常最集中的位置"""
        if not regions:
            return "无异常区域"

        # 按面积加权的位置投票
        position_scores: Dict[str, float] = {}
        for r in regions:
            pos = r.position_desc
            position_scores[pos] = position_scores.get(pos, 0) + r.area_ratio

        if position_scores:
            return max(position_scores, key=position_scores.get)
        return regions[0].position_desc

    @staticmethod
    def _describe_pattern(pattern: DistributionPattern, n_regions: int, ratio: float) -> str:
        """生成分布模式的自然语言描述"""
        descs = {
            DistributionPattern.NONE: "未检测到明显的播种异常区域",
            DistributionPattern.SCATTERED: f"检测到{n_regions}个零星分布的异常区域，总面积约占{ratio:.1%}",
            DistributionPattern.STRIP: f"检测到{n_regions}个异常区域呈条带状分布，可能沿播种行方向延伸，总面积约占{ratio:.1%}",
            DistributionPattern.CLUSTERED: f"检测到{n_regions}个异常区域集中分布在某一局部，总面积约占{ratio:.1%}",
            DistributionPattern.WIDESPREAD: f"检测到{n_regions}个异常区域大面积分布，覆盖面积约占{ratio:.1%}，情况较为严重",
        }
        return descs.get(pattern, f"检测到{n_regions}个异常区域")

    # ================================================================
    # 阶段3：严重程度量化
    # ================================================================

    def _stage3_quantify_severity(
        self,
        general_prediction: Dict[str, Any],
        filtered_hits: List[RetrievalHit],
        spatial: SpatialAnalysis,
    ) -> List[AnomalyDetail]:
        """
        为每种播种异常类型独立评估严重程度。

        综合信息源：
        1. 通用预测结果中的标签和置信度
        2. 筛选后证据的数量和距离
        3. 空间分析的面积和分布
        """
        details = []

        for label in self.HANDLED_LABELS:
            # 从通用预测中获取该类别的信息
            pred_info = self._get_label_from_prediction(general_prediction, label)
            detected = pred_info.get("detected", False)
            pred_confidence = pred_info.get("confidence", 0.0)

            # 从证据中统计
            label_hits = [h for h in filtered_hits
                          if label in self._parse_labels_from_metadata(h)]
            evidence_confidence = self._compute_evidence_confidence(label_hits)

            # 从空间分析中获取该类别的区域
            label_regions = [r for r in spatial.regions if r.anomaly_type == label]
            # Use spatial analysis regions for area (handles both self-patch and statistical modes)
            label_regions = [r for r in spatial.regions if r.anomaly_type == label]
            if label_regions and label_regions[0].bbox == (0, 0, 512, 512):
                # Statistical mode: use the estimated ratio directly
                area_ratio = min(1.0, label_regions[0].area_ratio) if label_regions else 0.0
            else:
                # Self-patch mode: pixel-dedup
                covered = set()
                for r in label_regions:
                    x1, y1, x2, y2 = r.bbox
                    for py in range(y1, min(y2, 512)):
                        for px in range(x1, min(x2, 512)):
                            covered.add((px, py))
                area_ratio = min(1.0, len(covered) / (512 * 512)) if label_regions else 0.0

            # 如果通用预测未检测到，但证据强烈支持，仍标记为检测到
            if not detected and len(label_hits) >= 2 and evidence_confidence > 0.5:
                detected = True
                logger.info(f"[PlantingAgent] {label}: 通用预测未检出，但证据支持（{len(label_hits)}条），标记为检出")

            # 综合置信度
            confidence = max(pred_confidence, evidence_confidence) if detected else 0.0

            # 确定严重度
            severity = SeverityLevel.NONE
            distribution = DistributionPattern.NONE
            if detected:
                severity = self._classify_severity(area_ratio)
                distribution = self._classify_distribution(label_regions, 512, 512)

                # 分布模式也影响严重度：条带状通常更严重
                if distribution == DistributionPattern.STRIP and severity == SeverityLevel.MINOR:
                    severity = SeverityLevel.MODERATE

            # 收集证据引用
            refs = [f"E{i+1}" for i, h in enumerate(filtered_hits) if h in label_hits]

            details.append(AnomalyDetail(
                label=label,
                detected=detected,
                confidence=confidence,
                affected_area_ratio=area_ratio,
                severity=severity,
                distribution=distribution,
                regions=label_regions,
                visual_evidence=pred_info.get("reason", ""),
                evidence_refs=refs,
            ))

        return details

    @staticmethod
    def _classify_severity(area_ratio: float) -> SeverityLevel:
        """根据面积占比分级"""
        for (lo, hi), level in _SEVERITY_THRESHOLDS.items():
            if lo <= area_ratio < hi:
                return level
        return SeverityLevel.MINOR

    @staticmethod
    def _compute_evidence_confidence(hits: List[RetrievalHit]) -> float:
        """
        根据检索到的相关证据计算置信度。

        逻辑：
        - 更多的证据 → 更高置信
        - 更近的距离 → 更高置信
        """
        if not hits:
            return 0.0

        # 距离转置信度：distance=0 → 1.0, distance=1.0 → 0.0
        confidences = [max(0, 1.0 - h.distance) for h in hits]

        # 取top-3的平均
        confidences.sort(reverse=True)
        top = confidences[:3]
        avg = sum(top) / len(top)

        # 证据数量加成：>3条+0.1, >5条+0.15
        bonus = 0.0
        if len(hits) >= 3:
            bonus += 0.1
        if len(hits) >= 5:
            bonus += 0.05

        return min(1.0, avg + bonus)

    @staticmethod
    def _get_label_from_prediction(prediction: Dict[str, Any], label: str) -> Dict[str, Any]:
        """从通用预测结果中提取特定标签的信息"""
        result: Dict[str, Any] = {"detected": False, "confidence": 0.0, "reason": ""}

        if not prediction:
            return result

        # 从evidence列表中查找
        evidence_list = prediction.get("evidence", [])
        if isinstance(evidence_list, list):
            for item in evidence_list:
                if isinstance(item, dict) and item.get("label", "").strip() == label:
                    result["detected"] = True
                    result["confidence"] = float(item.get("confidence", 0.0))
                    result["reason"] = item.get("reason", "")
                    return result

        # 从conclusion中检查
        conclusion = prediction.get("conclusion", "")
        if isinstance(conclusion, str) and label in conclusion.lower().replace(" ", "_"):
            result["detected"] = True
            result["confidence"] = 0.5  # 只从conclusion推断，给中等置信度
            return result

        return result

    @staticmethod
    def _compute_overall_severity(details: List[AnomalyDetail]) -> SeverityLevel:
        """取所有异常中最高的严重度"""
        severity_order = [
            SeverityLevel.NONE, SeverityLevel.MINOR,
            SeverityLevel.MODERATE, SeverityLevel.SEVERE,
            SeverityLevel.CRITICAL,
        ]
        max_idx = 0
        for d in details:
            if d.detected:
                idx = severity_order.index(d.severity) if d.severity in severity_order else 0
                max_idx = max(max_idx, idx)
        return severity_order[max_idx]

    # ================================================================
    # 阶段4：VLM二次推理（Phase 2）
    # ================================================================

    def _stage4_vlm_reasoning(
        self,
        image_path: str,
        filtered_hits: List[RetrievalHit],
        anomaly_details: List[AnomalyDetail],
        spatial: SpatialAnalysis,
        qwen_client: Any,
    ) -> str:
        """
        使用专项prompt调用Qwen2-VL进行播种异常的深度分析。

        Args:
            image_path: 待分析图像
            filtered_hits: 播种相关证据
            anomaly_details: 阶段3的量化结果
            spatial: 空间分析结果
            qwen_client: Qwen3VLClient实例

        Returns:
            VLM原始输出字符串
        """
        # 构建预分析上下文
        pre_analysis = {
            "detected_types": [d.label for d in anomaly_details if d.detected],
            "severity_estimate": self._compute_overall_severity(anomaly_details).value,
            "area_ratio": spatial.total_affected_ratio,
            "evidence_summary": [
                {
                    "original_index": i + 1,
                    "distance": h.distance,
                    "labels": self._parse_labels_from_metadata(h),
                    "tile_id": (h.metadata or {}).get("tile_id", ""),
                }
                for i, h in enumerate(filtered_hits[:6])
            ],
        }

        messages = build_planting_vlm_messages(
            image_path=image_path,
            filtered_hits=filtered_hits,
            pre_analysis=pre_analysis,
            spatial_info=spatial,
            max_evidence=4,
        )

        try:
            output = qwen_client.chat(messages)
            return output
        except Exception as e:
            logger.error(f"[PlantingAgent] VLM推理失败: {e}")
            return ""

    def _update_from_vlm_output(
        self, vlm_output: str, anomaly_details: List[AnomalyDetail]
    ) -> None:
        """用VLM输出更新分析详情（置信度、视觉描述等）"""
        if not vlm_output:
            return

        try:
            # 尝试解析JSON
            data = self._safe_parse_json(vlm_output)
            if not data:
                return

            pa = data.get("planting_analysis", {})
            for detail in anomaly_details:
                label_info = pa.get(detail.label, {})
                if not label_info:
                    continue

                # 更新检测状态
                vlm_detected = label_info.get("detected", None)
                if vlm_detected is not None:
                    if vlm_detected and not detail.detected:
                        detail.detected = True
                        logger.info(f"[PlantingAgent] VLM检出 {detail.label}（阶段3未检出）")

                # 更新置信度（取较高值）
                vlm_conf = label_info.get("confidence", 0.0)
                if isinstance(vlm_conf, (int, float)):
                    detail.confidence = max(detail.confidence, float(vlm_conf))

                # 更新视觉描述
                visual = label_info.get("visual_evidence", "")
                if visual:
                    detail.visual_evidence = visual

        except Exception as e:
            logger.warning(f"[PlantingAgent] 解析VLM输出失败: {e}")

    @staticmethod
    def _safe_parse_json(text: str) -> Optional[Dict]:
        """安全解析JSON，兼容代码块包裹"""
        text = text.strip()
        # 去除 ```json ... ``` 包裹
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试找到第一个 { 和最后一个 }
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
        return None

    # ================================================================
    # 阶段5：补救方案生成
    # ================================================================

    def _stage5_generate_recommendations(
        self, anomaly_details: List[AnomalyDetail]
    ) -> List[Recommendation]:
        """
        基于决策矩阵生成补救建议。

        每种检测到的异常类型根据其严重度匹配对应的建议模板。
        """
        recommendations = []

        for detail in anomaly_details:
            if not detail.detected or detail.severity == SeverityLevel.NONE:
                continue

            key = (detail.label, detail.severity.value)
            template = _RECOMMENDATION_MATRIX.get(key)

            if template is None:
                # 降级匹配：使用同类型的最近严重度
                template = self._fallback_recommendation(detail.label, detail.severity)

            if template:
                rec = Recommendation(
                    action=template["action"],
                    urgency=template.get("urgency", UrgencyLevel.MEDIUM),
                    target_anomaly=detail.label,
                    target_area=detail.regions[0].position_desc if detail.regions else "全域",
                    expected_effect=template.get("expected_effect", ""),
                    notes=template.get("notes", ""),
                )
                recommendations.append(rec)

        # 按紧急程度排序
        urgency_order = {
            UrgencyLevel.CRITICAL: 0,
            UrgencyLevel.HIGH: 1,
            UrgencyLevel.MEDIUM: 2,
            UrgencyLevel.LOW: 3,
        }
        recommendations.sort(key=lambda r: urgency_order.get(r.urgency, 99))

        return recommendations

    @staticmethod
    def _fallback_recommendation(label: str, severity: SeverityLevel) -> Optional[Dict]:
        """当精确匹配失败时，尝试降级匹配"""
        severity_order = ["minor", "moderate", "severe", "critical"]
        target_idx = severity_order.index(severity.value) if severity.value in severity_order else 0

        # 向下查找最近的
        for idx in range(target_idx, -1, -1):
            key = (label, severity_order[idx])
            if key in _RECOMMENDATION_MATRIX:
                return _RECOMMENDATION_MATRIX[key]
        return None

    # ================================================================
    # 辅助方法
    # ================================================================

    @staticmethod
    def _compute_overall_confidence(details: List[AnomalyDetail]) -> float:
        """计算整体置信度：取已检出异常的加权平均"""
        detected = [d for d in details if d.detected]
        if not detected:
            return 0.0
        return sum(d.confidence for d in detected) / len(detected)

    @staticmethod
    def _compute_overall_urgency(recommendations: List[Recommendation]) -> UrgencyLevel:
        """取最高紧急程度"""
        if not recommendations:
            return UrgencyLevel.LOW
        urgency_order = [UrgencyLevel.LOW, UrgencyLevel.MEDIUM, UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]
        max_idx = 0
        for r in recommendations:
            idx = urgency_order.index(r.urgency) if r.urgency in urgency_order else 0
            max_idx = max(max_idx, idx)
        return urgency_order[max_idx]

    @staticmethod
    def _collect_evidence_refs(hits: List[RetrievalHit]) -> List[str]:
        """收集证据引用列表"""
        return [f"E{i+1}" for i in range(len(hits))]

    @staticmethod
    def _extract_tile_id(hits: List[RetrievalHit]) -> str:
        """从检索结果中提取tile_id"""
        if hits:
            return (hits[0].metadata or {}).get("tile_id", "")
        return ""

    @staticmethod
    def _generate_summary(
        details: List[AnomalyDetail],
        severity: SeverityLevel,
        spatial: SpatialAnalysis,
    ) -> str:
        """生成一句话摘要"""
        detected = [d for d in details if d.detected]

        if not detected:
            return "未检测到播种异常（double_plant / planter_skip）"

        type_names = {
            "double_plant": "重复播种",
            "planter_skip": "跳播/漏播",
        }
        severity_names = {
            SeverityLevel.MINOR: "轻度",
            SeverityLevel.MODERATE: "中度",
            SeverityLevel.SEVERE: "重度",
            SeverityLevel.CRITICAL: "极重",
        }

        types_str = "、".join(type_names.get(d.label, d.label) for d in detected)
        sev_str = severity_names.get(severity, "")
        area_str = f"，受影响面积约{spatial.total_affected_ratio:.1%}" if spatial.total_affected_ratio > 0 else ""

        return f"检测到{sev_str}{types_str}异常{area_str}，{spatial.pattern_description}"