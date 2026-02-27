import React from 'react'
import type { NutrientOutput } from '../types'
import { assetUrl } from '../api'

export default function SidePanel(props: {
  out: NutrientOutput
  onToggle: (k: string, v: boolean) => void
  toggles: { showHeat: boolean; showPolys: boolean; showSampling: boolean; extrude: boolean }
}) {
  const { out, toggles } = props
  return (
    <div style={{
      width: 420,
      padding: 12,
      background: 'rgba(255,255,255,0.96)',
      overflow: 'auto',
      borderLeft: '1px solid #ddd'
    }}>
      <h3 style={{ margin: '6px 0 10px' }}>Nutrient Deficiency Agent</h3>
      <div style={{ fontSize: 13, color: '#333' }}>
        <div><b>is_applicable</b>: {String(out.decision.is_applicable)}</div>
        <div><b>up_conf</b>: {out.decision.upstream_label_conf.toFixed(3)}</div>
        <div><b>agent_score</b>: {out.decision.agent_validation_score.toFixed(3)}</div>
        <div style={{ marginTop: 6 }}><b>affected_ratio</b>: {out.area_stats.affected_ratio.toFixed(3)}</div>
        <div><b>patch_count</b>: {out.area_stats.patch_count}</div>
      </div>

      <h4>Layers</h4>
      <label><input type="checkbox" checked={toggles.showHeat} onChange={e=>props.onToggle('showHeat', e.target.checked)} /> Heatmap</label><br/>
      <label><input type="checkbox" checked={toggles.showPolys} onChange={e=>props.onToggle('showPolys', e.target.checked)} /> Polygons</label><br/>
      <label><input type="checkbox" checked={toggles.extrude} onChange={e=>props.onToggle('extrude', e.target.checked)} /> Extrude 3D</label><br/>
      <label><input type="checkbox" checked={toggles.showSampling} onChange={e=>props.onToggle('showSampling', e.target.checked)} /> Sampling points</label>

      <h4>Summary (CN)</h4>
      <div style={{ fontSize: 13, lineHeight: 1.35 }}>{out.diagnosis.summary_cn}</div>

      <h4>Summary (EN)</h4>
      <div style={{ fontSize: 13, lineHeight: 1.35 }}>{out.diagnosis.summary_en}</div>

      {out.decision.uncertainty_notes?.length ? (
        <>
          <h4>Uncertainty</h4>
          <ul>
            {out.decision.uncertainty_notes.map((t, i) => <li key={i} style={{ fontSize: 13 }}>{t}</li>)}
          </ul>
        </>
      ) : null}

      <h4>Visuals</h4>
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap: 8 }}>
        <a href={assetUrl(out.assets.query_preview || 'assets/query_preview.png')} target="_blank">
          <img src={assetUrl(out.assets.query_preview || 'assets/query_preview.png')} style={{ width:'100%', border:'1px solid #ddd' }} />
        </a>
        <a href={assetUrl(out.layers.index_preview_png)} target="_blank">
          <img src={assetUrl(out.layers.index_preview_png)} style={{ width:'100%', border:'1px solid #ddd' }} />
        </a>
        <a href={assetUrl(out.layers.severity_heatmap_png)} target="_blank">
          <img src={assetUrl(out.layers.severity_heatmap_png)} style={{ width:'100%', border:'1px solid #ddd' }} />
        </a>
        <a href={assetUrl(out.report_md)} target="_blank" style={{ fontSize: 13, alignSelf:'center' }}>Open report.md</a>
      </div>

      <h4>Suspected nutrients</h4>
      <ul>
        {out.diagnosis.suspected_nutrients.map((n, i) => (
          <li key={i} style={{ fontSize: 13 }}>
            {n.nutrient}: {(n.prob*100).toFixed(1)}%
          </li>
        ))}
      </ul>

      <div style={{ marginTop: 12, fontSize: 12, color:'#666' }}>
        Assets served from: <code>/nutrient_output</code>
      </div>
    </div>
  )
}
