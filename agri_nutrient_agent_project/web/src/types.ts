export type BBox = [number, number, number, number]

export type FrontendScene = {
  center: [number, number]
  default_zoom: number
  terrain: { enabled: boolean; exaggeration: number; dem_tiles_json: string }
  bbox: BBox
  layers: Record<string, string>
}

export type NutrientOutput = {
  schema: string
  time_utc: string
  run_id: string
  field_id: string
  upstream_path: string
  query_image_path: string
  decision: {
    is_applicable: boolean
    upstream_label_conf: number
    agent_validation_score: number
    uncertainty_notes: string[]
  }
  diagnosis: {
    summary_cn: string
    summary_en: string
    suspected_nutrients: { nutrient: string; prob: number; evidence: string[] }[]
  }
  area_stats: {
    affected_area_px: number
    affected_ratio: number
    patch_count: number
    largest_patch_px: number
  }
  layers: {
    severity_geojson: string
    sampling_points_geojson: string
    severity_heatmap_png: string
    index_preview_png: string
  }
  report_md: string
  assets: Record<string, any>
  frontend_scene: FrontendScene
}
