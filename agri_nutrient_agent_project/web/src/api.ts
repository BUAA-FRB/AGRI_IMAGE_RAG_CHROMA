import type { NutrientOutput } from './types'

const API_BASE = (import.meta as any).env?.VITE_API_BASE || 'http://localhost:8000'

export async function fetchLatest(): Promise<NutrientOutput> {
  const res = await fetch(`${API_BASE}/api/nutrient/latest`)
  if (!res.ok) throw new Error(`Failed to load: ${res.status}`)
  return res.json()
}

export function assetUrl(rel: string): string {
  // rel like "assets/xxx.png" or "geojson/xxx.geojson"
  const clean = rel.replace(/^\/+/, '')
  return `${API_BASE}/nutrient_output/${clean}`
}
