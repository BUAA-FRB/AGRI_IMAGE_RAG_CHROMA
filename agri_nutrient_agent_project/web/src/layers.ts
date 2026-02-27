import { BitmapLayer, GeoJsonLayer, ScatterplotLayer } from '@deck.gl/layers'
import type { Layer } from '@deck.gl/core'
import type { NutrientOutput } from './types'
import { assetUrl } from './api'

function bboxCorners(bbox: [number, number, number, number]) {
  const [minLng, minLat, maxLng, maxLat] = bbox
  return [
    [minLng, maxLat],
    [maxLng, maxLat],
    [maxLng, minLat],
    [minLng, minLat]
  ] as any
}

export async function buildDeckLayers(
  out: NutrientOutput,
  toggles: {
    showHeat: boolean
    showPolys: boolean
    showSampling: boolean
    extrude: boolean
  }
): Promise<Layer<any>[]> {
  const scene = out.frontend_scene
  const layers: Layer<any>[] = []

  // Field boundary bbox
  if (scene.layers.field_bbox_geojson) {
    layers.push(
      new GeoJsonLayer({
        id: 'field-bbox',
        data: assetUrl(scene.layers.field_bbox_geojson),
        stroked: true,
        filled: false,
        getLineColor: [255, 255, 255, 180],
        lineWidthMinPixels: 2
      })
    )
  }

  // Heatmap overlay as bitmap (transparent PNG)
  if (toggles.showHeat) {
    layers.push(
      new BitmapLayer({
        id: 'severity-heat',
        image: assetUrl(scene.layers.severity_heatmap_png),
        bounds: bboxCorners(scene.bbox),
        opacity: 0.75
      })
    )
  }

  // Severity polygons (extruded)
  if (toggles.showPolys) {
    layers.push(
      new GeoJsonLayer({
        id: 'severity-polys',
        data: assetUrl(scene.layers.severity_geojson),

        // appearance
        stroked: true,
        filled: true,
        opacity: 0.9,
        wireframe: false,

        // interaction
        pickable: true,
        autoHighlight: true,
        highlightColor: [255, 255, 255, 120],

        // 3D
        extruded: toggles.extrude,

        // colors (avoid black default)
        getFillColor: (f: any) => {
          // You can map severity if you add that property; for now unified warm color
          return [255, 120, 40, 120]
        },
        getLineColor: [255, 255, 255, 220],
        lineWidthMinPixels: 1,

        // elevation (avoid huge cubes)
        getElevation: (f: any) => {
          if (!toggles.extrude) return 0
          const area = f?.properties?.area_px ?? 0
          // 3..25 meters (visual scale)
          const h = Math.min(25, Math.max(3, Math.log10(area + 10) * 6))
          return h
        },

        // material makes it feel "3D" not flat
        material: {
          ambient: 0.25,
          diffuse: 0.6,
          shininess: 40,
          specularColor: [60, 60, 60]
        }
      })
    )
  }

  // Sampling points
  if (toggles.showSampling) {
    layers.push(
      new ScatterplotLayer({
        id: 'sampling-pts',
        data: assetUrl(scene.layers.sampling_points_geojson),
        getPosition: (d: any) => d.geometry.coordinates,
        getRadius: 8,
        radiusMinPixels: 4,
        pickable: true,
        getFillColor: [80, 200, 255, 200],
        getLineColor: [255, 255, 255, 220],
        lineWidthMinPixels: 1
      })
    )
  }

  return layers
}