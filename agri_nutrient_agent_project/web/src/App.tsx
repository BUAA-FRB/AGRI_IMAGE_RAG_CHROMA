import React, { useEffect, useMemo, useState } from 'react'
import DeckGL from '@deck.gl/react'
import { Map } from 'react-map-gl/maplibre'
import type { ViewState } from '@deck.gl/core'

import { fetchLatest } from './api'
import type { NutrientOutput } from './types'
import { buildMapStyle } from './mapStyle'
import { buildDeckLayers } from './layers'
import SidePanel from './components/SidePanel'

export default function App() {
  const [out, setOut] = useState<NutrientOutput | null>(null)
  const [layers, setLayers] = useState<any[]>([])
  const [toggles, setToggles] = useState({ showHeat: true, showPolys: true, showSampling: true, extrude: true })

  useEffect(() => {
    fetchLatest().then(setOut).catch(err => {
      console.error(err)
      alert('Failed to load nutrient output. Did you start serve.py and run_agent.py?')
    })
  }, [])

  useEffect(() => {
    if (!out) return
    buildDeckLayers(out, toggles).then(setLayers)
  }, [out, toggles])

  const viewState: ViewState | null = useMemo(() => {
    if (!out) return null
    return {
      longitude: out.frontend_scene.center[0],
      latitude: out.frontend_scene.center[1],
      zoom: out.frontend_scene.default_zoom,
      pitch: 60,
      bearing: 0
    }
  }, [out])

  const mapStyle = useMemo(() => {
    if (!out) return null
    return buildMapStyle(out.frontend_scene.terrain.dem_tiles_json, out.frontend_scene.terrain.exaggeration)
  }, [out])

  if (!out || !viewState || !mapStyle) {
    return <div style={{ padding: 18 }}>Loading...</div>
  }

  return (
    <div style={{ display: 'flex', height: '100%' }}>
      <div style={{ flex: 1, position: 'relative' }}>
        <DeckGL
          initialViewState={viewState as any}
          controller={true}
          layers={layers}
          style={{ position: 'absolute', inset: 0 }}
        >
          <Map
            mapStyle={mapStyle as any}
            reuseMaps={true}
            attributionControl={true}
          />
        </DeckGL>
      </div>

      <SidePanel
        out={out}
        toggles={toggles}
        onToggle={(k, v) => setToggles(prev => ({ ...prev, [k]: v }))}
      />
    </div>
  )
}
