/**
 * MapLibre 3D terrain style (no token required for DEM).
 * - Satellite raster basemap for a more "agri" look (demo)
 * - Terrain DEM from demotiles.maplibre.org
 *
 * Note:
 * - The satellite tile source below is for demo/showcase. In production,
 *   you should use a properly licensed/self-hosted tile source.
 */
export function buildMapStyle(demTilesJsonUrl: string, exaggeration: number) {
  return {
    version: 8,
    sources: {
      // Satellite basemap (demo)
      satellite: {
        type: 'raster',
        tiles: [
          'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
        ],
        tileSize: 256,
        maxzoom: 19
      },

      terrainSource: {
        type: 'raster-dem',
        url: demTilesJsonUrl,
        tileSize: 256
      },

      hillshadeSource: {
        type: 'raster-dem',
        url: demTilesJsonUrl,
        tileSize: 256
      }
    },
    layers: [
      { id: 'sat', type: 'raster', source: 'satellite' },
      {
        id: 'hills',
        type: 'hillshade',
        source: 'hillshadeSource',
        layout: { visibility: 'visible' }
      }
    ],
    terrain: {
      source: 'terrainSource',
      exaggeration
    },
    sky: {}
  }
}