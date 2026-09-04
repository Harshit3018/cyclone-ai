import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import { getCycloneTrack } from '../api/client';
import type { Cyclone } from '../types';

// Fix Leaflet default icon issue with bundlers
import iconUrl from 'leaflet/dist/images/marker-icon.png';
import iconRetinaUrl from 'leaflet/dist/images/marker-icon-2x.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

const DefaultIcon = L.icon({
  iconUrl,
  iconRetinaUrl,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});
L.Marker.prototype.options.icon = DefaultIcon;

interface Props {
  cyclones: Cyclone[];
  selectedId?: string;
  trackPoints?: Array<{ latitude: number; longitude: number; category?: string; point_index?: number; wind_kph?: number; timestamp?: string }>;
  /**
   * `uncertainty_radius_km` is nullable: the API reports null at lead times where the
   * cone was never measured rather than extrapolating one. The truthiness guards at the
   * circle-drawing site already handle it — no cone is drawn — but the type has to admit
   * null or callers cannot pass a real ForecastPoint.
   */
  forecastPoints?: Array<{ latitude: number; longitude: number; forecast_hour: number; uncertainty_radius_km?: number | null }>;
  height?: string;
}

const categoryColor = (cat?: string): string => {
  if (!cat) return '#64748b';
  if (cat.includes('Super')) return '#dc2626';
  if (cat.includes('Extremely')) return '#ea580c';
  if (cat.includes('Very Severe')) return '#f59e0b';
  if (cat.includes('Severe')) return '#eab308';
  if (cat.includes('Cyclonic Storm')) return '#22c55e';
  if (cat.includes('Deep')) return '#06b6d4';
  if (cat.includes('Depression')) return '#3b82f6';
  return '#64748b';
};

// OpenStreetMap standard tiles: genuinely keyless.
// Do not switch this to a CARTO basemap URL. Both basemaps.cartocdn.com and
// cartodb-basemaps-*.global.ssl.fastly.net now return a 10 KB placeholder tile
// stamped "API KEY REQUIRED" — HTTP 200 with a valid PNG, so nothing errors and the
// watermark just appears across the whole map. Verify any replacement by fetching a
// tile and checking its size, not by checking the status code.
// Swap this for a self-hosted tile path to run the demo fully offline.
const TILE_URL = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';

type OverviewTrack = {
  latlngs: L.LatLngTuple[];
  last: { latitude: number; longitude: number; category?: string };
};

export default function CycloneMap({ cyclones, selectedId, trackPoints, forecastPoints, height = '100%' }: Props) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<L.Map | null>(null);
  const [overviewTracks, setOverviewTracks] = useState<Record<string, OverviewTrack>>({});

  // The dashboard passes a cyclone list with no coordinates on it, so fetch each
  // cyclone's real track from the API. This replaces the previous placeholder that
  // laid markers out on a synthetic diagonal (baseLat = 12 + idx * 2.5).
  const needOverview = !trackPoints || trackPoints.length === 0;
  const overviewKey = needOverview ? cyclones.map(c => c.id).join(',') : '';

  useEffect(() => {
    if (!needOverview || cyclones.length === 0) return;
    let cancelled = false;

    Promise.all(
      cyclones.map(c =>
        getCycloneTrack(c.id)
          .then(r => {
            const pts = (r.data.track_points || []) as Array<{
              latitude: number; longitude: number; category?: string;
            }>;
            if (pts.length === 0) return null;
            return [c.id, {
              latlngs: pts.map(p => [p.latitude, p.longitude] as L.LatLngTuple),
              last: pts[pts.length - 1],
            }] as const;
          })
          .catch(() => null)
      )
    ).then(entries => {
      if (cancelled) return;
      const next: Record<string, OverviewTrack> = {};
      for (const e of entries) if (e) next[e[0]] = e[1];
      setOverviewTracks(next);
    });

    return () => { cancelled = true; };
    // overviewKey is the stable identity of the cyclone list
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [overviewKey, needOverview]);

  useEffect(() => {
    if (!mapRef.current) return;
    if (mapInstance.current) return;

    const map = L.map(mapRef.current, {
      center: [15, 82],
      zoom: 5,
      zoomControl: true,
      attributionControl: true,
    });

    // Keyless basemap — see the TILE_URL comment above before changing this.
    L.tileLayer(TILE_URL, {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 18,
    }).addTo(map);

    mapInstance.current = map;

    return () => {
      map.remove();
      mapInstance.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapInstance.current;
    if (!map) return;

    // Clear existing overlays (keep the tile layer). Collect first, then remove:
    // removing inside eachLayer mutates the collection being iterated and silently
    // skips layers, leaving stale markers behind on re-render.
    const stale: L.Layer[] = [];
    map.eachLayer(layer => {
      if (!(layer instanceof L.TileLayer)) stale.push(layer);
    });
    stale.forEach(layer => map.removeLayer(layer));

    // Draw track points if available
    if (trackPoints && trackPoints.length > 0) {
      const latlngs: L.LatLngExpression[] = trackPoints.map(p => [p.latitude, p.longitude]);

      // Track polyline
      L.polyline(latlngs, {
        color: '#0c8df0',
        weight: 2.5,
        opacity: 0.8,
        dashArray: undefined,
      }).addTo(map);

      // Track point markers
      trackPoints.forEach((p, i) => {
        const color = categoryColor(p.category);
        const isLast = i === trackPoints.length - 1;

        const circle = L.circleMarker([p.latitude, p.longitude], {
          radius: isLast ? 8 : 4,
          fillColor: color,
          fillOpacity: isLast ? 1 : 0.7,
          color: isLast ? '#fff' : color,
          weight: isLast ? 2 : 1,
        }).addTo(map);

        circle.bindPopup(`
          <div style="font-family:Inter,sans-serif;font-size:12px">
            <div style="font-weight:600;margin-bottom:4px">${p.timestamp || ''}</div>
            <div>Lat: ${p.latitude.toFixed(2)}° Lon: ${p.longitude.toFixed(2)}°</div>
            <div>Wind: ${p.wind_kph?.toFixed(0) || '--'} km/h</div>
            <div>Category: ${p.category || '--'}</div>
            <div style="color:#f59e0b;margin-top:4px;font-size:10px">OBSERVED • ${selectedId ? 'DEMO' : ''} DATA</div>
          </div>
        `);
      });

      // Fit bounds
      if (latlngs.length > 0) {
        map.fitBounds(L.latLngBounds(latlngs as L.LatLngTuple[]).pad(0.3));
      }
    }

    // Draw forecast points
    if (forecastPoints && forecastPoints.length > 0) {
      const lastTrack = trackPoints && trackPoints.length > 0
        ? trackPoints[trackPoints.length - 1]
        : null;

      const fcLatlngs: L.LatLngExpression[] = [];
      if (lastTrack) fcLatlngs.push([lastTrack.latitude, lastTrack.longitude]);

      forecastPoints.forEach(fp => {
        fcLatlngs.push([fp.latitude, fp.longitude]);

        // Forecast marker
        L.circleMarker([fp.latitude, fp.longitude], {
          radius: 6,
          fillColor: '#f59e0b',
          fillOpacity: 0.8,
          color: '#fff',
          weight: 1,
        }).addTo(map).bindPopup(`
          <div style="font-family:Inter,sans-serif;font-size:12px">
            <div style="font-weight:600;color:#f59e0b">FORECAST +${fp.forecast_hour}h</div>
            <div>Lat: ${fp.latitude.toFixed(2)}° Lon: ${fp.longitude.toFixed(2)}°</div>
            ${fp.uncertainty_radius_km ? `<div>Uncertainty: ±${fp.uncertainty_radius_km.toFixed(0)} km</div>` : ''}
            <div style="color:#f59e0b;margin-top:4px;font-size:10px">AI PREDICTION • BASELINE MODEL</div>
          </div>
        `);

        // Uncertainty circle
        if (fp.uncertainty_radius_km && fp.uncertainty_radius_km > 0) {
          L.circle([fp.latitude, fp.longitude], {
            radius: fp.uncertainty_radius_km * 1000,
            fillColor: '#f59e0b',
            fillOpacity: 0.08,
            color: '#f59e0b',
            weight: 1,
            opacity: 0.3,
            dashArray: '4 4',
          }).addTo(map);
        }
      });

      // Forecast line
      if (fcLatlngs.length > 1) {
        L.polyline(fcLatlngs, {
          color: '#f59e0b',
          weight: 2,
          opacity: 0.7,
          dashArray: '8 6',
        }).addTo(map);
      }
    }

    // Dashboard overview: draw each cyclone's real observed track.
    if (!trackPoints || trackPoints.length === 0) {
      if (cyclones.length > 0) {
        const bounds: L.LatLngTuple[] = [];

        cyclones.forEach(c => {
          const ov = overviewTracks[c.id];
          if (!ov) return;  // track not loaded yet — draw nothing rather than a guess

          const color = categoryColor(c.peak_category);

          // Full observed track
          L.polyline(ov.latlngs, {
            color,
            weight: 2,
            opacity: c.is_active ? 0.85 : 0.5,
          }).addTo(map);

          // Marker at the most recent position
          const marker = L.circleMarker([ov.last.latitude, ov.last.longitude], {
            radius: c.is_active ? 10 : 6,
            fillColor: color,
            fillOpacity: c.is_active ? 0.9 : 0.6,
            color: c.is_active ? '#fff' : color,
            weight: c.is_active ? 2 : 1,
          }).addTo(map);

          marker.bindPopup(`
            <div style="font-family:Inter,sans-serif;font-size:12px">
              <div style="font-weight:700;margin-bottom:4px;color:#06b6d4">${c.name.replace('DEMO_', '')}</div>
              <div>Season: ${c.season} • ${c.basin}</div>
              <div>Peak Wind: ${c.peak_wind_kph?.toFixed(0) || '--'} km/h</div>
              <div>Min Pressure: ${c.min_pressure_hpa?.toFixed(0) || '--'} hPa</div>
              <div>Last position: ${ov.last.latitude.toFixed(2)}°N, ${ov.last.longitude.toFixed(2)}°E</div>
              <div style="margin-top:4px">
                <span style="color:${c.is_active ? '#22c55e' : '#6b7280'};font-size:10px;font-weight:600">${c.is_active ? '● ACTIVE' : '○ HISTORICAL'}</span>
              </div>
              <div style="color:#f59e0b;margin-top:4px;font-size:10px">${c.data_type}</div>
            </div>
          `);

          bounds.push(...ov.latlngs);
        });

        if (bounds.length > 1) {
          map.fitBounds(L.latLngBounds(bounds).pad(0.15));
        } else if (bounds.length === 1) {
          map.setView(bounds[0], 6);
        } else {
          map.setView([15, 82], 5);
        }
      } else {
        // Default view: North Indian Ocean
        map.setView([15, 82], 5);
      }
    }

  }, [cyclones, trackPoints, forecastPoints, selectedId, overviewTracks]);

  return <div ref={mapRef} style={{ height, width: '100%' }} />;
}
