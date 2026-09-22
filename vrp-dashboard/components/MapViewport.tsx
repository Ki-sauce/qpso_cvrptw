"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { LocateFixed, Pause, Play, RotateCcw, Route, SlidersHorizontal } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import DraggablePanel from "@/components/DraggablePanel";

type RouteProperties = {
  vehicle?: number;
  color?: string;
  stops?: string[];
};

type RouteCollection = GeoJSON.FeatureCollection<GeoJSON.LineString, RouteProperties>;
type RouteNodeProperties = {
  vehicle: number;
  kind: "source" | "destination" | "intermediate";
  color: string;
};
type RouteNodeCollection = GeoJSON.FeatureCollection<GeoJSON.Point, RouteNodeProperties>;
type Coordinate = [number, number];
type VehicleFilter = number | "all";

export interface MapViewportProps {
  routesGeoJSON: RouteCollection | GeoJSON.FeatureCollection<GeoJSON.LineString, Record<string, unknown>> | null | undefined;
  customers: Array<{ id: string; lat: number; lng: number; demand?: number; ready_time?: number; due_time?: number }>;
  depot: { id: string; lat: number; lng: number };
  activeCity: "salt-lake" | "manhattan";
}

const CITY_PRESETS: Record<
  "salt-lake" | "manhattan",
  { center: [number, number]; zoom: number; bounds: maplibregl.LngLatBoundsLike }
> = {
  "salt-lake": {
    center: [88.433, 22.58],
    zoom: 12.7,
    bounds: [
      [88.25, 22.4],
      [88.6, 22.75],
    ],
  },
  manhattan: {
    center: [-73.9712, 40.7831],
    zoom: 12.2,
    bounds: [
      [-74.15, 40.6],
      [-73.75, 40.95],
    ],
  },
};

const EMPTY_COLLECTION: RouteCollection = {
  type: "FeatureCollection",
  features: [],
};

const EMPTY_NODE_COLLECTION: RouteNodeCollection = {
  type: "FeatureCollection",
  features: [],
};

function isCoordinate(value: unknown): value is Coordinate {
  return (
    Array.isArray(value) &&
    value.length >= 2 &&
    typeof value[0] === "number" &&
    typeof value[1] === "number" &&
    Number.isFinite(value[0]) &&
    Number.isFinite(value[1])
  );
}

function normalizeRoutes(routes: MapViewportProps["routesGeoJSON"]): RouteCollection {
  if (!routes) return EMPTY_COLLECTION;
  return {
    type: "FeatureCollection",
    features: routes.features.flatMap((feature) => {
      const coordinates = feature.geometry.coordinates.filter(isCoordinate) as Coordinate[];
      if (coordinates.length < 2) return [];
      return [
        {
          type: "Feature",
          properties: {
            vehicle: Number(feature.properties?.vehicle ?? 0),
            color: String(feature.properties?.color ?? "#a1a1aa"),
            stops: Array.isArray(feature.properties?.stops)
              ? feature.properties.stops.map(String)
              : [],
          },
          geometry: { type: "LineString", coordinates },
        },
      ];
    }),
  };
}

function pointAlongLine(coordinates: Coordinate[], progress: number): Coordinate {
  if (coordinates.length === 0) return [0, 0];
  if (coordinates.length === 1) return coordinates[0];

  const lengths = coordinates.slice(1).map((point, index) => {
    const previous = coordinates[index];
    return Math.hypot(point[0] - previous[0], point[1] - previous[1]);
  });
  const total = lengths.reduce((sum, length) => sum + length, 0);
  if (total === 0) return coordinates[0];

  let distance = Math.max(0, Math.min(1, progress)) * total;
  for (let index = 0; index < lengths.length; index += 1) {
    const segmentLength = lengths[index];
    if (distance <= segmentLength || index === lengths.length - 1) {
      const ratio = segmentLength === 0 ? 0 : distance / segmentLength;
      const start = coordinates[index];
      const end = coordinates[index + 1];
      return [
        start[0] + (end[0] - start[0]) * ratio,
        start[1] + (end[1] - start[1]) * ratio,
      ];
    }
    distance -= segmentLength;
  }
  return coordinates[coordinates.length - 1];
}

function linePrefix(coordinates: Coordinate[], progress: number): Coordinate[] {
  if (coordinates.length < 2) return coordinates;
  const clampedProgress = Math.max(0, Math.min(1, progress));
  if (clampedProgress <= 0) return [coordinates[0], coordinates[0]];
  if (clampedProgress >= 1) return coordinates;

  const lengths = coordinates.slice(1).map((point, index) =>
    Math.hypot(point[0] - coordinates[index][0], point[1] - coordinates[index][1]),
  );
  const total = lengths.reduce((sum, length) => sum + length, 0);
  let distance = clampedProgress * total;
  const prefix = [coordinates[0]];

  for (let index = 0; index < lengths.length; index += 1) {
    if (distance > lengths[index]) {
      prefix.push(coordinates[index + 1]);
      distance -= lengths[index];
      continue;
    }
    const ratio = lengths[index] === 0 ? 0 : distance / lengths[index];
    prefix.push([
      coordinates[index][0] + (coordinates[index + 1][0] - coordinates[index][0]) * ratio,
      coordinates[index][1] + (coordinates[index + 1][1] - coordinates[index][1]) * ratio,
    ]);
    break;
  }
  return prefix.length >= 2 ? prefix : [coordinates[0], coordinates[0]];
}

function buildRouteNodes(
  routes: RouteCollection,
  customers: MapViewportProps["customers"],
  depot: MapViewportProps["depot"],
): RouteNodeCollection {
  const stopCoordinates = new Map<string, Coordinate>([
    [depot.id, [depot.lng, depot.lat]],
    ...customers.map((customer) => [customer.id, [customer.lng, customer.lat] as Coordinate] as const),
  ]);

  return {
    type: "FeatureCollection",
    features: routes.features.flatMap((feature) => {
      const vehicle = Number(feature.properties?.vehicle ?? 0);
      const stops = feature.properties?.stops ?? [];
      const deliveryStops = stops.filter((stopId) => stopId !== depot.id);
      if (deliveryStops.length === 0 || !vehicle) return [];

      return stops.flatMap((stopId, index) => {
        const coordinate = stopCoordinates.get(stopId);
        if (!coordinate) return [];
        const isSource = index === 0;
        const isDestination = index === stops.length - 1;
        return [{
            type: "Feature" as const,
            properties: {
              vehicle,
              kind: isSource ? "source" as const : isDestination ? "destination" as const : "intermediate" as const,
              color: isSource ? "#22c55e" : isDestination ? "#f59e0b" : "#38bdf8",
            },
            geometry: { type: "Point" as const, coordinates: coordinate },
          }];
      });
    }),
  };
}

export default function MapViewport({ routesGeoJSON, customers, depot, activeCity }: MapViewportProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const movingMarkersRef = useRef<Map<number, maplibregl.Marker>>(new Map());
  const stopMarkersRef = useRef<maplibregl.Marker[]>([]);
  const progressRef = useRef(0);
  const speedRef = useRef(1);
  const animationFrameRef = useRef<number | null>(null);
  const overlayFrameRef = useRef<number | null>(null);
  const animationStartedAtRef = useRef<number | null>(null);
  const apiKey = process.env.NEXT_PUBLIC_STADIA_API_KEY || "";

  const [mapLoaded, setMapLoaded] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [vehicleFilter, setVehicleFilter] = useState<VehicleFilter>("all");
  const [selectedVehicle, setSelectedVehicle] = useState<number | null>(null);
  const [isCompactViewport] = useState(() => typeof window !== "undefined" && window.innerWidth <= 720);
  const [mapViewVersion, setMapViewVersion] = useState(0);
  const [mapInstance, setMapInstance] = useState<maplibregl.Map | null>(null);

  const normalizedRoutes = useMemo(() => normalizeRoutes(routesGeoJSON), [routesGeoJSON]);
  const routeNodes = useMemo(
    () => buildRouteNodes(normalizedRoutes, customers, depot),
    [customers, depot, normalizedRoutes],
  );
  const vehicleIds = useMemo(
    () => normalizedRoutes.features.map((feature) => Number(feature.properties?.vehicle ?? 0)).filter(Boolean),
    [normalizedRoutes],
  );

  useEffect(() => {
    speedRef.current = speed;
  }, [speed]);

  const updateSource = useCallback((sourceId: string, data: GeoJSON.GeoJSON) => {
    const map = mapRef.current;
    if (!map) return;
    const applyData = () => {
      const source = map.getSource(sourceId) as maplibregl.GeoJSONSource | undefined;
      if (!source) return;
      source.setData(data);
      map.triggerRepaint();
    };
    if (map.isStyleLoaded()) {
      applyData();
    } else {
      map.once("styledata", applyData);
    }
  }, []);

  const getProgressRoutes = useCallback((): RouteCollection => {
    return {
      type: "FeatureCollection",
      features: normalizedRoutes.features.flatMap((feature) => {
        const vehicle = Number(feature.properties?.vehicle ?? 0);
        if (vehicleFilter !== "all" && vehicle !== vehicleFilter) return [];
        return [
          {
            ...feature,
            geometry: {
              type: "LineString",
              coordinates: linePrefix(feature.geometry.coordinates as Coordinate[], progressRef.current),
            },
          },
        ];
      }),
    };
  }, [normalizedRoutes, vehicleFilter]);

  const updateMovingMarkers = useCallback(() => {
    const map = mapRef.current;
    if (!map) return;

    const activeVehicles = new Set<number>();
    normalizedRoutes.features.forEach((feature) => {
      const vehicle = Number(feature.properties?.vehicle ?? 0);
      if (vehicleFilter !== "all" && vehicle !== vehicleFilter) return;
      activeVehicles.add(vehicle);
      const color = String(feature.properties?.color ?? "#a1a1aa");
      const coordinate = pointAlongLine(feature.geometry.coordinates as Coordinate[], progressRef.current);
      let marker = movingMarkersRef.current.get(vehicle);
      if (!marker) {
        const element = document.createElement("div");
        element.className = "route-vehicle-marker";
        element.setAttribute("aria-label", `Vehicle ${vehicle} animation position`);
        element.innerHTML = `<span class="route-vehicle-marker__pulse"></span><span class="route-vehicle-marker__dot"></span>`;
        element.addEventListener("click", () => {
          setSelectedVehicle(vehicle);
          setVehicleFilter(vehicle);
        });
        marker = new maplibregl.Marker({ element, anchor: "center" }).setLngLat(coordinate).addTo(map);
        movingMarkersRef.current.set(vehicle, marker);
      }
      marker.getElement().style.setProperty("--route-color", color);
      marker.setLngLat(coordinate);
    });

    movingMarkersRef.current.forEach((marker, vehicle) => {
      if (!activeVehicles.has(vehicle)) {
        marker.remove();
        movingMarkersRef.current.delete(vehicle);
      }
    });
  }, [normalizedRoutes, vehicleFilter]);

  const updateProgress = useCallback(
    (nextProgress: number) => {
      progressRef.current = Math.max(0, Math.min(1, nextProgress));
      setProgress(progressRef.current);
      updateSource("routes-progress", getProgressRoutes());
      updateMovingMarkers();
    },
    [getProgressRoutes, updateMovingMarkers, updateSource],
  );

  const stopAnimation = useCallback(() => {
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    animationStartedAtRef.current = null;
    setIsPlaying(false);
  }, []);

  const resetAnimation = useCallback(() => {
    stopAnimation();
    updateProgress(0);
  }, [stopAnimation, updateProgress]);

  const startAnimation = useCallback(() => {
    if (normalizedRoutes.features.length === 0) return;
    setIsPlaying(true);
    animationStartedAtRef.current = performance.now() - progressRef.current * (7000 / speedRef.current);

    const animate = (now: number) => {
      const startedAt = animationStartedAtRef.current ?? now;
      const duration = 7000 / speedRef.current;
      const nextProgress = Math.min(1, (now - startedAt) / duration);
      updateProgress(nextProgress);
      if (nextProgress >= 1) {
        animationFrameRef.current = null;
        animationStartedAtRef.current = null;
        setIsPlaying(false);
        return;
      }
      animationFrameRef.current = requestAnimationFrame(animate);
    };
    animationFrameRef.current = requestAnimationFrame(animate);
  }, [normalizedRoutes.features.length, updateProgress]);

  const focusOnRoutes = useCallback(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;

    const routeCoordinates = normalizedRoutes.features
      .filter((feature) => vehicleFilter === "all" || Number(feature.properties?.vehicle ?? 0) === vehicleFilter)
      .flatMap((feature) => feature.geometry.coordinates) as Coordinate[];
    const stopCoordinates = routeNodes.features
      .filter((feature) => vehicleFilter === "all" || Number(feature.properties?.vehicle ?? 0) === vehicleFilter)
      .map((feature) => feature.geometry.coordinates as Coordinate);
    const coordinates = [...routeCoordinates, ...stopCoordinates];
    if (coordinates.length === 0) return;

    const bounds = coordinates.reduce(
      (current, coordinate) => current.extend(coordinate),
      new maplibregl.LngLatBounds(coordinates[0], coordinates[0]),
    );

    map.stop();
    map.resize();

    // Do not let the city viewport constraint cancel a fit operation when a
    // route reaches the edge of the selected service area.
    map.setMaxBounds(null);
    const longitudeSpan = bounds.getEast() - bounds.getWest();
    const latitudeSpan = bounds.getNorth() - bounds.getSouth();
    if (longitudeSpan < 1e-7 && latitudeSpan < 1e-7) {
      map.flyTo({ center: bounds.getCenter(), zoom: 15, duration: 700 });
    } else {
      map.fitBounds(bounds, {
        padding: { top: 120, right: 120, bottom: 120, left: 120 },
        duration: 700,
        maxZoom: 14,
      });
    }
    map.once("moveend", () => map.setMaxBounds(CITY_PRESETS[activeCity].bounds));
  }, [activeCity, mapLoaded, normalizedRoutes, routeNodes, vehicleFilter]);

  const projectedRouteData = useMemo(() => {
    const map = mapInstance;
    if (!mapLoaded || !map) return { routes: [], nodes: [] };
    const projectionVersion = mapViewVersion;

    const project = (coordinate: Coordinate) => {
      const point = map.project(coordinate);
      return `${point.x},${point.y}`;
    };

    const routes = normalizedRoutes.features.flatMap((feature) => {
      const vehicle = Number(feature.properties?.vehicle ?? 0);
      if (vehicleFilter !== "all" && vehicle !== vehicleFilter) return [];
      const coordinates = feature.geometry.coordinates as Coordinate[];
      if (coordinates.length < 2) return [];
      return [{
        vehicle,
        color: String(feature.properties?.color ?? "#38bdf8"),
        basePoints: coordinates.map(project).join(" "),
        progressPoints: linePrefix(coordinates, progress).map(project).join(" "),
      }];
    });

    const nodes = routeNodes.features.flatMap((feature) => {
      const vehicle = Number(feature.properties?.vehicle ?? 0);
      if (vehicleFilter !== "all" && vehicle !== vehicleFilter) return [];
      const coordinate = feature.geometry.coordinates as Coordinate;
      const point = map.project(coordinate);
      return [{
        x: point.x,
        y: point.y,
        vehicle,
        kind: feature.properties?.kind ?? "intermediate",
        color: feature.properties?.color ?? "#38bdf8",
      }];
    });

    return { routes, nodes, projectionVersion };
  }, [mapInstance, mapLoaded, mapViewVersion, normalizedRoutes, routeNodes, vehicleFilter, progress]);

  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return;
    const targetCity = CITY_PRESETS["salt-lake"];
    const style: maplibregl.StyleSpecification = {
      version: 8,
      sources: {
        "stadia-dark": {
          type: "raster",
          tiles: [`https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}.png?api_key=${apiKey}`],
          tileSize: 256,
          attribution: "&copy; Stadia Maps &copy; OpenMapTiles &copy; OpenStreetMap contributors",
        },
      },
      layers: [{ id: "stadia-dark-layer", type: "raster", source: "stadia-dark", minzoom: 0, maxzoom: 20 }],
    };

    const map = new maplibregl.Map({
      container: mapContainer.current,
      style,
      center: targetCity.center,
      zoom: targetCity.zoom,
      minZoom: 11,
      maxBounds: targetCity.bounds,
      attributionControl: false,
      pitchWithRotate: false,
    });

    const refreshProjectedOverlay = () => {
      if (overlayFrameRef.current !== null) return;
      overlayFrameRef.current = requestAnimationFrame(() => {
        overlayFrameRef.current = null;
        setMapViewVersion((version) => version + 1);
      });
    };
    const mapViewEvents = ["move", "zoom", "resize", "rotate", "pitch"] as const;
    mapViewEvents.forEach((eventName) => map.on(eventName, refreshProjectedOverlay));

    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.addControl(new maplibregl.FullscreenControl(), "top-right");
    map.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-right");

    const addRouteLayers = () => {
      if (!map.getSource("routes-base")) {
        map.addSource("routes-base", { type: "geojson", data: EMPTY_COLLECTION });
      }
      if (!map.getSource("routes-progress")) {
        map.addSource("routes-progress", { type: "geojson", data: EMPTY_COLLECTION });
      }
      if (!map.getSource("route-nodes")) {
        map.addSource("route-nodes", { type: "geojson", data: EMPTY_NODE_COLLECTION });
      }

      const addLayerIfMissing = (layer: Parameters<typeof map.addLayer>[0]) => {
        if (!map.getLayer(layer.id)) map.addLayer(layer);
      };

      addLayerIfMissing({
        id: "routes-base-glow",
        type: "line",
        source: "routes-base",
        paint: { "line-color": ["coalesce", ["get", "color"], "#38bdf8"], "line-width": 10, "line-opacity": 0.24, "line-blur": 3 },
      });
      addLayerIfMissing({
        id: "routes-base-line",
        type: "line",
        source: "routes-base",
        layout: { "line-join": "round", "line-cap": "round" },
        paint: { "line-color": ["coalesce", ["get", "color"], "#a1a1aa"], "line-width": 3, "line-opacity": 0.68, "line-dasharray": [2, 2] },
      });
      addLayerIfMissing({
        id: "routes-progress-glow",
        type: "line",
        source: "routes-progress",
        paint: { "line-color": ["coalesce", ["get", "color"], "#f8fafc"], "line-width": 14, "line-opacity": 0.48, "line-blur": 4 },
      });
      addLayerIfMissing({
        id: "routes-progress-line",
        type: "line",
        source: "routes-progress",
        layout: { "line-join": "round", "line-cap": "round" },
        paint: { "line-color": ["coalesce", ["get", "color"], "#f8fafc"], "line-width": 6, "line-opacity": 1 },
      });
      addLayerIfMissing({
        id: "route-nodes-intermediate",
        type: "circle",
        source: "route-nodes",
        filter: ["==", ["get", "kind"], "intermediate"],
        paint: {
          "circle-color": "#38bdf8",
          "circle-radius": 5,
          "circle-opacity": 1,
          "circle-stroke-color": "#09090b",
          "circle-stroke-width": 1.25,
        },
      });
      addLayerIfMissing({
        id: "route-nodes-source",
        type: "circle",
        source: "route-nodes",
        filter: ["==", ["get", "kind"], "source"],
        paint: {
          "circle-color": "#22c55e",
          "circle-radius": 6,
          "circle-opacity": 1,
          "circle-stroke-color": "#09090b",
          "circle-stroke-width": 2,
        },
      });
      addLayerIfMissing({
        id: "route-nodes-destination",
        type: "circle",
        source: "route-nodes",
        filter: ["==", ["get", "kind"], "destination"],
        paint: {
          "circle-color": "#f59e0b",
          "circle-radius": 6,
          "circle-translate": [5, -5],
          "circle-opacity": 1,
          "circle-stroke-color": "#09090b",
          "circle-stroke-width": 2,
        },
      });
      map.on("click", "routes-base-line", (event) => {
        const vehicle = Number(event.features?.[0]?.properties?.vehicle ?? 0);
        if (vehicle) {
          setSelectedVehicle(vehicle);
          setVehicleFilter(vehicle);
        }
      });
      map.on("mouseenter", "routes-base-line", () => {
        map.getCanvas().style.cursor = "pointer";
      });
      map.on("mouseleave", "routes-base-line", () => {
        map.getCanvas().style.cursor = "";
      });
    };

    const onMapReady = () => {
      addRouteLayers();
      setMapLoaded(true);
      map.resize();
    };
    if (map.isStyleLoaded()) {
      onMapReady();
    } else {
      map.once("load", onMapReady);
    }
    mapRef.current = map;
    setMapInstance(map);
    const resizeObserver = new ResizeObserver(() => map.resize());
    resizeObserver.observe(mapContainer.current);

    const movingMarkers = movingMarkersRef.current;
    return () => {
      stopAnimation();
      movingMarkers.forEach((marker) => marker.remove());
      movingMarkers.clear();
      stopMarkersRef.current.forEach((marker) => marker.remove());
      stopMarkersRef.current = [];
      if (overlayFrameRef.current !== null) {
        cancelAnimationFrame(overlayFrameRef.current);
        overlayFrameRef.current = null;
      }
      mapViewEvents.forEach((eventName) => map.off(eventName, refreshProjectedOverlay));
      resizeObserver.disconnect();
      map.remove();
      mapRef.current = null;
      setMapInstance(null);
      setMapLoaded(false);
    };
  }, [apiKey, stopAnimation]);

  useEffect(() => {
    if (!mapRef.current) return;
    const map = mapRef.current;
    const targetCity = CITY_PRESETS[activeCity];
    map.setMaxBounds(null);
    map.flyTo({ center: targetCity.center, zoom: targetCity.zoom, duration: 1200 });
    const reapplyBounds = () => map.setMaxBounds(targetCity.bounds);
    map.once("moveend", reapplyBounds);
    return () => {
      map.off("moveend", reapplyBounds);
    };
  }, [activeCity]);

  useEffect(() => {
    if (!mapLoaded) return;
    updateSource("routes-base", normalizedRoutes);
    updateSource("route-nodes", routeNodes);
    const playbackFrame = window.requestAnimationFrame(() => {
      stopAnimation();
      updateProgress(0);
      if (normalizedRoutes.features.length > 0) {
        startAnimation();
        window.setTimeout(focusOnRoutes, 100);
      }
    });
    return () => {
      window.cancelAnimationFrame(playbackFrame);
      stopAnimation();
    };
  }, [focusOnRoutes, mapLoaded, normalizedRoutes, routeNodes, startAnimation, stopAnimation, updateProgress, updateSource]);

  useEffect(() => {
    if (!mapLoaded) return;
    const map = mapRef.current;
    if (!map) return;
    const nodeFilter = (kind: RouteNodeProperties["kind"]): maplibregl.FilterSpecification => (
      vehicleFilter === "all"
        ? ["==", ["get", "kind"], kind]
        : ["all", ["==", ["get", "kind"], kind], ["==", ["get", "vehicle"], vehicleFilter]]
    ) as maplibregl.FilterSpecification;
    const vehicleRouteFilter = vehicleFilter === "all"
      ? null
      : (["==", ["get", "vehicle"], vehicleFilter] as maplibregl.FilterSpecification);
    map.setFilter("routes-base-line", vehicleRouteFilter);
    map.setFilter("routes-base-glow", vehicleRouteFilter);
    map.setFilter("route-nodes-intermediate", nodeFilter("intermediate"));
    map.setFilter("route-nodes-source", nodeFilter("source"));
    map.setFilter("route-nodes-destination", nodeFilter("destination"));
    updateProgress(progressRef.current);
  }, [mapLoaded, updateProgress, vehicleFilter]);

  useEffect(() => {
    if (!mapLoaded) return;
    const map = mapRef.current;
    if (!map) return;
    stopMarkersRef.current.forEach((marker) => marker.remove());
    stopMarkersRef.current = [];

    const addStopMarker = (id: string, lng: number, lat: number, type: "depot" | "customer", detail: string) => {
      const element = document.createElement("button");
      element.type = "button";
      element.className = `vrp-stop-marker vrp-stop-marker--${type}`;
      element.setAttribute("aria-label", detail);
      element.title = detail;
      const popup = new maplibregl.Popup({ offset: 14, closeButton: false }).setHTML(
        `<div class="map-popup"><strong>${id}</strong><span>${detail}</span></div>`,
      );
      const marker = new maplibregl.Marker({ element, anchor: "center" })
        .setLngLat([lng, lat])
        .setPopup(popup)
        .addTo(map);
      stopMarkersRef.current.push(marker);
    };

    if (isCoordinate([depot.lng, depot.lat])) {
      addStopMarker(depot.id, depot.lng, depot.lat, "depot", "Dispatch depot");
    }
    customers.forEach((customer) => {
      if (isCoordinate([customer.lng, customer.lat])) {
        addStopMarker(customer.id, customer.lng, customer.lat, "customer", `${customer.demand ?? 0} item demand`);
      }
    });
  }, [customers, depot, mapLoaded]);

  useEffect(() => {
    return () => {
      stopMarkersRef.current.forEach((marker) => marker.remove());
      stopMarkersRef.current = [];
      if (animationFrameRef.current !== null) cancelAnimationFrame(animationFrameRef.current);
    };
  }, []);

  const selectedLabel = selectedVehicle === null ? "All vehicles" : `Vehicle ${selectedVehicle}`;

    return (
    <div className="absolute inset-0 h-full w-full overflow-hidden bg-background">
      <div ref={mapContainer} className="absolute inset-0 h-full w-full outline-none" />

      <svg
        className="pointer-events-none absolute inset-0 z-[1] h-full w-full overflow-visible"
        aria-label="Animated algorithm route overlay"
        role="img"
      >
        {projectedRouteData.routes.map((route) => (
          <g key={`svg-route-${route.vehicle}`}>
            <polyline
              points={route.basePoints}
              fill="none"
              stroke={route.color}
              strokeWidth="3"
              strokeOpacity="0.58"
              strokeDasharray="7 8"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <polyline
              points={route.progressPoints}
              fill="none"
              stroke={route.color}
              strokeWidth="7"
              strokeOpacity="1"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </g>
        ))}
        {projectedRouteData.nodes.map((node, index) => (
          <circle
            key={`svg-node-${node.vehicle}-${node.kind}-${index}`}
            cx={node.x}
            cy={node.y}
            r={node.kind === "intermediate" ? 4.5 : 7}
            fill={node.color}
            stroke="#09090b"
            strokeWidth={node.kind === "intermediate" ? 1.5 : 2}
          />
        ))}
      </svg>

      <div className="pointer-events-none absolute inset-0 z-10">
        <DraggablePanel
          label="Route playback"
          initialPosition={isCompactViewport ? { x: 8, y: 180 } : { x: 250, y: 24 }}
          className="pointer-events-auto flex w-[min(28rem,calc(100vw-1rem))] min-w-[min(14rem,calc(100vw-1rem))] max-w-[calc(100vw-1rem)] resize flex-col gap-3 overflow-auto rounded-xl border border-border/80 bg-card/90 p-3 shadow-2xl shadow-black/30 backdrop-blur-xl"
        >
          <div className="flex items-center gap-2">
            <div className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
              <Route className="size-4" />
            </div>
            <div className="min-w-0">
              <div className="truncate text-xs font-semibold tracking-tight">Live route playback</div>
              <div className="truncate text-[11px] text-muted-foreground">{selectedLabel} · algorithm path geometry</div>
            </div>
            <Badge variant="outline" className="ml-auto font-mono text-[10px]">{Math.round(progress * 100)}%</Badge>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-muted">
            <div className="h-full rounded-full bg-primary transition-[width] duration-75" style={{ width: `${progress * 100}%` }} />
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            <Button size="sm" variant="secondary" onClick={isPlaying ? stopAnimation : startAnimation} disabled={normalizedRoutes.features.length === 0}>
              {isPlaying ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
              {isPlaying ? "Pause" : "Play"}
            </Button>
            <Button size="icon-sm" variant="ghost" onClick={resetAnimation} aria-label="Reset route playback" title="Reset playback">
              <RotateCcw className="size-3.5" />
            </Button>
            <Button size="icon-sm" variant="ghost" onClick={focusOnRoutes} aria-label="Fit routes to map" title="Fit routes to map">
              <LocateFixed className="size-3.5" />
            </Button>
            <div className="ml-auto flex items-center gap-1 rounded-md border border-border bg-background/60 p-0.5">
              {[1, 2, 4].map((value) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setSpeed(value)}
                  className={`rounded px-2 py-1 text-[10px] font-mono transition-colors ${speed === value ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted hover:text-foreground"}`}
                >
                  {value}×
                </button>
              ))}
            </div>
          </div>
          <input
            aria-label="Route playback progress"
            type="range"
            min="0"
            max="1"
            step="0.001"
            value={progress}
            onChange={(event) => {
              stopAnimation();
              updateProgress(Number(event.target.value));
            }}
            className="h-1 w-full cursor-pointer accent-[var(--primary)]"
          />
          <div className="flex items-center gap-2 border-t border-border/70 pt-2">
            <SlidersHorizontal className="size-3.5 text-muted-foreground" />
            <select
              aria-label="Select route vehicle"
              value={vehicleFilter}
              onChange={(event) => setVehicleFilter(event.target.value === "all" ? "all" : Number(event.target.value))}
              className="h-8 min-w-0 flex-1 rounded-md border border-input bg-background px-2 text-xs font-medium text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <option value="all">All vehicles</option>
              {vehicleIds.map((vehicle) => <option key={vehicle} value={vehicle}>Vehicle {vehicle}</option>)}
            </select>
          </div>
        </DraggablePanel>

      </div>

      <div className="pointer-events-none absolute bottom-4 left-4 z-10 flex flex-wrap items-center gap-3 rounded-lg border border-border/70 bg-card/85 px-3 py-2 text-[10px] text-muted-foreground shadow-xl backdrop-blur-md sm:bottom-6 sm:left-6">
        <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-emerald-500" /> Source</span>
        <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-amber-500" /> Destination</span>
        <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-sky-400" /> Intermediate</span>
        <span className="font-mono">{vehicleIds.length} routes loaded</span>
      </div>
    </div>
  );
}
