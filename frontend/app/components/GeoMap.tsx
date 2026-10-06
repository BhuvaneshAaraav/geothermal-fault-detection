"use client";

import { useEffect, useRef, useState } from "react";
import {
  Map,
  Marker,
  NavigationControl,
  FullscreenControl,
  Popup,
  setWorkerUrl,
} from "maplibre-gl";

import "maplibre-gl/dist/maplibre-gl.css";
setWorkerUrl("/maplibre/maplibre-gl-worker.mjs");/* =========================================================
   TYPES
========================================================= */

type Candidate = {
  type: "Feature";

  geometry: {
    type: "Point";

    coordinates: [
      number,
      number
    ];
  };

  properties: {
    rank: number;
    row: number;
    column: number;
    score: number;
    x: number;
    y: number;
  };
};

type CandidateCollection = {
  type: "FeatureCollection";

  features: Candidate[];
};

type LayerInfo = {
  index: number;
  name: string;
  description: string;
  file: string;
};

type LayerMetadata = {
  source: string;
  crs: string;
  source_crs: string;
  width: number;
  height: number;
  resolution: [
    number,
    number
  ];

  corners: {
    topLeft: [number, number];
    topRight: [number, number];
    bottomRight: [number, number];
    bottomLeft: [number, number];
  };

  layers: LayerInfo[];
};

type SearchEventDetail = {
  latitude: number;
  longitude: number;
  radiusKm?: number;
};

type CandidateResult = {
  rank: number;
  distanceKm: number;
  score: number;
  latitude: number;
  longitude: number;
  row: number;
  column: number;
};

/* =========================================================
   CONSTANTS
========================================================= */

const DEFAULT_LATITUDE = 39.1;
const DEFAULT_LONGITUDE = -118.5;
const DEFAULT_RADIUS = 10;

const DEFAULT_LOCATION =
  "Walker River Indian Reservation, Mineral County, Nevada, USA";

/* =========================================================
   DISTANCE
========================================================= */

function distanceKm(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): number {
  const R = 6371;

  const dLat =
    ((lat2 - lat1) * Math.PI) / 180;

  const dLon =
    ((lon2 - lon1) * Math.PI) / 180;

  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(
      (lat1 * Math.PI) / 180
    ) *
      Math.cos(
        (lat2 * Math.PI) / 180
      ) *
      Math.sin(dLon / 2) ** 2;

  return (
    R *
    2 *
    Math.atan2(
      Math.sqrt(a),
      Math.sqrt(1 - a)
    )
  );
}

/* =========================================================
   DISPLAY NAME
========================================================= */

function formatLayerName(
  name: string
): string {
  const firstUnderscore =
    name.indexOf("_");

  if (firstUnderscore === -1) {
    return name;
  }

  const shortName =
    name.slice(
      0,
      firstUnderscore
    );

  let description =
    name.slice(
      firstUnderscore + 1
    );

  description =
    description.replace(
      /_/g,
      " "
    );

  description =
    description.replace(
      /\s+/g,
      " "
    );

  description =
    description
      .replace(
        /\b\w/g,
        (c) => c.toUpperCase()
      );

  return `${shortName.toUpperCase()} — ${description}`;
}

/* =========================================================
   SEARCH CIRCLE
========================================================= */

function createCircle(
  longitude: number,
  latitude: number,
  radiusKm: number
) {
  const points: [
    number,
    number
  ][] = [];

  const earthRadius = 6371;

  const angularDistance =
    radiusKm / earthRadius;

  const lat1 =
    (latitude * Math.PI) /
    180;

  const lon1 =
    (longitude * Math.PI) /
    180;

  for (
    let i = 0;
    i <= 128;
    i++
  ) {
    const bearing =
      (i / 128) *
      Math.PI *
      2;

    const lat2 =
      Math.asin(
        Math.sin(lat1) *
          Math.cos(
            angularDistance
          ) +
          Math.cos(lat1) *
            Math.sin(
              angularDistance
            ) *
            Math.cos(bearing)
      );

    const lon2 =
      lon1 +
      Math.atan2(
        Math.sin(bearing) *
          Math.sin(
            angularDistance
          ) *
          Math.cos(lat1),
        Math.cos(
          angularDistance
        ) -
          Math.sin(lat1) *
            Math.sin(lat2)
      );

    points.push([
      (lon2 * 180) /
        Math.PI,

      (lat2 * 180) /
        Math.PI,
    ]);
  }

  return {
    type: "Feature",
    geometry: {
      type: "Polygon",
      coordinates: [
        points,
      ],
    },
    properties: {},
  };
}

/* =========================================================
   COMPONENT
========================================================= */

export default function GeoMap() {
  const mapContainerRef =
    useRef<HTMLDivElement | null>(
      null
    );


  const mapRef =
    useRef<Map | null>(null);

  const mapReadyRef =
    useRef(false);

  const dataReadyRef =
    useRef(false);

  const candidatesRef =
    useRef<Candidate[]>([]);

  const metadataRef =
    useRef<LayerMetadata | null>(
      null
    );

  const searchMarkerRef =
    useRef<Marker | null>(null);

  const candidateMarkersRef =
    useRef<Marker[]>([]);

    const selectedCandidateMarkerRef =
  useRef<Marker | null>(null);

const candidateFocusMarkerRef =
  useRef<Marker | null>(null);

  const popupRef =
    useRef<Popup | null>(null);

  const pendingSearchRef =
    useRef<SearchEventDetail | null>(
      null
    );

  const [candidates, setCandidates] =
    useState<Candidate[]>([]);

    const [selectedCandidateRank, setSelectedCandidateRank] =
  useState<number | null>(null);

  const [
    metadata,
    setMetadata,
  ] =
    useState<LayerMetadata | null>(
      null
    );

  const [
    selectedLayer,
    setSelectedLayer,
  ] =
    useState("1");

  const [
    layerOpacity,
    setLayerOpacity,
  ] =
    useState(0.35);

  const [
    loading,
    setLoading,
  ] =
    useState(true);

  const [
    error,
    setError,
  ] =
    useState<string | null>(null);

  const [
    searchInfo,
    setSearchInfo,
  ] =
    useState<{
      latitude: number;
      longitude: number;
      radiusKm: number;
      locationName: string;
      nearbyCount: number;
      nearestDistance: number | null;
    } | null>(null);


  /* =======================================================
     3D SUBSURFACE VIEW
  ======================================================= */

  const [
    subsurfaceCandidate,
    setSubsurfaceCandidate,
  ] = useState<{
    latitude: number;
    longitude: number;
    rank?: number;
    score?: number;
  } | null>(null);

  const [
    subsurfaceOpen,
    setSubsurfaceOpen,
  ] = useState(false);

  const [
    subsurfaceFullscreen,
    setSubsurfaceFullscreen,
  ] = useState(false);

  function openSubsurfaceView(
    latitude: number,
    longitude: number,
    rank?: number,
    score?: number
  ) {
    setSubsurfaceCandidate({
      latitude,
      longitude,
      rank,
      score,
    });

    setSubsurfaceFullscreen(false);
    setSubsurfaceOpen(true);
  }

  function closeSubsurfaceView() {
    setSubsurfaceFullscreen(false);
    setSubsurfaceOpen(false);
  }

  useEffect(() => {
    if (!subsurfaceFullscreen) return;

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setSubsurfaceFullscreen(false);
      }
    };

    window.addEventListener("keydown", handleEscape);

    return () => {
      window.removeEventListener("keydown", handleEscape);
    };
  }, [subsurfaceFullscreen]);

  function buildSubsurfaceProfile(
    candidate: {
      latitude: number;
      longitude: number;
      rank?: number;
      score?: number;
    }
  ) {
    const seedSource =
      Math.abs(candidate.latitude * 1000003) +
      Math.abs(candidate.longitude * 9176.37) +
      Math.abs((candidate.rank ?? 0) * 31.17) +
      Math.abs((candidate.score ?? 0) * 10000);

    const frac = (n: number) => {
      const x = Math.sin(seedSource + n * 12.9898) * 43758.5453;
      return x - Math.floor(x);
    };

    const surfaceX = 250 + frac(1) * 430;
    const surfaceY = 185 + frac(2) * 48;
    const dipDeg = 25 + frac(3) * 35;
    const strikeDeg = Math.round(frac(4) * 180);
    const reservoirDepthKm = 2.0 + frac(5) * 2.5;
    const faultWidth = 105 + frac(6) * 60;
    const faultBottomX =
      Math.max(250, Math.min(740, surfaceX + (frac(7) - 0.45) * 250));
    const reservoirX = Math.max(360, Math.min(760, faultBottomX + (frac(8) - 0.5) * 130));
    const reservoirY = 470 + frac(9) * 45;

    const topFaultY = surfaceY + 30;
    const bottomFaultY = 590;

    const topLeft = surfaceX - faultWidth / 2;
    const topRight = surfaceX + faultWidth / 2;
    const bottomLeft = faultBottomX - faultWidth * 0.72;
    const bottomRight = faultBottomX + faultWidth * 0.72;

    const flowXs = [0.18, 0.38, 0.58, 0.78].map(
      (t) =>
        surfaceX +
        (faultBottomX - surfaceX) * t +
        (frac(20 + t * 10) - 0.5) * 18
    );

    const flowYs = [0.22, 0.42, 0.62, 0.82].map(
      (t) => topFaultY + (bottomFaultY - topFaultY) * t
    );

    const layerJitter = [
      0,
      -8 + frac(30) * 16,
      -5 + frac(31) * 18,
      -7 + frac(32) * 20,
      -4 + frac(33) * 16,
    ];

    return {
      surfaceX,
      surfaceY,
      dipDeg,
      strikeDeg,
      reservoirDepthKm,
      reservoirX,
      reservoirY,
      topLeft,
      topRight,
      bottomLeft,
      faultBottomX,
      bottomRight,
      topFaultY,
      bottomFaultY,
      flowXs,
      flowYs,
      layerJitter,
    };
  }


/* =====================================================
   3D TERRAIN CONTROLS
===================================================== */

const enable3DTerrain = () => {
  const map = mapRef.current;

  if (!map || !mapReadyRef.current) {
    return;
  }

  try {
    // Add DEM source only once
    if (!map.getSource("geodawn-terrain")) {
      map.addSource("geodawn-terrain", {
        type: "raster-dem",
        url: "https://tiles.mapterhorn.com/tilejson.json",
        tileSize: 256,
      });
    }

    // Turn on real terrain
    map.setTerrain({
      source: "geodawn-terrain",
      exaggeration: 1.5,
    });

    // Tilt the live map
    map.easeTo({
      pitch: 60,
      bearing: 15,
      duration: 1200,
      essential: true,
    });

  } catch (error) {
    console.error(
      "GeoDAWN 3D terrain error:",
      error
    );
  }
};


const disable3DTerrain = () => {
  const map = mapRef.current;

  if (!map || !mapReadyRef.current) {
    return;
  }

  try {
    map.setTerrain(null);

    map.easeTo({
      pitch: 0,
      bearing: 0,
      duration: 1000,
      essential: true,
    });

  } catch (error) {
    console.error(
      "GeoDAWN 2D terrain error:",
      error
    );
  }
};

    
    /* =========================================================

     GEOPHYSICAL LAYER SELECTION

  ========================================================= */

  useEffect(() => {

    function handleLayerSelect(

      event: Event

    ) {

      const customEvent =

        event as CustomEvent<{

          layerIndex: number;

        }>;

      const layerIndex =

        customEvent.detail?.layerIndex;

      if (

        !Number.isInteger(layerIndex) ||

        layerIndex < 1 ||

        layerIndex > 19

      ) {

        return;

      }

      console.log(

        "🗺️ GeoDAWN layer selected:",

        layerIndex

      );

      setSelectedLayer(

        String(layerIndex)

      );

    }

    window.addEventListener(

      "geodawn-layer-select",

      handleLayerSelect

    );

    return () => {

      window.removeEventListener(

        "geodawn-layer-select",

        handleLayerSelect

      );

    };

  }, []);

  
  /* =======================================================
     CLEAR MARKERS
  ======================================================= */

  function clearCandidateMarkers() {
    candidateMarkersRef.current.forEach(
      (marker) => marker.remove()
    );

    candidateMarkersRef.current = [];
  }

  /* =======================================================
     CLEAR RADIUS
  ======================================================= */

  function clearRadius() {
    const map = mapRef.current;

    if (!map) return;

    if (
      map.getLayer(
        "geodawn-radius-fill"
      )
    ) {
      map.removeLayer(
        "geodawn-radius-fill"
      );
    }

    if (
      map.getLayer(
        "geodawn-radius-line"
      )
    ) {
      map.removeLayer(
        "geodawn-radius-line"
      );
    }

    if (
      map.getSource(
        "geodawn-radius"
      )
    ) {
      map.removeSource(
        "geodawn-radius"
      );
    }
  }

  /* =======================================================
     DRAW RADIUS
  ======================================================= */

 function drawRadius(
  latitude: number,
  longitude: number,
  radiusKm: number
) {
  const map = mapRef.current;

  if (!map) return;

  clearRadius();

  const circle = createCircle(
    longitude,
    latitude,
    radiusKm
  );

  map.addSource("geodawn-radius", {
    type: "geojson",
    data: circle as any,
  });

  /*
   * Analysis area
   */
  map.addLayer({
    id: "geodawn-radius-fill",
    type: "fill",
    source: "geodawn-radius",
    paint: {
      "fill-color": "#00bfff",
      "fill-opacity": 0.08,
    },
  });

  /*
   * 10 km boundary
   */
  map.addLayer({
    id: "geodawn-radius-line",
    type: "line",
    source: "geodawn-radius",
    paint: {
      "line-color": "#00e5ff",
      "line-width": 4,
      "line-opacity": 1,
    },
  });

  /*
   * Make absolutely sure the radius is above
   * the satellite and geophysical raster layers.
   */
  try {
    map.moveLayer("geodawn-radius-fill");
    map.moveLayer("geodawn-radius-line");
  } catch (error) {
    console.warn(
      "Could not move radius layers:",
      error
    );
  }

  console.log(
    `⭕ GeoDAWN analysis radius: ${radiusKm} km`
  );
}
 /* =======================================================
   SEARCH MARKER
======================================================= */

function setSearchMarker(
  latitude: number,
  longitude: number
) {
  const map = mapRef.current;

  if (!map) return;

  searchMarkerRef.current?.remove();

  searchMarkerRef.current =
    new Marker({
      color: "#2563eb",
    })
      .setLngLat([
        longitude,
        latitude,
      ])
      .addTo(map);
}

  /* =======================================================
     CANDIDATE POPUP
  ======================================================= */
function openCandidatePopup(
  candidate: Candidate,
  searchedLatitude: number,
  searchedLongitude: number
) {
  const map = mapRef.current;

  if (!map) return;

  const [longitude, latitude] =
    candidate.geometry.coordinates;

  const distance = distanceKm(
    searchedLatitude,
    searchedLongitude,
    latitude,
    longitude
  );

  /*
   * If the same candidate popup is already open,
   * clicking it again closes the popup.
   */
  const existingPopup = popupRef.current;

  if (existingPopup?.isOpen()) {
    const existingLocation =
      existingPopup.getLngLat();

    if (
      Math.abs(existingLocation.lng - longitude) < 0.000001 &&
      Math.abs(existingLocation.lat - latitude) < 0.000001
    ) {
      existingPopup.remove();
      popupRef.current = null;
      return;
    }

    /*
     * A different candidate was clicked.
     * Close the previous popup first.
     */
    existingPopup.remove();
    popupRef.current = null;
  }

  popupRef.current = new Popup({
    closeButton: true,
    closeOnClick: true,
    maxWidth: "350px",
    offset: 12,
  })
    .setLngLat([
      longitude,
      latitude,
    ])
    .setHTML(`
      <div style="
        font-family:
          system-ui,
          -apple-system,
          BlinkMacSystemFont,
          sans-serif;
        color:#111827;
        min-width:270px;
      ">

        <div style="
          font-size:16px;
          font-weight:700;
          margin-bottom:12px;
        ">
          GeoDAWN Fault Candidate
        </div>

        <div style="
          display:inline-block;
          padding:4px 8px;
          margin-bottom:10px;
          border-radius:6px;
          background:#fee2e2;
          color:#b91c1c;
          font-size:11px;
          font-weight:700;
        ">
          AI-GENERATED CANDIDATE
        </div>

        <div style="
          font-size:13px;
          line-height:1.65;
        ">

          <div>
            <strong>Rank:</strong>
            #${candidate.properties.rank}
          </div>

          <div style="
            margin-top:5px;
          ">
            <strong>Candidate score:</strong>
            ${candidate.properties.score.toFixed(6)}
          </div>

          <div style="
            margin-top:5px;
          ">
            <strong>Distance from search:</strong>
            ${distance.toFixed(2)} km
          </div>

          <div style="
            margin-top:8px;
            padding-top:8px;
            border-top:1px solid #e5e7eb;
          ">
            <strong>Candidate location</strong>
          </div>

          <div>
            <strong>Latitude:</strong>
            ${latitude.toFixed(6)}
          </div>

          <div>
            <strong>Longitude:</strong>
            ${longitude.toFixed(6)}
          </div>

          <div style="
            margin-top:8px;
            padding-top:8px;
            border-top:1px solid #e5e7eb;
          ">
            <strong>Raster position</strong>
          </div>

          <div>
            <strong>Row:</strong>
            ${candidate.properties.row}
          </div>

          <div>
            <strong>Column:</strong>
            ${candidate.properties.column}
          </div>

        </div>

        <div style="
          margin-top:12px;
          padding-top:9px;
          border-top:1px solid #e5e7eb;
          color:#64748b;
          font-size:11px;
          line-height:1.5;
        ">
          This is an AI-generated geological
          fault candidate and is not a confirmed
          geological fault.
        </div>

      </div>
    `)
    .addTo(map);
}

  /* =======================================================
     DRAW CANDIDATES
  ======================================================= */

 function drawCandidates(
  nearby: {
    candidate: Candidate;
    distance: number;
  }[],
  searchedLatitude: number,
  searchedLongitude: number
) {
  clearCandidateMarkers();

  /*
   * Show only the closest candidates.
   * The full 84,500-candidate dataset remains untouched.
   */
  const visible = nearby.slice(0, 150);

  visible.forEach(({ candidate, distance }) => {
    const [longitude, latitude] =
      candidate.geometry.coordinates;

    const element = document.createElement("div");

    element.style.width = "8px";
    element.style.height = "8px";
    element.style.borderRadius = "50%";
    element.style.background = "#ef4444";
    element.style.border = "1px solid rgba(255,255,255,0.9)";
    element.style.boxShadow =
      "0 1px 4px rgba(0,0,0,0.45)";
    element.style.cursor = "pointer";

    element.title =
      `Candidate #${candidate.properties?.rank ?? "—"} • ` +
      `${distance.toFixed(2)} km`;

    element.addEventListener("click", (event) => {
      event.stopPropagation();

      openSubsurfaceView(
        latitude,
        longitude,
        candidate.properties.rank,
        candidate.properties.score
      );

      openCandidatePopup(
        candidate,
        searchedLatitude,
        searchedLongitude
      );
    });

    const marker = new Marker({
      element,
    })
      .setLngLat([
        longitude,
        latitude,
      ])
      .addTo(mapRef.current!);

    candidateMarkersRef.current.push(marker);
  });
}
  /* =======================================================
     LOCATION NAME
  ======================================================= */

  async function getLocationName(
  latitude: number,
  longitude: number
): Promise<string> {
  /*
   * Reverse-geocode the searched coordinate.
   *
   * This means the displayed location is based on
   * the actual coordinate instead of being hardcoded.
   */

  try {
    const url =
      `https://nominatim.openstreetmap.org/reverse` +
      `?format=jsonv2` +
      `&lat=${encodeURIComponent(latitude)}` +
      `&lon=${encodeURIComponent(longitude)}` +
      `&zoom=10` +
      `&addressdetails=1`;

    const response = await fetch(url, {
      headers: {
        Accept: "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(
        `Reverse geocoding failed: ${response.status}`
      );
    }

    const data = await response.json();

    const address = data.address ?? {};

    const parts = [
      address.tourism,
      address.amenity,
      address.village,
      address.town,
      address.city,
      address.county,
      address.state,
      address.country,
    ].filter(Boolean);

    if (parts.length > 0) {
      return parts.join(", ");
    }

    if (data.display_name) {
      return data.display_name;
    }

    return `${latitude.toFixed(6)}, ${longitude.toFixed(6)}`;
  } catch (error) {
    console.warn(
      "⚠️ Reverse geocoding failed:",
      error
    );

    return `${latitude.toFixed(6)}, ${longitude.toFixed(6)}`;
  }
}
  /* =======================================================
     PERFORM SEARCH
  ======================================================= */

  async function performSearch(
    latitude: number,
    longitude: number,
    radiusKm: number
  ) {
    const map = mapRef.current;

    if (
      !map ||
      !mapReadyRef.current ||
      !dataReadyRef.current
    ) {
      /*
       * Store it and execute when both
       * map and data are ready.
       */

      pendingSearchRef.current = {
        latitude,
        longitude,
        radiusKm,
      };

      return;
    }

    console.log(
      "🔎 Searching:",
      latitude,
      longitude,
      `${radiusKm} km`
    );

    setSearchMarker(
      latitude,
      longitude
    );

    drawRadius(
      latitude,
      longitude,
      radiusKm
    );

    const nearby =
      candidatesRef.current
        .map((candidate) => {
          const [
            candidateLongitude,
            candidateLatitude,
          ] =
            candidate.geometry
              .coordinates;

          return {
            candidate,

            distance:
              distanceKm(
                latitude,
                longitude,
                candidateLatitude,
                candidateLongitude
              ),
          };
        })
        .filter(
          (item) =>
            item.distance <=
            radiusKm
        )
        .sort(
          (a, b) =>
            a.distance -
            b.distance
        );

    drawCandidates(
      nearby,
      latitude,
      longitude
    );

    const nearestDistance =
      nearby.length > 0
        ? nearby[0].distance
        : null;

    const locationName =
      await getLocationName(
        latitude,
        longitude
      );

    setSearchInfo({
      latitude,
      longitude,
      radiusKm,
      locationName,
      nearbyCount:
        nearby.length,
      nearestDistance,
    });

    const results: CandidateResult[] =
      nearby.map(
        (item) => ({
          rank:
            item.candidate
              .properties
              .rank,

          distanceKm:
            item.distance,

          score:
            item.candidate
              .properties
              .score,

          latitude:
            item.candidate
              .geometry
              .coordinates[1],

          longitude:
            item.candidate
              .geometry
              .coordinates[0],

          row:
            item.candidate
              .properties
              .row,

          column:
            item.candidate
              .properties
              .column,
        })
      );

    window.dispatchEvent(
      new CustomEvent(
        "geodawn-search-results",
        {
          detail: {
            latitude,
            longitude,
            radiusKm,
            count:
              nearby.length,
            nearestDistanceKm:
              nearestDistance,
            locationName,
          },
        }
      )
    );

    window.dispatchEvent(
      new CustomEvent(
        "geodawn-candidates-results",
        {
          detail:
            results,
        }
      )
    );

    map.flyTo({
      center: [
        longitude,
        latitude,
      ],

      zoom:
        radiusKm <= 1
          ? 13
          : radiusKm <= 5
          ? 11.5
          : radiusKm <= 10
          ? 10
          : radiusKm <= 25
          ? 9
          : 8,

      duration: 1000,
    });
  }

  /* =======================================================
     LOAD DATA
  ======================================================= */

  useEffect(() => {
    let cancelled = false;

    async function loadData() {
      try {
        const [
          candidateResponse,
          layerResponse,
        ] =
          await Promise.all([
            fetch(
              "/data/V1_19_candidates.geojson",
              {
                cache: "no-store",
              }
            ),

            fetch(
              "/layers/layers.json",
              {
                cache: "no-store",
              }
            ),
          ]);

        if (
          !candidateResponse.ok
        ) {
          throw new Error(
            `Candidate data failed: HTTP ${candidateResponse.status}`
          );
        }

        if (
          !layerResponse.ok
        ) {
          throw new Error(
            `Layer metadata failed: HTTP ${layerResponse.status}`
          );
        }

        const candidateData =
          (await candidateResponse.json()) as CandidateCollection;

        const layerData =
          (await layerResponse.json()) as LayerMetadata;

        if (cancelled) return;

        /*
         * Store candidates.
         */

        candidatesRef.current =
          candidateData.features;

        setCandidates(
          candidateData.features
        );

        /*
         * Store metadata.
         */

        metadataRef.current =
          layerData;

        setMetadata(
          layerData
        );

        if (
          layerData.layers.length
        ) {
          setSelectedLayer(
            String(
              layerData.layers[0].index
            )
          );
        }

        dataReadyRef.current =
          true;

        console.log(
          `✅ Loaded ${candidateData.features.length.toLocaleString()} candidates`
        );

        console.log(
          `✅ Loaded ${layerData.layers.length} geophysical layers`
        );

        setLoading(false);

        /*
         * Execute pending search if
         * map is already ready.
         */

        if (
          mapReadyRef.current &&
          pendingSearchRef.current
        ) {
          const search =
            pendingSearchRef.current;

          pendingSearchRef.current =
            null;

          void performSearch(
            search.latitude,
            search.longitude,
            search.radiusKm ??
              DEFAULT_RADIUS
          );
        }
      } catch (err) {
        console.error(
          "❌ GeoDAWN data loading error:",
          err
        );

        if (!cancelled) {
          setError(
            err instanceof Error
              ? err.message
              : "Unable to load GeoDAWN data."
          );

          setLoading(false);
        }
      }
    }

    void loadData();

    return () => {
      cancelled = true;
    };
  }, []);

  /* =======================================================
     CREATE MAP
  ======================================================= */

  useEffect(() => {
    if (
      !mapContainerRef.current
    ) {
      return;
    }

    if (mapRef.current) {
      return;
    }

    /*
     * Simple MapLibre style.
     *
     * This avoids external style JSON problems.
     */

    const style = {
      version: 8,

      sources: {
        osm: {
          type: "raster",

          tiles: [
            "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
          ],

          tileSize: 256,

          attribution:
            "© OpenStreetMap contributors",
        },
      },

      layers: [
        {
          id: "osm",

          type: "raster",

          source: "osm",
        },
      ],
    };

    const map = new Map({
  container: mapContainerRef.current,

  style: {
  version: 8,

  sources: {
    satellite: {
      type: "raster",
      tiles: [
        "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg",
      ],
      tileSize: 256,
      attribution: "Sentinel-2 cloudless imagery",
    },
  },

  layers: [
    {
      id: "satellite",
      type: "raster",
      source: "satellite",
    },
  ],
},

  center: [
    -118.5,
    39.1,
  ],

  zoom: 7,

  minZoom: 3,

  maxZoom: 18,

  attributionControl: true,
});

mapRef.current = map;

map.addControl(
  new NavigationControl(),
  "top-right"
);

map.addControl(
  new FullscreenControl(),
  "top-right"
);

map.on("load", () => {
  console.log("✅ GeoDAWN map loaded");
  mapReadyRef.current = true;

  if (
    dataReadyRef.current &&
    pendingSearchRef.current
  ) {
    const search =
      pendingSearchRef.current;

    pendingSearchRef.current = null;

    void performSearch(
      search.latitude,
      search.longitude,
      search.radiusKm ??
        DEFAULT_RADIUS
    );
  }
});

map.on("error", (event) => {
  console.error("❌ MapLibre error:", event);

  if (event?.error) {
    console.error(
      "MapLibre error message:",
      event.error.message
    );

    console.error(
      "MapLibre error stack:",
      event.error.stack
    );
  }
});

    return () => {
      mapReadyRef.current =
        false;

      clearCandidateMarkers();

      searchMarkerRef.current?.remove();

      popupRef.current?.remove();

      map.remove();

      mapRef.current =
        null;
    };
  }, []);

  /* =======================================================
     SEARCH EVENT
  ======================================================= */

  useEffect(() => {
    const handleSearch = (
      event: Event
    ) => {
      const custom =
        event as CustomEvent<SearchEventDetail>;

      const detail =
        custom.detail;

      if (!detail) return;

      const latitude =
        Number(
          detail.latitude
        );

      const longitude =
        Number(
          detail.longitude
        );

      const radiusKm =
        Number(
          detail.radiusKm ??
            DEFAULT_RADIUS
        );

      if (
        !Number.isFinite(
          latitude
        ) ||
        !Number.isFinite(
          longitude
        ) ||
        !Number.isFinite(
          radiusKm
        )
      ) {
        return;
      }

      void performSearch(
        latitude,
        longitude,
        radiusKm
      );
    };

    window.addEventListener(
      "geodawn-location-search",
      handleSearch
    );

    return () => {
      window.removeEventListener(
        "geodawn-location-search",
        handleSearch
      );
    };
  }, []);


useEffect(() => {
  const handleCandidateFocus = (
    event: Event
  ) => {
    const custom =
      event as CustomEvent<{
        latitude: number;
        longitude: number;
        rank?: number;
        score?: number;
      }>;

    const detail = custom.detail;

    if (!detail) return;

    const latitude = Number(detail.latitude);
    const longitude = Number(detail.longitude);

    if (
      !Number.isFinite(latitude) ||
      !Number.isFinite(longitude)
    ) {
      return;
    }

    const map = mapRef.current;

    if (!map || !mapReadyRef.current) {
      return;
    }

   enable3DTerrain();

map.flyTo({
  center: [longitude, latitude],
  zoom: 11,
  pitch: 60,
  bearing: 15,
  duration: 1600,
  essential: true,
});

    /*
     * Remove previous focus marker
     */
    if (candidateFocusMarkerRef.current) {
      candidateFocusMarkerRef.current.remove();
      candidateFocusMarkerRef.current = null;
    }

    /*
     * Remove any existing popup.
     * This is important because both:
     *  - red candidate markers
     *  - View on map
     * now share popupRef.
     */
    popupRef.current?.remove();
    popupRef.current = null;

    /*
     * Create highlighted candidate marker
     */
    const markerElement =
      document.createElement("div");

    markerElement.style.width = "18px";
    markerElement.style.height = "18px";
    markerElement.style.borderRadius = "50%";
    markerElement.style.background = "#ef4444";
    markerElement.style.border = "3px solid white";
    markerElement.style.boxShadow =
      "0 0 0 4px rgba(239,68,68,0.25), 0 4px 12px rgba(0,0,0,0.35)";

    const marker = new Marker({
      element: markerElement,
    })
      .setLngLat([longitude, latitude])
      .addTo(map);

    candidateFocusMarkerRef.current = marker;

    openSubsurfaceView(
      latitude,
      longitude,
      detail.rank,
      detail.score
    );

    const rank = detail.rank;
    const score = detail.score;

    /*
     * Same popup system used by normal candidate markers.
     */
    const popupContent = `
      <div style="
        font-family:
          system-ui,
          -apple-system,
          BlinkMacSystemFont,
          sans-serif;
        color:#111827;
        min-width:270px;
      ">

        <div style="
          font-size:16px;
          font-weight:700;
          margin-bottom:12px;
        ">
          GeoDAWN Fault Candidate
        </div>

        <div style="
          display:inline-block;
          padding:4px 8px;
          margin-bottom:10px;
          border-radius:6px;
          background:#fee2e2;
          color:#b91c1c;
          font-size:11px;
          font-weight:700;
        ">
          AI-GENERATED CANDIDATE
        </div>

        <div style="
          font-size:13px;
          line-height:1.65;
        ">

          ${
            rank !== undefined
              ? `
                <div>
                  <strong>Rank:</strong>
                  #${rank}
                </div>
              `
              : ""
          }

          ${
            score !== undefined
              ? `
                <div style="margin-top:5px;">
                  <strong>Candidate score:</strong>
                  ${score.toFixed(6)}
                </div>
              `
              : ""
          }

          <div style="
            margin-top:8px;
            padding-top:8px;
            border-top:1px solid #e5e7eb;
          ">
            <strong>Candidate location</strong>
          </div>

          <div>
            <strong>Latitude:</strong>
            ${latitude.toFixed(6)}
          </div>

          <div>
            <strong>Longitude:</strong>
            ${longitude.toFixed(6)}
          </div>

        </div>

        <div style="
          margin-top:12px;
          padding-top:9px;
          border-top:1px solid #e5e7eb;
          color:#64748b;
          font-size:11px;
          line-height:1.5;
        ">
          This is an AI-generated geological
          fault candidate and is not a confirmed
          geological fault.
        </div>

      </div>
    `;

    /*
     * IMPORTANT:
     * Store this popup in popupRef.
     */
    popupRef.current = new Popup({
      closeButton: true,
      closeOnClick: true,
      offset: 12,
      maxWidth: "350px",
    })
      .setLngLat([longitude, latitude])
      .setHTML(popupContent)
      .addTo(map);
  };

  window.addEventListener(
    "geodawn-candidate-focus",
    handleCandidateFocus
  );

  return () => {
    window.removeEventListener(
      "geodawn-candidate-focus",
      handleCandidateFocus
    );
  };
}, []);
  /* =======================================================
     TABLE → MAP
  ======================================================= */

  useEffect(() => {
    const handleInspect = (
      event: Event
    ) => {
      const custom =
        event as CustomEvent<{
          latitude: number;
          longitude: number;
        }>;

      const {
        latitude,
        longitude,
      } =
        custom.detail;

      const map =
        mapRef.current;

      if (!map) return;

      map.flyTo({
        center: [
          longitude,
          latitude,
        ],

        zoom: 13,

        duration: 1000,
      });

      const candidate =
        candidatesRef.current.find(
          (item) => {
            const [
              itemLongitude,
              itemLatitude,
            ] =
              item.geometry
                .coordinates;

            return (
              Math.abs(
                itemLatitude -
                  latitude
              ) < 0.000001 &&
              Math.abs(
                itemLongitude -
                  longitude
              ) < 0.000001
            );
          }
        );

      if (candidate) {
        openSubsurfaceView(
          latitude,
          longitude,
          candidate.properties.rank,
          candidate.properties.score
        );

        openCandidatePopup(
          candidate,

          searchInfo?.latitude ??
            latitude,

          searchInfo?.longitude ??
            longitude
        );
      } else {
        openSubsurfaceView(
          latitude,
          longitude
        );
      }
    };

    window.addEventListener(
      "geodawn-inspect-candidate",
      handleInspect
    );

    return () => {
      window.removeEventListener(
        "geodawn-inspect-candidate",
        handleInspect
      );
    };
  }, [
    searchInfo,
  ]);

  /* =======================================================
     GEOPHYSICAL RASTER
  ======================================================= */

 /* =========================================================
   GEOPHYSICAL RASTER OVERLAY
========================================================= */
/* =========================================================
   GEOPHYSICAL RASTER OVERLAY
========================================================= */

useEffect(() => {
  const map = mapRef.current;

  if (
    !map ||
    !metadata ||
    !selectedLayer
  ) {
    return;
  }

  const selected =
    metadata.layers.find(
      (layer) =>
        String(layer.index) ===
        String(selectedLayer)
    );

  if (!selected) {
    console.error(
      "❌ GeoDAWN layer not found:",
      selectedLayer
    );
    return;
  }

  const sourceId =
    "geodawn-geophysical-source";

  const rasterId =
    "geodawn-geophysical-raster";

  const removeRaster = () => {
    if (map.getLayer(rasterId)) {
      map.removeLayer(rasterId);
    }

    if (map.getSource(sourceId)) {
      map.removeSource(sourceId);
    }
  };

  const installRaster = () => {
    if (!map.isStyleLoaded()) {
      return;
    }

    removeRaster();

    const {
      topLeft,
      topRight,
      bottomRight,
      bottomLeft,
    } = metadata.corners;

    const imageUrl =
      `/layers/${encodeURIComponent(
        selected.file
      )}`;

    console.log(
      "🗺️ Loading GeoDAWN raster:",
      {
        band: selected.index,
        name: selected.name,
        file: selected.file,
        url: imageUrl,
        coordinates: [
          topLeft,
          topRight,
          bottomRight,
          bottomLeft,
        ],
      }
    );

    map.addSource(
      sourceId,
      {
        type: "image",
        url: imageUrl,
        coordinates: [
          topLeft,
          topRight,
          bottomRight,
          bottomLeft,
        ],
      }
    );

    map.addLayer({
      id: rasterId,
      type: "raster",
      source: sourceId,

      paint: {
        /*
         * Strong opacity temporarily so we can
         * clearly verify the raster is working.
         */
        "raster-opacity": 0.75,

        "raster-fade-duration": 0,

        "raster-resampling": "nearest",
      },
    });

    /*
     * Make absolutely sure the geophysical layer
     * is above the satellite layer.
     */
    try {
      map.moveLayer(rasterId);
    } catch {
      // Layer is already at the top.
    }

    console.log(
      "✅ GeoDAWN raster added:",
      `Band ${selected.index}`
    );
  };

  if (map.isStyleLoaded()) {
    installRaster();
  } else {
    map.once(
      "load",
      installRaster
    );
  }

  return () => {
    removeRaster();
  };
}, [
  metadata,
  selectedLayer,
]);

    /* =======================================================
       RENDER
    ======================================================= */

    return (
      <div className="relative w-full">

        {/* =====================================================
            LIVE MAP
        ===================================================== */}

        <div
          ref={mapContainerRef}
          className="relative h-[560px] w-full overflow-hidden rounded-xl border border-white/10 bg-[#07111f]"
        />

        {/* =====================================================
            3D SUBSURFACE / 2D CONTROLS
        ===================================================== */}

        <div className="absolute right-4 top-4 z-20 flex flex-col gap-2">

          <button
            type="button"
            onClick={() => {
              if (subsurfaceCandidate) {
                setSubsurfaceOpen(true);
              }
            }}
            disabled={!subsurfaceCandidate}
            className={`rounded-lg border px-3 py-2 text-xs font-semibold shadow-lg backdrop-blur-md transition ${
              subsurfaceCandidate
                ? "border-white/20 bg-[#07111f]/90 text-white hover:bg-[#10223a]"
                : "cursor-not-allowed border-white/10 bg-[#07111f]/60 text-slate-500"
            }`}
          >
            3D Subsurface
          </button>

          <button
            type="button"
            onClick={closeSubsurfaceView}
            className="rounded-lg border border-white/20 bg-white/95 px-3 py-2 text-xs font-semibold text-[#172554] shadow-lg backdrop-blur-md transition hover:bg-white"
          >
            2D Map
          </button>

        </div>

        {/* =====================================================
            ACTIVE GEOPHYSICAL LAYER
        ===================================================== */}

        {!loading && metadata && (() => {

          const activeLayer =
            metadata.layers.find(
              (layer) =>
                String(layer.index) ===
                String(selectedLayer)
            );

          if (!activeLayer) {
            return null;
          }

          const category =
            activeLayer.index === 10 ||
            activeLayer.index === 16
              ? "Seismic"
              : activeLayer.index === 17
              ? "Electrical"
              : activeLayer.index === 15
              ? "Geology"
              : activeLayer.index === 12 ||
                activeLayer.index === 19
              ? "Topography"
              : activeLayer.index === 4 ||
                activeLayer.index === 7 ||
                activeLayer.index === 8
              ? "Geodetic"
              : activeLayer.index === 5 ||
                activeLayer.index === 11 ||
                activeLayer.index === 13 ||
                activeLayer.index === 18
              ? "Gravity"
              : "Magnetic";

          const cleanName =
            activeLayer.name
              .split("_")
              .slice(1)
              .join(" ")
              .replace(/_/g, " ")
              .replace(/\s+/g, " ")
              .trim();

          return (
            <div className="absolute left-4 top-4 z-10 w-[320px] rounded-xl border border-white/70 bg-white/95 p-4 shadow-xl backdrop-blur-md">

              <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-500">
                Geophysical Layer
              </div>

              <div className="mt-2 flex items-start gap-3">

                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[#eef2ff] text-xs font-bold text-[#3157d5]">
                  {String(
                    activeLayer.index
                  ).padStart(2, "0")}
                </div>

                <div className="min-w-0">

                  <div className="text-sm font-bold text-[#10245c]">
                    Band {activeLayer.index}
                  </div>

                  <div className="mt-0.5 text-sm font-semibold leading-5 text-slate-700">
                    {cleanName}
                  </div>

                </div>

              </div>

              <div className="mt-4 rounded-xl border border-[#dbe5ff] bg-[#f5f7ff] p-4">

                <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-[#3157d5]">
                  What this layer represents
                </div>

                <p className="mt-2 text-sm font-semibold leading-6 text-[#172554]">
                  {activeLayer.description}
                </p>

              </div>

              <div className="mt-3 grid grid-cols-2 gap-2">

                <div className="rounded-lg bg-slate-50 px-3 py-2">

                  <div className="text-[9px] font-bold uppercase tracking-wider text-slate-400">
                    Category
                  </div>

                  <div className="mt-0.5 text-xs font-semibold text-slate-700">
                    {category}
                  </div>

                </div>

                <div className="rounded-lg bg-slate-50 px-3 py-2">

                  <div className="text-[9px] font-bold uppercase tracking-wider text-slate-400">
                    Resolution
                  </div>

                  <div className="mt-0.5 text-xs font-semibold text-slate-700">
                    100 m
                  </div>

                </div>

              </div>

            </div>
          );

        })()}

        {/* =====================================================
            LOADING
        ===================================================== */}

        {loading && (
          <div className="absolute inset-0 z-20 flex items-center justify-center rounded-xl bg-[#07111f]/90 backdrop-blur-sm">

            <div className="text-center">

              <div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-2 border-slate-700 border-t-blue-500" />

              <p className="text-sm font-medium text-slate-300">
                Loading GeoDAWN...
              </p>

              <p className="mt-1 text-xs text-slate-500">
                Loading candidates and geophysical layers
              </p>

            </div>

          </div>
        )}

        {/* =====================================================
            ERROR
        ===================================================== */}

        {error && (
          <div className="absolute left-4 right-4 top-4 z-30 rounded-lg border border-red-500/30 bg-red-950/90 px-4 py-3 text-sm text-red-200 backdrop-blur">

            <div className="font-semibold">
              GeoDAWN data error
            </div>

            <div className="mt-1 text-xs text-red-300">
              {error}
            </div>

          </div>
        )}

        {/* =====================================================
            GEOPHYSICAL LAYER CONTROL
        ===================================================== */}

        {!loading && metadata && (
          <div className="mt-4 rounded-xl border border-white/10 bg-[#07111f] p-5">

            <div className="mb-4 flex items-center justify-between">

              <div>

                <h3 className="text-sm font-semibold text-white">
                  Geophysical Layer
                </h3>

                <p className="mt-1 text-xs text-slate-400">
                  Visualize one of the 19 input geophysical bands.
                </p>

              </div>

              <div className="text-xs text-slate-500">
                {metadata.width} × {metadata.height}
              </div>

            </div>

            <div className="flex flex-col gap-3 md:flex-row md:items-center">

              <select
                value={selectedLayer}
                onChange={(event) =>
                  setSelectedLayer(
                    event.target.value
                  )
                }
                className="h-10 flex-1 rounded-lg border border-white/10 bg-[#0b1728] px-3 text-sm text-slate-200 outline-none transition focus:border-blue-500"
              >

                {metadata.layers.map(
                  (layer) => (

                    <option
                      key={layer.index}
                      value={String(layer.index)}
                    >
                      {String(
                        layer.index
                      ).padStart(2, "0")}{" "}
                      —{" "}
                      {layer.description}
                    </option>

                  )
                )}

              </select>

              <div className="flex items-center gap-3 md:w-64">

                <span className="whitespace-nowrap text-xs text-slate-400">
                  Opacity
                </span>

                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={layerOpacity}
                  onChange={(event) =>
                    setLayerOpacity(
                      Number(
                        event.target.value
                      )
                    )
                  }
                  className="w-full"
                />

                <span className="w-10 text-right text-xs text-slate-400">
                  {Math.round(
                    layerOpacity * 100
                  )}%
                </span>

              </div>

            </div>

            {metadata.layers.find(
              (layer) =>
                String(layer.index) ===
                selectedLayer
            ) && (
              <p className="mt-3 text-xs text-slate-500">

                {
                  metadata.layers.find(
                    (layer) =>
                      String(layer.index) ===
                      selectedLayer
                  )?.description
                }

              </p>
            )}

          </div>
        )}

        {/* =====================================================
            SEARCH INFORMATION
        ===================================================== */}

        {searchInfo && (
          <div className="mt-4 grid gap-3 md:grid-cols-4">

            <div className="rounded-xl border border-white/10 bg-[#07111f] p-4">

              <div className="text-[11px] uppercase tracking-wider text-slate-500">
                Location
              </div>

              <div className="mt-2 text-sm font-semibold text-white">
                {searchInfo.locationName}
              </div>

            </div>

            <div className="rounded-xl border border-white/10 bg-[#07111f] p-4">

              <div className="text-[11px] uppercase tracking-wider text-slate-500">
                Coordinates
              </div>

              <div className="mt-2 text-sm font-semibold text-white">
                {searchInfo.latitude.toFixed(6)}
                {", "}
                {searchInfo.longitude.toFixed(6)}
              </div>

            </div>

            <div className="rounded-xl border border-white/10 bg-[#07111f] p-4">

              <div className="text-[11px] uppercase tracking-wider text-slate-500">
                Search Radius
              </div>

              <div className="mt-2 text-sm font-semibold text-white">
                {searchInfo.radiusKm} km
              </div>

            </div>

            <div className="rounded-xl border border-white/10 bg-[#07111f] p-4">

              <div className="text-[11px] uppercase tracking-wider text-slate-500">
                Nearby Candidates
              </div>

              <div className="mt-2 text-sm font-semibold text-white">
                {searchInfo.nearbyCount.toLocaleString()}
              </div>

            </div>

          </div>
        )}

        {/* =====================================================
            CONCEPTUAL 3D GEOLOGICAL SUBSURFACE
        ===================================================== */}

        {subsurfaceOpen && subsurfaceCandidate && (() => {
          const profile = buildSubsurfaceProfile(subsurfaceCandidate);

          const layerY = [
            260,
            330 + profile.layerJitter[1],
            395 + profile.layerJitter[2],
            465 + profile.layerJitter[3],
            535 + profile.layerJitter[4],
            600,
          ];

          const blockLeft = 105;
          const blockRight = 895;
          const surfaceBackLeft = 240;
          const surfaceBackRight = 760;
          const surfaceBackY = 140;

          const fullscreenClass = subsurfaceFullscreen
            ? "fixed inset-0 z-[9999] overflow-hidden bg-[#07111f]"
            : "absolute inset-0 z-50 overflow-hidden rounded-xl bg-[#07111f]/96 backdrop-blur-md";

          return (
            <div className={fullscreenClass}>

              <div className="flex h-full flex-col">

                {/* HEADER */}

                <div className="flex shrink-0 items-center justify-between border-b border-white/10 px-5 py-4">

                  <div>
                    <div className="text-[10px] font-bold uppercase tracking-[0.16em] text-cyan-300">
                      Conceptual subsurface model
                    </div>

                    <h3 className="mt-1 text-lg font-bold text-white">
                      Candidate geological structure
                    </h3>

                    <p className="mt-1 max-w-3xl text-xs text-slate-400">
                      Approximate 3D geological cross-section generated from this candidate's location and model score.
                    </p>
                  </div>

                  <div className="flex items-center gap-2">

                    <button
                      type="button"
                      onClick={() =>
                        setSubsurfaceFullscreen(
                          !subsurfaceFullscreen
                        )
                      }
                      className="rounded-lg border border-white/15 bg-white/5 px-3 py-2 text-xs font-semibold text-slate-200 transition hover:bg-white/10"
                    >
                      {subsurfaceFullscreen
                        ? "Exit full view"
                        : "Full view"}
                    </button>

                    <button
                      type="button"
                      onClick={closeSubsurfaceView}
                      className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/15 bg-white/5 text-lg text-slate-300 transition hover:bg-white/10"
                      aria-label="Close 3D view"
                    >
                      ×
                    </button>

                  </div>

                </div>

                {/* CONTENT */}

                <div className="min-h-0 flex-1 overflow-auto p-3 md:p-5">

                  <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_300px]">

                    {/* 3D GEOLOGICAL BLOCK */}

                    <div className="overflow-hidden rounded-2xl border border-white/10 bg-[#091421]">

                      <div className="border-b border-white/10 px-4 py-3">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div className="text-xs font-semibold text-white">
                            Geological cross-section
                          </div>

                          <div className="rounded-full border border-cyan-400/20 bg-cyan-400/10 px-3 py-1 text-[10px] font-semibold text-cyan-200">
                            CONCEPTUAL — NOT MEASURED
                          </div>
                        </div>
                      </div>

                      <div className="p-2 md:p-4">

                        <svg
                          viewBox="0 0 1000 650"
                          className="h-auto w-full"
                          role="img"
                          aria-label="Approximate 3D geological cross-section for the selected GeoDAWN candidate"
                        >

                          <defs>
                            <linearGradient id="reservoirGradient" x1="0" x2="1" y1="0" y2="1">
                              <stop offset="0%" stopColor="#ffb36b" stopOpacity="0.95" />
                              <stop offset="100%" stopColor="#c2410c" stopOpacity="0.85" />
                            </linearGradient>
                          </defs>

                          <rect
                            x="0"
                            y="0"
                            width="1000"
                            height="650"
                            fill="#091421"
                          />

                          {/* BACK SURFACE */}

                          <polygon
                            points={`${surfaceBackLeft},${surfaceBackY} ${surfaceBackRight},${surfaceBackY - 25} 900,205 320,250`}
                            fill="#d8cfb3"
                            stroke="#e2e8f0"
                            strokeWidth="2"
                          />

                          {/* TOP SURFACE */}

                          <polygon
                            points={`100,210 ${surfaceBackLeft},${surfaceBackY} ${surfaceBackRight},${surfaceBackY - 25} 900,225 315,300`}
                            fill="#b69d70"
                            stroke="#f1ead8"
                            strokeWidth="2"
                          />

                          {/* SURFACE CONTOURS */}

                          <path
                            d={`M120 222 C270 197, 390 230, 510 195 S760 175, 875 226`}
                            fill="none"
                            stroke="#f7f0dc"
                            strokeWidth="7"
                            opacity="0.65"
                          />

                          <path
                            d={`M140 247 C280 223, 395 254, 530 220 S760 204, 850 247`}
                            fill="none"
                            stroke="#78694f"
                            strokeWidth="4"
                            opacity="0.7"
                          />

                          {/* LAYER FRONT FACES */}

                          <polygon
                            points={`100,210 315,300 900,225 895,330 315,395 100,300`}
                            fill="#c8a36e"
                            stroke="#8d6e49"
                            strokeWidth="1.5"
                          />

                          <polygon
                            points={`100,300 315,395 895,330 895,400 315,465 100,390`}
                            fill="#a97c58"
                            stroke="#78563d"
                            strokeWidth="1.5"
                          />

                          <polygon
                            points={`100,390 315,465 895,400 895,470 315,535 100,460`}
                            fill="#64777c"
                            stroke="#42545a"
                            strokeWidth="1.5"
                          />

                          <polygon
                            points={`100,460 315,535 895,470 895,535 315,600 100,525`}
                            fill="#4d626d"
                            stroke="#334650"
                            strokeWidth="1.5"
                          />

                          <polygon
                            points={`100,525 315,600 895,535 895,595 315,650 100,590`}
                            fill="#344454"
                            stroke="#202d3a"
                            strokeWidth="1.5"
                          />

                          {/* LEFT SIDE FACE */}

                          <polygon
                            points="100,210 315,300 315,600 100,525"
                            fill="#896f52"
                            stroke="#d7dde5"
                            strokeWidth="2"
                          />

                          <polygon
                            points="100,300 315,395 315,465 100,390"
                            fill="#9b704d"
                          />

                          <polygon
                            points="100,390 315,465 315,535 100,460"
                            fill="#52656b"
                          />

                          <polygon
                            points="100,460 315,535 315,600 100,525"
                            fill="#435763"
                          />

                          {/* DYNAMIC HOT RESERVOIR */}

                          <ellipse
                            cx={profile.reservoirX}
                            cy={profile.reservoirY}
                            rx={105 + profile.strikeDeg % 35}
                            ry="42"
                            fill="url(#reservoirGradient)"
                            opacity="0.9"
                          />

                          <ellipse
                            cx={profile.reservoirX - 12}
                            cy={profile.reservoirY - 4}
                            rx="55"
                            ry="20"
                            fill="#fde68a"
                            opacity="0.34"
                          />

                          {/* DYNAMIC FAULT PLANE */}

                          <polygon
                            points={`${profile.topLeft},${profile.topFaultY} ${profile.topRight},${profile.topFaultY - 4} ${profile.bottomRight},${profile.bottomFaultY} ${profile.bottomLeft},${profile.bottomFaultY}`}
                            fill="#ef4444"
                            fillOpacity="0.62"
                            stroke="#fecaca"
                            strokeWidth="2"
                          />

                          {/* FAULT CENTER LINE */}

                          <line
                            x1={profile.surfaceX}
                            y1={profile.topFaultY}
                            x2={profile.faultBottomX}
                            y2={profile.bottomFaultY}
                            stroke="#7f1d1d"
                            strokeWidth="5"
                            opacity="0.65"
                          />

                          {/* CONCEPTUAL FLOW */}

                          <g fill="#67e8f9" stroke="#083344" strokeWidth="1">
                            {profile.flowXs.map((x, i) => {
                              const y = profile.flowYs[i];
                              const nextY = y - 12;
                              return (
                                <path
                                  key={`flow-${i}`}
                                  d={`M ${x} ${y + 16} L ${x} ${nextY} l -9 11 h 18 z`}
                                  opacity={0.55 + i * 0.1}
                                />
                              );
                            })}
                          </g>

                          {/* CANDIDATE */}

                          <circle
                            cx={profile.surfaceX}
                            cy={profile.surfaceY}
                            r="14"
                            fill="#3b82f6"
                            stroke="white"
                            strokeWidth="5"
                          />

                          <circle
                            cx={profile.surfaceX}
                            cy={profile.surfaceY}
                            r="25"
                            fill="none"
                            stroke="#60a5fa"
                            strokeWidth="3"
                            opacity="0.55"
                            className="animate-pulse"
                          />

                          {/* DEPTH GUIDES */}

                          <g stroke="#94a3b8" strokeDasharray="6 8" opacity="0.35">
                            <line x1="65" y1={layerY[1]} x2="930" y2={layerY[1]} />
                            <line x1="65" y1={layerY[2]} x2="930" y2={layerY[2]} />
                            <line x1="65" y1={layerY[3]} x2="930" y2={layerY[3]} />
                            <line x1="65" y1={layerY[4]} x2="930" y2={layerY[4]} />
                          </g>

                          {/* LABELS */}

                          <g
                            fontFamily="system-ui, -apple-system, sans-serif"
                            fontSize="18"
                            fontWeight="700"
                          >
                            <text x="118" y="190" fill="#f8fafc">SURFACE</text>

                            <text
                              x={Math.min(760, profile.surfaceX + 28)}
                              y={profile.surfaceY - 12}
                              fill="#fca5a5"
                            >
                              CANDIDATE
                            </text>

                            <text x="665" y="290" fill="#f8fafc">
                              SEDIMENTARY / WEATHERED
                            </text>

                            <text x="690" y="380" fill="#e2e8f0">
                              FRACTURED ROCK
                            </text>

                            <text x="700" y="450" fill="#e2e8f0">
                              COMPETENT ROCK
                            </text>

                            <text
                              x={Math.max(620, profile.reservoirX - 90)}
                              y={profile.reservoirY + 7}
                              fill="#fed7aa"
                            >
                              HOT RESERVOIR
                            </text>

                            <text x="700" y="565" fill="#cbd5e1">
                              BASEMENT
                            </text>

                            <text
                              x={Math.max(300, profile.surfaceX - 55)}
                              y="355"
                              fill="#fff1f2"
                              transform={`rotate(${-(90 - profile.dipDeg)} ${Math.max(300, profile.surfaceX - 55)} 355)`}
                              fontSize="15"
                            >
                              CONCEPTUAL FAULT
                            </text>

                            <text
                              x={Math.max(330, profile.reservoirX - 135)}
                              y="610"
                              fill="#cffafe"
                              fontSize="14"
                            >
                              CONCEPTUAL HOT-FLUID PATHWAY
                            </text>
                          </g>

                          {/* AXIS / SCALE NOTE */}

                          <text
                            x="55"
                            y="630"
                            fill="#94a3b8"
                            fontFamily="system-ui, -apple-system, sans-serif"
                            fontSize="13"
                          >
                            Depth increases downward — schematic only
                          </text>

                        </svg>

                      </div>

                    </div>

                    {/* INFORMATION PANEL */}

                    <div className="space-y-3">

                      <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                        <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-cyan-300">
                          Selected AI candidate
                        </div>

                        <div className="mt-3 text-sm font-bold text-white">
                          Candidate
                          {subsurfaceCandidate.rank !== undefined
                            ? ` #${subsurfaceCandidate.rank}`
                            : ""}
                        </div>

                        <div className="mt-3 grid grid-cols-2 gap-2">
                          <div className="rounded-lg bg-black/20 px-3 py-2">
                            <div className="text-[9px] uppercase tracking-wider text-slate-500">
                              Latitude
                            </div>
                            <div className="mt-1 font-mono text-xs text-white">
                              {subsurfaceCandidate.latitude.toFixed(6)}
                            </div>
                          </div>

                          <div className="rounded-lg bg-black/20 px-3 py-2">
                            <div className="text-[9px] uppercase tracking-wider text-slate-500">
                              Longitude
                            </div>
                            <div className="mt-1 font-mono text-xs text-white">
                              {subsurfaceCandidate.longitude.toFixed(6)}
                            </div>
                          </div>
                        </div>

                        {subsurfaceCandidate.score !== undefined && (
                          <div className="mt-2 rounded-lg bg-black/20 px-3 py-2">
                            <div className="text-[9px] uppercase tracking-wider text-slate-500">
                              AI candidate score
                            </div>
                            <div className="mt-1 text-sm font-bold text-white">
                              {subsurfaceCandidate.score.toFixed(6)}
                            </div>
                          </div>
                        )}
                      </div>

                      <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                        <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-amber-300">
                          Approximate geometry
                        </div>

                        <div className="mt-3 grid grid-cols-2 gap-2">
                          <div className="rounded-lg bg-black/20 px-3 py-2">
                            <div className="text-[9px] uppercase tracking-wider text-slate-500">
                              Conceptual strike
                            </div>
                            <div className="mt-1 text-xs font-bold text-white">
                              {profile.strikeDeg}°
                            </div>
                          </div>

                          <div className="rounded-lg bg-black/20 px-3 py-2">
                            <div className="text-[9px] uppercase tracking-wider text-slate-500">
                              Conceptual dip
                            </div>
                            <div className="mt-1 text-xs font-bold text-white">
                              {Math.round(profile.dipDeg)}°
                            </div>
                          </div>

                          <div className="col-span-2 rounded-lg bg-black/20 px-3 py-2">
                            <div className="text-[9px] uppercase tracking-wider text-slate-500">
                              Conceptual reservoir depth
                            </div>
                            <div className="mt-1 text-xs font-bold text-white">
                              ~{profile.reservoirDepthKm.toFixed(1)} km
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                        <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-amber-300">
                          What the diagram shows
                        </div>

                        <div className="mt-3 space-y-2 text-xs leading-5 text-slate-300">
                          <div>
                            <span className="font-semibold text-red-200">
                              Red plane:
                            </span>{" "}
                            approximate conceptual fault geometry.
                          </div>

                          <div>
                            <span className="font-semibold text-cyan-200">
                              Cyan arrows:
                            </span>{" "}
                            conceptual upward hot-fluid pathway.
                          </div>

                          <div>
                            <span className="font-semibold text-orange-200">
                              Orange zone:
                            </span>{" "}
                            conceptual hot reservoir.
                          </div>
                        </div>
                      </div>

                      <div className="rounded-2xl border border-cyan-400/20 bg-cyan-400/5 p-4">
                        <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-cyan-300">
                          Scientific caution
                        </div>

                        <p className="mt-2 text-xs leading-5 text-slate-300">
                          This diagram is generated to help interpret the selected AI candidate. Its fault orientation, depth, reservoir and fluid pathway are approximate visualizations; the V1_19 model does not directly measure these underground properties.
                        </p>
                      </div>

                      <button
                        type="button"
                        onClick={closeSubsurfaceView}
                        className="w-full rounded-xl bg-[#3157d5] px-4 py-3 text-sm font-semibold text-white transition hover:bg-[#2647ba]"
                      >
                        Return to live map
                      </button>

                    </div>

                  </div>

                </div>

              </div>

            </div>
          );
        })()}

      </div>
    );
  }
