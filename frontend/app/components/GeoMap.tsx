"use client";

import { useEffect, useRef, useState } from "react";
import {
  Map,
  Marker,
  NavigationControl,
  FullscreenControl,
  Popup,
} from "maplibre-gl";

import "maplibre-gl/dist/maplibre-gl.css";

/* =========================================================
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

      openCandidatePopup(
        candidate,
        searchedLatitude,
        searchedLongitude,
        distance
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
        DEFAULT_RADIUS_KM
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

    map.flyTo({
      center: [longitude, latitude],
      zoom: 11,
      duration: 1200,
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
        openCandidatePopup(
          candidate,

          searchInfo?.latitude ??
            latitude,

          searchInfo?.longitude ??
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
          MAP
      ===================================================== */}

      <div className="relative">

  <div
    ref={mapContainerRef}
    className="relative h-[560px] w-full overflow-hidden rounded-xl border border-white/10 bg-[#07111f]"
  />

  {/* =====================================================
    ACTIVE GEOPHYSICAL LAYER
===================================================== */}

{!loading && metadata && (() => {
  const activeLayer = metadata.layers.find(
    (layer) =>
      String(layer.index) ===
      String(selectedLayer)
  );

  if (!activeLayer) return null;

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

  const cleanName = activeLayer.name
    .split("_")
    .slice(1)
    .join(" ")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim();
return (
  <div className="absolute left-4 top-4 z-10 w-[320px] rounded-xl border border-white/70 bg-white/95 p-4 shadow-xl backdrop-blur-md">

    {/* HEADER */}

    <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-500">
      Geophysical Layer
    </div>

    {/* BAND + NAME */}

    <div className="mt-2 flex items-start gap-3">

      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[#eef2ff] text-xs font-bold text-[#3157d5]">
        {String(activeLayer.index).padStart(2, "0")}
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

    {/* WHAT THIS LAYER REPRESENTS */}

    <div className="mt-4 rounded-xl border border-[#dbe5ff] bg-[#f5f7ff] p-4">

      <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-[#3157d5]">
        What this layer represents
      </div>

      <p className="mt-2 text-sm font-semibold leading-6 text-[#172554]">
        {activeLayer.description}
      </p>

    </div>

    {/* LAYER DETAILS */}

    <div className="mt-3 grid grid-cols-2 gap-2">

      {/* CATEGORY */}

      <div className="rounded-lg bg-slate-50 px-3 py-2">

        <div className="text-[9px] font-bold uppercase tracking-wider text-slate-400">
          Category
        </div>

        <div className="mt-0.5 text-xs font-semibold text-slate-700">
          {category}
        </div>

      </div>

      {/* RESOLUTION */}

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

</div>

      {/* =====================================================
          LOADING OVERLAY
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
                setSelectedLayer(event.target.value)
              }
              className="h-10 flex-1 rounded-lg border border-white/10 bg-[#0b1728] px-3 text-sm text-slate-200 outline-none transition focus:border-blue-500"
            >
              {metadata.layers.map((layer) => (
                <option
                  key={layer.index}
                  value={String(layer.index)}
                >
                  {String(layer.index).padStart(2, "0")} —{" "}
                  {layer.description}
                </option>
              ))}
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
                    Number(event.target.value)
                  )
                }
                className="w-full"
              />

              <span className="w-10 text-right text-xs text-slate-400">
                {Math.round(layerOpacity * 100)}%
              </span>

            </div>

          </div>

          {metadata.layers.find(
            (layer) =>
              String(layer.index) === selectedLayer
          ) && (
            <p className="mt-3 text-xs text-slate-500">
              {
                metadata.layers.find(
                  (layer) =>
                    String(layer.index) === selectedLayer
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
              {searchInfo.latitude.toFixed(6)},{" "}
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

    </div>
  );
}