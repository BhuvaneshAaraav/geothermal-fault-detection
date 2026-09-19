"use client";

import { useEffect, useState } from "react";
import GeoMap from "./components/GeoMap";

type CandidateResult = {
  rank: number;
  distanceKm: number;
  score: number;
  latitude: number;
  longitude: number;
  row: number;
  column: number;
};

function scrollToSection(id: string) {
  document.getElementById(id)?.scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
}

function LocationSearch() {
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [radius, setRadius] = useState("10");

  const handleSearch = () => {
    const lat = Number(latitude);
    const lon = Number(longitude);
    const radiusKm = Number(radius);

    if (
      !Number.isFinite(lat) ||
      !Number.isFinite(lon) ||
      lat < -90 ||
      lat > 90 ||
      lon < -180 ||
      lon > 180
    ) {
      alert("Please enter valid latitude and longitude.");
      return;
    }

    window.dispatchEvent(
      new CustomEvent("geodawn-location-search", {
        detail: {
          latitude: lat,
          longitude: lon,
          radiusKm,
        },
      })
    );

    scrollToSection("fault-map");
  };

  return (
    <div
      id="search"
      className="rounded-2xl border border-white/10 bg-[#0a1627] p-5"
    >
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-white">
          Search Location
        </h3>

        <p className="mt-1 text-sm text-slate-400">
          Find AI-predicted geothermal fault candidates near a location.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
        <input
          type="number"
          step="any"
          placeholder="Latitude"
          value={latitude}
          onChange={(e) => setLatitude(e.target.value)}
          className="rounded-lg border border-white/10 bg-[#07111f] px-4 py-3 text-sm text-white outline-none placeholder:text-slate-500 focus:border-blue-500"
        />

        <input
          type="number"
          step="any"
          placeholder="Longitude"
          value={longitude}
          onChange={(e) => setLongitude(e.target.value)}
          className="rounded-lg border border-white/10 bg-[#07111f] px-4 py-3 text-sm text-white outline-none placeholder:text-slate-500 focus:border-blue-500"
        />

        <select
          value={radius}
          onChange={(e) => setRadius(e.target.value)}
          className="rounded-lg border border-white/10 bg-[#07111f] px-4 py-3 text-sm text-white outline-none focus:border-blue-500"
        >
          <option value="1">Within 1 km</option>
          <option value="5">Within 5 km</option>
          <option value="10">Within 10 km</option>
          <option value="25">Within 25 km</option>
          <option value="50">Within 50 km</option>
        </select>

        <button
          type="button"
          onClick={handleSearch}
          className="rounded-lg bg-blue-600 px-6 py-3 text-sm font-semibold text-white transition hover:bg-blue-500"
        >
          Find Faults
        </button>
      </div>
    </div>
  );
}

function SearchResults() {
  const [result, setResult] = useState<{
    latitude: number;
    longitude: number;
    radiusKm: number;
    count: number;
    nearestDistanceKm: number | null;
    locationName?: string;
  } | null>(null);

  useEffect(() => {
    const handler = (event: Event) => {
      const customEvent =
        event as CustomEvent;

      setResult(customEvent.detail);
    };

    window.addEventListener(
      "geodawn-search-results",
      handler
    );

    return () => {
      window.removeEventListener(
        "geodawn-search-results",
        handler
      );
    };
  }, []);

  if (!result) {
    return (
      <section className="rounded-2xl border border-[#dbe2ef] bg-white p-6 shadow-sm">
        <h2 className="text-lg font-bold text-[#10245c]">
          Search Results
        </h2>

        <p className="mt-2 text-sm text-[#64748b]">
          Search a location to find nearby AI-generated
          geothermal fault candidates.
        </p>
      </section>
    );
  }

  const locationName =
    result.locationName ||
    `${result.latitude.toFixed(6)}, ${result.longitude.toFixed(6)}`;

  return (
    <section className="rounded-2xl border border-[#dbe2ef] bg-white p-6 shadow-sm">

      <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">

        <div>
          <h2 className="text-lg font-bold text-[#10245c]">
            Search Analysis
          </h2>

          <p className="mt-1 text-sm text-[#64748b]">
            AI-generated geothermal fault candidates
            within the selected search area.
          </p>
        </div>

        <div className="rounded-full bg-[#eef2ff] px-3 py-1.5 text-xs font-semibold text-[#3157d5]">
          {result.radiusKm} km analysis
        </div>

      </div>

      <div className="mt-6 grid gap-4 md:grid-cols-2 lg:grid-cols-4">

        {/* LOCATION */}

        <div className="rounded-xl border border-[#e2e8f0] bg-[#f8fafc] p-5 md:col-span-2">

          <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#94a3b8]">
            Location
          </p>

          <p className="mt-2 text-base font-semibold leading-6 text-[#172554]">
            {locationName}
          </p>

        </div>

        {/* COORDINATES */}

        <div className="rounded-xl border border-[#e2e8f0] bg-[#f8fafc] p-5">

          <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#94a3b8]">
            Coordinates
          </p>

          <p className="mt-3 text-sm font-semibold text-[#334155]">
            {result.latitude.toFixed(6)}
          </p>

          <p className="mt-1 text-sm font-semibold text-[#334155]">
            {result.longitude.toFixed(6)}
          </p>

        </div>

        {/* RADIUS */}

        <div className="rounded-xl border border-[#e2e8f0] bg-[#f8fafc] p-5">

          <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#94a3b8]">
            Search Radius
          </p>

          <p className="mt-3 text-2xl font-bold text-[#10245c]">
            {result.radiusKm} km
          </p>

        </div>

        {/* CANDIDATES */}

        <div className="rounded-xl border border-[#e2e8f0] bg-[#f8fafc] p-5">

          <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#94a3b8]">
            Nearby Candidates
          </p>

          <p className="mt-3 text-2xl font-bold text-[#10245c]">
            {result.count.toLocaleString()}
          </p>

        </div>

        {/* NEAREST */}

        <div className="rounded-xl border border-[#e2e8f0] bg-[#f8fafc] p-5">

          <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#94a3b8]">
            Nearest Candidate
          </p>

          <p className="mt-3 text-2xl font-bold text-[#10245c]">
            {result.nearestDistanceKm !== null
              ? `${result.nearestDistanceKm.toFixed(2)} km`
              : "—"}
          </p>

        </div>

      </div>

    </section>
  );
}

function CandidatesPanel() {
  const [selectedCandidateRank, setSelectedCandidateRank] =

    useState<number | null>(null);
  const [candidates, setCandidates] =
    useState<CandidateResult[]>([]);

const [minScore, setMinScore] = useState("");
const [maxDistance, setMaxDistance] = useState("");

  const [sortBy, setSortBy] =
    useState<"distance" | "rank" | "score">(
      "distance"
    );

  const [limit, setLimit] =
    useState(25);

  useEffect(() => {
    const handleCandidates = (
      event: Event
    ) => {
      const e =
        event as CustomEvent<CandidateResult[]>;

      setCandidates(e.detail || []);
    };

    window.addEventListener(
      "geodawn-candidates-results",
      handleCandidates
    );

    return () => {
      window.removeEventListener(
        "geodawn-candidates-results",
        handleCandidates
      );
    };
  }, []);

  /*
   * Sort candidates
   */

  const sortedCandidates =
    [...candidates].sort(
      (a, b) => {
        if (sortBy === "distance") {
          return (
            a.distanceKm -
            b.distanceKm
          );
        }

        if (sortBy === "rank") {
          return (
            a.rank -
            b.rank
          );
        }

        return (
          b.score -
          a.score
        );
      }
    );

  const visibleCandidates =
    sortedCandidates.slice(
      0,
      limit
    );

  /*
   * Empty state
   */

  if (candidates.length === 0) {
    return (
      <section
        id="candidates"
        className="rounded-2xl border border-[#dbe2ef] bg-white p-6 shadow-sm"
      >
        
        <div className="flex items-center justify-between">

          <div>
            <h2 className="text-lg font-bold text-[#10245c]">
              Nearby Fault Candidates
            </h2>

            <p className="mt-1 text-sm text-[#64748b]">
              Search a location to display nearby
              AI-generated geothermal fault candidates.
            </p>
          </div>

        </div>
       
<div className="flex items-center gap-3">

  <button
    type="button"
    onClick={exportCandidatesCSV}
    disabled={candidates.length === 0}
    className="
      rounded-lg
      border border-[#dbe2ee]
      bg-white
      px-3 py-2
      text-xs
      font-semibold
      text-[#3157d5]
      transition
      hover:border-[#b9c7e8]
      hover:bg-[#f8faff]
      disabled:cursor-not-allowed
      disabled:opacity-40
    "
  >
    Export CSV ↓
  </button>

  <div className="
    rounded-full
    bg-[#eef2ff]
    px-3 py-1.5
    text-xs
    font-semibold
    text-[#3157d5]
  ">
    {candidates.length} candidates
  </div>

</div>


      </section>
    );
  }

function exportCandidatesCSV() {
  if (candidates.length === 0) return;

  const headers = [
    "Rank",
    "AI Score",
    "Distance (km)",
    "Latitude",
    "Longitude",
    "Raster Row",
    "Raster Column",
  ];

  const rows = candidates.map((candidate) => [
    candidate.rank,
    candidate.score.toFixed(6),
    candidate.distanceKm.toFixed(4),
    candidate.latitude.toFixed(6),
    candidate.longitude.toFixed(6),
    candidate.row,
    candidate.column,
  ]);

  const csv = [
    headers.join(","),
    ...rows.map((row) => row.join(",")),
  ].join("\n");

  const blob = new Blob([csv], {
    type: "text/csv;charset=utf-8;",
  });

  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.href = url;
  link.download = "geodawn_fault_candidates.csv";

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  URL.revokeObjectURL(url);
}

  return (
    <section
      id="candidates"
      className="overflow-hidden rounded-2xl border border-[#dbe2ef] bg-white shadow-sm"
    >

      {/* =====================================================
          HEADER
      ===================================================== */}

      <div className="border-b border-[#e2e8f0] p-6">

        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">

          <div>

            <h2 className="text-lg font-bold text-[#10245c]">
              Nearby Fault Candidates
            </h2>

            <p className="mt-1 text-sm text-[#64748b]">
              AI-generated candidates within the
              selected search area.
            </p>

          </div>

          <div className="flex items-center gap-3">

  <button
    type="button"
    onClick={exportCandidatesCSV}
    disabled={candidates.length === 0}
    className="
      rounded-lg
      border border-[#dbe2ee]
      bg-white
      px-3 py-2
      text-xs
      font-semibold
      text-[#3157d5]
      transition
      hover:border-[#b9c7e8]
      hover:bg-[#f8faff]
      disabled:cursor-not-allowed
      disabled:opacity-40
    "
  >
    Export CSV ↓
  </button>

  <div className="rounded-full bg-[#eef2ff] px-4 py-2 text-xs font-bold text-[#3157d5]">
    {candidates.length.toLocaleString()} candidates
  </div>

</div>

        </div>

        {/* CONTROLS */}

        <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-end">

          <div className="flex items-center gap-2">

            <label className="text-xs font-medium text-[#64748b]">
              Sort by
            </label>

            <select
              value={sortBy}
              onChange={(event) =>
                setSortBy(
                  event.target.value as
                    | "distance"
                    | "rank"
                    | "score"
                )
              }
              className="h-9 rounded-lg border border-[#dbe2ef] bg-white px-3 text-xs font-medium text-[#334155] outline-none focus:border-[#3157d5]"
            >
              <option value="distance">
                Distance
              </option>

              <option value="rank">
                Rank
              </option>

              <option value="score">
                Candidate score
              </option>
            </select>

          </div>

          <div className="flex items-center gap-2">

            <label className="text-xs font-medium text-[#64748b]">
              Show
            </label>

            <select
              value={limit}
              onChange={(event) =>
                setLimit(
                  Number(
                    event.target.value
                  )
                )
              }
              className="h-9 rounded-lg border border-[#dbe2ef] bg-white px-3 text-xs font-medium text-[#334155] outline-none focus:border-[#3157d5]"
            >
              <option value={10}>
                10 rows
              </option>

              <option value={25}>
                25 rows
              </option>

              <option value={50}>
                50 rows
              </option>

              <option value={100}>
                100 rows
              </option>
            </select>

          </div>

        </div>

      </div>

     {/* =====================================================
    TABLE
===================================================== */}

<div className="overflow-x-auto">

  <table className="w-full min-w-[950px]">

    <thead className="bg-[#f8fafc]">

      <tr className="border-b border-[#e5eaf2]">

        <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-[#64748b]">
          Rank
        </th>

        <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-[#64748b]">
          AI Score
        </th>

        <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-[#64748b]">
          Distance
        </th>

        <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-[#64748b]">
          Coordinates
        </th>

        <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-[#64748b]">
          Raster Cell
        </th>

        <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wide text-[#64748b]">
          Action
        </th>

      </tr>

    </thead>


    <tbody>

      {visibleCandidates.map(
        (candidate, index) => (

          <tr
            key={`${candidate.rank}-${candidate.latitude}-${candidate.longitude}-${index}`}
            className={`border-b border-[#edf1f6] transition ${
  selectedCandidateRank === candidate.rank
    ? "bg-blue-50"
    : "hover:bg-[#f8faff]"
}`}
          >

            {/* RANK */}

            <td className="px-6 py-4">

              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-[#eef2ff] text-[10px] font-bold text-[#3157d5]">
                {candidate.rank}
              </div>

            </td>


            {/* AI SCORE */}

            <td className="px-6 py-4">

              <div className="flex items-center gap-3">

                <span className="text-sm font-bold text-[#172554]">
                  {candidate.score.toFixed(6)}
                </span>

                <div className="hidden h-1.5 w-20 overflow-hidden rounded-full bg-[#e2e8f0] sm:block">

                  <div
                    className="h-full rounded-full bg-[#3157d5]"
                    style={{
                      width: `${Math.min(
                        candidate.score * 100,
                        100
                      )}%`,
                    }}
                  />

                </div>

              </div>

            </td>


            {/* DISTANCE */}

            <td className="px-6 py-4">

              <span className="text-sm font-semibold text-[#172554]">
                {candidate.distanceKm.toFixed(2)} km
              </span>

            </td>


            {/* COORDINATES */}

            <td className="px-6 py-4">

              <div className="font-mono text-xs leading-5 text-[#475569]">

                <div>
                  {candidate.latitude.toFixed(6)}
                </div>

                <div>
                  {candidate.longitude.toFixed(6)}
                </div>

              </div>

            </td>


            {/* RASTER CELL */}

            <td className="px-6 py-4">

              <span className="font-mono text-xs text-[#64748b]">
                {candidate.row}
                {" × "}
                {candidate.column}
              </span>

            </td>


            {/* VIEW ON MAP */}

            <td className="px-6 py-4 text-right">

              <button
                type="button"
                onClick={() => {


                  setSelectedCandidateRank(candidate.rank);
                  window.dispatchEvent(
                    new CustomEvent(
                      "geodawn-candidate-focus",
                      {
                        detail: {
                          latitude:
                            candidate.latitude,

                          longitude:
                            candidate.longitude,

                          rank:
                            candidate.rank,

                          score:
                            candidate.score,
                        },
                      }
                    )
                  );

                  setTimeout(() => {
                    scrollToSection("fault-map");
                  }, 100);

                }}
                className="rounded-lg bg-[#eef2ff] px-3 py-2 text-xs font-semibold text-[#3157d5] transition hover:bg-[#dbe5ff]"
              >
                View on map →
              </button>

            </td>

          </tr>

        )
      )}

    </tbody>

  </table>

</div>

      {/* =====================================================
          FOOTER
      ===================================================== */}

      <div className="flex flex-col gap-2 border-t border-[#e2e8f0] bg-[#fafbfc] px-6 py-4 text-xs text-[#64748b] sm:flex-row sm:items-center sm:justify-between">

        <span>
          Showing{" "}
          <strong className="text-[#334155]">
            {visibleCandidates.length}
          </strong>{" "}
          of{" "}
          <strong className="text-[#334155]">
            {candidates.length.toLocaleString()}
          </strong>{" "}
          nearby candidates
        </span>

        <span>
          AI-generated candidates · Not confirmed geological faults
        </span>

      </div>

    </section>
  );
}

function GeophysicalLayers() {
  /*function viewLayerOnMap(layerIndex: number) {
  window.dispatchEvent(
    new CustomEvent("geodawn-layer-select", {
      detail: { layerIndex },
    })
  );

  document
    .getElementById("fault-map")
    ?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
}*/
  const bands = [
    {
      index: 1,
      name: "Magnetic anomaly",
      description:
        "Deviation from the expected Earth's magnetic field.",
      category: "Magnetic",
    },
    {
      index: 2,
      name: "Reduced to pole",
      description:
        "Magnetic anomaly corrected for latitude effects.",
      category: "Magnetic",
    },
    {
      index: 3,
      name: "TMI horizontal gradient",
      description:
        "Rate of change in total magnetic intensity in the horizontal direction.",
      category: "Magnetic",
    },
    {
      index: 4,
      name: "Geodetic 2nd invariant",
      description:
        "Magnitude of the geodetic strain-rate tensor.",
      category: "Geodetic",
    },
    {
      index: 5,
      name: "Isostatic gravity anomaly slope",
      description:
        "Gravity gradient after isostatic correction.",
      category: "Gravity",
    },
    {
      index: 6,
      name: "Total curvature",
      description:
        "Magnetic-field derivative used for edge detection.",
      category: "Magnetic",
    },
    {
      index: 7,
      name: "Geodetic shear rate",
      description:
        "Rate of angular deformation from geodetic observations.",
      category: "Geodetic",
    },
    {
      index: 8,
      name: "Geodetic dilatation rate",
      description:
        "Rate of volumetric strain expansion or contraction.",
      category: "Geodetic",
    },
    {
      index: 9,
      name: "TMI vertical gradient",
      description:
        "Rate of change in total magnetic intensity vertically.",
      category: "Magnetic",
    },
    {
      index: 10,
      name: "Distance to earthquakes",
      description:
        "Distance-related earthquake parameter using the specified search parameters.",
      category: "Seismic",
    },
    {
      index: 11,
      name: "Isostatic gravity vertical gradient",
      description:
        "Vertical rate of change of the isostatic gravity anomaly.",
      category: "Gravity",
    },
    {
      index: 12,
      name: "Detrended elevation",
      description:
        "Topography with regional trends removed.",
      category: "Topography",
    },
    {
      index: 13,
      name: "Isostatic gravity anomaly",
      description:
        "Gravity after compensating for topographic mass.",
      category: "Gravity",
    },
    {
      index: 14,
      name: "Total magnetic intensity",
      description:
        "Total strength of the Earth's magnetic field.",
      category: "Magnetic",
    },
    {
      index: 15,
      name: "Depth to basement surface",
      description:
        "Estimated thickness of the sedimentary cover above basement.",
      category: "Geology",
    },
    {
      index: 16,
      name: "Earthquake intensity / density",
      description:
        "Earthquake intensity or density calculated using the specified parameters.",
      category: "Seismic",
    },
    {
      index: 17,
      name: "Surface conductivity",
      description:
        "Electrical conductivity of the subsurface near the surface.",
      category: "Electrical",
    },
    {
      index: 18,
      name: "Isostatic gravity horizontal gradient",
      description:
        "Horizontal rate of change of the isostatic gravity anomaly.",
      category: "Gravity",
    },
    {
      index: 19,
      name: "Detrended elevation slope",
      description:
        "Elevation gradient after regional trends are removed.",
      category: "Topography",
    },
  ];

  const categoryStyle: Record<
    string,
    string
  > = {
    Magnetic:
      "bg-blue-50 text-blue-700 border-blue-100",
    Gravity:
      "bg-violet-50 text-violet-700 border-violet-100",
    Geodetic:
      "bg-emerald-50 text-emerald-700 border-emerald-100",
    Seismic:
      "bg-orange-50 text-orange-700 border-orange-100",
    Topography:
      "bg-amber-50 text-amber-700 border-amber-100",
    Geology:
      "bg-slate-100 text-slate-700 border-slate-200",
    Electrical:
      "bg-cyan-50 text-cyan-700 border-cyan-100",
  };

  const handleLayerSelect = (
    index: number
  ) => {
    /*
     * GeoMap will listen for this event.
     *
     * We keep the layer data in the map component
     * so the existing working map is not rebuilt.
     */
    window.dispatchEvent(
      new CustomEvent(
        "geodawn-layer-select",
        {
          detail: {
            layerIndex: index,
          },
        }
      )
    );

    scrollToSection("fault-map");
  };

  return (
    <section
      id="layers"
      className="scroll-mt-28 rounded-2xl border border-[#dbe2ef] bg-white p-6 shadow-sm"
    >

      {/* HEADER */}

      <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">

        <div>

          <div className="flex items-center gap-3">

            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[#eef2ff] text-lg text-[#3157d5]">
              ▱
            </div>

            <div>

              <h2 className="text-xl font-bold text-[#10245c]">
                Geophysical Layers
              </h2>

              <p className="mt-1 text-sm text-[#64748b]">
                Explore the 19 geophysical input bands used by
                the GeoDAWN V1_19 model.
              </p>

            </div>

          </div>

        </div>

        <div className="rounded-full border border-[#dbe2ef] bg-[#f8fafc] px-4 py-2 text-xs font-semibold text-[#475569]">
          19 input bands
        </div>

      </div>

      {/* INFO */}

      <div className="mt-6 rounded-xl border border-[#dbe5ff] bg-[#f5f7ff] p-4">

        <div className="flex gap-3">

          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#e0e7ff] text-sm font-bold text-[#3157d5]">
            i
          </div>

          <div>

            <p className="text-sm font-semibold text-[#1e3a8a]">
              Model input data
            </p>

            <p className="mt-1 text-xs leading-5 text-[#64748b]">
              These layers represent the geophysical variables
              provided to V1_19 during model inference. Select a
              layer to inspect it on the satellite map.
            </p>

          </div>

        </div>

      </div>

      {/* LAYERS */}

      <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2">

        {bands.map((band) => (

          <button
            key={band.index}
            type="button"
            onClick={() =>
              handleLayerSelect(
                band.index
              )
            }
            className="group rounded-xl border border-[#e2e8f0] bg-[#fbfcfe] p-4 text-left transition hover:-translate-y-0.5 hover:border-[#b8c7ef] hover:bg-[#f7f9ff] hover:shadow-sm"
          >

            <div className="flex gap-4">

              {/* NUMBER */}

              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[#eef2ff] text-sm font-bold text-[#3157d5]">
                {String(
                  band.index
                ).padStart(2, "0")}
              </div>

              {/* CONTENT */}

              <div className="min-w-0 flex-1">

                <div className="flex flex-wrap items-center gap-2">

                  <h3 className="text-sm font-bold text-[#172554]">
                    {band.name}
                  </h3>

                  <span
                    className={`rounded-full border px-2 py-0.5 text-[9px] font-bold uppercase tracking-wide ${
                      categoryStyle[
                        band.category
                      ]
                    }`}
                  >
                    {band.category}
                  </span>

                </div>

                <p className="mt-1.5 text-xs leading-5 text-[#64748b]">
                  {band.description}
                </p>

                <div className="mt-3 flex items-center justify-between">

                  <span className="text-[10px] font-medium text-[#94a3b8]">
                    Band {band.index} / 19
                  </span>

                  <span className="text-xs font-semibold text-[#3157d5] opacity-0 transition group-hover:opacity-100">
                    View on map →
                  </span>

                </div>

              </div>

            </div>

          </button>

        ))}

      </div>

    </section>
  );
}


function ModelInformation() {
  return (
    <section
      id="model"
      className="scroll-mt-28 rounded-2xl border border-[#dbe2ef] bg-white p-6 shadow-sm"
    >
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">

        <div className="flex items-center gap-3">

          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[#eef2ff] text-[#3157d5]">
            ⚙
          </div>

          <div>
            <h2 className="text-xl font-bold text-[#10245c]">
              Model Information
            </h2>

            <p className="mt-1 text-sm text-[#64748b]">
              Locked configuration of the GeoDAWN V1_19 production model.
            </p>
          </div>

        </div>

        <div className="rounded-full bg-[#eef2ff] px-4 py-2 text-xs font-bold text-[#3157d5]">
          V1_19
        </div>

      </div>

      <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">

        {/* ARCHITECTURE */}

        <div className="rounded-xl border border-[#e2e8f0] bg-[#f8fafc] p-5">
          <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#94a3b8]">
            Architecture
          </p>

          <p className="mt-3 text-lg font-bold text-[#172554]">
            U-Net
          </p>

          <p className="mt-1 text-xs text-[#64748b]">
            Spatial deep-learning model
          </p>
        </div>

        {/* INPUT FEATURES */}

        <div className="rounded-xl border border-[#e2e8f0] bg-[#f8fafc] p-5">
          <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#94a3b8]">
            Input Features
          </p>

          <p className="mt-3 text-lg font-bold text-[#172554]">
            19
          </p>

          <p className="mt-1 text-xs text-[#64748b]">
            Geophysical bands
          </p>
        </div>

        {/* PATCH SIZE */}

        <div className="rounded-xl border border-[#e2e8f0] bg-[#f8fafc] p-5">
          <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#94a3b8]">
            Patch Size
          </p>

          <p className="mt-3 text-lg font-bold text-[#172554]">
            31 × 31
          </p>

          <p className="mt-1 text-xs text-[#64748b]">
            Spatial context window
          </p>
        </div>

        {/* RESOLUTION */}

        <div className="rounded-xl border border-[#e2e8f0] bg-[#f8fafc] p-5">
          <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#94a3b8]">
            Spatial Resolution
          </p>

          <p className="mt-3 text-lg font-bold text-[#172554]">
            100 m
          </p>

          <p className="mt-1 text-xs text-[#64748b]">
            Input raster resolution
          </p>
        </div>

      </div>

      {/* SECOND ROW */}

      <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-3">

        <div className="rounded-xl border border-[#e2e8f0] bg-white p-5">
          <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#94a3b8]">
            Inference
          </p>

          <p className="mt-2 text-sm font-semibold text-[#172554]">
            Tiled raster inference
          </p>
        </div>

        <div className="rounded-xl border border-[#e2e8f0] bg-white p-5">
          <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#94a3b8]">
            Final Candidates
          </p>

          <p className="mt-2 text-sm font-semibold text-[#172554]">
            84,500
          </p>
        </div>

        <div className="rounded-xl border border-[#e2e8f0] bg-white p-5">
          <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#94a3b8]">
            Validation DTI
          </p>

          <p className="mt-2 text-sm font-semibold text-[#172554]">
            0.2438
          </p>
        </div>

      </div>

      {/* NOTICE */}

      <div className="mt-5 rounded-xl border border-[#dbe5ff] bg-[#f5f7ff] p-4">

        <div className="flex gap-3">

          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#e0e7ff] text-sm font-bold text-[#3157d5]">
            i
          </div>

          <div>
            <p className="text-sm font-semibold text-[#1e3a8a]">
              Production configuration
            </p>

            <p className="mt-1 text-xs leading-5 text-[#64748b]">
              These values describe the locked V1_19 configuration
              used for the final GeoDAWN candidate product.
              Candidate locations are AI-generated predictions and
              should be treated as targets for geological investigation,
              not confirmed faults.
            </p>
          </div>

        </div>

      </div>

    </section>
  );
}

export default function Home() {
  const [activeSection, setActiveSection] =
    useState("overview");

  const navigation = [
    {
      id: "overview",
      label: "Overview",
      icon: "⌂",
    },
    {
      id: "fault-map",
      label: "Fault Map",
      icon: "◈",
    },
    {
      id: "candidates",
      label: "Candidates",
      icon: "◎",
    },
    {
      id: "layers",
      label: "Geophysical Layers",
      icon: "▱",
    },
    {
      id: "model",
      label: "Model",
      icon: "▣",
    },
  ];

  const handleNavigation = (id: string) => {
    setActiveSection(id);
    scrollToSection(id);
  };

  return (
    <main className="min-h-screen bg-[#f6f8fc] text-[#172554]">

      {/* =====================================================
          SIDEBAR
      ===================================================== */}

      <aside className="fixed left-0 top-0 z-40 hidden h-screen w-[298px] border-r border-[#dbe2ef] bg-white lg:flex lg:flex-col">

        {/* BRAND */}

        <div className="flex h-[100px] items-center border-b border-[#e5eaf2] px-8">

          <div className="mr-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-[#e9edff] text-2xl text-[#3157d5]">
            ◈
          </div>

          <div>
            <h1 className="text-[25px] font-bold tracking-tight text-[#10245c]">
              GeoDAWN
            </h1>

            <p className="mt-0.5 text-xs font-medium text-[#64748b]">
              Geothermal Fault Detection
            </p>
          </div>

        </div>

        {/* NAVIGATION */}

        <nav className="flex-1 px-4 py-7">

          <p className="mb-3 px-4 text-[10px] font-bold uppercase tracking-[0.16em] text-[#94a3b8]">
            Workspace
          </p>

          <div className="space-y-1.5">

            {navigation.map((item) => {
              const active =
                activeSection === item.id;

              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() =>
                    handleNavigation(item.id)
                  }
                  className={`relative flex w-full items-center gap-4 rounded-xl px-4 py-3.5 text-left text-sm font-medium transition ${
                    active
                      ? "bg-[#e9efff] text-[#2855d9]"
                      : "text-[#64748b] hover:bg-[#f5f7fb] hover:text-[#172554]"
                  }`}
                >

                  {active && (
                    <span className="absolute left-0 top-2 bottom-2 w-1 rounded-r-full bg-[#3157d5]" />
                  )}

                  <span
                    className={`flex h-8 w-8 items-center justify-center rounded-lg text-lg ${
                      active
                        ? "bg-[#dbe5ff] text-[#3157d5]"
                        : "text-[#64748b]"
                    }`}
                  >
                    {item.icon}
                  </span>

                  {item.label}

                </button>
              );
            })}

          </div>

          {/* =====================================================
    RECENT PROJECTS
===================================================== */}

<div className="mt-10 border-t border-[#e5eaf2] pt-7">

  <p className="mb-4 px-4 text-[10px] font-bold uppercase tracking-[0.16em] text-[#94a3b8]">
    Recent Projects
  </p>

  <div className="space-y-2 px-2">

    {/* WALKER REGION */}

    <button
      type="button"
      onClick={() => {
        setActiveSection("fault-map");

       // scrollToSection("fault-map");

        /*
         * Restore the main Walker Region analysis.
         * This uses the existing GeoDAWN search system.
         */
        window.dispatchEvent(
          new CustomEvent(
            "geodawn-location-search",
            {
              detail: {
                latitude: 39.1,
                longitude: -118.5,
                radiusKm: 10,
              },
            }
          )
        );
        setTimeout(() => {

      scrollToSection("fault-map");

    }, 100);
      }}
      className="group flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left transition hover:bg-[#f5f7fb]"
    >

      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#f1f4fa] text-sm text-[#64748b] transition group-hover:bg-[#e9efff] group-hover:text-[#3157d5]">
        ◷
      </span>

      <div className="min-w-0">

        <p className="truncate text-sm font-medium text-[#334155] group-hover:text-[#2855d9]">
          Walker Region
        </p>

        <p className="text-xs text-[#94a3b8]">
          GeoDAWN analysis
        </p>

      </div>

    </button>


    {/* MODEL V1_19 */}

    <button
      type="button"
      onClick={() => {
        setActiveSection("model");
        scrollToSection("model");
      }}
      className="group flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left transition hover:bg-[#f5f7fb]"
    >

      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#f1f4fa] text-sm text-[#64748b] transition group-hover:bg-[#e9efff] group-hover:text-[#3157d5]">
        ◷
      </span>

      <div className="min-w-0">

        <p className="truncate text-sm font-medium text-[#334155] group-hover:text-[#2855d9]">
          Model V1_19
        </p>

        <p className="text-xs text-[#94a3b8]">
          Production results
        </p>

      </div>

    </button>

  </div>

</div>
        </nav>

        {/* =====================================================
            BOTTOM PROFILE
        ===================================================== */}

        <div className="border-t border-[#e5eaf2] p-5">

          <div className="rounded-2xl bg-[#f2f6ff] p-4">

            <div className="flex items-center gap-3">

              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-[#3157d5] text-sm font-bold text-white">
                BH
              </div>

              <div>
                <p className="text-sm font-semibold text-[#172554]">
                  Researcher
                </p>

                <p className="text-xs text-[#64748b]">
                  GeoDAWN
                </p>
              </div>

            </div>

          </div>

        </div>

      </aside>

       

      {/* =====================================================
          MAIN CONTENT
      ===================================================== */}

      <div className="lg:ml-[298px]">

        {/* HEADER */}

        <header className="sticky top-0 z-30 border-b border-[#dfe5ef] bg-white/95 backdrop-blur">

          <div className="flex h-[82px] items-center justify-between px-5 md:px-8">

            <div>

              <div className="flex items-center gap-2">

                <span className="text-lg">
                  ◉
                </span>

                <h2 className="text-base font-semibold text-[#1e3a8a]">
                  Mapping the Earth's Energy Potential
                </h2>

              </div>

              <p className="mt-1 text-xs text-[#64748b]">
                AI-driven analysis of multi-spectral
                geophysical data for sustainable geothermal exploration
              </p>

            </div>

            {/* SEARCH */}

            <div className="hidden w-[360px] items-center rounded-xl border border-[#dbe2ef] bg-[#f8fafc] px-4 py-2.5 xl:flex">

              <span className="mr-3 text-lg text-[#64748b]">
                ⌕
              </span>

              <span className="text-sm text-[#94a3b8]">
                Search location, coordinates, or region...
              </span>

              <span className="ml-auto rounded-md border border-[#dbe2ef] bg-white px-2 py-1 text-[10px] font-semibold text-[#94a3b8]">
                ⌘ K
              </span>

            </div>

            {/* STATUS */}

            <div className="flex items-center gap-3">

              <div className="hidden items-center gap-2 rounded-full border border-[#c9f0dc] bg-[#effcf5] px-4 py-2 sm:flex">

                <span className="h-2.5 w-2.5 rounded-full bg-[#22c55e]" />

                <span className="text-xs font-semibold text-[#15803d]">
                  Model Ready
                </span>

              </div>

              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-[#3157d5] text-xs font-bold text-white">
                BH
              </div>

            </div>

          </div>

        </header>

        {/* =====================================================
            PAGE
        ===================================================== */}

        <section className="space-y-7 p-5 md:p-8">

          {/* PAGE TITLE */}

          <section
            id="overview"
            className="scroll-mt-28"
          >

            <div className="flex flex-col justify-between gap-5 md:flex-row md:items-end">

              <div>

                <h1 className="text-3xl font-bold tracking-tight text-[#10245c]">
                  Overview
                </h1>

                <p className="mt-2 text-sm text-[#64748b]">
                  Geothermal fault detection for a cleaner,
                  more sustainable tomorrow.
                </p>

              </div>

              <div className="flex items-center gap-5 rounded-2xl border border-[#dbe2ef] bg-white px-5 py-4 shadow-sm">

                <div>
                  <p className="text-[10px] font-bold uppercase tracking-wider text-[#94a3b8]">
                    Model
                  </p>

                  <p className="mt-1 text-sm font-bold text-[#1e3a8a]">
                    V1_19
                  </p>
                </div>

                <div className="h-8 w-px bg-[#e2e8f0]" />

                <div>
                  <p className="text-[10px] font-bold uppercase tracking-wider text-[#94a3b8]">
                    Architecture
                  </p>

                  <p className="mt-1 text-sm font-bold text-[#1e3a8a]">
                    U-Net
                  </p>
                </div>

              </div>

            </div>

          </section>

          {/* =====================================================
              METRICS
          ===================================================== */}

          <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">

            {[
              {
                label: "AI Candidates",
                value: "84,500",
                detail: "Final fault candidates",
                icon: "◎",
              },
              {
                label: "Validation DTI",
                value: "0.2438",
                detail: "Locked V1_19 result",
                icon: "≈",
              },
              {
                label: "Input Features",
                value: "19",
                detail: "Geophysical bands",
                icon: "▱",
              },
              {
                label: "Spatial Resolution",
                value: "100 m",
                detail: "Raster resolution",
                icon: "⌖",
              },
            ].map((item) => (

              <div
                key={item.label}
                className="rounded-2xl border border-[#dbe2ef] bg-white p-5 shadow-sm"
              >

                <div className="flex items-start justify-between">

                  <div>
                    <p className="text-xs font-semibold uppercase tracking-wide text-[#64748b]">
                      {item.label}
                    </p>

                    <p className="mt-3 text-3xl font-bold text-[#10245c]">
                      {item.value}
                    </p>

                    <p className="mt-1 text-xs text-[#94a3b8]">
                      {item.detail}
                    </p>
                  </div>

                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#eef2ff] text-lg text-[#3157d5]">
                    {item.icon}
                  </div>

                </div>

              </div>

            ))}

          </section>

          {/* =====================================================
              SEARCH
          ===================================================== */}

          <div className="[&>div]:!border-[#dbe2ef] [&>div]:!bg-white [&_input]:!border-[#dbe2ef] [&_input]:!bg-[#f8fafc] [&_input]:!text-[#172554] [&_select]:!border-[#dbe2ef] [&_select]:!bg-[#f8fafc] [&_select]:!text-[#172554]">
            <LocationSearch />
          </div>

          {/* =====================================================
              SEARCH RESULTS
          ===================================================== */}

          <div className="[&>section]:!border-[#dbe2ef] [&>section]:!bg-white [&_div]:!border-[#e2e8f0]">
            <SearchResults />
          </div>

          {/* =====================================================
              MAP
          ===================================================== */}

          <section
            id="fault-map"
            className="scroll-mt-28 overflow-hidden rounded-2xl border border-[#dbe2ef] bg-white shadow-sm"
          >

            <div className="flex flex-col gap-4 border-b border-[#e5eaf2] px-6 py-5 md:flex-row md:items-center md:justify-between">

              <div>

                <h2 className="text-xl font-bold text-[#10245c]">
                  Fault Candidate Map
                </h2>

                <p className="mt-1 text-sm text-[#64748b]">
                  AI-generated geothermal fault candidates
                  within the selected analysis area.
                </p>

              </div>

              <div className="flex items-center gap-4">

                <div className="flex items-center gap-2 text-xs font-medium text-[#64748b]">
                  <span className="h-2.5 w-2.5 rounded-full bg-[#ef5b63]" />
                  Nearby candidates
                </div>

                <div className="rounded-lg border border-[#dbe2ef] bg-[#f8fafc] px-3 py-2 text-xs font-medium text-[#475569]">
                  V1_19
                </div>

              </div>

            </div>

            <div className="h-[560px] w-full">
              <GeoMap />
            </div>

          </section>

          {/* =====================================================
              CANDIDATES
          ===================================================== */}

          <div
            id="candidates"
            className="scroll-mt-28 [&>section]:!border-[#dbe2ef] [&>section]:!bg-white [&_thead]:!bg-[#f8fafc] [&_tr]:!border-[#e5eaf2] [&_th]:!text-[#64748b] [&_td]:!text-[#475569]"
          >
            <CandidatesPanel />
          </div>

          {/* =====================================================
              GEOPHYSICAL LAYERS
          ===================================================== */}

          <div
            id="layers"
            className="scroll-mt-28 [&>section]:!border-[#dbe2ef] [&>section]:!bg-white"
          >
            <GeophysicalLayers />
          </div>

          {/* =====================================================
              MODEL
          ===================================================== */}

          <div
            id="model"
            className="scroll-mt-28 [&>section]:!border-[#dbe2ef] [&>section]:!bg-white"
          >
            <ModelInformation />
          </div>

          {/* =====================================================
              FINAL SCIENTIFIC NOTE
          ===================================================== */}

          <section className="rounded-2xl border border-[#dbe2ef] bg-white p-6 shadow-sm">

            <div className="flex gap-4">

              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[#eef2ff] text-[#3157d5]">
                i
              </div>

              <div>

                <h3 className="font-bold text-[#10245c]">
                  GeoDAWN Interpretation
                </h3>

                <p className="mt-2 max-w-5xl text-sm leading-6 text-[#64748b]">
                  GeoDAWN identifies locations that the V1_19
                  model considers geothermal fault candidates
                  using 19 geophysical input bands. These
                  candidates are intended for geological
                  investigation and are not confirmed geological
                  faults.
                </p>

              </div>

            </div>

          </section>

          {/* FOOTER */}

          <footer className="flex flex-col justify-between gap-3 border-t border-[#dbe2ef] pt-5 text-xs text-[#64748b] md:flex-row">

            <p>
              GeoDAWN • AI-assisted geothermal exploration
            </p>

            <div className="flex gap-5">
              <span>Data: USGS</span>
              <span>Map: Satellite imagery</span>
              <span>Model: V1_19</span>
            </div>

          </footer>

        </section>

      </div>

    </main>
  );
}