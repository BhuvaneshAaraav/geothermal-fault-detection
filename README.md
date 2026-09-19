# GeoDAWN — AI-Based Geothermal Fault Candidate Mapping

## Overview

GeoDAWN is an AI-based geospatial system designed to identify and rank
candidate locations that may correspond to geological fault structures
associated with geothermal resources.

The system uses 19 geophysical raster features and a convolutional neural
network to generate a spatial fault-probability map. A spatial post-processing
stage then produces geographically separated candidate locations.

The system is intended as a decision-support and exploration tool.
AI predictions do not constitute geological confirmation.

---

## Problem

Geothermal exploration requires identifying geological structures that may
allow heat and fluids to move through the subsurface.

Faults and fracture zones can be important indicators because they may provide
pathways for geothermal fluids.

Traditional exploration can require substantial geological and geophysical
analysis.

GeoDAWN investigates whether multiple geophysical measurements can be combined
with deep learning to automatically identify spatial patterns associated with
known fault structures.

---

## Input Data

The model uses 19 geophysical raster bands.

Examples include:

- Magnetic anomaly
- Reduced-to-pole magnetic data
- Total magnetic intensity
- Gravity-derived features
- Geodetic second invariant
- Isostatic gravity anomaly slope
- Terrain correction
- Shear-rate related features
- Dilatation-rate related features
- Vertical-gradient magnetic features

Raster dimensions:

3730 × 3292 pixels

Coordinate reference system:

EPSG:32611

Spatial resolution:

100 m × 100 m

---

## Data Preprocessing

The preprocessing pipeline performs:

1. Raster loading
2. Invalid/nodata detection
3. Feature statistics calculation
4. Feature normalization
5. Spatial train/validation separation
6. Positive/negative sample balancing
7. 31 × 31 spatial patch extraction

Extreme raster nodata values such as:

-3.40282e+38

are treated as invalid values rather than normal numerical measurements.

---

## Model

The main model is a convolutional U-Net architecture.

### Input

19 geophysical channels

### Architecture

```text
19 channels
     ↓
Conv Block
     ↓
32 channels
     ↓
Conv Block
     ↓
64 channels
     ↓
Conv Block
     ↓
128 channels
     ↓
Bottleneck
     ↓
256 channels
     ↓
Decoder
     ↓
128
     ↓
64
     ↓
32
     ↓
1-channel output