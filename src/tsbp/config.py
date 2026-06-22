"""Run configuration.

For now this is the same dataclass the original single-file script used, moved
verbatim so the mechanical split is behaviour-preserving.  The next phase
replaces the hard-coded defaults with a YAML loader and a per-wavefront spec
list (see the project plan); nothing about the defaults changes here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # --- inputs ---------------------------------------------------------
    wf_path: str = "/Users/dmelgarm/Kamchatka2025/QGIS/WF1.geojson"
    bathy_path: str = "/Users/dmelgarm/DEMs/ETOPO2/ETOPO2v2c_f4.nc"
    bathy_negate: bool = True          # ETOPO/GEBCO: ocean negative -> negate

    # The bathymetry DEMs are global; trace_rays and grid_travel_times work on
    # whatever extent we hand them (and grid_travel_times tiles its output bins
    # over the WHOLE extent), so we subset to a domain that comfortably brackets
    # both WF1 and the candidate-source region before doing anything.
    domain_lon: tuple[float, float] = (150.0, 168.0)
    domain_lat: tuple[float, float] = (41.0, 55.0)

    # --- event (the 2025-07-30 M8.8 Kamchatka earthquake) ---------------
    # CONFIRM THESE against your own catalogue before trusting the numbers.
    # USGS hypocentre / origin used as the validation epicentre.
    epi_lon: float = 160.396           # deg E
    epi_lat: float = 52.473            # deg N
    origin_time_utc: str = "2025-07-30T23:24:52"   # ISO-8601, UTC

    # Rupture speed (km/s).  Adds the time for the rupture front to travel from
    # the epicentre (nucleation) to each candidate source before its tsunami is
    # launched: t_rup(S) = dist(epi, S) / v_rup.  Affects the ANCHORED map only.
    # None or 0 -> no rupture-delay term (instantaneous point-source assumption).
    rupture_speed_kms: float | None = 2.2

    # WF1 arrival time after origin.  Two modes:
    #  * swot_times_path set -> PER-PIXEL anchor: each WF1 point is matched to the
    #    nearest SWOT pixel and uses that pixel's observation time (SWOT images
    #    each pixel at a different time along its polar orbit).  This is the real
    #    case and supersedes the scalar below.
    #  * swot_times_path None -> uniform scalar anchor KNOWN_ARRIVAL_MIN (minutes).
    swot_times_path: str | None = "/Users/dmelgarm/code/GMT/Kamchatka2025/synthetic_swot_hhres_v2.csv"
    KNOWN_ARRIVAL_MIN: float = 69.5    # fallback uniform arrival (minutes)

    # Actual SWOT sea-surface-height anomaly (lon lat ssh, metres), shown in the
    # data panel next to each misfit map so the back-projected wavefield is
    # visible alongside the inferred source.  None -> data panel shows WF1 only.
    swot_ssh_path: str | None = "/Users/dmelgarm/Kamchatka2025/swot/filtered_swot_data.txt"

    # Resample the hand-digitised WF1 polyline to this many evenly spaced points
    # along its arc length.  None -> use the raw digitised vertices as-is.
    n_wf_points: int | None = 100

    # --- candidate-source grid -----------------------------------------
    # Wide enough to contain the whole constant-travel-time valley (a single
    # far-field wavefront constrains an ARC, not a point), so it is not clipped.
    cand_lon: tuple[float, float] = (153.0, 164.0)
    cand_lat: tuple[float, float] = (47.0, 55.0)
    cand_dlon: float = 0.1             # spacing in deg
    cand_dlat: float = 0.1

    # --- ray fan --------------------------------------------------------
    # For each WF1 point we fan rays about the bearing from that point toward
    # the candidate-region centroid.  Wide + dense enough that every candidate
    # cell is reached from every WF1 point.
    fan_halfwidth_deg: float = 45.0    # +/- about the centre bearing
    azimuth_step_deg: float = 0.05      # ray spacing in the fan

    # --- ray tracing ----------------------------------------------------
    dt: float = 5.0                   # integration step (s)
    max_time: float = 120 * 60         # max integration time (s)
    # grid_travel_times gridding resolution.  Keep this COARSER than the
    # candidate grid: the binned-min first-arrival estimator is noisy when bins
    # hold few rays (esp. at fan peripheries like the far south), and that noise
    # shows up as ragged misfit contours.  Coarser bins average more rays into a
    # stable first-arrival; bilinear sampling then restores candidate-grid
    # detail.  ~0.12 deg halves the southern high-frequency noise vs 0.05.
    bin_deg: float = 0.2
    # Dispersive wavelength (m), deep-water.  None -> shallow-water sqrt(g*h).
    # Plumbed through to trace_rays so this same engine can be reused on
    # short-wavelength / dispersive fronts later.
    wavelength: float | None = None

    # --- misfit ---------------------------------------------------------
    coverage_frac: float = 0.8         # min fraction of WF1 pts a cell must see

    # Misfit colour-scale clip (minutes).  Colours saturate at this value so the
    # low-misfit valley uses the full dynamic range; cells above it are clamped
    # to the end colour (colorbar drawn with an over-range arrow).  None -> auto.
    misfit_vmax: float | None = 10.0

    # --- output ---------------------------------------------------------
    out_dir: str = "/Users/dmelgarm/code/python/Kamchatka2025/backprojection"
    tag: str = "wf1_lowpass"           # filename stem
