"""Run configuration.

A run is described by shared settings (event, bathymetry, SWOT data, candidate
grid, tracing, misfit, output) plus a LIST of wavefronts to back-project.  The
``Config`` dataclass below holds the shared settings as flat attributes (the
names the engine/diagnostics/plotting modules read); ``WavefrontSpec`` describes
one wavefront.  ``load_config`` builds both from a YAML file so a run is defined
by a config file, not by editing Python.

If ``Config.wavefronts`` is empty, the CLI synthesises a single wavefront from
the legacy ``wf_path`` / ``wavelength`` / ``n_wf_points`` fields, which keeps the
no-config / single-wavefront command line behaving exactly as before.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields


@dataclass
class WavefrontSpec:
    """One observed wavefront to back-project independently."""
    name: str                          # short label, used in output filenames
    path: str                          # GeoJSON polyline (LineString/MultiLineString)
    wavelength: float | None = None    # deep-water wavelength (m); None=shallow-water
    n_points: int | None = None        # resample polyline to this many points
    local_wavelength: float | None = None  # in-situ (band-passed) wavelength (m)
    local_depth: float | None = None       # depth (m) where local_wavelength was measured


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
    # Shared reference depth (m) for the in-situ `local_wavelength` option, used
    # when a wavefront gives local_wavelength but no per-front local_depth.  None
    # -> derive it per wavefront as the median bathymetric depth at its points.
    local_depth: float | None = None

    # --- misfit ---------------------------------------------------------
    coverage_frac: float = 0.8         # min fraction of WF1 pts a cell must see

    # Misfit colour-scale clip (minutes).  Colours saturate at this value so the
    # low-misfit valley uses the full dynamic range; cells above it are clamped
    # to the end colour (colorbar drawn with an over-range arrow).  None -> auto.
    misfit_vmax: float | None = 10.0

    # Predicted-vs-digitised wavefront-fit figure: for each best-fit source trace
    # forward and draw the predicted wavefront (isochron) over the digitised
    # polyline, points coloured by travel-time residual.  A spatial goodness-of-
    # fit; adds a couple of cheap forward traces per wavefront.  <tag>_wffit.png.
    wavefront_fit: bool = True

    # --- emission-time search (a SEPARATE hypothesis from rupture-anchoring) ---
    # Opt-in.  Treat the front as radiating from a source S at a single emission
    # time tau = origin + {0, step, ..., max} minutes (tau >= 0), IGNORING the
    # rupture delay, and score candidates by rms_j( tau + T_j(S) - t_j ).  Finds
    # the (location, emission-time) that best explains the front.  Runs IN
    # ADDITION to the anchored/free maps; a cheap post-process over the stack.
    time_search: bool = False
    time_step_min: float = 5.0         # tau grid spacing (minutes)
    time_max_min: float = 60.0         # maximum tau (minutes after origin)

    # --- output ---------------------------------------------------------
    out_dir: str = "/Users/dmelgarm/code/python/Kamchatka2025/backprojection"
    tag: str = "wf1_lowpass"           # filename stem

    # --- wavefronts -----------------------------------------------------
    # The wavefronts to back-project (each independently).  Empty -> the CLI
    # synthesises one from wf_path / wavelength / n_wf_points (single-WF mode).
    wavefronts: list[WavefrontSpec] = field(default_factory=list)


# ======================================================================
#  YAML loader
# ======================================================================
# Maps the nested YAML groups onto the flat Config attribute names.  Keeping
# Config flat means engine/diagnostics/plotting need no changes.
#   yaml group : {yaml key -> Config attribute}
_YAML_MAP = {
    "event": {"epi_lon": "epi_lon", "epi_lat": "epi_lat",
              "origin_time_utc": "origin_time_utc",
              "rupture_speed_kms": "rupture_speed_kms"},
    "bathymetry": {"path": "bathy_path", "negate": "bathy_negate",
                   "domain_lon": "domain_lon", "domain_lat": "domain_lat"},
    "swot": {"times": "swot_times_path", "ssh": "swot_ssh_path",
             "known_arrival_min": "KNOWN_ARRIVAL_MIN"},
    "candidate": {"lon": "cand_lon", "lat": "cand_lat",
                  "dlon": "cand_dlon", "dlat": "cand_dlat"},
    "tracing": {"dt": "dt", "max_time": "max_time", "bin_deg": "bin_deg",
                "fan_halfwidth_deg": "fan_halfwidth_deg",
                "azimuth_step_deg": "azimuth_step_deg",
                "local_depth": "local_depth"},
    "misfit": {"coverage_frac": "coverage_frac"},
    "plot": {"misfit_vmax": "misfit_vmax", "wavefront_fit": "wavefront_fit"},
    "time_search": {"enabled": "time_search", "step_min": "time_step_min",
                    "max_min": "time_max_min"},
    "output": {"out_dir": "out_dir", "tag": "tag"},
}

# Config attributes that must be tuples (YAML gives lists).
_TUPLE_ATTRS = {"domain_lon", "domain_lat", "cand_lon", "cand_lat"}


def load_config(path):
    """Build a Config (with its wavefronts list) from a YAML file.

    Unspecified settings fall back to the Config defaults, so a config file only
    needs to state what differs.  See ``configs/`` for an example."""
    import yaml

    with open(path) as fh:
        doc = yaml.safe_load(fh) or {}

    valid = {f.name for f in fields(Config)}
    kwargs = {}
    for group, mapping in _YAML_MAP.items():
        section = doc.get(group, {}) or {}
        for ykey, attr in mapping.items():
            if ykey in section:
                val = section[ykey]
                if attr in _TUPLE_ATTRS and val is not None:
                    val = tuple(val)
                if attr == "origin_time_utc" and val is not None:
                    val = str(val)         # YAML may parse a bare timestamp
                kwargs[attr] = val

    specs = []
    for i, wf in enumerate(doc.get("wavefronts", []) or []):
        specs.append(WavefrontSpec(
            name=str(wf.get("name", f"WF{i + 1}")),
            path=wf["path"],
            wavelength=wf.get("wavelength"),
            n_points=wf.get("n_points"),
            local_wavelength=wf.get("local_wavelength"),
            local_depth=wf.get("local_depth"),
        ))
    kwargs["wavefronts"] = specs

    unknown = set(kwargs) - valid
    assert not unknown, f"internal: unmapped Config attrs {unknown}"
    return Config(**kwargs)
