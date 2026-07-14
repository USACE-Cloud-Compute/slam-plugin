import os
import sys
import shutil
import glob
import logging
import datetime as dt

import numpy as np

from cc.plugin_manager import PluginManager, DataSourceOpInput
from cc.datastore_s3 import S3DataStore

logger = logging.getLogger(__name__)

# Ensure the SLAM-SIGSIM submodule scripts are importable
# This allows the worker scripts to `from slam_functions import ...`
SLAM_SUBMODULE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "lib", "slam-sigsim", "sub_python"
)
if SLAM_SUBMODULE_PATH not in sys.path:
    sys.path.insert(0, SLAM_SUBMODULE_PATH)

# Public NOAA AORC v1.1 (AWS Open Data, anonymous). Zarr, one <year>.zarr per year.
PUBLIC_AORC_BUCKET = "noaa-nws-aorc-v1-1-1km"


# ---------------------------------------------------------------------------
# Datastore helpers
#
# Every action carries its OWN stores/inputs/outputs (the published cc_py_sdk
# v1.1.0 resolves action IO against action-private stores/datasources with no
# payload-level fallback), so all transfers below go through the SDK's built-in
# io manager on the Action object:
#
#   action.copy_file_to_local(DataSourceOpInput(name, pathkey, datakey), path)
#   action.copy_file_to_remote(DataSourceOpInput(name, pathkey, datakey), path)
#   action.copy_folder_to_remote(DataSourceOpInput(name, pathkey, datakey), dir)
#
# DataSourceOpInput selects a datasource by name and a single named path key,
# so a datasource that declares several named paths (an ESRI shapefile's
# shp/shx/dbf/prj) is fetched/staged one path key at a time.
#
# Two SDK gaps this module works around:
#   1. PluginManager.connect() only wires a live boto3 _session onto PAYLOAD
#      stores, never action stores -> _connect_action_stores() connects them so
#      the io-manager methods (which call store._session) work.
#   2. The SDK ships put_folder / copy_folder_to_remote but NO folder DOWNLOAD
#      (get_dir only lists; get_object is single-key) -> _get_folder() pages the
#      prefix on the connected session. It also uses action._iomgr.get_store(),
#      NOT action.get_store(), because Action.get_store() is a no-arg bug in the
#      SDK that returns the bound method instead of the store.
#
# Path-key convention (kept in lockstep with slam-compute-manifest.json):
#   single geojson file -> "geo-json"
#   ESRI shapefile      -> "shp","shx","dbf","prj"   (.prj REQUIRED: SLAM's
#                          shapefile_to_mask raises on a null CRS)
#   directory of files  -> "directory"
# ---------------------------------------------------------------------------
SHP_COMPONENTS = ("shp", "shx", "dbf", "prj")


def _connect_action_stores(action):
    """Connect a live S3 session onto every action-scoped store. PluginManager
    only connects payload-level stores, so without this the built-in io-manager
    methods would hit an unset store._session on action stores.

    Tolerant by design: a declared store may not be exercised by a given run
    (e.g. the AORC store is unused when aorc_to_daily_nc has source='public'),
    so a connect failure (missing creds env for that profile) is logged and
    skipped rather than aborting the action. A store that IS needed but failed
    to connect surfaces at first use."""
    for store in action._iomgr.stores or []:
        if getattr(store, "_session", None) is None:
            try:
                session = S3DataStore()
                session.connect(store)
                store._session = session
            except Exception as e:
                logger.warning(
                    f"store '{store.name}' (profile {store.profile}) not connected: {e}"
                )


def _get_file(action, ds_name, localpath, pathkey="geo-json"):
    """Download a single input object (one named path key) to localpath."""
    os.makedirs(os.path.dirname(localpath) or ".", exist_ok=True)
    action.copy_file_to_local(DataSourceOpInput(ds_name, pathkey, None), localpath)
    logger.info(f"downloaded input '{ds_name}'[{pathkey}] -> {localpath}")
    return localpath


def _get_shapefile(action, ds_name, localdir, stem="WS", components=SHP_COMPONENTS):
    """Download an ESRI shapefile datasource, one sidecar per named path key, to
    localdir/<stem>.<ext>. The local stem is fixed (SLAM hard-codes 'WS.shp'),
    independent of the remote object names."""
    os.makedirs(localdir, exist_ok=True)
    out = {}
    for ext in components:
        dest = os.path.join(localdir, f"{stem}.{ext}")
        action.copy_file_to_local(DataSourceOpInput(ds_name, ext, None), dest)
        out[ext] = dest
    logger.info(
        f"downloaded shapefile '{ds_name}' -> {localdir}/{stem}.{{{','.join(components)}}}"
    )
    return out


def _get_folder(action, ds_name, localdir, pathkey="directory"):
    """Download every object under a directory datasource prefix into localdir
    (flattened, preserving the key suffix below the prefix). The SDK has no
    folder download, so page the prefix on the connected store session."""
    ds = action.get_input_data_source(ds_name)
    if ds is None:
        raise ValueError(f"input data source '{ds_name}' not defined on action")
    _connect_action_stores(action)
    store = action._iomgr.get_store(ds.store_name)
    if store is None:
        raise ValueError(f"store '{ds.store_name}' not found on this action")
    prefix = store.full_path(ds.paths[pathkey]).removeprefix("/")
    if not prefix.endswith("/"):
        prefix += "/"
    fs = store._session.filestore
    os.makedirs(localdir, exist_ok=True)
    paginator = fs.client.get_paginator("list_objects_v2")
    files = []
    for page in paginator.paginate(Bucket=fs.bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix) :].lstrip("/")
            dest = os.path.join(localdir, rel)
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            reader = fs.get_object(key)
            with open(dest, "wb") as f:
                shutil.copyfileobj(reader, f)
            files.append(dest)
    logger.info(f"downloaded {len(files)} object(s) from {prefix} -> {localdir}")
    return files


def _get_first_from_folder(action, ds_name, localpath, pathkey="directory"):
    """Download the first object under a directory datasource prefix to
    localpath. Used when only a single sample file is needed (e.g. a precip
    grid reference), avoiding a full folder pull."""
    ds = action.get_input_data_source(ds_name)
    if ds is None:
        raise ValueError(f"input data source '{ds_name}' not defined on action")
    _connect_action_stores(action)
    store = action._iomgr.get_store(ds.store_name)
    if store is None:
        raise ValueError(f"store '{ds.store_name}' not found on this action")
    prefix = store.full_path(ds.paths[pathkey]).removeprefix("/")
    if not prefix.endswith("/"):
        prefix += "/"
    fs = store._session.filestore
    os.makedirs(os.path.dirname(localpath) or ".", exist_ok=True)
    paginator = fs.client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=fs.bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            reader = fs.get_object(key)
            with open(localpath, "wb") as f:
                shutil.copyfileobj(reader, f)
            logger.info(f"downloaded sample object {key} -> {localpath}")
            return localpath
    raise ValueError(f"no objects found under '{ds_name}' prefix {prefix}")


def _put_folder(action, ds_name, localdir, pathkey="directory"):
    """Upload the contents of localdir under an output directory datasource."""
    action.copy_folder_to_remote(DataSourceOpInput(ds_name, pathkey, None), localdir)
    logger.info(f"uploaded {localdir} -> output '{ds_name}'[{pathkey}]")


def _put_shapefile(action, ds_name, localdir, stem, components=SHP_COMPONENTS):
    """Upload an ESRI shapefile, one produced sidecar per named output path key."""
    for ext in components:
        src = os.path.join(localdir, f"{stem}.{ext}")
        if not os.path.exists(src):
            logger.warning(f"{ds_name}: expected sidecar {src} not found; skipping")
            continue
        action.copy_file_to_remote(DataSourceOpInput(ds_name, ext, None), src)
    logger.info(f"uploaded shapefile {localdir}/{stem}.* -> output '{ds_name}'")


# ---------------------------------------------------------------------------
# Pure cores (no cc plumbing) so the heavy logic is independently testable.
# ---------------------------------------------------------------------------
def zarr_to_daily_netcdf(
    source,
    bbox,
    start_date,
    end_date,
    precvar,
    outdir,
    resample_km=None,
    resample_agg="mean",
    aorc_bucket=None,
    aorc_prefix="",
    aorc_endpoint=None,
    aorc_key=None,
    aorc_secret=None,
):
    """Read an AORC Zarr record (public NOAA bucket or an internal mirror),
    clip to bbox (minx, miny, maxx, maxy in EPSG:4326), optionally resample the
    ~800 m native grid to a coarser resolution, and write one NetCDF per calendar
    day into outdir. Returns the list of written files.

    source: "public" -> anonymous s3://noaa-nws-aorc-v1-1-1km/<year>.zarr
            "mirror" -> s3://<aorc_bucket>/<aorc_prefix>/<year>.zarr on aorc_endpoint

    resample_km: target grid resolution in km (e.g. 4 or 16). None/0 -> keep the
        native grid. The block factor is derived from the actual grid spacing and
        the grid is coarsened by an integer factor (boundary trimmed).
    resample_agg: block reducer, "mean" (default; areal-average depth), "sum", or "max".
    """
    import s3fs
    import xarray as xr

    source = (source or "mirror").lower()
    if source == "public":
        fs = s3fs.S3FileSystem(anon=True)
        base = PUBLIC_AORC_BUCKET
    elif source == "mirror":
        fs = s3fs.S3FileSystem(
            key=aorc_key,
            secret=aorc_secret,
            client_kwargs={"endpoint_url": aorc_endpoint},
        )
        base = aorc_bucket + (("/" + aorc_prefix.strip("/")) if aorc_prefix else "")
    else:
        raise ValueError(
            f"unknown AORC source '{source}' (expected 'public' or 'mirror')"
        )

    minx, miny, maxx, maxy = bbox
    sd = dt.date.fromisoformat(str(start_date))
    ed = dt.date.fromisoformat(str(end_date))
    os.makedirs(outdir, exist_ok=True)

    written = []
    for year in range(sd.year, ed.year + 1):
        zuri = f"{base}/{year}.zarr"
        logger.info(f"opening AORC zarr {source}: s3://{zuri}")
        mapper = fs.get_mapper(zuri)
        try:
            ds = xr.open_zarr(mapper, consolidated=True)
        except Exception:
            ds = xr.open_zarr(mapper, consolidated=False)

        if precvar not in ds:
            raise KeyError(
                f"variable '{precvar}' not in {zuri}; have {list(ds.data_vars)}"
            )

        lats = ds["latitude"].values
        lons = ds["longitude"].values
        lat_idx = np.where((lats >= miny) & (lats <= maxy))[0]
        lon_idx = np.where((lons >= minx) & (lons <= maxx))[0]
        if lat_idx.size == 0 or lon_idx.size == 0:
            logger.warning(f"{year}: bbox does not intersect the grid; skipping")
            ds.close()
            continue

        sub = ds[[precvar]].isel(
            latitude=slice(int(lat_idx.min()), int(lat_idx.max()) + 1),
            longitude=slice(int(lon_idx.min()), int(lon_idx.max()) + 1),
        )

        # Optional spatial resample factor. Derived from the coordinate spacing
        # only; the coarsen itself is applied per-day AFTER load (below) so we
        # never materialize the full-year lazy array through CF decoding.
        factor = 1
        agg = resample_agg if resample_agg in ("mean", "sum", "max") else "mean"
        if resample_km:
            native_deg = float(np.abs(np.diff(sub["latitude"].values)).mean())
            native_km = native_deg * 111.32
            factor = max(1, int(round(float(resample_km) / native_km)))
            logger.info(
                f"resample target ~{resample_km} km: native ~{native_km:.2f} km, "
                f"coarsen factor x{factor} (agg={agg})"
            )

        day = max(sd, dt.date(year, 1, 1))
        last = min(ed, dt.date(year, 12, 31))
        while day <= last:
            d0 = f"{day.isoformat()}T00:00:00"
            d1 = f"{day.isoformat()}T23:59:59"
            day_da = sub[precvar].sel(time=slice(d0, d1))
            if day_da.time.size > 0:
                day_loaded = day_da.load()
                if factor > 1:
                    day_loaded = getattr(
                        day_loaded.coarsen(
                            latitude=factor, longitude=factor, boundary="trim"
                        ),
                        agg,
                    )()
                out = day_loaded.to_dataset(name=precvar)
                fname = os.path.join(outdir, f"AORC.{day.strftime('%Y%m%d')}.nc")
                out.to_netcdf(
                    fname,
                    encoding={precvar: {"zlib": True, "complevel": 4}},
                )
                out.close()
                written.append(fname)
                logger.info(f"wrote {fname} ({day_da.time.size} steps)")
            day += dt.timedelta(days=1)
        ds.close()

    if not written:
        raise RuntimeError(
            "no daily NetCDF files were written (empty date range or bbox miss)"
        )
    return written


def geojson_to_shapefile(geojson_path, out_shp, target_crs="EPSG:4326"):
    """Convert a geojson to an ESRI shapefile (WS.shp + sidecars) so downstream
    SLAM stages that read WS.shp can consume it. Returns the sidecar file list."""
    import geopandas as gpd

    gdf = gpd.read_file(geojson_path)
    if target_crs:
        if gdf.crs is None:
            gdf = gdf.set_crs(target_crs)
        else:
            gdf = gdf.to_crs(target_crs)
    os.makedirs(os.path.dirname(out_shp) or ".", exist_ok=True)
    gdf.to_file(out_shp, driver="ESRI Shapefile")
    stem = os.path.splitext(out_shp)[0]
    sidecars = sorted(glob.glob(stem + ".*"))
    logger.info(
        f"wrote shapefile {out_shp} with sidecars {[os.path.basename(s) for s in sidecars]}"
    )
    return sidecars


# ---------------------------------------------------------------------------
# New actions
# ---------------------------------------------------------------------------
def run_aorc_to_daily_nc(action):
    """Action: convert an AORC Zarr dataset (public NOAA data or an internal
    mirror) to daily NetCDF clipped to a geojson search filter, then stage the
    daily files to the destination datalocation for PP2WAP.

    Attributes:
      source        "mirror" (default) | "public"
      aorc_store    name of the declared AORC store for source="mirror" (default
                    "AORC"); its profile picks the bucket/endpoint/creds env and
                    its root is the cache prefix. Env AORC_* is used if absent.
      precvar       precip variable name (default APCP_surface)
      start_date    ISO date, first day to convert (inclusive)
      end_date      ISO date, last day to convert (inclusive)
      buffer_deg    degrees to expand the search-filter bbox (default 1.0)
      resample_km   coarsen the ~800 m grid to this resolution in km (e.g. 4 or 16);
                    omit / 0 to keep native
      resample_agg  block reducer for resampling: "mean" (default), "sum", "max"
    Input datasource:  search_filter  (a geojson defining the clip region)
    Output datasource: daily_netcdf   (destination prefix for AORC.YYYYMMDD.nc)
    Store (mirror):    AORC            (storm-cloud bucket, root = aorc-cache-conus)
    """
    import geopandas as gpd

    logger.info("ACTION: aorc_to_daily_nc")
    a = action.attributes
    source = str(a.get("source", "mirror")).lower()
    precvar = str(a.get("precvar", "APCP_surface"))
    buffer_deg = float(a.get("buffer_deg", 1.0))
    resample_km = a.get("resample_km")
    resample_agg = str(a.get("resample_agg", "mean"))
    start_date = a.get("start_date")
    end_date = a.get("end_date")
    if not start_date or not end_date:
        raise ValueError(
            "aorc_to_daily_nc requires 'start_date' and 'end_date' attributes"
        )

    _connect_action_stores(action)

    work = "/data/aorc"
    os.makedirs(work, exist_ok=True)
    filter_path = os.path.join(work, "search-filter.geojson")
    _get_file(action, "search_filter", filter_path, pathkey="geo-json")

    gdf = gpd.read_file(filter_path).to_crs(4326)
    minx, miny, maxx, maxy = gdf.total_bounds
    bbox = (minx - buffer_deg, miny - buffer_deg, maxx + buffer_deg, maxy + buffer_deg)
    logger.info(f"search-filter bbox (+{buffer_deg} deg): {bbox}")

    # The AORC mirror lives in its own store/bucket (NOT the FFRD model store).
    # Read the location from the declared AORC store: bucket comes from its
    # profile's <PROFILE>_AWS_S3_BUCKET, and the store root is the cache prefix.
    # Fall back to the AORC_* env convention if no store is declared. (source
    # "public" ignores these and reads the anonymous NOAA bucket.)
    aorc_store_name = str(a.get("aorc_store", "AORC"))
    store = action._iomgr.get_store(aorc_store_name)
    if store is not None:
        prof = store.profile
        aorc_bucket = os.environ.get(f"{prof}_AWS_S3_BUCKET")
        aorc_prefix = (store.params or {}).get("root", "")
        aorc_endpoint = os.environ.get(f"{prof}_AWS_ENDPOINT")
        aorc_key = os.environ.get(f"{prof}_AWS_ACCESS_KEY_ID")
        aorc_secret = os.environ.get(f"{prof}_AWS_SECRET_ACCESS_KEY")
        logger.info(
            f"AORC mirror store '{aorc_store_name}' (profile {prof}): "
            f"bucket={aorc_bucket} root={aorc_prefix}"
        )
    else:
        aorc_bucket = os.environ.get("AORC_AWS_S3_BUCKET")
        aorc_prefix = os.environ.get("AORC_S3_PREFIX", "")
        aorc_endpoint = os.environ.get("AORC_AWS_ENDPOINT")
        aorc_key = os.environ.get("AORC_AWS_ACCESS_KEY_ID")
        aorc_secret = os.environ.get("AORC_AWS_SECRET_ACCESS_KEY")

    outdir = "/data/precip-out"
    zarr_to_daily_netcdf(
        source=source,
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        precvar=precvar,
        outdir=outdir,
        resample_km=resample_km,
        resample_agg=resample_agg,
        aorc_bucket=aorc_bucket,
        aorc_prefix=aorc_prefix,
        aorc_endpoint=aorc_endpoint,
        aorc_key=aorc_key,
        aorc_secret=aorc_secret,
    )
    _put_folder(action, "daily_netcdf", outdir)


def run_geojson_to_shp(action):
    """Action (optional): convert a watershed geojson to an ESRI shapefile and
    stage WS.{shp,shx,dbf,prj} to the destination datalocation read as
    'watershed_shapefile' by PP2WAP / LMC / CLMPV.

    Attributes:
      shp_name    local shapefile stem written before upload (default WS); the
                  remote object names come from the output datasource path keys.
      target_crs  CRS to write (default EPSG:4326, to match the AORC grid)
    Input datasource:  watershed  (path key 'geo-json')
    Output datasource: watershed  (path keys 'shp','shx','dbf','prj')
    """
    logger.info("ACTION: geojson_to_shp")
    _connect_action_stores(action)
    a = action.attributes
    stem = str(a.get("shp_name", "WS"))
    target_crs = a.get("target_crs", "EPSG:4326")

    work = "/data/geojson2shp"
    shpdir = os.path.join(work, "shp")
    os.makedirs(shpdir, exist_ok=True)
    gj = os.path.join(work, "input.geojson")
    _get_file(action, "watershed", gj, pathkey="geo-json")

    geojson_to_shapefile(gj, os.path.join(shpdir, f"{stem}.shp"), target_crs=target_crs)
    _put_shapefile(action, "watershed", shpdir, stem)


# ---------------------------------------------------------------------------
# SLAM pipeline actions (folder-aware watershed + precip staging)
# ---------------------------------------------------------------------------
def run_pp2wap(action):
    logger.info("STAGE 1: PP2WAP")
    attrs = action.attributes

    _connect_action_stores(action)
    os.makedirs("/data", exist_ok=True)
    # daily precip NetCDFs (produced by aorc_to_daily_nc) + watershed sidecars,
    # both staged into the working dir where 1.PP2WAP.py globs '*.nc' and reads WS.shp.
    _get_folder(action, "precipitation", "/data")
    _get_shapefile(action, "watershed_shapefile", "/data", stem="WS")

    script = os.path.join(SLAM_SUBMODULE_PATH, "1.PP2WAP.py")
    cmd = [
        sys.executable,
        script,
        str(attrs.get("precvar", "precrate")),
        str(attrs.get("lon_name", "longitude")),
        str(attrs.get("lat_name", "latitude")),
        str(attrs.get("tpd", 24)),
        str(attrs.get("output_format", "")),
    ]
    result = subprocess_run(cmd, cwd="/data")
    if result != 0:
        raise RuntimeError(f"PP2WAP failed with return code {result}")

    _stage_dir_to_remote(
        action,
        "wap_output",
        "/data",
        glob.glob("/data/WAP.*.nc4") + glob.glob("/data/WS.*.PP2WAP.nc"),
    )


def run_amc(action):
    logger.info("STAGE 2: AMC")
    attrs = action.attributes

    _connect_action_stores(action)
    os.makedirs("/data/wap", exist_ok=True)
    _get_folder(action, "wap_input", "/data/wap")

    script = os.path.join(SLAM_SUBMODULE_PATH, "2.AMC.py")
    cmd = [
        sys.executable,
        script,
        str(attrs.get("storm_duration", 24)),
        str(attrs.get("year", "")),
        str(attrs.get("season_start", "0101")),
        str(attrs.get("season_end", "1231")),
        str(attrs.get("am_key", "")),
        str(attrs.get("out_key", "")),
    ]
    result = subprocess_run(cmd, cwd="/data/wap")
    if result != 0:
        raise RuntimeError(f"AMC failed with return code {result}")

    # AMC (2.AMC.py) writes "<amkey>Maximum.<year>.<outkey>.nc4", but LMC
    # (3.LMC.py:54) and the LMCol->LMs combine both glob "*Maxima.*.nc4" --
    # AMC's "Maximum" is the lone naming inconsistency in the SLAM chain.
    # Rename before staging so every downstream consumer finds the file.
    am_files = []
    for f in glob.glob("/data/wap/*Maximum*.nc4"):
        renamed = f.replace("Maximum", "Maxima")
        os.rename(f, renamed)
        am_files.append(renamed)
    _stage_dir_to_remote(action, "am_output", "/data/wap", am_files)


def run_lmc(action):
    logger.info("STAGE 3: LMC")
    attrs = action.attributes

    _connect_action_stores(action)
    os.makedirs("/data/lmc", exist_ok=True)
    _get_folder(action, "annual_maxima", "/data/lmc")
    # Raw precip must land in the LMC cwd (not a subdir): 3.LMC.py globs
    # "<precip_prefix>*<precip_suffix>" relative to cwd (lines 51, 96).
    _get_folder(action, "raw_precipitation", "/data/lmc")
    _get_shapefile(action, "watershed_shapefile", "/data/lmc", stem="WS")

    out_key = str(attrs.get("output_key", ""))
    script = os.path.join(SLAM_SUBMODULE_PATH, "3.LMC.py")

    # 3.LMC.py processes the domain one block of `group_size` (=15) valid-anchor
    # longitude columns at a time, selected by the chunk index; only the block
    # holding the basin's home anchor writes WSAM. The canonical pipeline runs
    # every block in parallel then merges (PostLMC.sh -> 3bp.CombineLMs.py). We
    # run all blocks sequentially in-process, then merge, so LMs covers the full
    # domain and WSAM is always produced.
    import math
    import xarray as xr

    _GROUP_SIZE = 15
    am_glob = "/data/lmc/*Maxima.*.nc4"
    with xr.open_mfdataset(glob.glob(am_glob)) as _am:
        _lonlist = np.unique(np.where(_am["WAP"].values > 0)[2])
    n_chunks = max(1, math.ceil(_lonlist.size / _GROUP_SIZE))
    logger.info(
        f"LMC: {_lonlist.size} valid-anchor longitude column(s) -> {n_chunks} chunk(s)"
    )

    for chunk in range(n_chunks):
        cmd = [
            sys.executable,
            script,
            str(chunk),
            str(attrs.get("duration", "24")),
            str(attrs.get("lat_name", "latitude")),
            str(attrs.get("lon_name", "longitude")),
            str(attrs.get("precvar", "precrate")),
            str(attrs.get("precip_prefix", "")),
            str(attrs.get("precip_suffix", "")),
            out_key,
        ]
        result = subprocess_run(cmd, cwd="/data/lmc")
        if result != 0:
            raise RuntimeError(f"LMC chunk {chunk} failed with return code {result}")

    # Merge per-chunk LMCol.<chunk>.<key>.<method>.nc into one LMs.<key>.<method>.nc
    # (reindex longitude onto the full annual-maxima grid), as 3bp.CombineLMs.py does.
    with xr.open_mfdataset(glob.glob(am_glob)) as _am:
        for method in ("mean", "median"):
            cols = sorted(glob.glob(f"/data/lmc/LMCol.*.{method}.nc"))
            if not cols:
                continue
            with xr.open_mfdataset(cols) as _lm:
                _lm.reindex(longitude=_am.longitude).to_netcdf(
                    f"/data/lmc/LMs.{out_key}.{method}.nc"
                )
            logger.info(
                f"merged {len(cols)} LMCol chunk(s) -> LMs.{out_key}.{method}.nc"
            )

    _stage_dir_to_remote(
        action,
        "lm_output",
        "/data/lmc",
        glob.glob("/data/lmc/LMs.*.nc")
        + glob.glob("/data/lmc/LMCol*.nc")
        + glob.glob("/data/lmc/WSAM.*.nc")
        + glob.glob("/data/lmc/WS.*.LMC*.nc"),
    )


def run_clmpv(action):
    logger.info("STAGE 4: CLMPV")
    attrs = action.attributes

    _connect_action_stores(action)
    os.makedirs("/data/clmpv", exist_ok=True)
    _get_folder(action, "lm_files", "/data/clmpv")
    _get_folder(action, "wsam_files", "/data/clmpv")
    _get_shapefile(action, "watershed_shapefile", "/data/clmpv", stem="WS")

    # 4.CLMPV.py opens exactly one "LMs.*.nc"; LMC emits both mean and median.
    # Keep only the configured composite method.
    comp = str(attrs.get("comp_method", "mean"))
    for f in glob.glob("/data/clmpv/LMs.*.nc"):
        if not os.path.basename(f).endswith(f".{comp}.nc"):
            os.remove(f)

    # 4.CLMPV.py (line 213 + get_or_create_wsarray) needs one "*.precip.nc" grid
    # reference carrying a "precrate" variable. AORC daily files are "AORC.*.nc"
    # with "APCP_surface", so build a single renamed reference; this reproduces
    # the same watershed mask LMC derives from a raw precip file.
    import xarray as xr

    _get_first_from_folder(action, "raw_precipitation", "/data/clmpv/_refsrc.nc")
    with xr.open_dataset("/data/clmpv/_refsrc.nc") as _ref:
        _dvar = (
            "APCP_surface"
            if "APCP_surface" in _ref.data_vars
            else list(_ref.data_vars)[0]
        )
        _ref.rename({_dvar: "precrate"}).to_netcdf("/data/clmpv/ref.precip.nc")
    os.remove("/data/clmpv/_refsrc.nc")

    values_path = "-"
    domain_mode = attrs.get("domain_mode", "SIM")
    values = attrs.get("values", [])
    if domain_mode in ("SIM", "SIG") and values:
        values_file = "/data/clmpv/values.txt"
        with open(values_file, "w") as vf:
            vf.write("\n".join(str(v) for v in values))
        values_path = values_file

    script = os.path.join(SLAM_SUBMODULE_PATH, "4.CLMPV.py")
    cmd = [
        sys.executable,
        script,
        str(attrs.get("huc", "")),
        str(attrs.get("duration", "24")),
        str(attrs.get("comp_method", "mean")),
        str(attrs.get("precip_key", "")),
        str(attrs.get("res_key", "")),
        str(domain_mode),
        values_path,
    ]
    result = subprocess_run(cmd, cwd="/data/clmpv")
    if result != 0:
        raise RuntimeError(f"CLMPV failed with return code {result}")

    _stage_dir_to_remote(
        action,
        "td_output",
        "/data/clmpv",
        glob.glob("/data/clmpv/CLMPVResult.*.nc")
        + glob.glob("/data/clmpv/TD.*")
        + glob.glob("/data/clmpv/WS.*.CLMPV.nc"),
    )


# ---------------------------------------------------------------------------
# small internal utilities
# ---------------------------------------------------------------------------
def subprocess_run(cmd, cwd):
    import subprocess

    logger.info(f"run: {' '.join(cmd)} (cwd={cwd})")
    return subprocess.run(cmd, cwd=cwd).returncode


def _stage_dir_to_remote(action, ds_name, base, files):
    """Upload a specific set of output files (already on disk under base) to the
    output datasource prefix, preserving basenames."""
    if not files:
        logger.warning(f"{ds_name}: no output files matched to upload")
        return
    staging = os.path.join(base, "_upload_" + ds_name)
    os.makedirs(staging, exist_ok=True)
    for f in files:
        shutil.copy2(f, os.path.join(staging, os.path.basename(f)))
    _put_folder(action, ds_name, staging)
    shutil.rmtree(staging, ignore_errors=True)


def main():
    pm = PluginManager()
    pl = pm.get_payload()

    logger.info("=== SLAM-SIGSIM Plugin Starting ===")
    logger.info(f"Attributes: {pl.attributes}")

    for action in pl.actions:
        logger.info(f"Running action: {action.type} - {action.description}")
        match action.type:
            case "aorc_to_daily_nc":
                run_aorc_to_daily_nc(action)
            case "geojson_to_shp":
                run_geojson_to_shp(action)
            case "pp2wap":
                run_pp2wap(action)
            case "amc":
                run_amc(action)
            case "lmc":
                run_lmc(action)
            case "clmpv":
                run_clmpv(action)
            case _:
                raise ValueError(f"Unknown action type: {action.type}")

    logger.info("=== SLAM-SIGSIM Plugin Complete ===")


if __name__ == "__main__":
    main()
