import os
import sys
import subprocess
import glob
import logging
from cc.plugin_manager import PluginManager, DataSourceOpInput

logger = logging.getLogger(__name__)

# Ensure the SLAM-SIGSIM submodule scripts are importable
# This allows the worker scripts to `from slam_functions import ...`
SLAM_SUBMODULE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "..", "lib", "slam-sigsim", "sub_python"
)
if SLAM_SUBMODULE_PATH not in sys.path:
    sys.path.insert(0, SLAM_SUBMODULE_PATH)


def main():
    pm = PluginManager()
    pl = pm.get_payload()

    logger.info("=== SLAM-SIGSIM Plugin Starting ===")
    logger.info(f"Attributes: {pl.attributes}")

    for action in pl.actions:
        logger.info(f"Running action: {action.type} - {action.description}")
        match action.type:
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


def run_pp2wap(action):
    logger.info("STAGE 1: PP2WAP")
    attrs = action.attributes
    
    # Setup local working directory
    os.makedirs("/data", exist_ok=True)
    action.copy_file_to_local(
        DataSourceOpInput(name="precipitation", pathkey="default", datakey=None),
        "/data/precip",
    )
    action.copy_file_to_local(
        DataSourceOpInput(name="watershed_shapefile", pathkey="default", datakey=None),
        "/data/WS.shp",
    )

    # Execute the submodule script
    script = os.path.join(SLAM_SUBMODULE_PATH, "1.PP2WAP.py")
    cmd = [
        sys.executable, script,
        str(attrs.get("precvar", "precrate")),
        str(attrs.get("lon_name", "longitude")),
        str(attrs.get("lat_name", "latitude")),
        str(attrs.get("tpd", 24)),
        str(attrs.get("output_format", ""))
    ]
    
    result = subprocess.run(cmd, cwd="/data")
    if result.returncode != 0:
        raise RuntimeError(f"PP2WAP failed with return code {result.returncode}")

    # Upload outputs
    for f in glob.glob("/data/WAP.*.nc4") + glob.glob("/data/WS.*.PP2WAP.nc"):
        action.copy_file_to_remote(
            DataSourceOpInput(name="wap_output", pathkey="default", datakey=None),
            f
        )


def run_amc(action):
    logger.info("STAGE 2: AMC")
    attrs = action.attributes
    
    os.makedirs("/data/wap", exist_ok=True)
    action.copy_file_to_local(
        DataSourceOpInput(name="wap_input", pathkey="default", datakey=None),
        "/data/wap",
    )

    script = os.path.join(SLAM_SUBMODULE_PATH, "2.AMC.py")
    cmd = [
        sys.executable, script,
        str(attrs.get("storm_duration", 24)),
        str(attrs.get("year", "")),
        str(attrs.get("season_start", "0101")),
        str(attrs.get("season_end", "1231")),
        str(attrs.get("am_key", "")),
        str(attrs.get("out_key", ""))
    ]

    result = subprocess.run(cmd, cwd="/data/wap")
    if result.returncode != 0:
        raise RuntimeError(f"AMC failed with return code {result.returncode}")

    for f in glob.glob("/data/wap/Maximum.*.nc4"):
        action.copy_file_to_remote(
            DataSourceOpInput(name="am_output", pathkey="default", datakey=None),
            f
        )


def run_lmc(action):
    logger.info("STAGE 3: LMC")
    attrs = action.attributes
    
    os.makedirs("/data/lmc", exist_ok=True)
    action.copy_file_to_local(
        DataSourceOpInput(name="annual_maxima", pathkey="default", datakey=None),
        "/data/lmc",
    )
    action.copy_file_to_local(
        DataSourceOpInput(name="raw_precipitation", pathkey="default", datakey=None),
        "/data/lmc/precip",
    )
    action.copy_file_to_local(
        DataSourceOpInput(name="watershed_shapefile", pathkey="default", datakey=None),
        "/data/lmc/WS.shp",
    )

    script = os.path.join(SLAM_SUBMODULE_PATH, "3.LMC.py")
    cmd = [
        sys.executable, script,
        str(attrs.get("chunk_index", 0)),
        str(attrs.get("duration", "24")),
        str(attrs.get("lat_name", "latitude")),
        str(attrs.get("lon_name", "longitude")),
        str(attrs.get("precvar", "precrate")),
        str(attrs.get("precip_prefix", "")),
        str(attrs.get("precip_suffix", "")),
        str(attrs.get("output_key", ""))
    ]

    result = subprocess.run(cmd, cwd="/data/lmc")
    if result.returncode != 0:
        raise RuntimeError(f"LMC failed with return code {result.returncode}")

    for f in glob.glob("/data/lmc/LMCol*.nc") + glob.glob("/data/lmc/WSAM.*.nc") + glob.glob("/data/lmc/WS.*.LMC*.nc"):
        action.copy_file_to_remote(
            DataSourceOpInput(name="lm_output", pathkey="default", datakey=None),
            f
        )


def run_clmpv(action):
    logger.info("STAGE 4: CLMPV")
    attrs = action.attributes
    
    os.makedirs("/data/clmpv", exist_ok=True)
    action.copy_file_to_local(
        DataSourceOpInput(name="lm_files", pathkey="default", datakey=None),
        "/data/clmpv",
    )
    action.copy_file_to_local(
        DataSourceOpInput(name="wsam_files", pathkey="default", datakey=None),
        "/data/clmpv",
    )
    action.copy_file_to_local(
        DataSourceOpInput(name="watershed_shapefile", pathkey="default", datakey=None),
        "/data/clmpv/WS.shp",
    )

    # Write values file if domain_mode requires it
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
        sys.executable, script,
        str(attrs.get("huc", "")),
        str(attrs.get("duration", "24")),
        str(attrs.get("comp_method", "mean")),
        str(attrs.get("precip_key", "")),
        str(attrs.get("res_key", "")),
        str(domain_mode),
        values_path
    ]

    result = subprocess.run(cmd, cwd="/data/clmpv")
    if result.returncode != 0:
        raise RuntimeError(f"CLMPV failed with return code {result.returncode}")

    for f in glob.glob("/data/clmpv/CLMPVResult.*.nc") + glob.glob("/data/clmpv/TD.*.shp") + glob.glob("/data/clmpv/WS.*.CLMPV.nc"):
        action.copy_file_to_remote(
            DataSourceOpInput(name="td_output", pathkey="default", datakey=None),
            f
        )


if __name__ == "__main__":
    main()