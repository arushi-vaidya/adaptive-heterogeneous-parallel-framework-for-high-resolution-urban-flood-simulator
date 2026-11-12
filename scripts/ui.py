#!/usr/bin/env python3
"""Local research UI; simulation physics remain in the C++ solver."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import csv
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import threading
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "output" / "ui-run"
SOLVER_BINARY = ROOT / "build-mpi" / "flood_sim"
ADAPTIVE_BINARY = ROOT / "build-mpi" / "flood_adaptive_benchmark"
PAGE_PATH = Path(__file__).with_name("ui.html")
SUMMARY_PATH = ROOT / "benchmarks" / "adaptive" / "final-summary.csv"
SCENARIOS = {"urban-district", "dam-breach", "catchment", "dam-break", "rain-drain",
             "localized-refinement"}
RAINY_SCENARIOS = {"urban-district", "catchment"}  # rain wets every cell: AMR would refine everything
BACKENDS = ("serial", "openmp", "mpi", "cuda", "openacc")
ADAPTIVE_BACKENDS = ("serial", "openmp", "mpi")
SWEEP_SCENARIOS = {"urban-district", "dam-break", "rain-drain"}


def detect_capabilities():
    unavailable = {
        name: {"status": "UNAVAILABLE", "implemented": False,
               "detail": "solver executable is not built"}
        for name in BACKENDS
    }
    if not SOLVER_BINARY.is_file():
        return unavailable
    try:
        result = subprocess.run(
            [str(SOLVER_BINARY), "--capabilities"], check=True,
            capture_output=True, text=True, timeout=5)
        capabilities = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        return {
            name: {"status": "UNAVAILABLE", "implemented": False,
                   "detail": f"capability detection failed: {error}"}
            for name in BACKENDS
        }
    return capabilities


def detect_mpi_launcher():
    cache = SOLVER_BINARY.parent / "CMakeCache.txt"
    launcher = ""
    numproc_flag = "-n"
    preflags = ""
    postflags = ""
    if cache.is_file():
        for line in cache.read_text(errors="replace").splitlines():
            if line.startswith("MPIEXEC_EXECUTABLE:") and "=" in line:
                executable = line.split("=", 1)[1]
                if Path(executable).is_file():
                    launcher = executable
            elif line.startswith("MPIEXEC_NUMPROC_FLAG:") and "=" in line:
                numproc_flag = line.split("=", 1)[1] or numproc_flag
            elif line.startswith("MPIEXEC_PREFLAGS:") and "=" in line:
                preflags = line.split("=", 1)[1]
            elif line.startswith("MPIEXEC_POSTFLAGS:") and "=" in line:
                postflags = line.split("=", 1)[1]
    launcher = launcher or shutil.which("mpiexec") or shutil.which("mpirun") or ""
    return launcher, numproc_flag, preflags, postflags


SHOWCASE_DIR = ROOT / "output" / "ui-showcase-v2"


def ensure_showcase():
    """Pre-compute an urban-district run for the landing-page visuals."""
    if (SHOWCASE_DIR / "depth.csv").is_file() or not SOLVER_BINARY.is_file():
        return
    subprocess.run([str(SOLVER_BINARY), "--scenario", "urban-district", "--rows", "160",
                    "--cols", "160", "--duration", "120", "--frames", "40",
                    "--output", str(SHOWCASE_DIR)],
                   capture_output=True, text=True, check=False)


CAPABILITIES = detect_capabilities()
MPI_LAUNCHER, MPI_NUMPROC_FLAG, MPI_PREFLAGS, MPI_POSTFLAGS = detect_mpi_launcher()
PROCESS = None
SWEEP_PROCESS = None
LOCK = threading.Lock()
LAST_REQUEST_STATE = "NOT RUN"
LAST_REQUEST_ERROR = ""


def json_response(handler, payload, status=200):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def csv_rows(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required data source is missing: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def benchmark_rows(scenario):
    return [
        row for row in csv_rows(SUMMARY_PATH)
        if row.get("scenario") == scenario
    ]


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def last_summary():
    summary_path = RUN_DIR / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def output_file(filename):
    current = RUN_DIR / "current" / filename
    if current.is_file():
        if filename in ("refinement.csv", "terrain.csv", "stats.csv"):
            summary = last_summary()
            adaptive = bool(summary and summary.get("adaptive"))
            # Adaptive runs do not export these maps; any file present is left over from an earlier run.
            if filename == "refinement.csv" and not adaptive:
                return None
        return current
    if filename in ("depth.csv", "terrain.csv", "stats.csv"):
        reference = ROOT / "output" / "benchmark-dam-break" / filename
        if reference.is_file():
            return reference
    return None


def validation_summary():
    rows = csv_rows(SUMMARY_PATH)
    raw_rows = csv_rows(ROOT / "benchmarks" / "adaptive" / "final-uniform-raw.csv")

    def maximum_absolute(field):
        values = []
        for row in rows:
            try:
                value = float(row.get(field, ""))
            except (TypeError, ValueError):
                continue
            values.append(abs(value))
        return max(values) if values else None

    test_log = ROOT / "build-mpi" / "Testing" / "Temporary" / "LastTest.log"
    suite_passed = False
    tested_at = None
    passed_tests = 0
    total_tests = 0
    failed_tests = []
    if test_log.is_file():
        log_text = test_log.read_text(errors="replace")
        passed_tests = len(re.findall(r"^Test Passed\.$", log_text, re.MULTILINE))
        total_tests = len(re.findall(r"^\d+/\d+ Testing:", log_text, re.MULTILINE))
        failed_path = test_log.with_name("LastTestsFailed.log")
        if passed_tests != total_tests and failed_path.is_file():
            failed_tests = [
                line.split(":", 1)[1].strip()
                for line in failed_path.read_text(errors="replace").splitlines()
                if ":" in line
            ]
        suite_passed = total_tests > 0 and passed_tests == total_tests
        tested_at = test_log.stat().st_mtime
    return {
        "suitePassed": suite_passed,
        "testLogAvailable": test_log.is_file(),
        "testLogModified": tested_at,
        "passedTests": passed_tests,
        "totalTests": total_tests,
        "failedTests": failed_tests,
        "depthToleranceM": raw_rows[0].get("tolerance_m") if raw_rows else None,
        "momentumToleranceM2S": (
            raw_rows[0].get("momentum_tolerance_m2_s") if raw_rows else None),
        "conservationToleranceM3": (
            raw_rows[0].get("conservation_tolerance_m3") if raw_rows else None),
        "maxMassBalanceResidualM3": maximum_absolute("mass_balance_residual"),
        "maxInterfaceResidual": maximum_absolute("interface_conservation_residual"),
        "measuredRows": len(rows),
        "source": "benchmarks/adaptive/final-summary.csv",
    }


def status_payload():
    with LOCK:
        process = PROCESS
        running = process is not None and process.poll() is None
        result = {"running": running,
                  "state": "RUNNING" if running else LAST_REQUEST_STATE}
        if running:
            result["progress"] = read_json(RUN_DIR / "progress.json")
            return result
        if LAST_REQUEST_STATE == "UNAVAILABLE":
            result["error"] = LAST_REQUEST_ERROR
        elif process is None and (RUN_DIR / "summary.json").is_file():
            try:
                result["summary"] = json.loads(
                    (RUN_DIR / "summary.json").read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                result["error"] = f"Could not read simulation summary: {error}"
                result["state"] = "FAIL"
            else:
                result["state"] = (
                    "PASS" if result["summary"].get("validation_pass") else "FAIL")
        elif process is not None:
            if process.returncode:
                log_path = RUN_DIR / "ui.log"
                result["error"] = (
                    log_path.read_text(errors="replace")[-2000:]
                    if log_path.is_file() else "Simulation process failed.")
                result["state"] = "FAIL"
            else:
                summary_path = RUN_DIR / "summary.json"
                if not summary_path.is_file():
                    result["error"] = "Simulation exited without writing summary.json."
                    result["state"] = "FAIL"
                else:
                    try:
                        result["summary"] = json.loads(
                            summary_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError as error:
                        result["error"] = f"Could not read simulation summary: {error}"
                        result["state"] = "FAIL"
                    else:
                        result["state"] = (
                            "PASS" if result["summary"].get("validation_pass") else "FAIL")
        return result


def backend_options():
    options = []
    for backend, label in (("serial", "Serial"), ("openmp", "OpenMP"), ("mpi", "MPI")):
        available = CAPABILITIES.get(backend, {}).get("status") == "AVAILABLE"
        suffix = "" if available else " — unavailable in this build"
        selected = " selected" if backend == "serial" else ""
        disabled = "" if available else " disabled"
        options.append(
            f'<option value="{backend}"{selected}{disabled}>{label}{suffix}</option>')
    return "".join(options)


# User-adjustable scenario and physics parameters: form field -> (flag, min, max, scenarios).
PARAMETERS = {
    "rain_intensity": ("--rain-intensity", 0.0, 500.0, None),
    "rain_duration": ("--rain-duration", 0.0, 3600.0, None),
    "manning": ("--manning", 0.0, 0.2, None),
    "infiltration": ("--infiltration-mm-h", 0.0, 200.0, None),
    "surge_height": ("--surge-height", 0.0, 8.0, {"urban-district"}),
    "reservoir_depth": ("--reservoir-depth", 0.05, 10.0, {"dam-break", "dam-breach"}),
    "breach_width": ("--breach-width", 0.05, 1.0, {"dam-breach"}),
    "pond_depth": ("--pond-depth", 0.01, 5.0, {"localized-refinement"}),
}


def scenario_arguments(form, scenario):
    """Validate optional parameters; blank fields keep the scenario defaults."""
    arguments = []
    for field, (flag, low, high, scenarios) in PARAMETERS.items():
        raw = form.get(field, [""])[0].strip()
        if not raw or (scenarios and scenario not in scenarios):
            continue
        value = float(raw)
        if not low <= value <= high:
            raise ValueError(f"{field.replace('_', ' ').capitalize()} must be between {low:g} and {high:g}.")
        if field == "rain_duration" and not form.get("rain_intensity", [""])[0].strip():
            continue
        arguments.extend((flag, repr(value)))
    boundary = form.get("boundary", [""])[0]
    if boundary in ("closed", "outflow"):
        arguments.extend(("--boundary", boundary))
    elif boundary:
        raise ValueError("Boundary must be closed or outflow.")
    return arguments


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            try:
                page = PAGE_PATH.read_text(encoding="utf-8")
            except OSError as error:
                json_response(self, {"error": f"Could not load UI: {error}"}, 500)
                return
            body = page.replace("<!-- BACKEND_OPTIONS -->", backend_options()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/ui.js":
            body = PAGE_PATH.with_name("ui.js").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/javascript; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/status":
            json_response(self, status_payload())
        elif parsed.path == "/api/capabilities":
            json_response(self, {
                "backends": CAPABILITIES,
                "adaptiveBinaryAvailable": ADAPTIVE_BINARY.is_file(),
                "mpiLauncherAvailable": bool(MPI_LAUNCHER),
            "cpuCount": os.cpu_count() or 1,
            })
        elif parsed.path == "/api/benchmarks":
            query = urllib.parse.parse_qs(parsed.query)
            scenario = query.get("scenario", ["dam-break"])[0]
            if scenario not in SCENARIOS:
                json_response(self, {"error": "Unknown benchmark scenario."}, 400)
                return
            try:
                json_response(self, {
                    "scenario": scenario,
                    "source": "benchmarks/adaptive/final-summary.csv",
                    "rows": benchmark_rows(scenario),
                })
            except OSError as error:
                json_response(self, {"error": f"Could not read benchmark data: {error}"}, 500)
        elif parsed.path == "/api/validation":
            try:
                json_response(self, validation_summary())
            except OSError as error:
                json_response(self, {"error": f"Could not read validation data: {error}"}, 500)
        elif parsed.path == "/reference-depth.csv":
            source = RUN_DIR / "reference-depth.csv"
            # run_ui_comparison.py deletes this file when a new run starts.
            if not source.is_file():
                self.send_error(404, "No reference output is available.")
                return
            body = source.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path in ("/frames.bin", "/frames.json", "/landuse.csv") or (
                parsed.path.startswith("/showcase/") and parsed.path.rsplit("/", 1)[1] in (
                    "depth.csv", "terrain.csv", "hu.csv", "hv.csv", "landuse.csv",
                    "frames.bin", "frames.json")):
            name = parsed.path.rsplit("/", 1)[1]
            if parsed.path.startswith("/showcase/"):
                source = SHOWCASE_DIR / name
            else:
                summary = last_summary()
                source = RUN_DIR / "current" / name
                # adaptive runs export land use but record no time-lapse frames
                if summary and summary.get("adaptive") and name != "landuse.csv":
                    source = None
            if source is None or not source.is_file():
                self.send_error(404, "Output is not available.")
                return
            body = source.read_bytes()
            self.send_response(200)
            types = {".csv": "text/csv; charset=utf-8", ".json": "application/json",
                     ".bin": "application/octet-stream"}
            self.send_header("Content-Type", types[Path(name).suffix])
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/sweep":
            with LOCK:
                running = SWEEP_PROCESS is not None and SWEEP_PROCESS.poll() is None
            data = read_json(RUN_DIR / "sweep.json") or {"points": [], "complete": False}
            data["running"] = running
            json_response(self, data)
        elif parsed.path in ("/depth.csv", "/refinement.csv", "/terrain.csv", "/stats.csv"):
            filename = parsed.path.lstrip("/")
            source = output_file(filename)
            if source is None:
                self.send_error(404, "No generated output is available.")
                return
            body = source.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def start_sweep(self):
        global SWEEP_PROCESS
        try:
            length = int(self.headers.get("Content-Length", 0))
            form = urllib.parse.parse_qs(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            json_response(self, {"error": "Could not parse sweep settings."}, 400)
            return
        scenario = form.get("scenario", ["urban-district"])[0]
        if scenario not in SWEEP_SCENARIOS:
            json_response(self, {"error": "Unknown sweep scenario."}, 400)
            return
        backends = [name for name in ("serial", "openmp", "mpi", "cuda", "openacc")
                    if CAPABILITIES.get(name, {}).get("status") == "AVAILABLE" and
                    (name != "mpi" or MPI_LAUNCHER)]
        command = [
            sys.executable, str(ROOT / "scripts" / "run_ui_sweep.py"),
            "--binary", str(SOLVER_BINARY), "--scenario", scenario,
            "--backends", ",".join(backends), "--mpi-exec", MPI_LAUNCHER,
            f"--mpi-numproc-flag={MPI_NUMPROC_FLAG}",
            "--mpi-preflags", MPI_PREFLAGS, "--mpi-postflags", MPI_POSTFLAGS,
            "--output", str(RUN_DIR),
        ]
        with LOCK:
            if SWEEP_PROCESS is not None and SWEEP_PROCESS.poll() is None:
                json_response(self, {"error": "A scaling sweep is already running."}, 409)
                return
            if PROCESS is not None and PROCESS.poll() is None:
                json_response(self, {"error": "Wait for the current simulation to finish."}, 409)
                return
            RUN_DIR.mkdir(parents=True, exist_ok=True)
            log = (RUN_DIR / "sweep.log").open("w", encoding="utf-8")
            SWEEP_PROCESS = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                             stderr=subprocess.STDOUT)
        json_response(self, {"state": "RUNNING", "backends": backends}, 202)

    def do_POST(self):
        global PROCESS, LAST_REQUEST_STATE, LAST_REQUEST_ERROR
        if self.path == "/sweep":
            self.start_sweep()
            return
        if self.path != "/run":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            form = urllib.parse.parse_qs(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            json_response(self, {"error": "Could not parse simulation settings."}, 400)
            return

        def value(name, default):
            return form.get(name, [default])[0]

        scenario = value("scenario", "dam-break")
        if scenario not in SCENARIOS:
            json_response(self, {"error": "Unknown simulation scenario."}, 400)
            return
        backend = value("backend", "serial")
        adaptive = value("adaptive", "") == "true"
        with LOCK:
            if PROCESS is not None and PROCESS.poll() is None:
                json_response(self, {"error": "A simulation is already running."}, 409)
                return
            if SWEEP_PROCESS is not None and SWEEP_PROCESS.poll() is None:
                json_response(self, {"error": "Wait for the scaling sweep to finish."}, 409)
                return
        if backend not in BACKENDS:
            json_response(self, {"error": "Unknown backend."}, 400)
            return
        capability = CAPABILITIES.get(backend)
        if (capability is None or capability.get("status") != "AVAILABLE" or
                (backend == "mpi" and not MPI_LAUNCHER)):
            with LOCK:
                LAST_REQUEST_STATE = "UNAVAILABLE"
                LAST_REQUEST_ERROR = (
                    f"{backend.upper()} backend unavailable on this machine: "
                    + str((capability or {}).get("detail", "not built")))
            json_response(self, {"error": LAST_REQUEST_ERROR}, 409)
            return
        try:
            rows, cols = int(value("rows", "64")), int(value("cols", "64"))
            duration = float(value("duration", "0.12"))
            threads = int(value("threads", "1"))
            processes = int(value("processes", "1"))
            max_level = int(value("max_level", "2"))
            patch_extent = int(value("patch_extent", "8"))
            regrid_interval = int(value("regrid_interval", "2"))
            refine_threshold = float(value("refine_threshold", "0.10"))
            coarsen_threshold = float(value("coarsen_threshold", "0.05"))
            imbalance_threshold = float(value("imbalance_threshold", "1.20"))
            rebalance_cooldown = int(value("rebalance_cooldown", "10"))
            dynamic_load_balancing = value("dynamic_load_balancing", "") == "true"
            if not 8 <= rows <= 1024 or not 8 <= cols <= 1024:
                raise ValueError("Grid dimensions must be between 8 and 1024.")
            if not 0.01 <= duration <= 600:
                raise ValueError("Duration must be between 0.01 and 600 seconds.")
            if threads not in (1, 2, 4, 8) or processes not in (1, 2, 4):
                raise ValueError("Threads must be 1, 2, 4 or 8 and MPI ranks 1, 2 or 4.")
            if adaptive and backend not in ADAPTIVE_BACKENDS:
                raise ValueError("Adaptive mesh runs support the Serial, OpenMP and MPI backends.")
            if adaptive and scenario in RAINY_SCENARIOS:
                raise ValueError("Rain wets every cell in this scenario, so adaptive refinement would "
                                 "refine the whole domain; use the uniform solver.")
            if max_level not in (0, 1, 2) or patch_extent not in (4, 8, 16):
                raise ValueError("Adaptive level or patch extent is unsupported.")
            if regrid_interval < 1 or not 0 <= coarsen_threshold < refine_threshold:
                raise ValueError("Adaptive thresholds or interval are invalid.")
            if not 1.0 <= imbalance_threshold <= 10 or rebalance_cooldown < 0:
                raise ValueError("MPI imbalance threshold or cooldown is invalid.")
            if dynamic_load_balancing and (not adaptive or backend != "mpi"):
                raise ValueError("Dynamic load balancing requires adaptive MPI.")
            solver_args = scenario_arguments(form, scenario)
        except ValueError as error:
            json_response(self, {"error": str(error)}, 400)
            return

        command = [
            sys.executable, str(ROOT / "scripts" / "run_ui_comparison.py"),
            "--solver-args", shlex.join(solver_args),
            "--binary", str(SOLVER_BINARY),
            "--adaptive-binary", str(ADAPTIVE_BINARY),
            "--scenario", scenario, "--rows", str(rows), "--cols", str(cols),
            "--duration", str(duration), "--backend", backend,
            "--threads", str(threads), "--processes", str(processes),
            "--mpi-exec", MPI_LAUNCHER,
            f"--mpi-numproc-flag={MPI_NUMPROC_FLAG}",
            "--mpi-preflags", MPI_PREFLAGS, "--mpi-postflags", MPI_POSTFLAGS,
            "--imbalance-threshold", str(imbalance_threshold),
            "--rebalance-cooldown", str(rebalance_cooldown),
            "--output", str(RUN_DIR),
            "--reference-threads", "4",
        ]
        if value("reference", "true") != "true":
            command.append("--no-reference")
        if adaptive:
            command.extend((
                "--adaptive", "--patch-extent", str(patch_extent),
                "--max-level", str(max_level),
                "--regrid-interval", str(regrid_interval),
                "--refine-threshold", str(refine_threshold),
                "--coarsen-threshold", str(coarsen_threshold),
            ))
        if dynamic_load_balancing:
            command.append("--dynamic-load-balancing")
        with LOCK:
            if PROCESS is not None and PROCESS.poll() is None:
                json_response(self, {"error": "A simulation is already running."}, 409)
                return
            LAST_REQUEST_STATE = "RUNNING"
            LAST_REQUEST_ERROR = ""
            RUN_DIR.mkdir(parents=True, exist_ok=True)
            log = (RUN_DIR / "ui.log").open("w", encoding="utf-8")
            try:
                PROCESS = subprocess.Popen(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            except OSError as error:
                log.close()
                LAST_REQUEST_STATE = "FAIL"
                LAST_REQUEST_ERROR = f"Could not start simulation: {error}"
                json_response(self, {"error": LAST_REQUEST_ERROR}, 500)
                return
        json_response(self, {"state": "RUNNING"}, 202)

    def log_message(self, format_string, *args):
        print(f"{self.address_string()} - {format_string % args}")


def main():
    threading.Thread(target=ensure_showcase, daemon=True).start()
    address = ("127.0.0.1", 8000)
    print(f"Adaptive Flood UI: http://{address[0]}:{address[1]}")
    ThreadingHTTPServer(address, Handler).serve_forever()


if __name__ == "__main__":
    main()
