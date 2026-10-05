#!/usr/bin/env python3
"""Small local UI; all hydrodynamics are delegated to the C++ flood_sim binary."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import csv
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import threading
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "output" / "ui-run"
SOLVER_BINARY = ROOT / "build-mpi" / "flood_sim"
def detectCapabilities():
    if not SOLVER_BINARY.is_file():
        return {name: {"status": "UNAVAILABLE", "implemented": False,
                       "detail": "solver executable is not built"}
                for name in ("serial", "openmp", "mpi")}
    try:
        result = subprocess.run([str(SOLVER_BINARY), "--capabilities"], check=True,
                                capture_output=True, text=True, timeout=5)
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {name: {"status": "UNAVAILABLE", "implemented": False,
                       "detail": "capability detection failed"}
                for name in ("serial", "openmp", "mpi")}


def detectMpiLauncher():
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


CAPABILITIES = detectCapabilities()
OPENMP_AVAILABLE = CAPABILITIES["openmp"]["status"] == "AVAILABLE"
MPI_LAUNCHER, MPI_NUMPROC_FLAG, MPI_PREFLAGS, MPI_POSTFLAGS = detectMpiLauncher()
PROCESS = None
LOCK = threading.Lock()
LAST_REQUEST_STATE = "NOT RUN"
LAST_REQUEST_ERROR = ""

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Urban Flood | Backend Bench</title><style>
:root{color-scheme:light;--ink:#17232a;--muted:#637176;--paper:#f4f5f1;--line:#d3d8d3;--water:#156d88;--green:#bfd4bd;--orange:#d87942}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.45 Georgia,serif}
header{padding:20px 4vw;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center}
.mark{font:700 14px/1.2 ui-monospace,monospace;letter-spacing:0;text-transform:uppercase}.status{color:var(--muted);font:12px ui-monospace,monospace}
main{max-width:1200px;margin:auto;padding:30px 4vw}.top{display:grid;grid-template-columns:minmax(240px,330px) 1fr;gap:36px;align-items:start}
h1{font-size:34px;line-height:1.05;margin:0 0 24px;font-weight:500}.controls{display:grid;gap:14px}
label{display:grid;gap:5px;color:var(--muted);font:12px ui-monospace,monospace}
select,input{width:100%;height:40px;border:1px solid var(--line);background:#fff;color:var(--ink);padding:0 10px;font:15px Georgia,serif;border-radius:2px}
button{height:42px;border:0;background:var(--ink);color:#fff;font:14px ui-monospace,monospace;cursor:pointer}button:disabled{opacity:.5}
.viz{border-top:3px solid var(--water);padding-top:12px}.vizhead{display:flex;justify-content:space-between;color:var(--muted);font:12px ui-monospace,monospace;margin-bottom:8px}
canvas{display:block;width:100%;height:auto;aspect-ratio:1.45;background:#e2e8e5;border:1px solid var(--line)}
.metrics{display:grid;grid-template-columns:repeat(6,1fr);margin-top:28px;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
.metric{padding:14px 12px 16px 0}.metric+.metric{padding-left:12px;border-left:1px solid var(--line)}.label{color:var(--muted);font:11px ui-monospace,monospace}.value{font-size:22px;margin-top:6px}
table{width:100%;border-collapse:collapse;margin-top:12px;font:12px ui-monospace,monospace}th,td{text-align:left;padding:8px;border-bottom:1px solid var(--line)}th{color:var(--muted)}
.note{font:12px ui-monospace,monospace;color:var(--muted);margin-top:12px}
@media(max-width:760px){.top{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}.metric:nth-child(odd){padding-left:0;border-left:0}}
</style></head><body><header><div class="mark">Urban Flood / Backend Bench</div><div id="status" class="status">NOT RUN</div></header>
<main><div class="top"><section><h1>Run simulation</h1><form id="run" class="controls">
<label>SCENARIO<select name="scenario"><option>dam-break</option><option>rain-drain</option><option>localized-refinement</option></select></label>
<label>BACKEND<select name="backend" id="backend"><!-- BACKEND_OPTIONS --></select></label>
<label id="threadControl" hidden>OPENMP THREADS<select name="threads"><option>1</option><option>2</option><option selected>4</option></select></label>
<label id="processControl" hidden>MPI RANKS<select name="processes"><option>1</option><option>2</option><option selected>4</option></select></label>
<label>ROWS<input name="rows" type="number" min="8" max="1024" value="64"></label><label>COLUMNS<input name="cols" type="number" min="8" max="1024" value="64"></label>
<label>DURATION (SECONDS)<input name="duration" type="number" min="0.01" max="600" step="0.01" value="0.12"></label>
<label><span>ADAPTIVE REFINEMENT</span><input id="adaptive" name="adaptive" type="checkbox" value="true"></label>
<div id="adaptiveControls" hidden>
<label>MAX REFINEMENT LEVEL<select name="max_level"><option>0</option><option>1</option><option selected>2</option></select></label>
<label>PATCH SIZE<select name="patch_extent"><option>4</option><option selected>8</option><option>16</option></select></label>
<label>REGRID INTERVAL (STEPS)<input name="regrid_interval" type="number" min="1" value="2"></label>
<label>REFINE THRESHOLD<input name="refine_threshold" type="number" min="0.000001" step="0.01" value="0.10"></label>
<label>COARSEN THRESHOLD<input name="coarsen_threshold" type="number" min="0" step="0.01" value="0.05"></label>
<label id="dynamicControl" hidden><span>DYNAMIC MPI LOAD BALANCING</span><input name="dynamic_load_balancing" type="checkbox" value="true"></label>
</div>
<button id="start">RUN SIMULATION</button></form>
<div id="message" class="note">No simulation has run</div></section>
<section class="viz"><div class="vizhead"><span>FINAL WATER DEPTH / METRES</span><span id="mapMeta">NO OUTPUT</span></div><canvas id="map" width="900" height="620"></canvas><div id="refinementView" hidden><div class="vizhead"><span>REFINEMENT LEVEL</span><span id="refinementMeta">NO OUTPUT</span></div><canvas id="refinementMap" width="900" height="620"></canvas></div></section></div>
<section class="metrics"><div class="metric"><div class="label">BACKEND / THREADS</div><div class="value" id="backendUsed">—</div></div><div class="metric"><div class="label">RUNTIME</div><div class="value" id="runtime">—</div></div><div class="metric"><div class="label">SPEEDUP</div><div class="value" id="speedup">—</div></div><div class="metric"><div class="label">EFFICIENCY</div><div class="value" id="efficiency">—</div></div><div class="metric"><div class="label">MAX DEPTH ERROR</div><div class="value" id="error">—</div></div><div class="metric"><div class="label">TOTAL VOLUME</div><div class="value" id="volume">—</div></div>
<div class="metric"><div class="label">MAX DEPTH</div><div class="value" id="maxDepth">—</div></div><div class="metric"><div class="label">FLOODED AREA</div><div class="value" id="floodedArea">—</div></div><div class="metric"><div class="label">WET CELLS</div><div class="value" id="wetCells">—</div></div><div class="metric"><div class="label">RAIN INPUT</div><div class="value" id="rain">—</div></div><div class="metric"><div class="label">MASS RESIDUAL</div><div class="value" id="residual">—</div></div><div class="metric"><div class="label">CONSERVATION</div><div class="value" id="conservation">—</div></div></section>
<section class="metrics"><div class="metric"><div class="label">ACTIVE LEAF CELLS</div><div class="value" id="activeCells">—</div></div><div class="metric"><div class="label">TIMESTEPS</div><div class="value" id="steps">—</div></div><div class="metric"><div class="label">MPI COMMUNICATION</div><div class="value" id="communication">—</div></div><div class="metric"><div class="label">WORK IMBALANCE</div><div class="value" id="imbalance">—</div></div><div class="metric"><div class="label">MIGRATIONS</div><div class="value" id="migrations">—</div></div><div class="metric"><div class="label">MIGRATION TIME</div><div class="value" id="migrationTime">—</div></div></section>
<section><h2>Measured benchmark medians</h2><label>SCENARIO<select id="comparisonScenario"><option>dam-break</option><option>rain-drain</option><option>localized-refinement</option></select></label><div id="comparisonMessage" class="note">Final benchmark summary not available yet.</div><table><thead><tr><th>Family</th><th>Backend</th><th>Count</th><th>MPI mode</th><th>Median (s)</th><th>Speedup</th><th>Efficiency</th><th>Imbalance</th></tr></thead><tbody id="comparisonRows"></tbody></table></section>
<div class="note">UI orchestrates C++ serial reference and selected backend runs; it does not compute the physical model.</div></main>
<script>
const form=document.querySelector('#run'),button=document.querySelector('#start'),status=document.querySelector('#status'),message=document.querySelector('#message');
const backendSelect=document.querySelector('#backend'),threadControl=document.querySelector('#threadControl'),processControl=document.querySelector('#processControl');
const adaptiveControl=document.querySelector('#adaptive'),adaptiveControls=document.querySelector('#adaptiveControls'),dynamicControl=document.querySelector('#dynamicControl');
function updateControls(){threadControl.hidden=backendSelect.value!=='openmp';processControl.hidden=backendSelect.value!=='mpi';adaptiveControls.hidden=!adaptiveControl.checked;dynamicControl.hidden=!adaptiveControl.checked||backendSelect.value!=='mpi'}
updateControls();backendSelect.addEventListener('change',updateControls);adaptiveControl.addEventListener('change',updateControls);
function rowsFromCsv(text){return text.trim().split('\n').map(line=>line.split(',').map(Number))}
function paint(rows,canvasId,mode){const c=document.querySelector(canvasId),ctx=c.getContext('2d'),h=rows.length,w=rows[0].length;let max=0;for(const row of rows)for(const value of row)max=Math.max(max,value);ctx.clearRect(0,0,c.width,c.height);const cw=c.width/w,ch=c.height/h;for(let y=0;y<h;y++)for(let x=0;x<w;x++){const value=rows[y][x],t=max>0?Math.min(value/max,1):0;if(mode==='level'){ctx.fillStyle=['#e6ebe6','#64a6a2','#dc995c'][value]||'#273d52'}else{ctx.fillStyle=value<=1e-6?'#dce4df':`rgb(${Math.round(228-150*t)},${Math.round(239-75*t)},${Math.round(238-26*t)})`}ctx.fillRect(x*cw,y*ch,Math.ceil(cw),Math.ceil(ch))}return {h,w,max}}
async function poll(){const r=await fetch('/status'),d=await r.json();if(d.running){status.textContent='RUNNING';message.textContent='Running simulation and reference...';setTimeout(poll,700);return}button.disabled=false;status.textContent=d.state||'NOT RUN';if(d.error){message.textContent=d.error;return}if(!d.summary){message.textContent='No simulation has run';return}const q=d.summary,s=q.stats;message.textContent=`Completed at ${Number(s.time_s).toFixed(3)} s`;document.querySelector('#backendUsed').textContent=q.backend+(q.backend==='openmp'?' / '+q.threads+' threads':q.backend==='mpi'?' / '+q.processes+' ranks':'')+(q.adaptive?' / adaptive':'');document.querySelector('#runtime').textContent=Number(q.backend_runtime).toFixed(4)+' s';document.querySelector('#speedup').textContent=Number(q.speedup).toFixed(3)+'x';document.querySelector('#efficiency').textContent=Number(q.efficiency).toFixed(3);document.querySelector('#error').textContent=q.adaptive?'n/a':Number(q.max_error).toExponential(2)+' m';document.querySelector('#volume').textContent=Number(s.stored_m3).toFixed(3)+' m³';document.querySelector('#maxDepth').textContent=Number(s.max_depth_m).toFixed(4)+' m';document.querySelector('#floodedArea').textContent=s.flooded_area_m2===''?'n/a':Number(s.flooded_area_m2).toFixed(1)+' m²';document.querySelector('#wetCells').textContent=s.wet_cells;document.querySelector('#rain').textContent=Number(s.rainfall_m3).toFixed(3)+' m³';document.querySelector('#residual').textContent=Number(s.mass_residual_m3).toExponential(2)+' m³';document.querySelector('#conservation').textContent=q.conservation_pass?'PASS':'FAIL';document.querySelector('#activeCells').textContent=Number(s.active_leaf_cells||s.wet_cells).toLocaleString();document.querySelector('#steps').textContent=s.steps||'—';document.querySelector('#communication').textContent=s.communication_seconds===undefined?'—':Number(s.communication_seconds).toFixed(4)+' s';document.querySelector('#imbalance').textContent=s.load_imbalance===undefined?'—':Number(s.load_imbalance).toFixed(3)+'×';document.querySelector('#migrations').textContent=s.migration_count===undefined?'—':s.migration_count;document.querySelector('#migrationTime').textContent=s.migration_time_seconds===undefined?'—':Number(s.migration_time_seconds).toFixed(4)+' s';const depth=rowsFromCsv(await(await fetch('/depth.csv')).text()),map=paint(depth,'#map','depth');document.querySelector('#mapMeta').textContent=`${map.h} x ${map.w} CELLS | MAX ${map.max.toFixed(3)} M`;const refinementView=document.querySelector('#refinementView');refinementView.hidden=!q.adaptive;if(q.adaptive){const levels=rowsFromCsv(await(await fetch('/refinement.csv')).text()),refinement=paint(levels,'#refinementMap','level');document.querySelector('#refinementMeta').textContent=`${refinement.h} x ${refinement.w} CELLS | LEVEL 0–${q.max_level||2}`}}
form.addEventListener('submit',async e=>{e.preventDefault();button.disabled=true;status.textContent='RUNNING';message.textContent='Preparing simulation...';document.querySelectorAll('.value').forEach(value=>value.textContent='—');for(const id of ['map','refinementMap'])document.querySelector(id).getContext('2d').clearRect(0,0,900,620);document.querySelector('#mapMeta').textContent='NO OUTPUT';const response=await fetch('/run',{method:'POST',body:new URLSearchParams(new FormData(form))});if(!response.ok){const result=await response.json();button.disabled=false;status.textContent='UNAVAILABLE';message.textContent=result.error;return}poll()});
async function loadComparison(){const scenario=document.querySelector('#comparisonScenario').value,response=await fetch('/benchmark-summary?scenario='+encodeURIComponent(scenario)),data=await response.json(),body=document.querySelector('#comparisonRows');body.replaceChildren();if(!response.ok||!data.rows.length){document.querySelector('#comparisonMessage').textContent='No measured final-study rows for this scenario.';return}document.querySelector('#comparisonMessage').textContent=`${data.rows.length} median configurations from the recorded final study.`;for(const row of data.rows){const tr=document.createElement('tr'),count=row.backend==='openmp'?row.threads:row.backend==='mpi'?row.ranks:'1';for(const value of [row.family,row.backend,count,row.backend==='mpi'?(row.dynamic_load_balancing==='true'?'dynamic':'static'):'—',Number(row.median_runtime_seconds).toPrecision(4),row.serial_speedup===''?'—':Number(row.serial_speedup).toFixed(3)+'×',row.parallel_efficiency===''?'—':Number(row.parallel_efficiency).toFixed(3),row.workload_imbalance===''?'—':Number(row.workload_imbalance).toFixed(3)]){const td=document.createElement('td');td.textContent=value;tr.appendChild(td)}body.appendChild(tr)}}
document.querySelector('#comparisonScenario').addEventListener('change',loadComparison);loadComparison();
poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            body = PAGE.replace("<!-- BACKEND_OPTIONS -->", backendOptions()).encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        elif self.path == "/status":
            with LOCK:
                running = PROCESS is not None and PROCESS.poll() is None
                result = {"running": running,
                          "state": "RUNNING" if running else LAST_REQUEST_STATE}
                if not running and LAST_REQUEST_STATE == "UNAVAILABLE":
                    result["error"] = LAST_REQUEST_ERROR
                elif PROCESS is not None and not running:
                    if PROCESS.returncode:
                        result["error"] = (RUN_DIR / "ui.log").read_text(errors="replace")[-1000:]
                        result["state"] = "FAIL"
                    else:
                        with open(RUN_DIR / "summary.json", encoding="utf-8") as stream:
                            result["summary"] = json.load(stream)
                        result["state"] = "PASS" if result["summary"].get("validation_pass") else "FAIL"
            body = json.dumps(result).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        elif self.path in ("/depth.csv", "/refinement.csv"):
            filename = self.path.lstrip("/")
            source = RUN_DIR / "current" / filename
            if not source.exists():
                self.send_error(404)
                return
            body = source.read_bytes()
            self.send_response(200); self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        elif self.path.startswith("/benchmark-summary"):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            scenario = query.get("scenario", [""])[0]
            summary_path = ROOT / "benchmarks" / "adaptive" / "final-summary.csv"
            rows = []
            if scenario in {"dam-break", "rain-drain", "localized-refinement"} and summary_path.is_file():
                with summary_path.open(newline="", encoding="utf-8") as stream:
                    rows = [row for row in csv.DictReader(stream)
                            if row["scenario"] == scenario]
            body = json.dumps({"rows": rows}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        global PROCESS, LAST_REQUEST_STATE, LAST_REQUEST_ERROR
        if self.path != "/run":
            self.send_error(404); return
        length = int(self.headers.get("Content-Length", 0))
        form = urllib.parse.parse_qs(self.rfile.read(length).decode())
        def value(name, default): return form.get(name, [default])[0]
        scenario = value("scenario", "flat-basin")
        if scenario not in {"dam-break", "rain-drain", "localized-refinement"}:
            self.send_error(400); return
        backend = value("backend", "serial")
        adaptive = value("adaptive", "") == "true"
        with LOCK:
            if PROCESS is not None and PROCESS.poll() is None:
                self.send_error(409, "Simulation is already running"); return
        capability = CAPABILITIES.get(backend)
        if (capability is None or capability["status"] != "AVAILABLE" or
            (backend == "mpi" and not MPI_LAUNCHER)):
            with LOCK:
                LAST_REQUEST_STATE = "UNAVAILABLE"
                LAST_REQUEST_ERROR = "Backend unavailable in current environment"
            body = json.dumps({"error": "Backend unavailable in current environment"}).encode()
            self.send_response(409); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            return
        try:
            rows, cols = int(value("rows", "64")), int(value("cols", "64"))
            duration = float(value("duration", "60"))
            threads = int(value("threads", "1"))
            processes = int(value("processes", "1"))
            if not 8 <= rows <= 1024 or not 8 <= cols <= 1024 or not 0.01 <= duration <= 600:
                raise ValueError
            if threads not in (1, 2, 4):
                raise ValueError
            if processes not in (1, 2, 4):
                raise ValueError
            max_level = int(value("max_level", "2"))
            patch_extent = int(value("patch_extent", "8"))
            regrid_interval = int(value("regrid_interval", "2"))
            refine_threshold = float(value("refine_threshold", "0.10"))
            coarsen_threshold = float(value("coarsen_threshold", "0.05"))
            dynamic_load_balancing = value("dynamic_load_balancing", "") == "true"
            if max_level not in (0, 1, 2) or patch_extent not in (4, 8, 16):
                raise ValueError
            if regrid_interval < 1 or not 0 <= coarsen_threshold < refine_threshold:
                raise ValueError
            if dynamic_load_balancing and (not adaptive or backend != "mpi"):
                raise ValueError
        except ValueError:
            self.send_error(400, "One or more simulation settings are out of range"); return
        command = [sys.executable, str(ROOT / "scripts" / "run_ui_comparison.py"),
               "--binary", str(SOLVER_BINARY),
               "--adaptive-binary", str(ROOT / "build-mpi" / "flood_adaptive_benchmark"),
               "--scenario", scenario, "--rows", str(rows), "--cols", str(cols),
               "--duration", str(duration), "--backend", backend, "--threads", str(threads),
             "--processes", str(processes), "--mpi-exec", MPI_LAUNCHER,
             f"--mpi-numproc-flag={MPI_NUMPROC_FLAG}", "--mpi-preflags", MPI_PREFLAGS,
             "--mpi-postflags", MPI_POSTFLAGS, "--output", str(RUN_DIR)]
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
                self.send_error(409, "Simulation is already running"); return
            LAST_REQUEST_STATE = "RUNNING"
            LAST_REQUEST_ERROR = ""
            RUN_DIR.mkdir(parents=True, exist_ok=True)
            log = open(RUN_DIR / "ui.log", "w", encoding="utf-8")
            PROCESS = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        self.send_response(202); self.end_headers()

    def log_message(self, *_):
        pass


def backendOptions():
    return "".join(
        f'<option value="{backend}">{label}' +
        ("" if CAPABILITIES[backend]["status"] == "AVAILABLE"
         else " | Backend unavailable in current environment") +
        "</option>"
        for backend, label in (("serial", "Serial"), ("openmp", "OpenMP"), ("mpi", "MPI"))
    )


if __name__ == "__main__":
    address = ("127.0.0.1", 8000)
    print(f"Urban Flood UI: http://{address[0]}:{address[1]}")
    HTTPServer(address, Handler).serve_forever()