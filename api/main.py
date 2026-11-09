"""FastAPI application — the serving layer (Step 5).

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
    # then open http://127.0.0.1:8000/docs  (auto-generated OpenAPI)

Endpoints
---------
GET  /health           liveness probe — the container's HEALTHCHECK polls this
GET  /model            which model, which code (git_sha), which classes
POST /detect           image in -> bounding boxes out
GET  /monitor/features captured input features — the drift monitor's (Step 9)
                       current batch, pulled by `python -m src.monitoring.drift --url`
GET  /                 landing page (dark HTML; /meta has the JSON metadata)

Design notes
------------
* The model is loaded **once** in the lifespan handler, not per request.
  Loading a 76 MB checkpoint on every call would blow the latency budget.
* ``/health`` returns 200 even if the model failed to load, but reports
  ``model_loaded: false``. A container that can't load its model should be
  replaced rather than left to fail every request — but the *reason* must
  stay visible, so ``/model`` returns ``status: "error"`` with the message.
* Score threshold is a request parameter because the operating point is a
  deployment decision, not a training decision: the notebook chose
  score >= 0.71 for the safety operating point, and a tuner may want to
  change it without retraining.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse

from api.schemas import (
    Detection,
    DetectResponse,
    HealthResponse,
    ImageInfo,
    ModelInfo,
    MonitorFeaturesResponse,
)
from api.service import DEFAULT_THRESHOLD, ModelNotLoadedError, ModelService

TITLE = "Pedestrian Detection API"
VERSION = "1.0.0"

# Dark theme shared by the landing page and the Swagger docs. Injected into
# /docs as a real <style> element (a custom /docs route below) - NOT via the
# "customCss" config option, which the bundled Swagger UI build silently
# ignored (verified live on 2026-11-07).
_CSS = (
    ":root{--bg:#0f172a;--card:#111c30;--ink:#e2e8f0;--dim:#94a3b8;"
    "--accent:#22d3ee;--accent2:#818cf8;--line:#334155;--good:#34d399;"
    "--warn:#fbbf24;--tag:#334155;--code:#0b1220}"
    "html,body{background:var(--bg)!important;color:var(--ink)!important}"
)
_LANDING = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<style>
{_CSS}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:radial-gradient(1100px 620px at 76% -9%,#1e3a5f 0%,#0f172a 56%)
     ,var(--bg);color:var(--ink);font-family:"Segoe UI",-apple-system,sans-serif;
     min-height:100vh}}
.wrap{{max-width:880px;margin:0 auto;padding:44px 22px 30px}}
.eyebrow{{color:var(--accent);font-size:.78rem;letter-spacing:.14em;
         text-transform:uppercase;font-weight:600}}
h1{{font-size:2rem;margin:.25em 0 .35rem;letter-spacing:-.01em}}
.sub{{color:var(--dim);max-width:52ch;line-height:1.55}}
.badge{{display:inline-block;border:1px solid var(--line);border-radius:999px;
       padding:.18rem .6rem;font-size:.72rem;color:var(--dim);margin-right:.4rem}}
.metric{{display:flex;gap:.55rem;flex-wrap:wrap;margin:1.6rem 0}}
.metric>div{{background:var(--card);border:1px solid var(--line);
            border-radius:14px;padding:.85rem 1.05rem;min-width:140px;flex:1}}
.metric .v{{font-size:1.25rem;font-weight:700;color:var(--accent)}}
.metric .k{{font-size:.72rem;color:var(--dim);margin-top:.15rem;
           letter-spacing:.05em}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));
      gap:.7rem}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;
      padding:.85rem .95rem}}
.card a{{text-decoration:none}}
.ep{{color:var(--ink);font-weight:600;font-size:.92rem}}
.method{{display:inline-block;font-size:.62rem;font-weight:700;padding:.14rem
        .45rem;border-radius:6px;margin-right:.5rem;background:var(--tag);
        color:var(--accent)}}
.m-get{{color:var(--good)}}.m-post{{color:var(--warn)}}
.card p{{color:var(--dim);font-size:.8rem;margin-top:.35rem;line-height:1.45}}
.cta{{display:inline-block;margin:1.6rem .35rem .2rem;padding:.6rem 1.15rem;
     border-radius:10px;font-weight:600;font-size:.9rem;text-decoration:none;
     background:linear-gradient(135deg,var(--accent2),var(--accent));color:#082f49}}
.cta.ghost{{background:none;border:1px solid var(--line);color:var(--ink)}}
footer{{margin-top:2.2rem;padding-top:1rem;border-top:1px solid var(--line);
       color:var(--dim);font-size:.74rem;line-height:1.6}}
code{{background:#0b1220;border:1px solid var(--line);padding:.08rem .35rem;
     border-radius:6px;font-size:.82em;color:var(--accent)}}
</style></head><body>
<div class="wrap">
  <div class="eyebrow">MLOps pipeline & middot Faster R-CNN &middot ONNX on CPU</div>
  <h1>{TITLE}</h1>
  <p class="sub">Serving pedestrian / cyclist / vehicle detection behind a
  5-job CI pipeline; the drift monitor watches every served frame and
  retraining happens behind a human-approval gate.</p>
  <div class="metric">
    <div><div class="v">0.8854</div><div class="k">RECALL @ FPR &le; 1%</div></div>
    <div><div class="v">0.8167</div><div class="k">AP</div></div>
    <div><div class="v">~266&nbsp;ms</div><div class="k">CPU INFERENCE</div></div>
  </div>
  <div class="grid">
    <div class="card"><span class="method m-get">GET</span>
      <a class="ep" href="https://peds-detect.duckdns.org:8000/health">/health</a>
      <p>Liveness probe &mdash; the container HEALTHCHECK and any future ALB use it.</p></div>
    <div class="card"><span class="method m-get">GET</span>
      <a class="ep" href="https://peds-detect.duckdns.org:8000/model">/model</a>
      <p>Which model and <em>git_sha</em> is serving &mdash; the deploy proof.</p></div>
    <div class="card"><span class="method m-post">POST</span>
      <span class="ep">/detect</span>
      <p>PNG/JPEG in &rarr; bounding boxes out. <code>score_thresh</code> is
      request-level &mdash; the notebook's safety point is 0.71.</p></div>
    <div class="card"><span class="method m-get">GET</span>
      <a class="ep" href="https://peds-detect.duckdns.org:8000/monitor/features">/monitor/features</a>
      <p>9 aggregated input features per served frame &mdash; what the drift job
      compares to the training baseline.</p></div>
    <div class="card"><span class="method m-get">GET</span>
      <a class="ep" href="https://peds-detect.duckdns.org:8000/meta">/meta</a>
      <p>Service metadata as JSON (moved off <code>/</code>, which is now this
      page).</p></div>
    <div class="card"><span class="method m-get">GET</span>
      <span class="ep">/docs</span>
      <p>Interactive OpenAPI, restyled to match this page.</p></div>
  </div>
  <div>
    <a class="cta" href="/docs">Open interactive docs</a>
    <a class="cta ghost" href="/health">Health</a>
  </div>
  <footer>
    <span class="badge">zero stored secrets</span>
    <span class="badge">DuckDNS-updated domain</span>
    <span class="badge">self-redeploying via GHCR</span>
    <br>Demo API - unauthenticated; put it behind a reverse proxy or an ALB
    before serving real traffic.
  </footer>
</div>
</body></html>"""

# The docs page: inject the real dark <style> into the HTML that FastAPI's
# standard get_swagger_ui_html() produces, before </head> so it lands after
# swagger-ui.css and wins the cascade. Kept out of the OpenAPI schema.
_SWAGGER_DARK = (
    _CSS
    + """
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-thumb{background:var(--line);border-radius:5px}
::-webkit-scrollbar-track{background:var(--bg)}
.swagger-ui .information-container,.swagger-ui .wrapper,.swagger-ui .responses,
.swagger-ui .models,.swagger-ui #models,.swagger-ui .models-control
{background:transparent!important}
.swagger-ui .info .title{color:#f1f5f9!important}
.swagger-ui .info .title small{background:var(--line)!important;color:var(--ink)}
.swagger-ui .info .title small pre{color:var(--dim)}
.swagger-ui .info .description p,.swagger-ui .info .description div,
.swagger-ui .info .description li,.swagger-ui .info .description em,
.swagger-ui .info em,.swagger-ui .info a{color:#cbd5e1!important}
.swagger-ui .info .description code{background:var(--code)!important;
    color:var(--accent)!important;padding:.1rem .35rem;border-radius:4px!important}
.swagger-ui .scheme-container{background:var(--card)!important;
    box-shadow:none!important;border:none!important}
.swagger-ui .scheme-container .auth-wrapper .authorize{background:transparent}
.swagger-ui .auth-wrapper .authorize svg{fill:var(--accent)}
.swagger-ui .opblock-tag{color:var(--ink)!important;
    border-bottom:1px solid var(--line)!important;background:transparent!important}
.swagger-ui .opblock-tag svg{fill:var(--dim)}
.swagger-ui .opblock{background:var(--card)!important;
    border:1px solid var(--line)!important;box-shadow:0 1px 2px #0009!important}
.swagger-ui .opblock.opblock-get{background:rgba(56,189,248,.06)!important;
    border-color:#164e63!important}
.swagger-ui .opblock.opblock-post{background:rgba(74,222,128,.05)!important;
    border-color:#14532d!important}
.swagger-ui .opblock .opblock-summary-path,
.swagger-ui .opblock .opblock-summary-path__deprecated
{color:var(--ink)!important;text-shadow:none}
.swagger-ui .opblock .opblock-summary-description{color:var(--dim)!important}
.swagger-ui .opblock-section-header{background:#0f1d33!important;
    box-shadow:none!important;border-bottom:1px solid var(--line)!important}
.swagger-ui .opblock-section-header h4{color:#cbd5e1!important}
.swagger-ui .opblock-section-header .btn{border-color:var(--line)!important;
    color:var(--accent)!important;background:var(--card)!important}
.swagger-ui .parameter__name{color:var(--ink)!important}
.swagger-ui .parameter__name span{color:var(--dim)!important}
.swagger-ui .parameter__type{color:var(--dim)!important}
.swagger-ui .parameter__in,.swagger-ui .parameter__extension
{color:var(--dim)!important;background:#334155!important}
.swagger-ui table thead th,.swagger-ui table tbody td
{color:#cbd5e1!important;border-bottom:1px solid var(--line)!important}
.swagger-ui .response-col_status,.swagger-ui .response-col_description
{color:#cbd5e1!important}
.swagger-ui input,.swagger-ui select,.swagger-ui textarea,
.swagger-ui .body-param__text{background:var(--code)!important;
    color:var(--ink)!important;border:1px solid var(--line)!important}
.swagger-ui .btn{background:var(--card)!important;color:var(--ink)!important;
    border:1px solid var(--line)!important;box-shadow:none}
.swagger-ui .btn.try-out__btn{color:var(--accent)!important;
    background:transparent!important}
.swagger-ui .btn.cancel{color:#f87171!important;background:transparent!important;
    border:0!important}
.swagger-ui .execute{background:#0ea5e9!important;color:#04283b!important;
    border-color:#0ea5e9!important}
.swagger-ui .execute:hover{background:#38bdf8!important;color:#04283b!important}
.swagger-ui .markdown p,.swagger-ui .md,.swagger-ui .text{color:#cbd5e1!important}
.swagger-ui pre,.swagger-ui .highlight-code,.swagger-ui .microlight
{background:var(--code)!important;color:var(--ink)!important;
    border:1px solid var(--line)!important}
.swagger-ui .CodeMirror{background:var(--code)!important;
    color:var(--ink)!important}
.swagger-ui .cm-property{color:#7dd3fc!important}
.swagger-ui .cm-string{color:#a5b4fc!important}
.swagger-ui .cm-number{color:#fbbf24!important}
.swagger-ui .cm-atom,.swagger-ui .cm-keyword{color:#f472b6!important}
.swagger-ui .models h4,.swagger-ui #models h4,.swagger-ui .models-title
{color:var(--ink)!important}
.swagger-ui .model-box{background:var(--card)!important;
    border:1px solid var(--line)!important;padding:10px 12px!important;
    border-radius:8px}
.swagger-ui .model-title{color:var(--ink)!important}
.swagger-ui .model .prop .prop-name{color:#7dd3fc!important}
.swagger-ui .model .prop .prop-type{color:#fbbf24!important}
.swagger-ui .model .property.primitive .prop-type,
.swagger-ui .model .prop .prop-format{color:#a5b4fc!important}
.swagger-ui .model .model-toggle svg{fill:var(--line)}
.swagger-ui .dialog-ux .backdrop-ux{background:#020617cc}
.swagger-ui .dialog-ux .modal-ux{background:var(--card)!important;
    border:1px solid var(--line)}
.swagger-ui .dialog-ux .modal-ux-header h3,
.swagger-ui .dialog-ux .modal-ux-content h4{color:var(--ink)!important}
.swagger-ui .dialog-ux .modal-ux-content p{color:var(--dim)!important}
.swagger-ui .auth-container{border:1px solid var(--line)!important;
    background:#0f1d33!important}
.swagger-ui .auth-container input{background:var(--code)!important;
    color:var(--ink)!important}
.swagger-ui .btn.authorize{background:transparent!important;
    border:1px solid #4ade8080!important;color:#4ade80!important}
.swagger-ui .errors .error,.swagger-ui .errors .error h3{color:#fca5a5!important}"""
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, release it at shutdown."""
    device = os.getenv("INFERENCE_DEVICE", "cpu")
    app.state.service = ModelService.from_env(device=device)
    svc = app.state.service
    if svc.loaded:
        print(f"[api] model loaded: {svc.source} | classes={svc.classes} "
              f"| device={svc.device}")
    else:
        # Do NOT crash the container: /health must stay answerable so the
        # failure is observable, and the orchestrator can act on it.
        print(f"[api] MODEL FAILED TO LOAD: {svc.error}")
    yield
    app.state.service = None


app = FastAPI(
    title=TITLE,
    version=VERSION,
    description=(
        "Serves a Faster R-CNN pedestrian / cyclist / vehicle detector.\n\n"
        "Runs on EC2 behind the deployer poller; `/health` is the container "
        "HEALTHCHECK and the endpoint any future load balancer would use."
    ),
    lifespan=lifespan,
    docs_url=None,  # replaced by the themed /docs route below
    swagger_ui_parameters={
        "docExpansion": "list",
        "tryItOutEnabled": True,
    },
)


@app.get("/docs", include_in_schema=False, tags=["meta"],
         summary="Interactive docs (dark-themed)")
def docs() -> HTMLResponse:
    """The stock Swagger UI HTML, with the dark <style> injected before
    </head>. The customCss *config* option was silently ignored by the
    bundled Swagger UI build (verified live on 2026-11-07) — editing the
    served HTML is version-proof and costs no extra files."""
    page = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{TITLE} - Swagger UI",
        oauth2_redirect_url=None,
        swagger_ui_parameters={"docExpansion": "list", "tryItOutEnabled": True},
    )
    # FastAPI >= 0.93 returns an HTMLResponse, older versions a plain str -
    # normalise to the HTML text either way.
    html = page.body.decode("utf-8") if isinstance(page, Response) else page
    assert "</head>" in html, "swagger html shape changed"
    return HTMLResponse(html.replace("</head>", f"<style>{_SWAGGER_DARK}</style></head>"))


def _service() -> ModelService:
    svc = getattr(app.state, "service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="service is starting up")
    return svc


def _capture(image, detections: list, elapsed_ms: float) -> None:
    """Append one feature row per served frame (Step 9's input to drift).

    Opt-in via FEATURES_LOG; *nothing here may ever affect inference* — a
    failed or disabled capture is swallowed. Only aggregates are stored
    (9 image features + counts), never the image itself.
    """
    path = os.getenv("FEATURES_LOG")
    if not path:
        return
    try:
        from src.monitoring.features import append_row, extract_features

        row = extract_features(image)
        row["n_detections"] = len(detections)
        row["mean_score"] = round(
            max((d.get("score", 0.0) for d in detections), default=0.0), 4)
        row["inference_ms"] = round(elapsed_ms, 2)
        row["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        append_row(path, row)
    except Exception as exc:  # noqa: BLE001 — monitoring must not break serving
        print(f"[api] feature capture failed (ignored): {exc}")


@app.get("/", tags=["meta"], summary="Landing page", response_class=HTMLResponse)
def root() -> str:
    return _LANDING


@app.get("/meta", tags=["meta"], summary="Service metadata")
def meta() -> dict:
    return {"service": TITLE, "version": VERSION,
            "landing": "/", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["ops"],
         summary="Liveness probe for ECS / ALB")
def health() -> HealthResponse:
    svc = _service()
    return HealthResponse(
        status="ok" if svc.loaded else "degraded",
        model_loaded=svc.loaded,
    )


@app.get("/model", response_model=ModelInfo, tags=["ops"],
         summary="Which model is serving, and with which classes")
def model_info() -> ModelInfo:
    # git_sha comes from the image environment (CI sets GIT_SHA at build),
    # so this endpoint answers "which CODE is live" — the observable proof
    # that a push-to-main deploy actually reached the box.
    import os

    return ModelInfo(**_service().info(),
                     git_sha=os.environ.get("GIT_SHA", "unknown"))


@app.post("/detect", response_model=DetectResponse, tags=["inference"],
          summary="Detect pedestrians, cyclists and vehicles in an image")
async def detect(
    file: UploadFile = File(  # noqa: B008 - FastAPI declares dependencies this way
        ..., description="PNG or JPEG image"),
    score_thresh: float = Query(
        DEFAULT_THRESHOLD, ge=0.0, le=1.0,
        description="Minimum confidence. Raise to reduce false positives; "
                    "the notebook's safety operating point is 0.71.",
    ),
) -> DetectResponse:
    svc = _service()

    if not svc.loaded:
        raise HTTPException(
            status_code=503,
            detail=f"model not loaded: {svc.error}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        image = ModelService.read_image(data)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"could not decode image: {type(exc).__name__}",
        ) from exc

    try:
        detections, elapsed_ms = svc.detect(image, score_thresh)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:                      # tensor shape errors etc.
        raise HTTPException(
            status_code=500, detail=f"inference failed: {exc}"
        ) from exc

    _capture(image, detections, elapsed_ms)

    return DetectResponse(
        detections=[Detection(**d) for d in detections],
        image=ImageInfo(width=image.width, height=image.height),
        threshold=score_thresh,
        inference_ms=elapsed_ms,
        model_source=svc.source,
        model_version=svc.model_version,
        classes=svc.classes,
    )


@app.get("/monitor/features", response_model=MonitorFeaturesResponse,
         tags=["ops"],
         summary="Captured input features (the drift monitor's current batch)")
def monitor_features(
    limit: int = Query(
        1000, ge=1, le=20000,
        description="Return at most this many of the most recent rows.",
    ),
) -> MonitorFeaturesResponse:
    """What the drift job reads: `drift.py --url .../monitor/features`.

    Returns `capture: "off"` when FEATURES_LOG is unset — a monitor must be
    able to tell 'no drift' apart from 'not watching'.
    """
    from src.monitoring.features import read_rows

    path = os.getenv("FEATURES_LOG")
    if not path:
        return MonitorFeaturesResponse(capture="off", count=0, rows=[])
    rows = read_rows(path, limit=limit)
    return MonitorFeaturesResponse(capture="on", source=path,
                                   count=len(rows), rows=rows)
