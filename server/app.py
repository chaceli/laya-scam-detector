"""FastAPI app for the Laya scam-detection playground."""
from contextlib import asynccontextmanager
import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import config
from server.schemas import (
    HealthResponse,
    PRIMITIVE_TO_KEY,
    PredictRequest,
    PredictResponse,
    Sample,
    SamplesResponse,
)

SAMPLES = [
    Sample(id="zh-scam", label="中文·快递理赔诈骗", lang="zh",
           text="您好，我是XX快递客服，您有一个包裹在运输途中丢失，请点击链接填写个人信息进行理赔。"),
    Sample(id="zh-benign", label="中文·家人问候", lang="zh",
           text="妈，我今晚回家吃饭，大概6点到家。"),
    Sample(id="en-scam", label="英文·PayPal 钓鱼", lang="en",
           text="URGENT: Your PayPal account has been limited. Click here to verify: http://paypa1-secure.tk/login"),
    Sample(id="en-benign", label="英文·会议确认", lang="en",
           text="Hi, just confirming our meeting at 3pm tomorrow. Looking forward to it."),
]


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.router = None
        app.state.load_error = None
        try:
            from src.router import Router
            app.state.router = Router(
                english_dir=config.ENGLISH_DIR,
                multilingual_dir=config.MULTILINGUAL_DIR,
            )
        except Exception as e:
            app.state.load_error = str(e)
        yield
        app.state.router = None

    app = FastAPI(title="Laya Scam Playground", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        router = app.state.router
        return HealthResponse(
            ok=router is not None,
            models={
                "english": bool(router and router.english),
                "multilingual": bool(router and getattr(router, "multilingual", None)),
            },
            multilingual_dir=config.MULTILINGUAL_DIR,
            load_error=app.state.load_error,
        )

    @app.get("/api/defaults")
    def defaults() -> dict:
        return {"questions": json.loads(config.SCHEMA_PATH.read_text())}

    @app.get("/api/samples", response_model=SamplesResponse)
    def samples() -> SamplesResponse:
        return SamplesResponse(samples=SAMPLES)

    @app.post("/api/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        router = app.state.router
        if router is None:
            raise HTTPException(
                status_code=503,
                detail=f"model not loaded: {app.state.load_error}",
            )

        if req.questions is not None:
            questions = {
                PRIMITIVE_TO_KEY[p]: req.questions[PRIMITIVE_TO_KEY[p]]
                for p in req.primitives
                if PRIMITIVE_TO_KEY[p] in req.questions
            }
            if not questions:
                raise HTTPException(
                    status_code=422, detail="no matching questions provided"
                )
        else:
            schema = json.loads(config.SCHEMA_PATH.read_text())
            questions = {
                PRIMITIVE_TO_KEY[p]: schema[PRIMITIVE_TO_KEY[p]]
                for p in req.primitives
            }

        model = None if req.model == "auto" else req.model
        try:
            result = router.predict(
                req.text, questions, max_len=req.max_len, model=model
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"inference failed: {e}")

        return PredictResponse(
            answers=result["answers"],
            routing=result["routing"],
            latency_ms=result["latency_ms"],
        )

    if config.WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(config.WEB_DIR)), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(str(config.WEB_DIR / "index.html"))

    return app


app = create_app()