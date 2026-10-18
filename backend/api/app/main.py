from fastapi import FastAPI

app = FastAPI(title="RadarSense API")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "api"}
