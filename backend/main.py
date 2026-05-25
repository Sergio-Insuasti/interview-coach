from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")

@app.websocket("/ws/interview/{session_id}")
async def interview_stream(websocket: WebSocket, session_id: str):
    await websocket.app()
    while True:
        data = await websocket.receive_bytes()
        
        await websocket.send_json({"status": "received", "bytes": len(data)})
