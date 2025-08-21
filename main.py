from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from db.connection import initDB
from routes.user_routes import router, app_router
from routes.auth import auth_router
from routes.courses import router as c_router, app_router as c2_router
from routes.job import router as j_router, app_router as j2_router
from routes.youtube_routes import router as yt_router
from dotenv import load_dotenv
from ping_render import lifespan
from settings import BASE_URL, FRONTEND_URL
from services.websocket_manager import ws_manager
from fastapi import WebSocket, WebSocketDisconnect

load_dotenv()
initDB()


# origins = [
#     "http://localhost:3000",
#     "http://localhost:3001",
#     BASE_URL,
#     FRONTEND_URL,
#     "https://68a3f0b0adbb71e3ea5921f4--skill-sage-dashboard.netlify.app",
#     "https://skill-sage-dashboard-updated.onrender.com",
# ]


origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    BASE_URL,
    FRONTEND_URL,
    "https://68a3f0b0adbb71e3ea5921f4--skill-sage-dashboard.netlify.app",
    "https://skill-sage-dashboard-updated.onrender.com",
    "http://10.71.62.28",     # Your Flutter client IP
    "ws://10.71.62.28",       # WebSocket from Flutter client
    "http://10.71.62.222:8004",  # Your server
    "ws://10.71.62.222:8004",    # WebSocket server
]


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(j2_router)
app.include_router(c2_router)
app.include_router(c_router)
app.include_router(j_router)
app.include_router(auth_router)
app.include_router(router)
app.include_router(app_router)
app.include_router(yt_router)
app.include_router(app_router)


# @app.get("/")
# def hello_world():
#     return "Hello"


@app.websocket("/ws/{user_id}/")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    import logging
    logging.warning(f"WebSocket connection attempt for user: {user_id}")

    try:
        await ws_manager.connect(user_id, websocket)

        while True:
            try:
                # Wait for any message from client
                data = await websocket.receive_text()
                print(f"Received from user {user_id}: {data}")

                # Example: send JSON instead of plain text
                await websocket.send_json({
                    "type": "jobs_updated",
                    "message": "Upload complete!"
                })

            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error for user {user_id}: {e}")
                break

    except Exception as e:
        print(f"WebSocket connection error for user {user_id}: {e}")
    finally:
        ws_manager.disconnect(user_id)


@app.get("/healthcheck")
def healthcheck():
    return "OK"
