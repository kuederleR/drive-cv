"""
Mobile Web Application Server for DriveCV ADAS & 3D HUD.
Serves Flask HTTP endpoints, MJPEG live video feeds, and WebSocket JSON telemetry.
"""

import asyncio
import json
import logging
import os
import threading
import time
from typing import Optional, Set
import cv2
from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.serving import WSGIRequestHandler
import websockets
from drivecv.config import PipelineConfig
from drivecv.pipeline import ADASPipeline, ScaledVideoCapture
from drivecv.types import FrameData

# Suppress verbose web logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


class QuietWSGIRequestHandler(WSGIRequestHandler):
    """Custom WSGI request handler that handles HTTPS handshake attempts gracefully on HTTP port."""

    def log_error(self, format, *args):
        if args and isinstance(args[0], str) and ("Bad request version" in args[0] or "Bad request syntax" in args[0]):
            client_ip = self.client_address[0] if self.client_address else "Client"
            print(
                f"[WARNING] HTTPS connection attempted from {client_ip} on HTTP port {self.server.server_port}.\n"
                f"          👉 On Mobile Safari/iPhone, please type 'http://' explicitly: http://<jetson-ip>:{self.server.server_port}"
            )
            return
        super().log_error(format, *args)


class ADASWebServer:
    """
    Web Application Server:
    - Runs ADAS processing pipeline in a background thread.
    - Serves 3D HUD Web UI (HTML/CSS/Three.js/WebAudio).
    - Streams live MJPEG camera feed on /video_feed.
    - Broadcasts high-frequency 3D telemetry over WebSocket.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        host: str = "0.0.0.0",
        port: int = 5000,
        ws_port: Optional[int] = None,
        output_path: Optional[str] = None,
        default_source: str = "camera",
    ):
        self.config = config or PipelineConfig()
        self.host = host
        self.port = port
        self.ws_port = ws_port or (port + 1)
        self.output_path = output_path
        self.default_source = default_source

        self.static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
        self.app = Flask(__name__, static_folder=self.static_dir)

        self.pipeline = ADASPipeline(config=self.config, headless=True, output_path=self.output_path)
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()

        self._latest_jpeg: Optional[bytes] = None
        self._latest_telemetry: Optional[dict] = None
        self._lock = threading.Lock()
        self._running = False
        self._is_paused = False
        self._step_frame = False
        self._active_source = default_source  # "camera" or "video"
        self._pending_source_change: Optional[str] = None
        self._pending_video_path: Optional[str] = None
        self._current_video_path: Optional[str] = None
        self.demo_video_path: Optional[str] = None
        self.camera_device = os.environ.get("CAMERA_DEVICE", "0")

        self._setup_routes()

    def _get_available_recordings(self) -> list:
        """Discovers local video recordings (e.g. .avi, .mp4) from Jetson dashcam recordings directory."""
        recordings = []

        if self.demo_video_path and os.path.exists(self.demo_video_path):
            recordings.append({
                "name": f"Demo Video ({os.path.basename(self.demo_video_path)})",
                "path": self.demo_video_path,
                "is_default": True,
                "size_mb": round(os.path.getsize(self.demo_video_path) / (1024 * 1024), 1) if os.path.exists(self.demo_video_path) else 0.0
            })

        candidate_dirs = []
        env_dir = os.environ.get("RECORDINGS_DIR")
        if env_dir:
            candidate_dirs.append(env_dir)

        default_dashcam_dir = "/home/ryan/beaterai/dashcam_app/recordings"
        if default_dashcam_dir not in candidate_dirs:
            candidate_dirs.append(default_dashcam_dir)

        container_recordings_dir = "/recordings"
        if container_recordings_dir not in candidate_dirs:
            candidate_dirs.append(container_recordings_dir)

        local_rec = os.path.join(os.getcwd(), "recordings")
        if local_rec not in candidate_dirs:
            candidate_dirs.append(local_rec)

        seen_paths = {r["path"] for r in recordings}
        valid_exts = (".avi", ".mp4", ".mkv", ".mov")

        for d in candidate_dirs:
            if os.path.exists(d) and os.path.isdir(d):
                try:
                    for f in sorted(os.listdir(d)):
                        if f.lower().endswith(valid_exts):
                            full_p = os.path.join(d, f)
                            if full_p not in seen_paths:
                                seen_paths.add(full_p)
                                try:
                                    sz_mb = round(os.path.getsize(full_p) / (1024 * 1024), 1)
                                except Exception:
                                    sz_mb = 0.0
                                recordings.append({
                                    "name": f,
                                    "path": full_p,
                                    "size_mb": sz_mb,
                                    "dir": d
                                })
                except Exception as e:
                    print(f"[WARNING] Error scanning recordings dir '{d}': {e}")

        return recordings

    def _setup_routes(self):
        @self.app.route("/")
        def index():
            return send_from_directory(self.static_dir, "index.html")

        @self.app.route("/static/<path:filename>")
        def serve_static(filename):
            return send_from_directory(self.static_dir, filename)

        @self.app.route("/manifest.json")
        def manifest():
            return send_from_directory(self.static_dir, "manifest.json")

        @self.app.route("/video_feed")
        def video_feed():
            return Response(
                self._generate_mjpeg(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @self.app.route("/api/recordings", methods=["GET"])
        def api_recordings():
            recordings = self._get_available_recordings()
            with self._lock:
                current_v = self._current_video_path or self.demo_video_path
            return jsonify({
                "status": "ok",
                "recordings": recordings,
                "current_video_path": current_v,
            })

        @self.app.route("/api/control", methods=["POST"])
        def api_control():
            data = request.get_json() or {}
            action = data.get("action")
            if action == "pause":
                self._is_paused = True
            elif action == "play":
                self._is_paused = False
            elif action == "step":
                self._step_frame = True
            return jsonify({"status": "ok", "paused": self._is_paused})

        @self.app.route("/api/source", methods=["GET", "POST"])
        def api_source():
            if request.method == "POST":
                data = request.get_json() or {}
                new_source = data.get("source")
                target_path = data.get("video_path")

                if new_source in ("camera", "video"):
                    with self._lock:
                        self._pending_source_change = new_source
                        if target_path:
                            self._pending_video_path = target_path
                    return jsonify({
                        "status": "ok",
                        "requested_source": new_source,
                        "video_path": target_path or self._current_video_path,
                        "active_source": self._active_source
                    })
                return jsonify({"status": "error", "message": "Invalid source mode."}), 400

            with self._lock:
                active = self._active_source
                curr_video = self._current_video_path or self.demo_video_path
            return jsonify({
                "status": "ok",
                "active_source": active,
                "current_video_path": curr_video,
                "demo_video": self.demo_video_path,
                "camera_device": str(self.camera_device),
            })

        @self.app.route("/api/calibrate", methods=["POST"])
        def api_calibrate():
            data = request.get_json() or {}
            action = data.get("action")
            side = data.get("side")
            if action == "start" and side in ("left", "right"):
                self.pipeline.adas.ldw.start_calibration(side)
            elif action == "reset":
                self.pipeline.adas.ldw.reset_calibration()
            calib = self.pipeline.adas.ldw.get_calibration_dict()
            return jsonify({"status": "ok", "calibration": calib})

        @self.app.route("/api/telemetry")
        def api_telemetry():
            with self._lock:
                telemetry = self._latest_telemetry
            return jsonify(telemetry or {})

    def _generate_mjpeg(self):
        """Generates MJPEG multipart stream from latest processed frame."""
        while self._running:
            with self._lock:
                jpeg = self._latest_jpeg
            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            time.sleep(0.033)

    def _pipeline_loop(self, video_path: str, max_frames: Optional[int] = None):
        """Background thread running ADAS pipeline processing."""
        self.demo_video_path = video_path
        with self._lock:
            if not self._current_video_path:
                self._current_video_path = video_path

        requested = self._active_source

        def _open_source(src_type: str, custom_video_path: Optional[str] = None):
            if src_type == "camera":
                device = os.environ.get("CAMERA_DEVICE", "0")
                print(f"[INFO] Attempting to open Live USB Camera device '{device}'...")
                try:
                    c = ScaledVideoCapture(device, self.config.width, self.config.height)
                    print(f"[INFO] Successfully connected to Live Camera (device={device})!")
                    return c, "camera"
                except Exception as e:
                    print(f"[WARNING] Live camera access failed: {e}")
                    src_type = "video"

            v_file = custom_video_path or self._current_video_path or video_path
            print(f"[INFO] Opening video recording '{v_file}'...")
            try:
                c = ScaledVideoCapture(v_file, self.config.width, self.config.height, use_ffmpeg=self.config.use_ffmpeg_scale)
                return c, "video"
            except Exception as e:
                print(f"[WARNING] Could not open requested video '{v_file}': {e}")
                if v_file != video_path and os.path.exists(video_path):
                    print(f"[INFO] Falling back to default demo video '{video_path}'...")
                    try:
                        self._current_video_path = video_path
                        c = ScaledVideoCapture(video_path, self.config.width, self.config.height, use_ffmpeg=self.config.use_ffmpeg_scale)
                        return c, "video"
                    except Exception as e2:
                        print(f"[ERROR] Could not open fallback demo video '{video_path}': {e2}")

            # Ultimate fallback if video opening fails
            print("[WARNING] Using camera/stub fallback...")
            try:
                c = ScaledVideoCapture(0, self.config.width, self.config.height)
                return c, "camera"
            except Exception:
                class StubCap:
                    fps = 30.0
                    frame_count = 1000
                    is_live = False
                    width = self.config.width
                    height = self.config.height
                    def read(self):
                        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                        return True, frame
                    def release(self):
                        pass
                return StubCap(), "video"

        cap, current_src = _open_source(requested)
        with self._lock:
            self._active_source = current_src

        video_fps = cap.fps
        self.config.fps = video_fps
        frame_delay = 1.0 / max(1.0, video_fps)
        frame_idx = 1

        ret, frame = cap.read()
        if not ret or frame is None:
            if current_src == "camera":
                print("[INFO] Live camera gave no initial frame. Falling back to video recording...")
                cap.release()
                cap, current_src = _open_source("video")
                with self._lock:
                    self._active_source = current_src
                video_fps = cap.fps
                self.config.fps = video_fps
                frame_delay = 1.0 / max(1.0, video_fps)
                ret, frame = cap.read()

        while self._running:
            t_start = time.perf_counter()

            # Check for pending source or video path change request from API
            with self._lock:
                pending_src = self._pending_source_change
                pending_vpath = self._pending_video_path
                self._pending_source_change = None
                self._pending_video_path = None

            if pending_vpath:
                self._current_video_path = pending_vpath

            if (pending_src and pending_src != current_src) or pending_vpath:
                target_src = pending_src or current_src
                print(f"[INFO] Switching input source -> '{target_src}' (file: {self._current_video_path})...")
                try:
                    cap.release()
                except Exception:
                    pass
                self.pipeline.tracker.clear()
                cap, current_src = _open_source(target_src, custom_video_path=self._current_video_path)
                with self._lock:
                    self._active_source = current_src
                video_fps = cap.fps
                self.config.fps = video_fps
                frame_delay = 1.0 / max(1.0, video_fps)
                frame_idx = 1
                ret, frame = cap.read()

            if not self._is_paused or self._step_frame:
                if not ret or frame is None:
                    if cap.is_live:
                        time.sleep(0.01)
                        ret, frame = cap.read()
                        continue
                    else:
                        print("[INFO] Reached end of video recording. Looping back to start...")
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap, current_src = _open_source("video", custom_video_path=self._current_video_path)
                        frame_idx = 1
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            print("[WARNING] Loop back read failed. Re-opening fallback video...")
                            self._current_video_path = video_path
                            cap, current_src = _open_source("video", custom_video_path=video_path)
                            frame_idx = 1
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                time.sleep(0.05)
                                continue

                try:
                    timestamp = frame_idx / max(1.0, video_fps)
                    proc_frame, frame_data = self.pipeline.process_frame(frame, frame_idx, timestamp)
                    self._step_frame = False

                    # Render panoptic visualization overlay for 2D camera feed
                    vis_frame = self.pipeline.visualizer.render(proc_frame, frame_data)
                    if self.config.visualizer.show_hud:
                        self.pipeline.hud.draw(
                            vis_frame, frame_data, total_frames=cap.frame_count, auto_schedule=True
                        )

                    # Encode JPEG for MJPEG stream
                    ret_jpg, encoded_jpg = cv2.imencode(".jpg", vis_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    telemetry = ADASPipeline.get_telemetry_dict(frame_data, self.pipeline.adas)
                    telemetry["active_source"] = current_src
                    telemetry["current_video_path"] = self._current_video_path

                    with self._lock:
                        if ret_jpg and encoded_jpg is not None and len(encoded_jpg) > 500:
                            self._latest_jpeg = encoded_jpg.tobytes()
                        self._latest_telemetry = telemetry

                    # Broadcast telemetry via WebSocket event loop
                    if self._loop is not None and self.connected_clients:
                        asyncio.run_coroutine_threadsafe(
                            self._broadcast_telemetry(telemetry), self._loop
                        )

                    # Advance frame
                    ret, frame = cap.read()
                    frame_idx += 1
                except Exception as e:
                    print(f"[ERROR] Error processing frame {frame_idx}: {e}")
                    time.sleep(0.05)
                    ret, frame = cap.read()
                    frame_idx += 1

                if max_frames and frame_idx > max_frames:
                    print(f"[INFO] Reached max frame limit ({max_frames}). Stopping loop...")
                    break

            elapsed = time.perf_counter() - t_start
            sleep_time = max(0.001, frame_delay - elapsed)
            time.sleep(sleep_time)

        cap.release()
        print("[INFO] Pipeline loop terminated.")

    async def _broadcast_telemetry(self, telemetry: dict):
        """Pushes telemetry JSON to all connected WebSockets."""
        if not self.connected_clients:
            return
        msg = json.dumps(telemetry)
        stale_clients = set()
        for client in list(self.connected_clients):
            try:
                await client.send(msg)
            except Exception:
                stale_clients.add(client)
        self.connected_clients -= stale_clients

    async def _ws_handler(self, websocket):
        """Handles incoming WebSocket connections."""
        self.connected_clients.add(websocket)
        print(f"[INFO] Client connected to 3D HUD WebSocket (Total: {len(self.connected_clients)})")
        try:
            # Send latest telemetry snapshot upon initial connection
            with self._lock:
                snapshot = self._latest_telemetry
            if snapshot:
                await websocket.send(json.dumps(snapshot))
            async for _ in websocket:
                pass  # Keep connection open
        except Exception:
            pass
        finally:
            self.connected_clients.remove(websocket)
            print(f"[INFO] Client disconnected from WebSocket (Total: {len(self.connected_clients)})")

    def _start_ws_server(self):
        """Runs the WebSocket server event loop."""
        def process_request(connection, request):
            if request.headers.get("Upgrade", "").lower() != "websocket":
                host_hdr = request.headers.get("Host", self.host).split(":")[0]
                target_url = f"http://{host_hdr}:{self.port}/"
                return connection.respond(
                    302,
                    [("Location", target_url), ("Content-Type", "text/html")],
                    b"<html><body>Redirecting to <a href='" + target_url.encode() + b"'>DriveCV 3D HUD</a>...</body></html>",
                )
            return None

        async def main():
            self._loop = asyncio.get_running_loop()
            async with websockets.serve(
                self._ws_handler, self.host, self.ws_port, process_request=process_request
            ):
                print(f"[INFO] WebSocket Telemetry Server running on ws://{self.host}:{self.ws_port}/ws")
                await asyncio.Future()  # Keep server alive

        try:
            asyncio.run(main())
        except Exception as e:
            print(f"[WARNING] WebSocket server stopped: {e}")



    def run(self, video_path: str, max_frames: Optional[int] = None):
        """Starts web server and pipeline execution."""
        self._running = True

        # 1. Start WebSocket Thread
        self.ws_thread = threading.Thread(target=self._start_ws_server, daemon=True)
        self.ws_thread.start()

        # 2. Start Pipeline Thread
        self.pipeline_thread = threading.Thread(
            target=self._pipeline_loop, args=(video_path, max_frames), daemon=True
        )
        self.pipeline_thread.start()

        print(f"\n=======================================================")
        print(f" 🚀 DriveCV 3D Ego-Lane & ADAS Web Server Active!")
        print(f" 🌐 Web UI: http://{self.host}:{self.port} or http://localhost:{self.port}")
        print(f" 📡 WebSocket: ws://{self.host}:{self.ws_port}/ws")
        print(f"=======================================================\n")

        # 3. Start Flask HTTP Server
        try:
            self.app.run(
                host=self.host,
                port=self.port,
                debug=False,
                use_reloader=False,
                threaded=True,
                request_handler=QuietWSGIRequestHandler,
            )
        finally:
            self._running = False
            print("[INFO] Web server shutting down...")
