/**
 * DriveCV 3D HUD Application
 * Three.js 3D Scene + Web Audio API Synthesizer + Calibration & Controls Drawer
 */

// --- 1. Sound Engine (Web Audio API) ---
class SoundEngine {
    constructor() {
        this.ctx = null;
        this.isEnabled = false;
        this.lastAlertTime = 0;
    }

    init() {
        if (!this.ctx) {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            this.ctx = new AudioCtx();
        }
        if (this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
        this.isEnabled = true;
    }

    playBeep(freq = 440, type = 'sine', duration = 0.15, gainVal = 0.2) {
        if (!this.isEnabled || !this.ctx) return;
        try {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = type;
            osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
            gain.gain.setValueAtTime(gainVal, this.ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + duration);

            osc.connect(gain);
            gain.connect(this.ctx.destination);
            osc.start();
            osc.stop(this.ctx.currentTime + duration);
        } catch (e) {
            console.error("Audio playback error:", e);
        }
    }

    playLDWWarning(isLeft) {
        const now = Date.now();
        if (now - this.lastAlertTime < 400) return;
        this.lastAlertTime = now;

        const baseFreq = isLeft ? 520 : 660;
        this.playBeep(baseFreq, 'square', 0.1, 0.15);
        setTimeout(() => this.playBeep(baseFreq * 1.2, 'square', 0.12, 0.15), 120);
    }

    playFCWAlert(level) {
        const now = Date.now();
        if (level === 'CAUTION' && now - this.lastAlertTime > 1500) {
            this.lastAlertTime = now;
            this.playBeep(440, 'sine', 0.2, 0.15);
        } else if (level === 'WARNING' && now - this.lastAlertTime > 600) {
            this.lastAlertTime = now;
            this.playBeep(750, 'sawtooth', 0.15, 0.25);
            setTimeout(() => this.playBeep(750, 'sawtooth', 0.15, 0.25), 180);
        } else if (level === 'CRITICAL' && now - this.lastAlertTime > 300) {
            this.lastAlertTime = now;
            this.playBeep(900, 'sawtooth', 0.1, 0.35);
            setTimeout(() => this.playBeep(1100, 'sawtooth', 0.1, 0.35), 100);
        }
    }
}

// --- 2. 3D Vehicle Generator (Three.js Low-Poly Procedural Models) ---
class VehicleFactory {
    static createEgoCar() {
        const egoGroup = new THREE.Group();

        // Car Body (Clean dark slate sedan)
        const bodyMat = new THREE.MeshPhongMaterial({
            color: 0x4c566a,
            shininess: 40,
            flatShading: true
        });
        const bodyGeo = new THREE.BoxGeometry(1.9, 0.75, 4.2);
        const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
        bodyMesh.position.y = 0.55;
        egoGroup.add(bodyMesh);

        // Cabin Glass
        const cabinMat = new THREE.MeshPhongMaterial({ color: 0x1e222a });
        const cabinGeo = new THREE.BoxGeometry(1.6, 0.65, 2.2);
        const cabinMesh = new THREE.Mesh(cabinGeo, cabinMat);
        cabinMesh.position.set(0, 1.15, -0.2);
        egoGroup.add(cabinMesh);

        // Wheels
        const wheelGeo = new THREE.CylinderGeometry(0.32, 0.32, 0.25, 16);
        const wheelMat = new THREE.MeshBasicMaterial({ color: 0x181a20 });
        wheelGeo.rotateZ(Math.PI / 2);

        const wheelPositions = [
            [-0.95, 0.32, 1.3],
            [0.95, 0.32, 1.3],
            [-0.95, 0.32, -1.3],
            [0.95, 0.32, -1.3]
        ];

        wheelPositions.forEach(pos => {
            const wheel = new THREE.Mesh(wheelGeo, wheelMat);
            wheel.position.set(...pos);
            egoGroup.add(wheel);
        });

        // Headlights (Clean White)
        const lightGeo = new THREE.BoxGeometry(0.35, 0.12, 0.1);
        const lightMat = new THREE.MeshBasicMaterial({ color: 0xeceff4 });
        const leftHead = new THREE.Mesh(lightGeo, lightMat);
        leftHead.position.set(-0.7, 0.6, -2.12);
        const rightHead = new THREE.Mesh(lightGeo, lightMat);
        rightHead.position.set(0.7, 0.6, -2.12);
        egoGroup.add(leftHead);
        egoGroup.add(rightHead);

        // Taillights (Muted Red)
        const tailMat = new THREE.MeshBasicMaterial({ color: 0xbf616a });
        const leftTail = new THREE.Mesh(lightGeo, tailMat);
        leftTail.position.set(-0.7, 0.6, 2.12);
        const rightTail = new THREE.Mesh(lightGeo, tailMat);
        rightTail.position.set(0.7, 0.6, 2.12);
        egoGroup.add(leftTail);
        egoGroup.add(rightTail);

        return egoGroup;
    }

    static createVehicleModel(type = 'car', colorHex = 0xd08770) {
        const group = new THREE.Group();
        let w = 1.9, h = 1.4, l = 4.2;

        if (type === 'suv') {
            w = 2.0; h = 1.7; l = 4.6;
        } else if (type === 'truck') {
            w = 2.4; h = 2.4; l = 6.8;
        }

        const bodyMat = new THREE.MeshPhongMaterial({
            color: colorHex,
            shininess: 30,
            flatShading: true
        });
        const bodyGeo = new THREE.BoxGeometry(w, h * 0.5, l);
        const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
        bodyMesh.position.y = h * 0.35;
        group.add(bodyMesh);

        // Cabin
        const cabinMat = new THREE.MeshPhongMaterial({ color: 0x1e222a });
        const cabinGeo = new THREE.BoxGeometry(w * 0.9, h * 0.5, l * 0.5);
        const cabinMesh = new THREE.Mesh(cabinGeo, cabinMat);
        cabinMesh.position.set(0, h * 0.75, -l * 0.05);
        group.add(cabinMesh);

        // Taillights
        const tailMat = new THREE.MeshBasicMaterial({ color: 0xbf616a });
        const tailGeo = new THREE.BoxGeometry(0.35, 0.15, 0.05);
        const tLeft = new THREE.Mesh(tailGeo, tailMat);
        tLeft.position.set(-w * 0.35, h * 0.4, l * 0.5);
        const tRight = new THREE.Mesh(tailGeo, tailMat);
        tRight.position.set(w * 0.35, h * 0.4, l * 0.5);
        group.add(tLeft);
        group.add(tRight);

        // FCW Target Reticle Bounding Ring
        const ringGeo = new THREE.RingGeometry(w * 0.7, w * 0.85, 32);
        ringGeo.rotateX(-Math.PI / 2);
        const ringMat = new THREE.MeshBasicMaterial({
            color: 0x81a1c1,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.8
        });
        const reticle = new THREE.Mesh(ringGeo, ringMat);
        reticle.position.y = 0.05;
        reticle.name = "reticle";
        group.add(reticle);

        return group;
    }
}

// --- 2b. 3D Lane Line Ribbon Generator (Solid, Dashed, Double, Colors) ---
class LaneRibbonRenderer {
    constructor(scene) {
        this.scene = scene;
        this.leftGroup = new THREE.Group();
        this.rightGroup = new THREE.Group();
        this.scene.add(this.leftGroup);
        this.scene.add(this.rightGroup);
    }

    renderLane(isLeft, xOffset, curvature, lineType, isWarning) {
        const group = isLeft ? this.leftGroup : this.rightGroup;

        while (group.children.length > 0) {
            const obj = group.children[0];
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) {
                if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
                else obj.material.dispose();
            }
            group.remove(obj);
        }

        const typeStr = (lineType || (isLeft ? 'solid_yellow' : 'solid_white')).toLowerCase();
        const isYellow = typeStr.includes('yellow');
        const isDashed = typeStr.includes('dashed');
        const isDouble = typeStr.includes('double');

        let hexColor = isYellow ? 0xffc700 : 0xebf2fa;
        if (isWarning) {
            hexColor = 0xff4444;
        }

        const baseMat = new THREE.MeshBasicMaterial({
            color: hexColor,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.92
        });

        if (isDouble) {
            const sep = 0.12;
            const leftSubX = xOffset - sep;
            const rightSubX = xOffset + sep;

            this.buildRibbonMesh(group, leftSubX, curvature, baseMat, 0.09, typeStr.includes('solid_dashed') ? false : isDashed);
            this.buildRibbonMesh(group, rightSubX, curvature, baseMat, 0.09, isDashed);
        } else {
            this.buildRibbonMesh(group, xOffset, curvature, baseMat, 0.16, isDashed);
        }
    }

    buildRibbonMesh(group, xOffset, curvature, material, width = 0.16, isDashed = false) {
        const points = [];
        const indices = [];
        const numPts = 60;
        const length = 140;

        if (!isDashed) {
            for (let i = 0; i < numPts; i++) {
                const t = i / (numPts - 1);
                const z = -t * length;
                const x = xOffset + Math.pow(t, 1.8) * curvature;

                points.push(x - width / 2, 0.02, z);
                points.push(x + width / 2, 0.02, z);
            }
            for (let i = 0; i < numPts - 1; i++) {
                const base = i * 2;
                indices.push(base, base + 1, base + 2);
                indices.push(base + 1, base + 3, base + 2);
            }
        } else {
            const cycleLen = 8.0;
            const dashLen = 3.8;
            let currentVertex = 0;

            for (let zDist = 0; zDist < length; zDist += 0.5) {
                const cyclePos = zDist % cycleLen;
                if (cyclePos <= dashLen) {
                    const t = zDist / length;
                    const z = -zDist;
                    const x = xOffset + Math.pow(t, 1.8) * curvature;

                    points.push(x - width / 2, 0.02, z);
                    points.push(x + width / 2, 0.02, z);
                    currentVertex += 2;

                    const nextZDist = zDist + 0.5;
                    const nextCyclePos = nextZDist % cycleLen;
                    if (nextCyclePos <= dashLen && nextZDist < length) {
                        const b = currentVertex - 2;
                        indices.push(b, b + 1, b + 2);
                        indices.push(b + 1, b + 3, b + 2);
                    }
                }
            }
        }

        if (points.length >= 6) {
            const geo = new THREE.BufferGeometry();
            geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(points), 3));
            if (indices.length > 0) {
                geo.setIndex(indices);
            }
            geo.computeVertexNormals();
            const mesh = new THREE.Mesh(geo, material);
            group.add(mesh);
        }
    }
}

// --- 3. Main 3D HUD Application ---
class DriveHUDApp {
    constructor() {
        this.container = document.getElementById('canvas-container');
        this.soundEngine = new SoundEngine();
        this.targetOffset = 0;
        this.currentOffset = 0;
        this.currentCurvature = 0;
        this.trackedVehicles = new Map();

        this.initThree();
        this.initEventListeners();
        this.connectWebSocket();
        this.animate();
    }

    initThree() {
        // Scene setup - Dark gray plane-less environment
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x121418);
        this.scene.fog = new THREE.FogExp2(0x121418, 0.010);

        // Higher Camera Perspective (Above & Behind Ego Car looking down highway)
        this.camera = new THREE.PerspectiveCamera(
            58,
            window.innerWidth / window.innerHeight,
            0.1,
            220
        );
        this.setCameraDefaultPOV();

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.container.appendChild(this.renderer.domElement);

        // Ambient & Directional Lighting
        const ambLight = new THREE.AmbientLight(0xffffff, 0.7);
        this.scene.add(ambLight);

        const dirLight = new THREE.DirectionalLight(0xd8dee9, 0.7);
        dirLight.position.set(10, 25, 10);
        this.scene.add(dirLight);

        // Ego Car
        this.egoCar = VehicleFactory.createEgoCar();
        this.scene.add(this.egoCar);

        // 3D Lane Ribbon Renderer
        this.laneRenderer = new LaneRibbonRenderer(this.scene);
        // Render initial default lane ribbons immediately
        this.laneRenderer.renderLane(true, -1.85, 0, 'solid_yellow', false);
        this.laneRenderer.renderLane(false, 1.85, 0, 'solid_white', false);
    }

    setCameraDefaultPOV() {
        // Higher POV as requested by user
        this.camera.position.set(0, 4.2, 7.5);
        this.camera.lookAt(0, 0.8, -30.0);
    }

    updateScene(data) {
        // Sync active input source UI button state
        if (data.active_source) {
            this.updateSourceUI(data.active_source);
        }

        // 1. Update Ego Car Position and Road Curvature
        if (data.adas) {
            this.targetOffset = data.adas.ldw_offset_m || 0.0;
        }

        // Calculate dynamic curve from vanishing point
        if (data.lanes && data.lanes.vanish_x !== null) {
            const center_x = 480;
            const delta_x = data.lanes.vanish_x - center_x;
            this.currentCurvature += ((delta_x * 0.06) - this.currentCurvature) * 0.1;
        }

        // Smooth lerp ego lateral offset
        this.currentOffset += (this.targetOffset - this.currentOffset) * 0.15;
        this.egoCar.position.x = this.currentOffset;
        this.egoCar.rotation.z = (this.targetOffset - this.currentOffset) * -0.12;

        // 2. Update 3D Lane Line Geometry & Colors
        const leftType = (data.lanes && data.lanes.left_type) ? data.lanes.left_type : 'solid_yellow';
        const rightType = (data.lanes && data.lanes.right_type) ? data.lanes.right_type : 'solid_white';

        const ldwState = data.adas ? data.adas.ldw_state : 'NORMAL';
        const isLeftWarning = (ldwState === 'WARNING_LEFT');
        const isRightWarning = (ldwState === 'WARNING_RIGHT');

        if (isLeftWarning) {
            this.soundEngine.playLDWWarning(true);
        } else if (isRightWarning) {
            this.soundEngine.playLDWWarning(false);
        }

        // Render 3D Lane Ribbons matching actual detected line types & colors
        this.laneRenderer.renderLane(true, -1.85, this.currentCurvature, leftType, isLeftWarning);
        this.laneRenderer.renderLane(false, 1.85, this.currentCurvature, rightType, isRightWarning);

        // Update 2D Telemetry Badges in Navigation Bar
        this.updateLaneBadges(leftType, rightType, isLeftWarning, isRightWarning);


        // 3. Update 3D Tracked Vehicles
        const activeIds = new Set();
        if (data.tracks) {
            data.tracks.forEach(track => {
                activeIds.add(track.track_id);
                let vehMesh = this.trackedVehicles.get(track.track_id);
                if (!vehMesh) {
                    const color = track.is_lead ? 0xd08770 : 0x5e81ac;
                    vehMesh = VehicleFactory.createVehicleModel(track.class_name, color);
                    this.scene.add(vehMesh);
                    this.trackedVehicles.set(track.track_id, vehMesh);
                }

                // Position in 3D space: Z = -distance_m, X = lateral_offset_m + curve offset
                const targetZ = -(track.distance_m || 10.0);
                const normDist = Math.abs(targetZ) / 140.0;
                const curveShift = Math.pow(normDist, 1.8) * this.currentCurvature;
                const targetX = (track.lateral_offset_m || 0.0) + curveShift;

                vehMesh.position.x += (targetX - vehMesh.position.x) * 0.2;
                vehMesh.position.z += (targetZ - vehMesh.position.z) * 0.2;

                // Reticle update
                const reticle = vehMesh.getObjectByName("reticle");
                if (reticle) {
                    if (track.is_lead) {
                        reticle.visible = true;
                        const fcwLevel = data.adas ? data.adas.fcw_level : 'SAFE';
                        if (fcwLevel === 'CRITICAL') {
                            reticle.material.color.setHex(0xbf616a);
                        } else if (fcwLevel === 'WARNING') {
                            reticle.material.color.setHex(0xd08770);
                        } else {
                            reticle.material.color.setHex(0xa3be8c);
                        }
                    } else {
                        reticle.visible = false;
                    }
                }
            });
        }

        // Remove stale tracks
        for (const [id, mesh] of this.trackedVehicles.entries()) {
            if (!activeIds.has(id)) {
                this.scene.remove(mesh);
                this.trackedVehicles.delete(id);
            }
        }

        // 4. FCW Audio Chimes
        if (data.adas && data.adas.fcw_level !== 'SAFE') {
            this.soundEngine.playFCWAlert(data.adas.fcw_level);
        }

        // 5. Update UI Alerts & Calibration Progress
        this.updateHUD(data);
    }

    updateHUD(data) {
        // Warning Banners
        const ldwAlert = document.getElementById('ldw-alert');
        const ldwText = document.getElementById('ldw-text');
        if (data.adas && data.adas.ldw_state !== 'NORMAL') {
            ldwAlert.classList.remove('hidden');
            ldwText.innerText = data.adas.ldw_state === 'WARNING_LEFT' ? 
                'LANE DEPARTURE: LEFT' : 'LANE DEPARTURE: RIGHT';
        } else {
            ldwAlert.classList.add('hidden');
        }

        const fcwAlert = document.getElementById('fcw-alert');
        const fcwText = document.getElementById('fcw-text');
        if (data.adas && data.adas.fcw_level !== 'SAFE') {
            fcwAlert.classList.remove('hidden');
            fcwAlert.className = `alert-banner fcw-alert ${data.adas.fcw_level.toLowerCase()}`;
            fcwText.innerText = data.adas.warning_message || `FCW ${data.adas.fcw_level}`;
        } else {
            fcwAlert.classList.add('hidden');
        }

        // Calibration Controls Drawer State
        if (data.adas && data.adas.calibration) {
            const c = data.adas.calibration;
            const leftEl = document.getElementById('val-calib-left');
            const rightEl = document.getElementById('val-calib-right');
            const widthEl = document.getElementById('val-calib-width');
            const biasEl = document.getElementById('val-calib-bias');
            if (leftEl) leftEl.innerText = `${c.calibrated_left_m.toFixed(2)} m`;
            if (rightEl) rightEl.innerText = `+${c.calibrated_right_m.toFixed(2)} m`;
            if (widthEl) widthEl.innerText = `${(c.vehicle_width_m || 1.90).toFixed(2)} m`;
            if (biasEl) biasEl.innerText = `${(c.camera_bias_m || 0.0).toFixed(2)} m`;


            const progBox = document.getElementById('calib-progress-box');
            const progBar = document.getElementById('calib-progress-bar');
            const progMsg = document.getElementById('calib-status-msg');
            const progPct = document.getElementById('calib-pct-text');

            if (c.is_calibrating) {
                progBox.classList.remove('hidden');
                const pct = Math.round(c.calibration_progress * 100);
                progBar.style.width = `${pct}%`;
                progPct.innerText = `${pct}%`;
                progMsg.innerText = `Recording ${c.calibration_side.toUpperCase()} wheel edge...`;
            } else {
                progBox.classList.add('hidden');
            }
        }
    }

    updateLaneBadges(leftType, rightType, isLeftWarn, isRightWarn) {
        const leftPill = document.getElementById('left-lane-pill');
        const rightPill = document.getElementById('right-lane-pill');
        if (leftPill) {
            const label = leftType.replace(/_/g, ' ').toUpperCase();
            leftPill.innerText = `LEFT: ${label}`;
            leftPill.className = `lane-pill ${isLeftWarn ? 'pill-warning' : (leftType.includes('yellow') ? 'pill-yellow' : 'pill-white')}`;
        }
        if (rightPill) {
            const label = rightType.replace(/_/g, ' ').toUpperCase();
            rightPill.innerText = `RIGHT: ${label}`;
            rightPill.className = `lane-pill ${isRightWarn ? 'pill-warning' : (rightType.includes('yellow') ? 'pill-yellow' : 'pill-white')}`;
        }
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const hostname = window.location.hostname || 'localhost';
        const currentPort = window.location.port ? parseInt(window.location.port, 10) : 5000;

        const wsPort = currentPort === 5000 ? 5001 : currentPort;
        const wsUrl = `${protocol}//${hostname}:${wsPort}/ws`;

        const pill = document.getElementById('status-pill');
        const statusText = document.getElementById('status-text');

        let isWsConnected = false;

        const startHttpFallback = () => {
            if (this.httpPollInterval) return;
            this.httpPollInterval = setInterval(async () => {
                try {
                    const resp = await fetch('/api/telemetry');
                    if (resp.ok) {
                        const data = await resp.json();
                        if (data && Object.keys(data).length > 0) {
                            if (!isWsConnected) {
                                pill.className = 'status-pill connected';
                                statusText.innerText = 'ONLINE';
                            }
                            this.updateScene(data);
                        }
                    }
                } catch (err) {
                    if (!isWsConnected) {
                        pill.className = 'status-pill disconnected';
                        statusText.innerText = 'OFFLINE';
                    }
                }
            }, 60);
        };

        try {
            this.ws = new WebSocket(wsUrl);
        } catch(e) {
            console.warn("WebSocket init error, starting HTTP polling fallback...", e);
            startHttpFallback();
            setTimeout(() => this.connectWebSocket(), 3000);
            return;
        }

        this.ws.onopen = () => {
            isWsConnected = true;
            if (this.httpPollInterval) {
                clearInterval(this.httpPollInterval);
                this.httpPollInterval = null;
            }
            pill.className = 'status-pill connected';
            statusText.innerText = 'ONLINE';
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.updateScene(data);
            } catch (e) {
                console.error("Failed to parse telemetry:", e);
            }
        };

        this.ws.onclose = () => {
            isWsConnected = false;
            startHttpFallback();
            setTimeout(() => this.connectWebSocket(), 3000);
        };

        this.ws.onerror = (err) => {
            console.warn("WebSocket error, falling back to HTTP polling:", err);
            isWsConnected = false;
            startHttpFallback();
        };

        // Immediately trigger initial HTTP poll so data renders right away
        startHttpFallback();
    }

    loadRecordings() {
        fetch('/api/recordings')
            .then(res => res.json())
            .then(data => {
                if (data.status === 'ok' && data.recordings) {
                    const select = document.getElementById('select-recording-file');
                    if (!select) return;

                    select.innerHTML = '';
                    if (data.recordings.length === 0) {
                        select.innerHTML = '<option value="">No recordings found</option>';
                        return;
                    }

                    data.recordings.forEach(rec => {
                        const opt = document.createElement('option');
                        opt.value = rec.path;
                        opt.textContent = rec.size_mb > 0 
                            ? `${rec.name} (${rec.size_mb} MB)` 
                            : rec.name;
                        if (rec.path === data.current_video_path) {
                            opt.selected = true;
                        }
                        select.appendChild(opt);
                    });
                }
            })
            .catch(err => console.error("Error loading recordings:", err));
    }

    updateSourceUI(sourceType) {
        if (this.currentActiveSource === sourceType) return;
        this.currentActiveSource = sourceType;

        const navCam = document.getElementById('btn-src-camera');
        const navVid = document.getElementById('btn-src-video');
        const drawerCam = document.getElementById('drawer-src-camera');
        const drawerVid = document.getElementById('drawer-src-video');

        if (sourceType === 'camera') {
            if (navCam) navCam.classList.add('active');
            if (navVid) navVid.classList.remove('active');
            if (drawerCam) { drawerCam.className = 'btn btn-primary'; }
            if (drawerVid) { drawerVid.className = 'btn btn-secondary'; }
        } else {
            if (navCam) navCam.classList.remove('active');
            if (navVid) navVid.classList.add('active');
            if (drawerCam) { drawerCam.className = 'btn btn-secondary'; }
            if (drawerVid) { drawerVid.className = 'btn btn-primary'; }
        }
    }

    initEventListeners() {
        // Source Selector Toggle Handlers
        const setSource = (sourceType, videoPath = null) => {
            const payload = { source: sourceType };
            if (videoPath) payload.video_path = videoPath;

            fetch('/api/source', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'ok') {
                    this.updateSourceUI(sourceType);
                }
            })
            .catch(err => console.error("Error setting source:", err));
        };

        const navCam = document.getElementById('btn-src-camera');
        const navVid = document.getElementById('btn-src-video');
        const drawerCam = document.getElementById('drawer-src-camera');
        const drawerVid = document.getElementById('drawer-src-video');
        const selectRec = document.getElementById('select-recording-file');

        if (navCam) navCam.addEventListener('click', () => setSource('camera'));
        if (navVid) navVid.addEventListener('click', () => setSource('video'));
        if (drawerCam) drawerCam.addEventListener('click', () => setSource('camera'));
        if (drawerVid) {
            drawerVid.addEventListener('click', () => {
                const selectedPath = selectRec ? selectRec.value : null;
                setSource('video', selectedPath);
            });
        }
        if (selectRec) {
            selectRec.addEventListener('change', (e) => {
                if (e.target.value) {
                    setSource('video', e.target.value);
                }
            });
        }

        // Fetch list of recordings on init
        this.loadRecordings();

        // Controls Drawer Open / Close
        const modal = document.getElementById('controls-modal');
        const openBtn = document.getElementById('btn-open-controls');
        const closeBtn = document.getElementById('btn-close-controls');

        openBtn.addEventListener('click', () => modal.classList.remove('hidden'));
        closeBtn.addEventListener('click', () => modal.classList.add('hidden'));

        // Audio Toggle
        const audioBtn = document.getElementById('audio-toggle');
        const audioText = document.getElementById('audio-btn-text');
        audioBtn.addEventListener('click', () => {
            this.soundEngine.init();
            audioBtn.classList.add('btn-primary');
            audioBtn.classList.remove('btn-secondary');
            audioText.innerText = 'Audio Warnings Active ✓';
            this.soundEngine.playBeep(600, 'sine', 0.15, 0.2);
        });

        // 2D Camera View Toggle
        const viewBtn = document.getElementById('view-toggle');
        const topNav2dBtn = document.getElementById('btn-toggle-2d');
        const videoOverlay = document.getElementById('video-overlay');
        const closeVideoBtn = document.getElementById('btn-close-video');
        const mjpegImg = document.getElementById('mjpeg-stream');

        const toggle2D = () => {
            const isHidden = videoOverlay.classList.contains('hidden');
            if (isHidden) {
                if (mjpegImg) {
                    mjpegImg.src = '/video_feed?t=' + Date.now();
                }
                videoOverlay.classList.remove('hidden');
            } else {
                videoOverlay.classList.add('hidden');
            }
        };

        if (viewBtn) viewBtn.addEventListener('click', toggle2D);
        if (topNav2dBtn) topNav2dBtn.addEventListener('click', toggle2D);
        if (closeVideoBtn) closeVideoBtn.addEventListener('click', () => videoOverlay.classList.add('hidden'));

        // Play / Pause Controls
        const playBtn = document.getElementById('btn-play-pause');
        const playIcon = document.getElementById('play-icon');
        let isPaused = false;

        playBtn.addEventListener('click', () => {
            isPaused = !isPaused;
            playIcon.innerText = isPaused ? '▶️' : '⏸️';
            fetch('/api/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: isPaused ? 'pause' : 'play' })
            });
        });

        const stepBtn = document.getElementById('btn-step');
        stepBtn.addEventListener('click', () => {
            fetch('/api/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'step' })
            });
        });

        const resetCamBtn = document.getElementById('btn-reset-cam');
        resetCamBtn.addEventListener('click', () => this.setCameraDefaultPOV());

        // Calibration Triggers
        const calibLeftBtn = document.getElementById('btn-calib-left');
        const calibRightBtn = document.getElementById('btn-calib-right');
        const calibResetBtn = document.getElementById('btn-reset-calib');

        calibLeftBtn.addEventListener('click', () => {
            fetch('/api/calibrate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'start', side: 'left' })
            });
        });

        calibRightBtn.addEventListener('click', () => {
            fetch('/api/calibrate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'start', side: 'right' })
            });
        });

        calibResetBtn.addEventListener('click', () => {
            fetch('/api/calibrate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'reset' })
            });
        });

        // Responsive Resize
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.renderer.render(this.scene, this.camera);
    }
}

// Start application when DOM is ready
window.addEventListener('DOMContentLoaded', () => {
    window.app = new DriveHUDApp();

    // Prevent default iOS Safari bounce scrolling outside of scrollable drawer
    document.addEventListener('touchmove', (e) => {
        if (e.target.closest('.drawer-body')) {
            return;
        }
        e.preventDefault();
    }, { passive: false });
});
