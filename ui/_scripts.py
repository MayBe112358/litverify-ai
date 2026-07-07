"""Inline JavaScript snippets injected by the chat shell via st.iframe.

These live here (not in ``chat_shell.py``) so the Python module stays focused
on the Streamlit layout, and so the JS itself is easier to scan without 900
lines of Python noise on either side.
"""
from __future__ import annotations


AURORA_FLUID_SCRIPT = """
<script>
(function () {
    var win = window.parent || window;
    var doc = win.document || document;
    var version = "aurora-fluid-v8";
    if (win.__dwAurora && win.__dwAurora.version === version) {
        return;
    }
    if (win.__dwAurora && win.__dwAurora.cleanup) {
        try { win.__dwAurora.cleanup(); } catch (e) {}
    }
    var root = doc.documentElement;
    // Clear leftovers from the retired v1 cursor-glow bridge.
    ["--dw-aur-x", "--dw-aur-y", "--dw-aur-nx", "--dw-aur-ny"].forEach(function (k) {
        try { root.style.removeProperty(k); } catch (e) {}
    });
    root.removeAttribute("data-dw-aurora-active");

    // Reduced motion → stay on the static CSS fallback blobs.
    if (win.matchMedia && win.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        return;
    }

    var stale = doc.getElementById("dw-aurora-canvas");
    if (stale && stale.parentNode) { stale.parentNode.removeChild(stale); }
    var canvas = doc.createElement("canvas");
    canvas.id = "dw-aurora-canvas";
    canvas.className = "dw-aurora-canvas";
    doc.body.appendChild(canvas);

    var gl = null;
    try {
        gl = canvas.getContext("webgl", {
            alpha: false, antialias: false, depth: false,
            stencil: false, powerPreference: "low-power"
        }) || canvas.getContext("experimental-webgl");
    } catch (e) { gl = null; }
    if (!gl) {
        if (canvas.parentNode) { canvas.parentNode.removeChild(canvas); }
        return; // CSS fallback blobs take over
    }

    // ---- shaders: 6 drifting colour blobs blended into a full-screen
    // gradient, domain-warped for the liquid feel; up to 16 mouse-stroke
    // ripples displace the sampling coords so the colours get "plucked"
    // aside where the cursor passes, then settle back.
    var VSH = "attribute vec2 a;void main(){gl_Position=vec4(a,0.,1.);}";
    var FSH = [
        "precision mediump float;",
        "uniform vec2 u_res;",
        "uniform float u_time;",
        "uniform vec3 u_bg;",
        "uniform float u_strength;",
        "uniform vec3 u_cursor;",      // xy: lerped cursor (uv), z: activity 0..1
        "uniform vec3 u_colors[6];",
        "uniform vec3 u_blobs[6];",   // xy: centre (uv), z: radius
        "void main() {",
        "  vec2 uv = gl_FragCoord.xy / u_res;",
        "  float aspect = u_res.x / u_res.y;",
        "  vec2 p = uv;",
        //  Cursor field: one wide, soft gaussian. No trail points — a
        //  stroke leaves no line, the fluid just sways where the
        //  pointer is and calms down when it rests.
        "  vec2 dc = p - u_cursor.xy;",
        "  dc.x *= aspect;",
        "  float stir = u_cursor.z * exp(-dot(dc, dc) * 60.0);",
        //  Liquid undulation — two wave octaves, fast enough to be
        //  clearly alive, and amplified around the cursor so a stroke
        //  reads as extra turbulence in the fluid, not a drawn mark.
        "  float wa = 0.045 * (1.0 + 1.8 * stir);",
        "  p += wa * vec2(",
        "    sin(p.y * 3.4 + u_time * 0.70) + 0.5 * sin(p.y * 6.4 - u_time * 0.47),",
        "    cos(p.x * 3.1 + u_time * 0.62) + 0.5 * cos(p.x * 5.6 + u_time * 0.40));",
        //  Gentle parting so the colours also make way under the pointer.
        "  p += (0.045 * stir) * (dc / max(length(dc), 1e-4));",
        "  vec2 q = vec2(p.x * aspect, p.y);",
        "  vec3 acc = vec3(0.0);",
        "  float wsum = 0.0;",
        "  for (int i = 0; i < 6; i++) {",
        "    vec2 bp = vec2(u_blobs[i].x * aspect, u_blobs[i].y);",
        "    float dd = length(q - bp) / u_blobs[i].z;",
        "    float w = exp(-dd * dd * 2.0);",
        "    acc += w * u_colors[i];",
        "    wsum += w;",
        "  }",
        "  vec3 grad = acc / max(wsum, 1e-4);",
        "  float cover = 1.0 - exp(-wsum * 2.2);",
        "  gl_FragColor = vec4(mix(u_bg, grad, cover * u_strength), 1.0);",
        "}"
    ].join("\\n");

    function compile(type, src) {
        var sh = gl.createShader(type);
        gl.shaderSource(sh, src);
        gl.compileShader(sh);
        if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) { return null; }
        return sh;
    }
    var vs = compile(gl.VERTEX_SHADER, VSH);
    var fs = compile(gl.FRAGMENT_SHADER, FSH);
    var prog = vs && fs ? gl.createProgram() : null;
    if (prog) {
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
    }
    if (!prog || !gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        if (canvas.parentNode) { canvas.parentNode.removeChild(canvas); }
        return;
    }
    gl.useProgram(prog);
    var buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    var aLoc = gl.getAttribLocation(prog, "a");
    gl.enableVertexAttribArray(aLoc);
    gl.vertexAttribPointer(aLoc, 2, gl.FLOAT, false, 0, 0);
    var loc = {
        res: gl.getUniformLocation(prog, "u_res"),
        time: gl.getUniformLocation(prog, "u_time"),
        bg: gl.getUniformLocation(prog, "u_bg"),
        strength: gl.getUniformLocation(prog, "u_strength"),
        cursor: gl.getUniformLocation(prog, "u_cursor"),
        colors: gl.getUniformLocation(prog, "u_colors[0]"),
        blobs: gl.getUniformLocation(prog, "u_blobs[0]")
    };

    // ---- palettes (pre-pastelised; picked per frame off data-dw-theme)
    function rgb(hex) {
        var n = parseInt(hex.slice(1), 16);
        return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
    }
    function flat(hexes) {
        var out = new Float32Array(hexes.length * 3);
        hexes.forEach(function (h, i) { out.set(rgb(h), i * 3); });
        return out;
    }
    var LIGHT = {
        bg: rgb("#FFFFFF"), strength: 0.95,
        colors: flat(["#C5C4F8", "#FAD0E8", "#E3DAFC", "#C8EEF9", "#FFE4D8", "#D6F3E7"])
    };
    var DARK = {
        bg: rgb("#212327"), strength: 0.85,
        colors: flat(["#38366E", "#6D2F5B", "#47408C", "#175263", "#5A3153", "#1F5A50"])
    };

    // ---- 6 blobs on lissajous paths spanning the whole viewport.
    // Periods of ~15-30s per axis: slow enough to stay calm, fast
    // enough that the flow is clearly visible within a few seconds.
    var BLOBS = [
        {x: 0.15, y: 0.25, r: 0.60, ax: 0.14, ay: 0.11, wx: 0.38, wy: 0.30, px: 0.0, py: 1.7},
        {x: 0.85, y: 0.20, r: 0.55, ax: 0.13, ay: 0.14, wx: 0.28, wy: 0.42, px: 2.1, py: 0.6},
        {x: 0.55, y: 0.55, r: 0.65, ax: 0.17, ay: 0.14, wx: 0.21, wy: 0.28, px: 4.0, py: 2.9},
        {x: 0.15, y: 0.80, r: 0.55, ax: 0.11, ay: 0.13, wx: 0.35, wy: 0.24, px: 1.2, py: 5.1},
        {x: 0.85, y: 0.85, r: 0.60, ax: 0.14, ay: 0.11, wx: 0.31, wy: 0.38, px: 3.3, py: 1.1},
        {x: 0.50, y: 0.05, r: 0.50, ax: 0.15, ay: 0.10, wx: 0.24, wy: 0.35, px: 5.4, py: 3.8}
    ];
    var blobArr = new Float32Array(18);

    var t0 = win.performance.now();
    function now() { return (win.performance.now() - t0) / 1000; }
    // Smoothed cursor state: position lerps toward the raw pointer,
    // activity rises with movement speed and relaxes when the pointer
    // rests — the fluid sways harder near a moving cursor, no trail.
    var curX = 0.5, curY = 0.5, act = 0;
    var tgtX = 0.5, tgtY = 0.5;
    var lastX = null, lastY = null;
    function onPointerMove(e) {
        var x = e.clientX, y = e.clientY;
        var w = Math.max(1, win.innerWidth), h = Math.max(1, win.innerHeight);
        tgtX = x / w;
        tgtY = 1 - y / h;
        if (lastX === null) {
            lastX = x; lastY = y;
            curX = tgtX; curY = tgtY;
            return;
        }
        var dx = x - lastX, dy = y - lastY;
        act = Math.min(1, act + Math.sqrt(dx * dx + dy * dy) / 130);
        lastX = x;
        lastY = y;
    }
    function resize() {
        // Half-resolution render — it's a soft gradient, upscaling is free.
        canvas.width = Math.max(1, Math.round(win.innerWidth * 0.5));
        canvas.height = Math.max(1, Math.round(win.innerHeight * 0.5));
    }

    var rafId = null, timerId = null, stopped = false;
    function schedule() {
        if (!stopped) { rafId = win.requestAnimationFrame(frame); }
    }
    function frame() {
        rafId = null;
        if (stopped) { return; }
        // Not on the empty home / tab hidden → idle-poll instead of drawing.
        if (doc.hidden || root.getAttribute("data-dw-empty") !== "1") {
            timerId = win.setTimeout(schedule, 300);
            return;
        }
        var t = now();
        var pal = root.getAttribute("data-dw-theme") === "dark" ? DARK : LIGHT;
        for (var i = 0; i < 6; i++) {
            var b = BLOBS[i];
            blobArr[i * 3] = b.x + b.ax * Math.sin(t * b.wx + b.px);
            blobArr[i * 3 + 1] = b.y + b.ay * Math.cos(t * b.wy + b.py);
            blobArr[i * 3 + 2] = b.r;
        }
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.uniform2f(loc.res, canvas.width, canvas.height);
        gl.uniform1f(loc.time, t);
        gl.uniform3f(loc.bg, pal.bg[0], pal.bg[1], pal.bg[2]);
        gl.uniform1f(loc.strength, pal.strength);
        curX += (tgtX - curX) * 0.16;
        curY += (tgtY - curY) * 0.16;
        act *= 0.94;
        gl.uniform3f(loc.cursor, curX, curY, act);
        gl.uniform3fv(loc.colors, pal.colors);
        gl.uniform3fv(loc.blobs, blobArr);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        schedule();
    }

    doc.addEventListener("pointermove", onPointerMove, { passive: true });
    win.addEventListener("resize", resize);
    resize();
    root.setAttribute("data-dw-aurora-gl", "1");  // hides the CSS fallback blobs
    schedule();

    win.__dwAurora = {
        version: version,
        cleanup: function () {
            stopped = true;
            if (rafId !== null) { win.cancelAnimationFrame(rafId); }
            if (timerId !== null) { win.clearTimeout(timerId); }
            doc.removeEventListener("pointermove", onPointerMove);
            win.removeEventListener("resize", resize);
            root.removeAttribute("data-dw-aurora-gl");
            if (canvas.parentNode) { canvas.parentNode.removeChild(canvas); }
        }
    };
})();
</script>
"""


DRAG_BRIDGE_SCRIPT = """
<script>
(function () {
    var doc = window.parent && window.parent.document
        ? window.parent.document : document;
    var win = window.parent || window;
    var root = doc.documentElement;
    var bridgeVersion = "home-drop-native-v10";
    if (win.__dwDragBridge && win.__dwDragBridge.version === bridgeVersion) {
        return;
    }
    if (win.__dwDragBridge && win.__dwDragBridge.handlers) {
        try {
            var old = win.__dwDragBridge.handlers;
            doc.removeEventListener("dragenter", old.dragenter, true);
            doc.removeEventListener("dragover", old.dragover, true);
            doc.removeEventListener("dragleave", old.dragleave, true);
            doc.removeEventListener("drop", old.drop, true);
            doc.removeEventListener("dragend", old.dragend, true);
            win.removeEventListener("blur", old.blur);
            if (win.__dwDragBridge.resizeObserver) {
                win.__dwDragBridge.resizeObserver.disconnect();
            }
            if (win.__dwDragBridge.mutationObserver) {
                win.__dwDragBridge.mutationObserver.disconnect();
            }
            if (win.__dwDragBridge.sidebarInterval) {
                win.clearInterval(win.__dwDragBridge.sidebarInterval);
            }
        } catch (e) {}
    }
    root.dataset.dwDragBridgeReady = bridgeVersion;
    var depth = 0;
    var hideTimer = null;
    var resizeObserver = null;
    var routingDrop = false;
    var sidebarInterval = null;
    function sidebarOffset() {
        var expanded = doc.querySelector('[data-testid="stSidebar"][aria-expanded="true"]');
        if (expanded) {
            var rect = expanded.getBoundingClientRect();
            if (rect && rect.width > 80) {
                return Math.max(64, Math.round(rect.right || rect.width));
            }
        }
        return 64;
    }
    function syncSidebarOffset() {
        root.style.setProperty("--dw-sidebar-offset", sidebarOffset() + "px");
    }
    function hasFiles(event) {
        var types = event.dataTransfer && event.dataTransfer.types;
        if (!types) {
            return false;
        }
        return Array.prototype.indexOf.call(types, "Files") !== -1;
    }
    function isHomeArea(event) {
        return !event.clientX || event.clientX >= sidebarOffset();
    }
    function show() {
        syncSidebarOffset();
        if (hideTimer) {
            window.clearTimeout(hideTimer);
            hideTimer = null;
        }
        root.setAttribute("data-dw-file-drag", "1");
    }
    function getUploaderHost() {
        return doc.querySelector('[class*="st-key-home_drop_upload"]');
    }
    function getUploaderDropzone() {
        return doc.querySelector(
            '[class*="st-key-home_drop_upload"] [data-testid="stFileUploaderDropzone"]'
        ) || getUploaderHost();
    }
    function getUploaderInput() {
        return doc.querySelector(
            '[class*="st-key-home_drop_upload"] input[type="file"]'
        );
    }
    function cloneTransfer(dataTransfer) {
        if (!dataTransfer || !dataTransfer.files || !dataTransfer.files.length) {
            return null;
        }
        try {
            var transfer = new win.DataTransfer();
            Array.prototype.forEach.call(dataTransfer.files, function (file) {
                transfer.items.add(file);
            });
            return transfer;
        } catch (cloneError) {
            return dataTransfer;
        }
    }
    function filesAlreadyShown(dataTransfer) {
        if (!dataTransfer || !dataTransfer.files || !dataTransfer.files.length || !doc.body) {
            return false;
        }
        var text = doc.body.innerText || "";
        return Array.prototype.every.call(dataTransfer.files, function (file) {
            return file && file.name && text.indexOf(file.name) !== -1;
        });
    }
    function pushFilesToUploader(dataTransfer) {
        if (!dataTransfer || !dataTransfer.files || !dataTransfer.files.length) {
            return false;
        }
        var input = getUploaderInput();
        if (!input) {
            return false;
        }
        var transfer = cloneTransfer(dataTransfer);
        var files = transfer && transfer.files ? transfer.files : dataTransfer.files;
        try { input.value = ""; } catch (clearError) {}
        try {
            var descriptor = Object.getOwnPropertyDescriptor(
                win.HTMLInputElement.prototype,
                "files"
            );
            if (descriptor && descriptor.set) {
                descriptor.set.call(input, files);
            } else {
                input.files = files;
            }
        } catch (error) {
            try {
                Object.defineProperty(input, "files", {
                    value: files,
                    configurable: true
                });
            } catch (innerError) {
                return false;
            }
        }
        input.dispatchEvent(new win.Event("input", { bubbles: true, composed: true }));
        input.dispatchEvent(new win.Event("change", { bubbles: true, composed: true }));
        return true;
    }
    function forwardDropToUploader(dataTransfer) {
        var dropzone = getUploaderDropzone();
        var transfer = cloneTransfer(dataTransfer);
        if (!dropzone || !transfer || !transfer.files || !transfer.files.length) {
            return false;
        }
        routingDrop = true;
        try {
            ["dragenter", "dragover", "drop"].forEach(function (type) {
                var event = new win.DragEvent(type, {
                    bubbles: true,
                    cancelable: true,
                    composed: true,
                    dataTransfer: transfer
                });
                dropzone.dispatchEvent(event);
            });
            return true;
        } catch (error) {
            return false;
        } finally {
            routingDrop = false;
        }
    }
    function hide() {
        root.removeAttribute("data-dw-file-drag");
        depth = 0;
    }
    function hideSoon() {
        if (hideTimer) {
            window.clearTimeout(hideTimer);
        }
        hideTimer = window.setTimeout(hide, 80);
    }
    function onDragEnter(event) {
        if (!hasFiles(event) || !isHomeArea(event)) {
            return;
        }
        depth += 1;
        show();
    }
    // We must call preventDefault on dragover so the browser knows the
    // page is a valid drop target — without it, drop never fires.
    function onDragOver(event) {
        if (!hasFiles(event) || !isHomeArea(event)) {
            return;
        }
        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
        syncSidebarOffset();
        show();
    }
    function onDragLeave(event) {
        if (!hasFiles(event)) {
            return;
        }
        depth = Math.max(0, depth - 1);
        if (depth <= 0) {
            hideSoon();
        }
    }
    function onDrop(event) {
        if (routingDrop) {
            return;
        }
        if (!hasFiles(event) || !isHomeArea(event)) {
            hide();
            return;
        }
        syncSidebarOffset();
        var transfer = cloneTransfer(event.dataTransfer) || event.dataTransfer;
        var host = getUploaderHost();
        var isNativeTarget = host && host.contains(event.target);
        event.preventDefault();
        try { event.dataTransfer.dropEffect = "copy"; } catch (e) {}

        if (isNativeTarget) {
            // Let Streamlit/react-dropzone receive the original drop event too.
            window.setTimeout(function () {
                if (!filesAlreadyShown(transfer)) {
                    pushFilesToUploader(transfer);
                }
            }, 220);
            window.setTimeout(hide, 520);
            return;
        }

        event.stopPropagation();
        if (!forwardDropToUploader(transfer)) {
            pushFilesToUploader(transfer);
        }
        window.setTimeout(function () {
            if (!filesAlreadyShown(transfer)) {
                pushFilesToUploader(transfer);
            }
            hide();
        }, 220);
    }
    syncSidebarOffset();
    sidebarInterval = win.setInterval(syncSidebarOffset, 250);
    try {
        resizeObserver = new ResizeObserver(syncSidebarOffset);
        var sidebar = doc.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            resizeObserver.observe(sidebar);
        }
        resizeObserver.observe(doc.body);
    } catch (e) {}
    var mutationObserver = null;
    try {
        mutationObserver = new MutationObserver(syncSidebarOffset);
        mutationObserver.observe(doc.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ["aria-expanded", "style", "class"]
        });
    } catch (e) {}
    doc.addEventListener("dragenter", onDragEnter, true);
    doc.addEventListener("dragover", onDragOver, true);
    doc.addEventListener("dragleave", onDragLeave, true);
    doc.addEventListener("drop", onDrop, true);
    doc.addEventListener("dragend", hide, true);
    try {
        win.addEventListener("blur", hide);
    } catch (e) {}
    win.__dwDragBridge = {
        version: bridgeVersion,
        handlers: {
            dragenter: onDragEnter,
            dragover: onDragOver,
            dragleave: onDragLeave,
            drop: onDrop,
            dragend: hide,
            blur: hide
        },
        resizeObserver: resizeObserver,
        mutationObserver: mutationObserver,
        sidebarInterval: sidebarInterval
    };
})();
</script>
"""
