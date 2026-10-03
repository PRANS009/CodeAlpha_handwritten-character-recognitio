/**
 * Handwritten Alphabet Recognition — Frontend Logic
 * ===================================================
 * Canvas drawing, image capture, API calls, and results rendering.
 */

(function () {
    "use strict";

    // ── DOM Elements ─────────────────────────
    const canvas        = document.getElementById("draw-canvas");
    const ctx           = canvas.getContext("2d");
    const canvasWrapper = document.getElementById("canvas-wrapper");
    const canvasHint    = document.getElementById("canvas-hint");
    const penSlider     = document.getElementById("pen-size");
    const penDisplay    = document.getElementById("pen-size-display");
    const btnUndo       = document.getElementById("btn-undo");
    const btnClear      = document.getElementById("btn-clear");
    const btnPredict    = document.getElementById("btn-predict");

    // Result panels
    const resultsPlaceholder = document.getElementById("results-placeholder");
    const resultsLoading     = document.getElementById("results-loading");
    const resultsContent     = document.getElementById("results-content");
    const resultsError       = document.getElementById("results-error");
    const topChar            = document.getElementById("top-char");
    const topConfidence      = document.getElementById("top-confidence");
    const runnersList        = document.getElementById("runners-list");
    const errorText          = document.getElementById("error-text");

    // ── State ────────────────────────────────
    let drawing    = false;
    let penSize    = parseInt(penSlider.value, 10);
    let strokes    = [];       // Array of ImageData snapshots for undo
    let hasContent = false;

    // ── Canvas Setup ─────────────────────────
    function initCanvas() {
        ctx.fillStyle = "#0d0d15";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.lineCap   = "round";
        ctx.lineJoin  = "round";
        ctx.strokeStyle = "#f0f0f5";
        ctx.lineWidth   = penSize;
    }
    initCanvas();

    // ── Drawing Handlers ─────────────────────
    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        if (e.touches && e.touches.length > 0) {
            return {
                x: (e.touches[0].clientX - rect.left) * scaleX,
                y: (e.touches[0].clientY - rect.top) * scaleY,
            };
        }
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY,
        };
    }

    function startDraw(e) {
        e.preventDefault();
        drawing = true;
        // Save current state for undo
        strokes.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
        const pos = getPos(e);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);

        // Hide hint
        if (!hasContent) {
            hasContent = true;
            canvasHint.classList.add("hidden");
            canvasWrapper.classList.add("active");
        }
    }

    function draw(e) {
        if (!drawing) return;
        e.preventDefault();
        const pos = getPos(e);
        ctx.lineWidth = penSize;
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
    }

    function endDraw(e) {
        if (!drawing) return;
        e.preventDefault();
        drawing = false;
        ctx.closePath();
    }

    // Mouse events
    canvas.addEventListener("mousedown",  startDraw);
    canvas.addEventListener("mousemove",  draw);
    canvas.addEventListener("mouseup",    endDraw);
    canvas.addEventListener("mouseleave", endDraw);

    // Touch events
    canvas.addEventListener("touchstart", startDraw, { passive: false });
    canvas.addEventListener("touchmove",  draw,      { passive: false });
    canvas.addEventListener("touchend",   endDraw,   { passive: false });
    canvas.addEventListener("touchcancel",endDraw,    { passive: false });

    // ── Pen Size ─────────────────────────────
    penSlider.addEventListener("input", () => {
        penSize = parseInt(penSlider.value, 10);
        penDisplay.textContent = penSize;
    });

    // ── Undo ─────────────────────────────────
    btnUndo.addEventListener("click", () => {
        if (strokes.length === 0) return;
        const prev = strokes.pop();
        ctx.putImageData(prev, 0, 0);

        // Check if canvas is now empty
        if (strokes.length === 0) {
            hasContent = false;
            canvasHint.classList.remove("hidden");
            canvasWrapper.classList.remove("active");
        }
    });

    // ── Clear ────────────────────────────────
    btnClear.addEventListener("click", () => {
        strokes = [];
        hasContent = false;
        initCanvas();
        canvasHint.classList.remove("hidden");
        canvasWrapper.classList.remove("active");
        showPanel("placeholder");
    });

    // ── Panel Visibility ─────────────────────
    function showPanel(name) {
        resultsPlaceholder.classList.toggle("hidden", name !== "placeholder");
        resultsLoading.classList.toggle("hidden",     name !== "loading");
        resultsContent.classList.toggle("hidden",     name !== "content");
        resultsError.classList.toggle("hidden",       name !== "error");
    }

    // ── Predict ──────────────────────────────
    btnPredict.addEventListener("click", async () => {
        if (!hasContent) return;

        showPanel("loading");
        btnPredict.disabled = true;

        try {
            const dataURL = canvas.toDataURL("image/png");
            const res = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: dataURL }),
            });

            const data = await res.json();

            if (!res.ok || data.error) {
                throw new Error(data.error || "Prediction failed");
            }

            renderResults(data.predictions);
        } catch (err) {
            errorText.textContent = err.message || "Something went wrong";
            showPanel("error");
        } finally {
            btnPredict.disabled = false;
        }
    });

    // ── Render Results ───────────────────────
    function renderResults(predictions) {
        if (!predictions || predictions.length === 0) {
            errorText.textContent = "No predictions returned";
            showPanel("error");
            return;
        }

        // Top prediction
        const top = predictions[0];
        topChar.textContent       = top.character;
        topConfidence.textContent = top.confidence.toFixed(1) + "%";

        // Runner-ups
        runnersList.innerHTML = "";
        predictions.slice(1).forEach((pred, i) => {
            const item = document.createElement("div");
            item.className = "runner-item";
            item.innerHTML = `
                <div class="runner-char">${escapeHTML(pred.character)}</div>
                <div class="runner-bar-wrapper">
                    <div class="runner-bar-label">
                        <span>${escapeHTML(pred.character)}</span>
                        <span>${pred.confidence.toFixed(1)}%</span>
                    </div>
                    <div class="runner-bar">
                        <div class="runner-bar-fill" style="width: 0%"></div>
                    </div>
                </div>
            `;
            runnersList.appendChild(item);

            // Animate bar after a frame
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    item.querySelector(".runner-bar-fill").style.width =
                        Math.max(pred.confidence, 2) + "%";
                });
            });
        });

        showPanel("content");
    }

    function escapeHTML(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

})();
