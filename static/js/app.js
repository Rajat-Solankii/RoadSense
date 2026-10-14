/**
 * RoadSense: Machine Learning Collision Severity & Casualty Frontend Controller
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------------------
    // 1. TAB NAVIGATION
    // -------------------------------------------------------------------------
    const tabButtons = document.querySelectorAll(".tab-nav .tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            tabButtons.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            const targetId = btn.getAttribute("data-tab");
            const targetContent = document.getElementById(targetId);
            if (targetContent) {
                targetContent.classList.add("active");
            }
        });
    });

    // -------------------------------------------------------------------------
    // 2. DOM ELEMENTS & STATE
    // -------------------------------------------------------------------------
    const presetsGrid = document.getElementById("presetsGrid");
    const scenarioForm = document.getElementById("scenarioForm");
    const btnReset = document.getElementById("btnReset");
    const speedDeltaInput = document.getElementById("speed_delta_kmh");
    const speedDeltaLabel = document.getElementById("speedDeltaLabel");
    const scenarioLabel = document.getElementById("scenarioLabel");
    const loadingIndicator = document.getElementById("loadingIndicator");

    // Output Cards & Metrics
    const severityCard = document.getElementById("severityCard");
    const severityValue = document.getElementById("severityValue");
    const severityBadge = document.getElementById("severityBadge");
    const binaryValue = document.getElementById("binaryValue");
    const binaryBadge = document.getElementById("binaryBadge");
    const casualtiesValue = document.getElementById("casualtiesValue");

    // Probabilities
    const stackFatal = document.getElementById("stackFatal");
    const stackGrievous = document.getElementById("stackGrievous");
    const stackMinor = document.getElementById("stackMinor");
    const fatalPct = document.getElementById("fatalPct");
    const grievousPct = document.getElementById("grievousPct");
    const minorPct = document.getElementById("minorPct");

    // Counters & History
    const sessionEvaluationsCount = document.getElementById("sessionEvaluationsCount");
    const sessionDangerousCount = document.getElementById("sessionDangerousCount");
    const sessionHistoryBody = document.getElementById("sessionHistoryBody");

    let evaluationCount = 0;
    let dangerousCount = 0;
    let historyRows = [];

    // -------------------------------------------------------------------------
    // 3. LOAD FORM OPTIONS & PRESETS
    // -------------------------------------------------------------------------
    async function loadFormOptions() {
        try {
            const res = await fetch("/api/options");
            const data = await res.json();

            populateSelect("road_category", data.road_categories, "National Highway (NH)");
            populateSelect("road_type", data.road_types, "Divided Multi-Lane (4/6-Lane)");
            populateSelect("weather_condition", data.weather_conditions, "Clear / Sunny");
            populateSelect("road_surface", data.road_surfaces, "Dry Asphalt / Concrete");
            populateSelect("light_condition", data.light_conditions, "Daylight");
            populateSelect("junction_type", data.junction_types, "Straight Road (Not at Junction)");
            populateSelect("primary_vehicle_type", data.vehicle_types, "Car / Hatchback / Sedan");

            const secVehicles = ["None", ...data.vehicle_types];
            populateSelect("secondary_vehicle_type", secVehicles, "Heavy Goods Vehicle (Multi-Axle Truck / Dumper / Trailer)");
            populateSelect("collision_type", data.collision_types, "Rear-End / Chain Pileup");

            const speedSelect = document.getElementById("speed_limit_kmh");
            if (speedSelect) {
                speedSelect.innerHTML = "";
                data.speed_limits.forEach(spd => {
                    const opt = document.createElement("option");
                    opt.value = spd;
                    opt.textContent = `${spd} km/h`;
                    if (spd === 80) opt.selected = true;
                    speedSelect.appendChild(opt);
                });
            }
        } catch (err) {
            console.error("Failed to load options:", err);
        }
    }

    function populateSelect(elemId, items, defaultValue) {
        const el = document.getElementById(elemId);
        if (!el || !items) return;
        el.innerHTML = "";
        items.forEach(item => {
            const opt = document.createElement("option");
            opt.value = item;
            opt.textContent = item;
            if (item === defaultValue) opt.selected = true;
            el.appendChild(opt);
        });
    }

    async function loadPresets() {
        try {
            const res = await fetch("/api/presets");
            const presets = await res.json();
            if (!presetsGrid) return;
            presetsGrid.innerHTML = "";

            presets.forEach(p => {
                const btn = document.createElement("button");
                btn.type = "button";
                btn.className = "preset-btn";
                btn.innerHTML = `
                    <div class="preset-icon">${p.icon || '🚗'}</div>
                    <div class="preset-info">
                        <div class="preset-name">${p.name.split(':')[0]}</div>
                        <div class="preset-hint">${p.severity_hint || 'PRESET'}</div>
                    </div>
                `;
                btn.addEventListener("click", () => {
                    loadScenarioIntoForm(p.scenario, p.name);
                    runPrediction(p.name);
                });
                presetsGrid.appendChild(btn);
            });
        } catch (err) {
            console.error("Failed to load presets:", err);
        }
    }

    function loadScenarioIntoForm(sc, labelName) {
        if (!sc) return;
        setFieldValue("road_category", sc.road_category);
        setFieldValue("road_type", sc.road_type);
        setFieldValue("weather_condition", sc.weather_condition);
        setFieldValue("road_surface", sc.road_surface);
        setFieldValue("light_condition", sc.light_condition);
        setFieldValue("junction_type", sc.junction_type);
        setFieldValue("primary_vehicle_type", sc.primary_vehicle_type);
        setFieldValue("secondary_vehicle_type", sc.secondary_vehicle_type);
        setFieldValue("collision_type", sc.collision_type);
        setFieldValue("speed_limit_kmh", sc.speed_limit_kmh);

        if (sc.number_of_vehicles) setFieldValue("number_of_vehicles", sc.number_of_vehicles);
        if (sc.occupants_count) setFieldValue("occupants_count", sc.occupants_count);

        if (speedDeltaInput) {
            const delta = sc.speed_delta_kmh !== undefined ? sc.speed_delta_kmh : 0;
            speedDeltaInput.value = delta;
            updateSpeedDeltaLabel(delta);
        }

        setCheckbox("is_wrong_side", sc.is_wrong_side);
        setCheckbox("is_overloaded", sc.is_overloaded);
        setCheckbox("safety_gear_violation", sc.safety_gear_violation);
        setCheckbox("high_beam_hazard", sc.high_beam_hazard);
        setCheckbox("stray_animal_hazard", sc.stray_animal_hazard);
        setCheckbox("drowsy_window", sc.drowsy_window);
        setCheckbox("pothole_or_speedbreaker", sc.pothole_or_speedbreaker);

        if (scenarioLabel) {
            scenarioLabel.textContent = labelName || "Custom Scenario";
        }
    }

    function setFieldValue(id, val) {
        const el = document.getElementById(id);
        if (el && val !== undefined) el.value = val;
    }

    function setCheckbox(id, val) {
        const el = document.getElementById(id);
        if (el) el.checked = Boolean(val);
    }

    function getFormScenario() {
        return {
            road_category: document.getElementById("road_category")?.value || "National Highway (NH)",
            road_type: document.getElementById("road_type")?.value || "Divided Multi-Lane (4/6-Lane)",
            weather_condition: document.getElementById("weather_condition")?.value || "Clear / Sunny",
            road_surface: document.getElementById("road_surface")?.value || "Dry Asphalt / Concrete",
            light_condition: document.getElementById("light_condition")?.value || "Daylight",
            junction_type: document.getElementById("junction_type")?.value || "Straight Road (Not at Junction)",
            primary_vehicle_type: document.getElementById("primary_vehicle_type")?.value || "Car / Hatchback / Sedan",
            secondary_vehicle_type: document.getElementById("secondary_vehicle_type")?.value || "None",
            collision_type: document.getElementById("collision_type")?.value || "Rear-End / Chain Pileup",
            speed_limit_kmh: parseInt(document.getElementById("speed_limit_kmh")?.value || "80"),
            speed_delta_kmh: parseInt(speedDeltaInput?.value || "0"),
            number_of_vehicles: parseInt(document.getElementById("number_of_vehicles")?.value || "2"),
            occupants_count: parseInt(document.getElementById("occupants_count")?.value || "3"),
            is_wrong_side: document.getElementById("is_wrong_side")?.checked ? 1 : 0,
            is_overloaded: document.getElementById("is_overloaded")?.checked ? 1 : 0,
            safety_gear_violation: document.getElementById("safety_gear_violation")?.checked ? 1 : 0,
            high_beam_hazard: document.getElementById("high_beam_hazard")?.checked ? 1 : 0,
            stray_animal_hazard: document.getElementById("stray_animal_hazard")?.checked ? 1 : 0,
            drowsy_window: document.getElementById("drowsy_window")?.checked ? 1 : 0,
            pothole_or_speedbreaker: document.getElementById("pothole_or_speedbreaker")?.checked ? 1 : 0,
        };
    }

    // -------------------------------------------------------------------------
    // 4. PREDICTION INFERENCE & RENDERING
    // -------------------------------------------------------------------------
    async function runPrediction(label = "Custom Scenario") {
        if (loadingIndicator) loadingIndicator.classList.remove("hidden");

        const scenario = getFormScenario();
        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(scenario),
            });
            const data = await res.json();
            if (data.status === "success" && data.results) {
                renderResults(data.results, scenario, label);
            }
        } catch (err) {
            console.error("Prediction error:", err);
        } finally {
            if (loadingIndicator) loadingIndicator.classList.add("hidden");
        }
    }

    function renderResults(res, scenario, label) {
        const sev = res.prediction || "MINOR / PROPERTY DAMAGE";
        const probs = res.probabilities || { Fatal: 0.1, Grievous: 0.3, Minor: 0.6 };
        const pFatal = probs.Fatal || 0.0;
        const pGrievous = probs.Grievous || 0.0;
        const pMinor = probs.Minor || 0.0;
        const casualties = res.predicted_casualties || 1.0;
        const isDangerous = res.is_dangerous;

        // KPI Card
        if (severityValue) severityValue.textContent = sev;
        if (severityBadge) {
            if (sev.includes("FATAL")) {
                severityBadge.textContent = "High probability of fatal trauma";
                if (severityCard) severityCard.className = "kpi-banner kpi-fatal";
            } else if (sev.includes("GRIEVOUS")) {
                severityBadge.textContent = "Severe bodily injury & hospitalization probable";
                if (severityCard) severityCard.className = "kpi-banner kpi-grievous";
            } else {
                severityBadge.textContent = "Minor injury / property damage only";
                if (severityCard) severityCard.className = "kpi-banner kpi-minor";
            }
        }

        if (binaryValue) {
            binaryValue.textContent = isDangerous ? "DANGEROUS" : "LOW RISK";
            binaryValue.style.color = isDangerous ? "#ef4444" : "#10b981";
        }
        if (binaryBadge) {
            const dangerProb = res.task_a ? (res.task_a.probability * 100).toFixed(1) : (isDangerous ? "75" : "20");
            binaryBadge.textContent = `${dangerProb}% severe injury probability`;
        }

        if (casualtiesValue) casualtiesValue.textContent = casualties.toFixed(1);

        // Probabilities Breakdown
        if (stackFatal) stackFatal.style.width = `${(pFatal * 100).toFixed(1)}%`;
        if (stackGrievous) stackGrievous.style.width = `${(pGrievous * 100).toFixed(1)}%`;
        if (stackMinor) stackMinor.style.width = `${(pMinor * 100).toFixed(1)}%`;

        if (fatalPct) fatalPct.innerHTML = `<strong>${(pFatal * 100).toFixed(1)}%</strong>`;
        if (grievousPct) grievousPct.innerHTML = `<strong>${(pGrievous * 100).toFixed(1)}%</strong>`;
        if (minorPct) minorPct.innerHTML = `<strong>${(pMinor * 100).toFixed(1)}%</strong>`;

        // Update Session History & Counters
        evaluationCount++;
        if (isDangerous) dangerousCount++;
        if (sessionEvaluationsCount) sessionEvaluationsCount.textContent = evaluationCount;
        if (sessionDangerousCount) sessionDangerousCount.textContent = dangerousCount;

        addHistoryRecord(label, scenario, sev, casualties, isDangerous);
    }

    function addHistoryRecord(label, sc, sev, cas, isDangerous) {
        if (!sessionHistoryBody) return;
        const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

        historyRows.unshift({
            time: now,
            label: label,
            road: sc.road_category,
            severity: sev,
            casualties: cas.toFixed(1),
            isDangerous: isDangerous,
        });

        if (historyRows.length > 10) historyRows.pop();

        sessionHistoryBody.innerHTML = historyRows.map((r, i) => `
            <tr>
                <td>${i + 1}</td>
                <td><small class="text-muted">${r.time}</small></td>
                <td><strong>${r.label.split(':')[0]}</strong> <br><small class="text-muted">${r.road}</small></td>
                <td><span class="status-tag ${r.severity.includes('FATAL') ? 'tag-fatal' : (r.severity.includes('GRIEVOUS') ? 'tag-grievous' : 'tag-minor')}">${r.severity}</span></td>
                <td><strong>${r.casualties}</strong> victims</td>
                <td>${r.isDangerous ? '<span class="status-tag tag-fatal">🔴 Dangerous</span>' : '<span class="status-tag tag-minor">🟢 Low Risk</span>'}</td>
            </tr>
        `).join("");
    }

    function updateSpeedDeltaLabel(val) {
        if (!speedDeltaLabel) return;
        const num = parseInt(val);
        speedDeltaLabel.textContent = num >= 0 ? `+${num} km/h` : `${num} km/h`;
        speedDeltaLabel.style.color = num > 15 ? "#ef4444" : (num > 0 ? "#f59e0b" : "#10b981");
    }

    if (speedDeltaInput) {
        speedDeltaInput.addEventListener("input", (e) => {
            updateSpeedDeltaLabel(e.target.value);
            runPrediction("Custom Scenario");
        });
    }

    if (scenarioForm) {
        scenarioForm.addEventListener("submit", (e) => {
            e.preventDefault();
            runPrediction("Custom Scenario");
        });

        // Reactive inputs change triggers instant prediction
        scenarioForm.querySelectorAll("select, input[type='checkbox']").forEach(el => {
            el.addEventListener("change", () => runPrediction("Custom Scenario"));
        });
    }

    if (btnReset) {
        btnReset.addEventListener("click", () => {
            scenarioForm.reset();
            updateSpeedDeltaLabel(0);
            runPrediction("Reset Scenario");
        });
    }

    // -------------------------------------------------------------------------
    // 5. TAB 2: BATCH CSV AUDIT
    // -------------------------------------------------------------------------
    const dropzone = document.getElementById("dropzone");
    const batchFileInput = document.getElementById("batchFileInput");
    const batchLoading = document.getElementById("batchLoading");
    const batchResultsSummary = document.getElementById("batchResultsSummary");
    const batchTableWrapper = document.getElementById("batchTableWrapper");
    const batchTableBody = document.getElementById("batchTableBody");

    const batchTotalCount = document.getElementById("batchTotalCount");
    const batchFatalCount = document.getElementById("batchFatalCount");
    const batchGrievousCount = document.getElementById("batchGrievousCount");
    const batchMinorCount = document.getElementById("batchMinorCount");

    if (dropzone && batchFileInput) {
        dropzone.addEventListener("click", () => batchFileInput.click());
        dropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropzone.classList.add("dragover");
        });
        dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
        dropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropzone.classList.remove("dragover");
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                processBatchFile(e.dataTransfer.files[0]);
            }
        });

        batchFileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files[0]) {
                processBatchFile(e.target.files[0]);
            }
        });
    }

    async function processBatchFile(file) {
        if (!file.name.endsWith(".csv")) {
            alert("Please upload a valid .csv file.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        if (batchLoading) batchLoading.classList.remove("hidden");
        if (batchResultsSummary) batchResultsSummary.classList.add("hidden");
        if (batchTableWrapper) batchTableWrapper.classList.add("hidden");

        try {
            const res = await fetch("/api/batch", { method: "POST", body: formData });
            const data = await res.json();
            if (data.status === "success") {
                renderBatchResults(data);
            } else {
                alert("Batch error: " + (data.message || "Unknown error"));
            }
        } catch (err) {
            console.error("Batch processing error:", err);
            alert("Error running batch audit.");
        } finally {
            if (batchLoading) batchLoading.classList.add("hidden");
        }
    }

    function renderBatchResults(data) {
        const sum = data.summary;
        if (batchTotalCount) batchTotalCount.textContent = sum.total_records;
        if (batchFatalCount) batchFatalCount.textContent = sum.fatal_count;
        if (batchGrievousCount) batchGrievousCount.textContent = sum.grievous_count;
        if (batchMinorCount) batchMinorCount.textContent = sum.minor_count;

        if (batchResultsSummary) batchResultsSummary.classList.remove("hidden");

        if (batchTableBody && data.records) {
            batchTableBody.innerHTML = data.records.map(r => `
                <tr>
                    <td>${r.record_id}</td>
                    <td><strong>${r.label}</strong></td>
                    <td>${r.road_category}</td>
                    <td>${r.weather}</td>
                    <td>${r.primary_vehicle}</td>
                    <td><span class="status-tag ${r.severity.includes('FATAL') ? 'tag-fatal' : (r.severity.includes('GRIEVOUS') ? 'tag-grievous' : 'tag-minor')}">${r.severity}</span></td>
                    <td><strong>${r.casualties}</strong></td>
                    <td>${r.is_dangerous ? '<span class="status-tag tag-fatal">Dangerous</span>' : '<span class="status-tag tag-minor">Minor</span>'}</td>
                </tr>
            `).join("");
            if (batchTableWrapper) batchTableWrapper.classList.remove("hidden");
        }
    }

    // -------------------------------------------------------------------------
    // 6. TAB 3: MORTH OFFICIAL BENCHMARKS
    // -------------------------------------------------------------------------
    async function loadMorthStats() {
        const tableBody = document.getElementById("morthStatsBody");
        if (!tableBody) return;
        try {
            const res = await fetch("/api/morth_stats");
            const data = await res.json();
            if (data.status === "success" && data.records) {
                tableBody.innerHTML = data.records.map(r => `
                    <tr>
                        <td><strong>${r.hazard_category || r.category}</strong></td>
                        <td>${r.factor || r.subcategory}</td>
                        <td>${r.fatalities ? Number(r.fatalities).toLocaleString() : 'N/A'}</td>
                        <td><strong>${r.share_pct || r.national_share_pct}%</strong></td>
                        <td><span class="status-tag ${(r.share_pct || r.national_share_pct) > 25 ? 'tag-fatal' : ((r.share_pct || r.national_share_pct) > 10 ? 'tag-grievous' : 'tag-minor')}">${(r.share_pct || r.national_share_pct) > 25 ? 'High Impact' : 'Moderate'}</span></td>
                    </tr>
                `).join("");
            }
        } catch (err) {
            console.error("Failed to load MoRTH stats:", err);
        }
    }

    // -------------------------------------------------------------------------
    // INITIALIZATION
    // -------------------------------------------------------------------------
    loadFormOptions();
    loadPresets();
    loadMorthStats();

    // Run initial prediction after options load
    setTimeout(() => {
        runPrediction("Initial Scenario");
    }, 400);
});
