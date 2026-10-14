/**
 * RoadSense: Basic Frontend Application Logic
 * ===========================================
 * Simple preset scenario buttons and clear output rendering.
 */

document.addEventListener("DOMContentLoaded", async () => {
    // DOM Elements
    const presetsGrid = document.getElementById("presetsGrid");
    const scenarioForm = document.getElementById("scenarioForm");
    const speedDeltaInput = document.getElementById("speed_delta_kmh");
    const speedDeltaValue = document.getElementById("speedDeltaValue");
    const scenarioLabel = document.getElementById("scenarioLabel");
    const loadingIndicator = document.getElementById("loadingIndicator");
    const btnReset = document.getElementById("btnReset");

    // Output DOM Elements
    const severityCard = document.getElementById("severityCard");
    const severityValue = document.getElementById("severityValue");
    const severityBadge = document.getElementById("severityBadge");

    const driValue = document.getElementById("driValue");
    const driBand = document.getElementById("driBand");

    const casualtiesValue = document.getElementById("casualtiesValue");
    const occupantsAnalyzed = document.getElementById("occupantsAnalyzed");

    const fatalPct = document.getElementById("fatalPct");
    const grievousPct = document.getElementById("grievousPct");
    const minorPct = document.getElementById("minorPct");

    const urgencyText = document.getElementById("urgencyText");
    const threatsList = document.getElementById("threatsList");
    const directivesList = document.getElementById("directivesList");
    const secondaryList = document.getElementById("secondaryList");

    let presetCache = [];

    // 1. Initialize Options and Presets
    try {
        await Promise.all([loadOptions(), loadPresets()]);
        if (presetCache.length > 0) {
            selectPreset(0);
        }
    } catch (err) {
        console.error("Initialization error:", err);
    }

    // 2. Fetch & Populate Form Dropdowns
    async function loadOptions() {
        const res = await fetch("/api/options");
        const options = await res.json();

        populateSelect("road_category", options.road_categories);
        populateSelect("road_type", options.road_types);
        populateSelect("weather_condition", options.weather_conditions);
        populateSelect("road_surface", options.road_surfaces);
        populateSelect("light_condition", options.light_conditions);
        populateSelect("junction_type", options.junction_types);
        populateSelect("primary_vehicle_type", options.vehicle_types);
        
        const secondaryVehicles = ["None", ...options.vehicle_types];
        populateSelect("secondary_vehicle_type", secondaryVehicles);

        populateSelect("collision_type", options.collision_types);
        populateSelect("speed_limit_kmh", options.speed_limits, true);
    }

    function populateSelect(elemId, items, isNumeric = false) {
        const select = document.getElementById(elemId);
        if (!select) return;
        select.innerHTML = "";
        items.forEach(item => {
            const opt = document.createElement("option");
            opt.value = item;
            opt.textContent = isNumeric ? `${item} km/h` : item;
            select.appendChild(opt);
        });
    }

    // 3. Fetch & Render Preset Buttons
    async function loadPresets() {
        const res = await fetch("/api/presets");
        presetCache = await res.json();

        presetsGrid.innerHTML = "";
        presetCache.forEach((preset, idx) => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "preset-btn";
            btn.id = `preset-btn-${idx}`;
            btn.innerHTML = `
                <div class="preset-top">
                    <span class="preset-icon">${preset.icon}</span>
                    <span class="preset-badge ${preset.color}">${preset.badge}</span>
                </div>
                <div class="preset-title">${preset.name}</div>
            `;
            btn.addEventListener("click", () => selectPreset(idx));
            presetsGrid.appendChild(btn);
        });
    }

    // 4. Select a Preset and Auto-run Prediction
    function selectPreset(idx) {
        const preset = presetCache[idx];
        if (!preset) return;

        // Highlight selected preset button
        document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
        const activeBtn = document.getElementById(`preset-btn-${idx}`);
        if (activeBtn) activeBtn.classList.add("active");

        if (scenarioLabel) scenarioLabel.textContent = preset.name;

        // Populate Form Fields
        const s = preset.scenario;
        setFieldValue("road_category", s.road_category);
        setFieldValue("road_type", s.road_type);
        setFieldValue("weather_condition", s.weather_condition);
        setFieldValue("road_surface", s.road_surface);
        setFieldValue("light_condition", s.light_condition);
        setFieldValue("junction_type", s.junction_type);
        setFieldValue("primary_vehicle_type", s.primary_vehicle_type);
        setFieldValue("secondary_vehicle_type", s.secondary_vehicle_type);
        setFieldValue("collision_type", s.collision_type);
        setFieldValue("speed_limit_kmh", s.speed_limit_kmh);

        setFieldValue("number_of_vehicles", s.number_of_vehicles);
        setFieldValue("occupants_count", s.occupants_count);

        if (speedDeltaInput) {
            speedDeltaInput.value = s.speed_delta_kmh || 0;
            updateSpeedDeltaLabel(s.speed_delta_kmh || 0);
        }

        setCheckbox("is_wrong_side", s.is_wrong_side);
        setCheckbox("is_overloaded", s.is_overloaded);
        setCheckbox("safety_gear_violation", s.safety_gear_violation);
        setCheckbox("high_beam_hazard", s.high_beam_hazard);
        setCheckbox("stray_animal_hazard", s.stray_animal_hazard);
        setCheckbox("drowsy_window", s.drowsy_window);
        setCheckbox("pothole_or_speedbreaker", s.pothole_or_speedbreaker);

        runPrediction();
    }

    function setFieldValue(id, val) {
        const el = document.getElementById(id);
        if (el && val !== undefined) el.value = val;
    }

    function setCheckbox(id, val) {
        const el = document.getElementById(id);
        if (el) el.checked = Boolean(val);
    }

    // 5. Speed Delta Slider Interaction
    if (speedDeltaInput) {
        speedDeltaInput.addEventListener("input", (e) => {
            updateSpeedDeltaLabel(e.target.value);
        });
    }

    function updateSpeedDeltaLabel(val) {
        if (!speedDeltaValue) return;
        const num = parseInt(val, 10);
        if (num > 0) {
            speedDeltaValue.textContent = `+${num} km/h (Overspeeding)`;
            speedDeltaValue.style.color = "#dc2626";
        } else if (num < 0) {
            speedDeltaValue.textContent = `${num} km/h (Under limit)`;
            speedDeltaValue.style.color = "#16a34a";
        } else {
            speedDeltaValue.textContent = "Exact Limit (0 km/h)";
            speedDeltaValue.style.color = "#0f172a";
        }
    }

    // 6. Form Submission & Reset
    if (scenarioForm) {
        scenarioForm.addEventListener("submit", (e) => {
            e.preventDefault();
            document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
            if (scenarioLabel) scenarioLabel.textContent = "Custom User Inputs";
            runPrediction();
        });
    }

    if (btnReset) {
        btnReset.addEventListener("click", () => {
            if (presetCache.length > 0) {
                selectPreset(0);
            } else if (scenarioForm) {
                scenarioForm.reset();
                updateSpeedDeltaLabel(0);
                runPrediction();
            }
        });
    }

    // 7. Core Prediction Execution
    async function runPrediction() {
        if (loadingIndicator) loadingIndicator.classList.remove("hidden");

        const payload = {
            road_category: document.getElementById("road_category").value,
            road_type: document.getElementById("road_type").value,
            weather_condition: document.getElementById("weather_condition").value,
            road_surface: document.getElementById("road_surface").value,
            light_condition: document.getElementById("light_condition").value,
            junction_type: document.getElementById("junction_type").value,
            primary_vehicle_type: document.getElementById("primary_vehicle_type").value,
            secondary_vehicle_type: document.getElementById("secondary_vehicle_type").value,
            collision_type: document.getElementById("collision_type").value,
            speed_limit_kmh: parseInt(document.getElementById("speed_limit_kmh").value, 10),
            speed_delta_kmh: speedDeltaInput ? parseInt(speedDeltaInput.value, 10) : 0,
            number_of_vehicles: parseInt(document.getElementById("number_of_vehicles").value, 10),
            occupants_count: parseInt(document.getElementById("occupants_count").value, 10),
            is_wrong_side: document.getElementById("is_wrong_side").checked ? 1 : 0,
            is_overloaded: document.getElementById("is_overloaded").checked ? 1 : 0,
            safety_gear_violation: document.getElementById("safety_gear_violation").checked ? 1 : 0,
            high_beam_hazard: document.getElementById("high_beam_hazard").checked ? 1 : 0,
            stray_animal_hazard: document.getElementById("stray_animal_hazard").checked ? 1 : 0,
            drowsy_window: document.getElementById("drowsy_window").checked ? 1 : 0,
            pothole_or_speedbreaker: document.getElementById("pothole_or_speedbreaker").checked ? 1 : 0,
        };

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (data.status === "success") {
                renderResults(data.results, payload);
            } else {
                console.error("Prediction error:", data.message);
                alert("Prediction Error: " + data.message);
            }
        } catch (err) {
            console.error("Network error:", err);
        } finally {
            if (loadingIndicator) loadingIndicator.classList.add("hidden");
        }
    }

    // 8. Render Results Dashboard (Simple and Basic)
    function renderResults(results, payload) {
        const taskB = results.task_b || {};
        const taskC = results.task_c || {};
        const disaster = results.disaster || {};

        // 1. Severity
        const predSeverity = taskB.prediction || "UNKNOWN";
        if (severityValue) severityValue.textContent = predSeverity;

        if (severityCard) {
            severityCard.className = "output-box";
            if (predSeverity === "FATAL") {
                severityCard.classList.add("fatal-border");
                if (severityValue) severityValue.style.color = "#dc2626";
                if (severityBadge) severityBadge.textContent = "High lethality / fatal impact probability";
            } else if (predSeverity === "GRIEVOUS INJURY") {
                severityCard.classList.add("grievous-border");
                if (severityValue) severityValue.style.color = "#d97706";
                if (severityBadge) severityBadge.textContent = "Severe trauma / major hospital care required";
            } else {
                severityCard.classList.add("minor-border");
                if (severityValue) severityValue.style.color = "#16a34a";
                if (severityBadge) severityBadge.textContent = "Minor injury / property damage only";
            }
        }

        // 2. Disaster Risk Index (DRI)
        const dri = disaster.disaster_risk_index !== undefined ? disaster.disaster_risk_index : 0;
        if (driValue) {
            driValue.textContent = `${dri.toFixed(1)}%`;
            if (dri >= 75) driValue.style.color = "#dc2626";
            else if (dri >= 50) driValue.style.color = "#d97706";
            else driValue.style.color = "#16a34a";
        }
        if (driBand) driBand.textContent = disaster.risk_band || "MODERATE RISK";

        // 3. Expected Casualties
        const casualties = taskC.predicted_casualties !== undefined ? taskC.predicted_casualties : 1.0;
        if (casualtiesValue) casualtiesValue.textContent = casualties.toFixed(1);
        if (occupantsAnalyzed) occupantsAnalyzed.textContent = `from ${payload.occupants_count} occupants involved`;

        // 4. Granular Probabilities in Table
        const probs = taskB.probabilities || { Fatal: 0.33, Grievous: 0.33, Minor: 0.34 };
        if (fatalPct) fatalPct.textContent = `${(probs.Fatal * 100).toFixed(1)}%`;
        if (grievousPct) grievousPct.textContent = `${(probs.Grievous * 100).toFixed(1)}%`;
        if (minorPct) minorPct.textContent = `${(probs.Minor * 100).toFixed(1)}%`;

        // 5. Threat Assessment Text & List
        if (urgencyText) urgencyText.textContent = disaster.urgency_assessment || "Standard road precautions apply.";
        if (threatsList) {
            threatsList.innerHTML = "";
            const threats = disaster.escalation_threats || [];
            if (threats.length === 0) {
                threatsList.innerHTML = "<li>No immediate acute escalation threats detected</li>";
            } else {
                threats.forEach(t => {
                    const li = document.createElement("li");
                    li.textContent = t;
                    threatsList.appendChild(li);
                });
            }
        }

        // 6. Directives
        if (directivesList) {
            directivesList.innerHTML = "";
            const directives = disaster.mitigation_directives || [];
            if (directives.length === 0) {
                directivesList.innerHTML = "<li>Maintain safe headway and adhere to posted speed limits.</li>";
            } else {
                directives.forEach(d => {
                    const li = document.createElement("li");
                    li.textContent = d;
                    directivesList.appendChild(li);
                });
            }
        }

        // 7. Secondary Containment
        if (secondaryList) {
            secondaryList.innerHTML = "";
            const secondary = disaster.secondary_prevention_protocols || [];
            if (secondary.length === 0) {
                secondaryList.innerHTML = "<li>Move to shoulder and call emergency services if needed.</li>";
            } else {
                secondary.forEach(s => {
                    const li = document.createElement("li");
                    li.textContent = s;
                    secondaryList.appendChild(li);
                });
            }
        }
    }
});
