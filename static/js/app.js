/**
 * RoadSense AI: Frontend Application Logic
 * ========================================
 * Handles interactive preset scenario buttons, custom input changes,
 * and live AI risk & disaster assessment updates.
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
    const severityIcon = document.getElementById("severityIcon");
    const severityValue = document.getElementById("severityValue");
    const severityBadge = document.getElementById("severityBadge");

    const driCard = document.getElementById("driCard");
    const driValue = document.getElementById("driValue");
    const driBand = document.getElementById("driBand");

    const casualtiesValue = document.getElementById("casualtiesValue");
    const occupantsAnalyzed = document.getElementById("occupantsAnalyzed");

    const barFatal = document.getElementById("barFatal");
    const fatalPct = document.getElementById("fatalPct");
    const barGrievous = document.getElementById("barGrievous");
    const grievousPct = document.getElementById("grievousPct");
    const barMinor = document.getElementById("barMinor");
    const minorPct = document.getElementById("minorPct");

    const urgencyLevel = document.getElementById("urgencyLevel");
    const urgencyText = document.getElementById("urgencyText");
    const threatsList = document.getElementById("threatsList");
    const directivesList = document.getElementById("directivesList");
    const secondaryList = document.getElementById("secondaryList");

    let presetCache = [];

    // 1. Initialize Options and Presets
    try {
        await Promise.all([loadOptions(), loadPresets()]);
        // Trigger initial prediction for preset 0 if available
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
        
        // Secondary vehicle includes "None" option
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

        scenarioLabel.textContent = preset.name;

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

        // Speed Delta
        speedDeltaInput.value = s.speed_delta_kmh || 0;
        updateSpeedDeltaLabel(s.speed_delta_kmh || 0);

        // Behavioral Checkboxes
        setCheckbox("is_wrong_side", s.is_wrong_side);
        setCheckbox("is_overloaded", s.is_overloaded);
        setCheckbox("safety_gear_violation", s.safety_gear_violation);
        setCheckbox("high_beam_hazard", s.high_beam_hazard);
        setCheckbox("stray_animal_hazard", s.stray_animal_hazard);
        setCheckbox("drowsy_window", s.drowsy_window);
        setCheckbox("pothole_or_speedbreaker", s.pothole_or_speedbreaker);

        // Immediately trigger assessment
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
    speedDeltaInput.addEventListener("input", (e) => {
        updateSpeedDeltaLabel(e.target.value);
    });

    function updateSpeedDeltaLabel(val) {
        const num = parseInt(val, 10);
        if (num > 0) {
            speedDeltaValue.textContent = `+${num} km/h (Overspeeding)`;
            speedDeltaValue.style.color = "#f87171";
        } else if (num < 0) {
            speedDeltaValue.textContent = `${num} km/h (Caution Slow)`;
            speedDeltaValue.style.color = "#34d399";
        } else {
            speedDeltaValue.textContent = "Exact Limit (0 km/h)";
            speedDeltaValue.style.color = "var(--accent-blue)";
        }
    }

    // 6. Form Submission & API Request
    scenarioForm.addEventListener("submit", (e) => {
        e.preventDefault();
        // Clear preset highlight when manually submitted
        document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
        scenarioLabel.textContent = "Custom User-Configured Scenario";
        runPrediction();
    });

    // Reset Form
    btnReset.addEventListener("click", () => {
        if (presetCache.length > 0) {
            selectPreset(0);
        } else {
            scenarioForm.reset();
            updateSpeedDeltaLabel(0);
            runPrediction();
        }
    });

    // 7. Core Prediction Execution
    async function runPrediction() {
        loadingIndicator.classList.remove("hidden");

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
            speed_delta_kmh: parseInt(speedDeltaInput.value, 10),
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
                console.error("Prediction failed:", data.message);
                alert("Prediction Error: " + data.message);
            }
        } catch (err) {
            console.error("Network or inference error:", err);
        } finally {
            loadingIndicator.classList.add("hidden");
        }
    }

    // 8. Render Results Dashboard
    function renderResults(results, payload) {
        const taskA = results.task_a || {};
        const taskB = results.task_b || {};
        const taskC = results.task_c || {};
        const disaster = results.disaster || {};

        // 1. Severity KPI Card
        const predSeverity = taskB.prediction || "UNKNOWN";
        severityValue.textContent = predSeverity;

        // Reset classes
        severityCard.className = "metric-card";
        if (predSeverity === "FATAL") {
            severityCard.classList.add("fatal-border");
            severityIcon.textContent = "💀";
            severityBadge.textContent = "CRITICAL TRAUMA / HIGH LETHALITY";
            severityBadge.style.color = "#f87171";
        } else if (predSeverity === "GRIEVOUS INJURY") {
            severityCard.classList.add("grievous-border");
            severityIcon.textContent = "🩸";
            severityBadge.textContent = "SEVERE INJURY / HOSPITALIZATION";
            severityBadge.style.color = "#fbbf24";
        } else {
            severityCard.classList.add("minor-border");
            severityIcon.textContent = "🩹";
            severityBadge.textContent = "MINOR INJURY / PROPERTY DAMAGE";
            severityBadge.style.color = "#34d399";
        }

        // 2. Disaster Risk Index (DRI)
        const dri = disaster.disaster_risk_index !== undefined ? disaster.disaster_risk_index : 0;
        driValue.textContent = `${dri.toFixed(1)}%`;
        driBand.textContent = disaster.risk_band || "MODERATE RISK";
        
        driCard.className = "metric-card";
        if (dri >= 75) {
            driCard.classList.add("fatal-border");
            driBand.style.color = "#f87171";
        } else if (dri >= 50) {
            driCard.classList.add("grievous-border");
            driBand.style.color = "#fbbf24";
        } else {
            driCard.classList.add("minor-border");
            driBand.style.color = "#34d399";
        }

        // 3. Expected Casualties
        const casualties = taskC.predicted_casualties !== undefined ? taskC.predicted_casualties : 1.0;
        casualtiesValue.textContent = casualties.toFixed(1);
        occupantsAnalyzed.textContent = `from ${payload.occupants_count} occupants involved`;

        // 4. Granular Probability Breakdown
        const probs = taskB.probabilities || { Fatal: 0.33, Grievous: 0.33, Minor: 0.34 };
        const fatalPctVal = (probs.Fatal * 100).toFixed(1);
        const grievousPctVal = (probs.Grievous * 100).toFixed(1);
        const minorPctVal = (probs.Minor * 100).toFixed(1);

        fatalPct.textContent = `${fatalPctVal}%`;
        barFatal.style.width = `${fatalPctVal}%`;

        grievousPct.textContent = `${grievousPctVal}%`;
        barGrievous.style.width = `${grievousPctVal}%`;

        minorPct.textContent = `${minorPctVal}%`;
        barMinor.style.width = `${minorPctVal}%`;

        // 5. Urgency & Threats
        urgencyLevel.textContent = disaster.risk_band || "CODE YELLOW";
        urgencyText.textContent = disaster.urgency_assessment || "Standard road precautions apply.";

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

        // 6. Actionable In-Transit Directives
        directivesList.innerHTML = "";
        const directives = disaster.mitigation_directives || [];
        if (directives.length === 0) {
            directivesList.innerHTML = "<li>Maintain defensive headway and adhere to posted speed limits.</li>";
        } else {
            directives.forEach(d => {
                const li = document.createElement("li");
                li.textContent = d;
                directivesList.appendChild(li);
            });
        }

        // 7. Secondary Prevention & Emergency Protocol
        secondaryList.innerHTML = "";
        const secondary = disaster.secondary_prevention_protocols || [];
        if (secondary.length === 0) {
            secondaryList.innerHTML = "<li>Move to shoulder and dial 112 if assistance is needed.</li>";
        } else {
            secondary.forEach(s => {
                const li = document.createElement("li");
                li.textContent = s;
                secondaryList.appendChild(li);
            });
        }
    }
});
