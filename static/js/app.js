/**
 * RoadSense: Professional Decision-Support Application Logic
 * ==========================================================
 * Manages:
 *   1. Tab navigation (Single Assessment, What-If, Batch Audit, MoRTH)
 *   2. Single incident prediction & session history tracking
 *   3. What-If counterfactual intervention simulation
 *   4. Batch CSV upload and corridor audit register
 */

document.addEventListener("DOMContentLoaded", async () => {
    // Session State
    let sessionEvaluations = 0;
    let sessionCritical = 0;
    let presetCache = [];
    let optionsCache = null;
    let currentBatchRecords = [];

    // DOM Elements - Navigation & Header
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");
    const sessionEvaluationsCount = document.getElementById("sessionEvaluationsCount");
    const sessionCriticalCount = document.getElementById("sessionCriticalCount");
    const sessionHistoryBody = document.getElementById("sessionHistoryBody");

    // DOM Elements - Tab 1 Single
    const presetsGrid = document.getElementById("presetsGrid");
    const scenarioForm = document.getElementById("scenarioForm");
    const speedDeltaInput = document.getElementById("speed_delta_kmh");
    const speedDeltaValue = document.getElementById("speedDeltaValue");
    const scenarioLabel = document.getElementById("scenarioLabel");
    const loadingIndicator = document.getElementById("loadingIndicator");
    const btnReset = document.getElementById("btnReset");

    const severityCard = document.getElementById("severityCard");
    const severityValue = document.getElementById("severityValue");
    const severityBadge = document.getElementById("severityBadge");
    const driValue = document.getElementById("driValue");
    const driMeter = document.getElementById("driMeter");
    const driBand = document.getElementById("driBand");
    const casualtiesValue = document.getElementById("casualtiesValue");
    const occupantsAnalyzed = document.getElementById("occupantsAnalyzed");

    const stackFatal = document.getElementById("stackFatal");
    const stackGrievous = document.getElementById("stackGrievous");
    const stackMinor = document.getElementById("stackMinor");
    const fatalPct = document.getElementById("fatalPct");
    const grievousPct = document.getElementById("grievousPct");
    const minorPct = document.getElementById("minorPct");

    const urgencyText = document.getElementById("urgencyText");
    const threatsList = document.getElementById("threatsList");
    const directivesList = document.getElementById("directivesList");
    const secondaryList = document.getElementById("secondaryList");

    // DOM Elements - Tab 2 What-If
    const btnRunWhatIf = document.getElementById("btnRunWhatIf");
    const btnCopyFromSingle = document.getElementById("btnCopyFromSingle");
    const whatifResultsBox = document.getElementById("whatifResultsBox");
    const impactFatalReduction = document.getElementById("impactFatalReduction");
    const impactLivesSaved = document.getElementById("impactLivesSaved");
    const impactDriReduction = document.getElementById("impactDriReduction");
    const impactSeverityShift = document.getElementById("impactSeverityShift");
    const whatifTableBody = document.getElementById("whatifTableBody");

    // DOM Elements - Tab 3 Batch
    const dropzone = document.getElementById("dropzone");
    const batchFileInput = document.getElementById("batchFileInput");
    const btnRunBatchSample = document.getElementById("btnRunBatchSample");
    const batchSummarySection = document.getElementById("batchSummarySection");
    const batchTableSection = document.getElementById("batchTableSection");
    const batchTotalCount = document.getElementById("batchTotalCount");
    const batchFatalCount = document.getElementById("batchFatalCount");
    const batchGrievousCount = document.getElementById("batchGrievousCount");
    const batchMinorCount = document.getElementById("batchMinorCount");
    const batchAvgDri = document.getElementById("batchAvgDri");
    const batchResultsBody = document.getElementById("batchResultsBody");
    const btnExportBatchCsv = document.getElementById("btnExportBatchCsv");

    // -------------------------------------------------------------------------
    // 1. INITIALIZATION & TAB SWITCHING
    // -------------------------------------------------------------------------
    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-tab");
            tabButtons.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));
            btn.classList.add("active");
            const targetContent = document.getElementById(targetId);
            if (targetContent) targetContent.classList.add("active");
        });
    });

    try {
        await Promise.all([loadOptions(), loadPresets()]);
        if (presetCache.length > 0) {
            selectPreset(0);
        }
    } catch (err) {
        console.error("Initialization error:", err);
    }

    // -------------------------------------------------------------------------
    // 2. DROPDOWN POPULATION & PRESETS
    // -------------------------------------------------------------------------
    async function loadOptions() {
        const res = await fetch("/api/options");
        optionsCache = await res.json();

        // Single Assessment Form
        populateSelect("road_category", optionsCache.road_categories);
        populateSelect("road_type", optionsCache.road_types);
        populateSelect("weather_condition", optionsCache.weather_conditions);
        populateSelect("road_surface", optionsCache.road_surfaces);
        populateSelect("light_condition", optionsCache.light_conditions);
        populateSelect("junction_type", optionsCache.junction_types);
        populateSelect("primary_vehicle_type", optionsCache.vehicle_types);
        populateSelect("secondary_vehicle_type", ["None", ...optionsCache.vehicle_types]);
        populateSelect("collision_type", optionsCache.collision_types);
        populateSelect("speed_limit_kmh", optionsCache.speed_limits, true);

        // What-If Form
        populateSelect("wi_base_road", optionsCache.road_categories);
        populateSelect("wi_mit_road", optionsCache.road_categories);
        populateSelect("wi_base_weather", optionsCache.weather_conditions);
        populateSelect("wi_mit_weather", optionsCache.weather_conditions);
        populateSelect("wi_base_veh", optionsCache.vehicle_types);
        populateSelect("wi_mit_veh", optionsCache.vehicle_types);
        populateSelect("wi_base_sec_veh", ["None", ...optionsCache.vehicle_types]);
        populateSelect("wi_mit_sec_veh", ["None", ...optionsCache.vehicle_types]);
        populateSelect("wi_base_limit", optionsCache.speed_limits, true);
        populateSelect("wi_mit_limit", optionsCache.speed_limits, true);

        // Pre-fill What-If defaults
        setFieldValue("wi_base_road", "Expressway");
        setFieldValue("wi_mit_road", "Expressway");
        setFieldValue("wi_base_weather", "Dense Winter Fog / Smog");
        setFieldValue("wi_mit_weather", "Dense Winter Fog / Smog");
        setFieldValue("wi_base_veh", "Car / Hatchback / Sedan");
        setFieldValue("wi_mit_veh", "Car / Hatchback / Sedan");
        setFieldValue("wi_base_sec_veh", "Heavy Goods Vehicle (Multi-Axle Truck / Dumper / Trailer)");
        setFieldValue("wi_mit_sec_veh", "Heavy Goods Vehicle (Multi-Axle Truck / Dumper / Trailer)");
        setFieldValue("wi_base_limit", 120);
        setFieldValue("wi_mit_limit", 120);
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

    function selectPreset(idx) {
        const preset = presetCache[idx];
        if (!preset) return;

        document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
        const activeBtn = document.getElementById(`preset-btn-${idx}`);
        if (activeBtn) activeBtn.classList.add("active");

        if (scenarioLabel) scenarioLabel.textContent = preset.name;

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

        runPrediction(preset.name);
    }

    function setFieldValue(id, val) {
        const el = document.getElementById(id);
        if (el && val !== undefined) el.value = val;
    }

    function setCheckbox(id, val) {
        const el = document.getElementById(id);
        if (el) el.checked = Boolean(val);
    }

    // Speed Slider
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
            speedDeltaValue.textContent = `${num} km/h (Caution Slow)`;
            speedDeltaValue.style.color = "#16a34a";
        } else {
            speedDeltaValue.textContent = "Exact Limit (0 km/h)";
            speedDeltaValue.style.color = "#0f172a";
        }
    }

    // -------------------------------------------------------------------------
    // 3. TAB 1: SINGLE PREDICTION EXECUTION & HISTORY
    // -------------------------------------------------------------------------
    if (scenarioForm) {
        scenarioForm.addEventListener("submit", (e) => {
            e.preventDefault();
            document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
            if (scenarioLabel) scenarioLabel.textContent = "Custom Evaluated Incident";
            runPrediction("Custom Scenario");
        });
    }

    if (btnReset) {
        btnReset.addEventListener("click", () => {
            if (presetCache.length > 0) {
                selectPreset(0);
            } else if (scenarioForm) {
                scenarioForm.reset();
                updateSpeedDeltaLabel(0);
                runPrediction("Default Form");
            }
        });
    }

    async function runPrediction(scenarioTitle = "Scenario") {
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
                renderSingleResults(data.results, payload, scenarioTitle);
            } else {
                alert("Prediction Error: " + data.message);
            }
        } catch (err) {
            console.error("Prediction error:", err);
        } finally {
            if (loadingIndicator) loadingIndicator.classList.add("hidden");
        }
    }

    function renderSingleResults(results, payload, scenarioTitle) {
        const taskB = results.task_b || {};
        const taskC = results.task_c || {};
        const disaster = results.disaster || {};

        sessionEvaluations += 1;
        if (sessionEvaluationsCount) sessionEvaluationsCount.textContent = sessionEvaluations;

        const predSeverity = taskB.prediction || "UNKNOWN";
        if (severityValue) severityValue.textContent = predSeverity;

        if (severityCard) {
            severityCard.className = "kpi-banner";
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

        const dri = disaster.disaster_risk_index !== undefined ? disaster.disaster_risk_index : 0;
        if (dri >= 75) {
            sessionCritical += 1;
            if (sessionCriticalCount) sessionCriticalCount.textContent = sessionCritical;
        }

        if (driValue) {
            driValue.textContent = `${dri.toFixed(1)}%`;
            if (dri >= 75) driValue.style.color = "#dc2626";
            else if (dri >= 50) driValue.style.color = "#d97706";
            else driValue.style.color = "#16a34a";
        }
        if (driMeter) {
            driMeter.style.width = `${Math.min(100, dri)}%`;
            if (dri >= 75) driMeter.style.backgroundColor = "#dc2626";
            else if (dri >= 50) driMeter.style.backgroundColor = "#d97706";
            else driMeter.style.backgroundColor = "#16a34a";
        }
        if (driBand) driBand.textContent = disaster.risk_band || "MODERATE RISK";

        const casualties = taskC.predicted_casualties !== undefined ? taskC.predicted_casualties : 1.0;
        if (casualtiesValue) casualtiesValue.textContent = casualties.toFixed(1);
        if (occupantsAnalyzed) occupantsAnalyzed.textContent = `from ${payload.occupants_count} occupants involved`;

        // Stacked probability bar
        const probs = taskB.probabilities || { Fatal: 0.33, Grievous: 0.33, Minor: 0.34 };
        const fatalW = (probs.Fatal * 100).toFixed(1);
        const grievousW = (probs.Grievous * 100).toFixed(1);
        const minorW = (probs.Minor * 100).toFixed(1);

        if (stackFatal) stackFatal.style.width = `${fatalW}%`;
        if (stackGrievous) stackGrievous.style.width = `${grievousW}%`;
        if (stackMinor) stackMinor.style.width = `${minorW}%`;

        if (fatalPct) fatalPct.textContent = `${fatalW}%`;
        if (grievousPct) grievousPct.textContent = `${grievousW}%`;
        if (minorPct) minorPct.textContent = `${minorW}%`;

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

        // Append to Session Log Table
        appendSessionHistory(scenarioTitle, payload.road_category, predSeverity, dri, casualties, (disaster.escalation_threats || ["Standard road condition"])[0]);
    }

    function appendSessionHistory(title, road, severity, dri, casualties, hazard) {
        if (!sessionHistoryBody) return;
        if (sessionEvaluations === 1) sessionHistoryBody.innerHTML = "";

        const timeStr = new Date().toLocaleTimeString();
        let badgeClass = "tag-minor";
        if (severity === "FATAL") badgeClass = "tag-fatal";
        else if (severity === "GRIEVOUS INJURY") badgeClass = "tag-grievous";

        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><strong>#${sessionEvaluations}</strong></td>
            <td>${timeStr}</td>
            <td><strong>${title}</strong> <span class="text-xs text-muted">(${road})</span></td>
            <td><span class="status-tag ${badgeClass}">${severity}</span></td>
            <td><strong>${dri.toFixed(1)}%</strong></td>
            <td>${casualties.toFixed(1)}</td>
            <td class="text-xs">${hazard}</td>
        `;
        sessionHistoryBody.insertBefore(tr, sessionHistoryBody.firstChild);
    }

    // -------------------------------------------------------------------------
    // 4. TAB 2: "WHAT-IF" INTERVENTION SIMULATOR
    // -------------------------------------------------------------------------
    if (btnCopyFromSingle) {
        btnCopyFromSingle.addEventListener("click", () => {
            // Baseline copies current Single tab inputs
            setFieldValue("wi_base_road", document.getElementById("road_category").value);
            setFieldValue("wi_base_weather", document.getElementById("weather_condition").value);
            setFieldValue("wi_base_veh", document.getElementById("primary_vehicle_type").value);
            setFieldValue("wi_base_sec_veh", document.getElementById("secondary_vehicle_type").value);
            setFieldValue("wi_base_limit", document.getElementById("speed_limit_kmh").value);
            setFieldValue("wi_base_delta", speedDeltaInput ? speedDeltaInput.value : 10);
            setCheckbox("wi_base_wrong", document.getElementById("is_wrong_side").checked);
            setCheckbox("wi_base_gear", document.getElementById("safety_gear_violation").checked);
            setCheckbox("wi_base_overload", document.getElementById("is_overloaded").checked);

            // Mitigated sets a safe intervention
            setFieldValue("wi_mit_road", document.getElementById("road_category").value);
            setFieldValue("wi_mit_weather", document.getElementById("weather_condition").value);
            setFieldValue("wi_mit_veh", document.getElementById("primary_vehicle_type").value);
            setFieldValue("wi_mit_sec_veh", document.getElementById("secondary_vehicle_type").value);
            setFieldValue("wi_mit_limit", document.getElementById("speed_limit_kmh").value);
            setFieldValue("wi_mit_delta", -10); // Slower
            setCheckbox("wi_mit_wrong", false); // No wrong way
            setCheckbox("wi_mit_gear", false);  // Full helmet/seatbelt
            setCheckbox("wi_mit_overload", false);

            runWhatIfComparison();
        });
    }

    if (btnRunWhatIf) {
        btnRunWhatIf.addEventListener("click", () => {
            runWhatIfComparison();
        });
    }

    async function runWhatIfComparison() {
        const baseline = {
            road_category: document.getElementById("wi_base_road").value,
            road_type: "Divided Multi-Lane (4/6-Lane)",
            weather_condition: document.getElementById("wi_base_weather").value,
            road_surface: "Dry Asphalt / Concrete",
            light_condition: "Daylight",
            junction_type: "Straight Road (Not at Junction)",
            primary_vehicle_type: document.getElementById("wi_base_veh").value,
            secondary_vehicle_type: document.getElementById("wi_base_sec_veh").value,
            collision_type: "Head-On Collision",
            speed_limit_kmh: parseInt(document.getElementById("wi_base_limit").value, 10),
            speed_delta_kmh: parseInt(document.getElementById("wi_base_delta").value, 10),
            number_of_vehicles: 2,
            occupants_count: 5,
            is_wrong_side: document.getElementById("wi_base_wrong").checked ? 1 : 0,
            is_overloaded: document.getElementById("wi_base_overload").checked ? 1 : 0,
            safety_gear_violation: document.getElementById("wi_base_gear").checked ? 1 : 0,
            high_beam_hazard: 0,
            stray_animal_hazard: 0,
            drowsy_window: 0,
            pothole_or_speedbreaker: 0,
        };

        const mitigated = {
            road_category: document.getElementById("wi_mit_road").value,
            road_type: "Divided Multi-Lane (4/6-Lane)",
            weather_condition: document.getElementById("wi_mit_weather").value,
            road_surface: "Dry Asphalt / Concrete",
            light_condition: "Daylight",
            junction_type: "Straight Road (Not at Junction)",
            primary_vehicle_type: document.getElementById("wi_mit_veh").value,
            secondary_vehicle_type: document.getElementById("wi_mit_sec_veh").value,
            collision_type: "Head-On Collision",
            speed_limit_kmh: parseInt(document.getElementById("wi_mit_limit").value, 10),
            speed_delta_kmh: parseInt(document.getElementById("wi_mit_delta").value, 10),
            number_of_vehicles: 2,
            occupants_count: 5,
            is_wrong_side: document.getElementById("wi_mit_wrong").checked ? 1 : 0,
            is_overloaded: document.getElementById("wi_mit_overload").checked ? 1 : 0,
            safety_gear_violation: document.getElementById("wi_mit_gear").checked ? 1 : 0,
            high_beam_hazard: 0,
            stray_animal_hazard: 0,
            drowsy_window: 0,
            pothole_or_speedbreaker: 0,
        };

        try {
            const res = await fetch("/api/whatif", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ baseline, mitigated })
            });

            const data = await res.json();
            if (data.status === "success") {
                renderWhatIfResults(data);
            }
        } catch (err) {
            console.error("What-If comparison error:", err);
        }
    }

    function renderWhatIfResults(data) {
        const impact = data.impact || {};
        const b = data.baseline || {};
        const m = data.mitigated || {};

        if (impactFatalReduction) impactFatalReduction.textContent = `-${impact.fatal_prob_reduction_pct}%`;
        if (impactLivesSaved) impactLivesSaved.textContent = `${impact.casualties_saved}`;
        if (impactDriReduction) impactDriReduction.textContent = `-${impact.dri_reduction_pct}%`;
        if (impactSeverityShift) impactSeverityShift.textContent = `${impact.baseline_severity} ➔ ${impact.mitigated_severity}`;

        if (whatifTableBody) {
            const bProb = b.task_b?.probabilities?.Fatal ? (b.task_b.probabilities.Fatal * 100).toFixed(1) : 0;
            const mProb = m.task_b?.probabilities?.Fatal ? (m.task_b.probabilities.Fatal * 100).toFixed(1) : 0;
            const bCas = b.task_c?.predicted_casualties?.toFixed(1) || 1.0;
            const mCas = m.task_c?.predicted_casualties?.toFixed(1) || 1.0;
            const bDri = b.disaster?.disaster_risk_index?.toFixed(1) || 0;
            const mDri = m.disaster?.disaster_risk_index?.toFixed(1) || 0;

            whatifTableBody.innerHTML = `
                <tr>
                    <td><strong>Severity Grade</strong></td>
                    <td><span class="status-tag tag-fatal">${impact.baseline_severity}</span></td>
                    <td><span class="status-tag tag-minor">${impact.mitigated_severity}</span></td>
                    <td class="text-success font-bold">Severity downgraded to manageable trauma</td>
                </tr>
                <tr>
                    <td><strong>Fatal Probability</strong></td>
                    <td>${bProb}%</td>
                    <td>${mProb}%</td>
                    <td class="text-success font-bold">-${impact.fatal_prob_reduction_pct}% lethality probability drop</td>
                </tr>
                <tr>
                    <td><strong>Expected Casualties</strong></td>
                    <td>${bCas} persons</td>
                    <td>${mCas} persons</td>
                    <td class="text-success font-bold">${impact.casualties_saved} persons saved from trauma</td>
                </tr>
                <tr>
                    <td><strong>Disaster Risk Index</strong></td>
                    <td>${bDri}% (${b.disaster?.risk_band || 'HIGH'})</td>
                    <td>${mDri}% (${m.disaster?.risk_band || 'LOW'})</td>
                    <td class="text-success font-bold">-${impact.dri_reduction_pct}% composite risk reduction</td>
                </tr>
            `;
        }
    }

    // -------------------------------------------------------------------------
    // 5. TAB 3: BATCH CSV AUDIT
    // -------------------------------------------------------------------------
    if (dropzone && batchFileInput) {
        dropzone.addEventListener("click", () => batchFileInput.click());
        dropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropzone.style.borderColor = "#0284c7";
            dropzone.style.backgroundColor = "#f0f9ff";
        });
        dropzone.addEventListener("dragleave", () => {
            dropzone.style.borderColor = "#94a3b8";
            dropzone.style.backgroundColor = "#f8fafc";
        });
        dropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropzone.style.borderColor = "#94a3b8";
            dropzone.style.backgroundColor = "#f8fafc";
            if (e.dataTransfer.files.length > 0) {
                uploadBatchFile(e.dataTransfer.files[0]);
            }
        });

        batchFileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                uploadBatchFile(e.target.files[0]);
            }
        });
    }

    async function uploadBatchFile(file) {
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/batch", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            if (data.status === "success") {
                renderBatchResults(data);
            } else {
                alert("Batch Upload Error: " + data.message);
            }
        } catch (err) {
            console.error("Batch file error:", err);
        }
    }

    if (btnRunBatchSample) {
        btnRunBatchSample.addEventListener("click", async () => {
            try {
                // Fetch sample CSV template and post directly to batch API
                const sampleRes = await fetch("/api/sample_csv");
                const sampleBlob = await sampleRes.blob();
                const file = new File([sampleBlob], "roadsense_sample.csv", { type: "text/csv" });
                uploadBatchFile(file);
            } catch (err) {
                console.error("Sample batch error:", err);
            }
        });
    }

    function renderBatchResults(data) {
        const sum = data.summary || {};
        currentBatchRecords = data.records || [];

        if (batchSummarySection) batchSummarySection.classList.remove("hidden");
        if (batchTableSection) batchTableSection.classList.remove("hidden");

        if (batchTotalCount) batchTotalCount.textContent = sum.total_records || 0;
        if (batchFatalCount) batchFatalCount.textContent = sum.fatal_count || 0;
        if (batchGrievousCount) batchGrievousCount.textContent = sum.grievous_count || 0;
        if (batchMinorCount) batchMinorCount.textContent = sum.minor_count || 0;
        if (batchAvgDri) batchAvgDri.textContent = `${sum.average_dri || 0}%`;

        if (batchResultsBody) {
            batchResultsBody.innerHTML = "";
            currentBatchRecords.forEach(r => {
                let badgeClass = "tag-minor";
                if (r.severity === "FATAL") badgeClass = "tag-fatal";
                else if (r.severity === "GRIEVOUS INJURY") badgeClass = "tag-grievous";

                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td><strong>#${r.record_id}</strong></td>
                    <td><strong>${r.label}</strong></td>
                    <td>${r.road_category}</td>
                    <td>${r.weather}</td>
                    <td>${r.primary_vehicle}</td>
                    <td>${r.speed_delta_kmh > 0 ? '+' + r.speed_delta_kmh + ' km/h' : r.speed_delta_kmh + ' km/h'}</td>
                    <td><span class="status-tag ${badgeClass}">${r.severity}</span></td>
                    <td><strong>${r.casualties}</strong></td>
                    <td><strong>${r.dri}%</strong></td>
                    <td class="text-xs">${r.primary_directive}</td>
                `;
                batchResultsBody.appendChild(tr);
            });
        }
    }

    // Export Assessed Table to CSV
    if (btnExportBatchCsv) {
        btnExportBatchCsv.addEventListener("click", () => {
            if (currentBatchRecords.length === 0) return;
            const headers = ["record_id", "label", "road_category", "weather", "primary_vehicle", "speed_delta_kmh", "severity", "casualties", "dri", "risk_band", "primary_directive"];
            let csvContent = headers.join(",") + "\n";
            currentBatchRecords.forEach(r => {
                const row = headers.map(h => `"${String(r[h] || '').replace(/"/g, '""')}"`);
                csvContent += row.join(",") + "\n";
            });

            const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "roadsense_assessed_audit_results.csv";
            a.click();
            URL.revokeObjectURL(url);
        });
    }

    // -------------------------------------------------------------------------
    // 6. TAB 5: IRAD REAL-TIME INCIDENT FEED
    // -------------------------------------------------------------------------
    const iradStateFilter = document.getElementById("iradStateFilter");
    const btnRefreshIrad = document.getElementById("btnRefreshIrad");
    const btnSimulateIradLive = document.getElementById("btnSimulateIradLive");
    const iradTotalIncidents = document.getElementById("iradTotalIncidents");
    const iradTotalFatal = document.getElementById("iradTotalFatal");
    const iradTotalGrievous = document.getElementById("iradTotalGrievous");
    const iradIncidentsBody = document.getElementById("iradIncidentsBody");

    let iradCache = [];

    async function loadIradIncidents() {
        const state = iradStateFilter ? iradStateFilter.value : "All";
        try {
            const res = await fetch(`/api/irad/incidents?state=${encodeURIComponent(state)}`);
            const data = await res.json();
            if (data.status === "success") {
                renderIradIncidents(data);
            }
        } catch (err) {
            console.error("iRAD feed error:", err);
        }
    }

    function renderIradIncidents(data) {
        iradCache = data.incidents || [];
        const sum = data.summary || {};

        if (iradTotalIncidents) iradTotalIncidents.textContent = sum.total_incidents || 0;
        if (iradTotalFatal) iradTotalFatal.textContent = sum.actual_fatalities || 0;
        if (iradTotalGrievous) iradTotalGrievous.textContent = sum.actual_grievous || 0;

        if (!iradIncidentsBody) return;
        iradIncidentsBody.innerHTML = "";

        if (iradCache.length === 0) {
            iradIncidentsBody.innerHTML = `<tr><td colspan="9" class="text-center text-muted">No incidents found for ${iradStateFilter?.value || 'selected state'}.</td></tr>`;
            return;
        }

        iradCache.forEach(inc => {
            let badgeClass = "tag-minor";
            if (inc.ai_severity_predicted === "FATAL") badgeClass = "tag-fatal";
            else if (inc.ai_severity_predicted === "GRIEVOUS INJURY") badgeClass = "tag-grievous";

            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong class="font-mono text-xs">${inc.accident_id}</strong></td>
                <td class="text-xs">${inc.reported_at}</td>
                <td><strong>${inc.state}</strong> <span class="text-xs text-muted">(${inc.district})</span></td>
                <td><strong>${inc.corridor_name}</strong> <span class="text-xs text-muted">KM ${inc.chainage_km}</span></td>
                <td>${inc.primary_vehicle_type} <br><span class="text-xs text-muted">vs ${inc.secondary_vehicle_type}</span></td>
                <td><strong>${inc.actual_fatalities}</strong> Dead / <strong>${inc.actual_grievous}</strong> Grievous</td>
                <td><span class="status-tag ${badgeClass}">${inc.ai_severity_predicted}</span></td>
                <td><strong>${inc.ai_disaster_risk_index}%</strong></td>
                <td><button type="button" class="btn btn-xs btn-outline btn-load-irad" data-id="${inc.accident_id}">⚡ Load & Assess</button></td>
            `;
            iradIncidentsBody.appendChild(tr);
        });

        // Wire load buttons
        document.querySelectorAll(".btn-load-irad").forEach(btn => {
            btn.addEventListener("click", () => {
                const accId = btn.getAttribute("data-id");
                loadIradIntoSingleAssessment(accId);
            });
        });
    }

    function loadIradIntoSingleAssessment(accidentId) {
        const inc = iradCache.find(x => x.accident_id === accidentId);
        if (!inc) return;

        // Switch to Tab 1
        const tabSingleBtn = document.querySelector('[data-tab="tab-single"]');
        if (tabSingleBtn) tabSingleBtn.click();

        // Populate fields
        setFieldValue("road_category", inc.road_category);
        setFieldValue("road_type", inc.road_type);
        setFieldValue("weather_condition", inc.weather_condition);
        setFieldValue("road_surface", inc.road_surface);
        setFieldValue("light_condition", inc.light_condition);
        setFieldValue("junction_type", inc.junction_type);
        setFieldValue("primary_vehicle_type", inc.primary_vehicle_type);
        setFieldValue("secondary_vehicle_type", inc.secondary_vehicle_type);
        setFieldValue("collision_type", inc.collision_type);
        setFieldValue("speed_limit_kmh", inc.speed_limit_kmh);
        setFieldValue("number_of_vehicles", inc.number_of_vehicles);
        setFieldValue("occupants_count", inc.occupants_count);

        if (speedDeltaInput) {
            speedDeltaInput.value = inc.speed_delta_kmh || 0;
            updateSpeedDeltaLabel(inc.speed_delta_kmh || 0);
        }

        setCheckbox("is_wrong_side", inc.is_wrong_side);
        setCheckbox("is_overloaded", inc.is_overloaded);
        setCheckbox("safety_gear_violation", inc.safety_gear_violation);
        setCheckbox("high_beam_hazard", inc.high_beam_hazard);
        setCheckbox("stray_animal_hazard", inc.stray_animal_hazard);
        setCheckbox("drowsy_window", inc.drowsy_window);
        setCheckbox("pothole_or_speedbreaker", inc.pothole_or_speedbreaker);

        if (scenarioLabel) scenarioLabel.textContent = `${inc.accident_id} &bull; ${inc.corridor_name}`;

        runPrediction(`${inc.accident_id} (${inc.corridor_name})`);
    }

    if (iradStateFilter) {
        iradStateFilter.addEventListener("change", () => loadIradIncidents());
    }

    if (btnRefreshIrad) {
        btnRefreshIrad.addEventListener("click", () => loadIradIncidents());
    }

    if (btnSimulateIradLive) {
        btnSimulateIradLive.addEventListener("click", async () => {
            try {
                btnSimulateIradLive.textContent = "⏳ Receiving Dispatch...";
                const res = await fetch("/api/irad/simulate_live", { method: "POST" });
                const data = await res.json();
                if (data.status === "success") {
                    await loadIradIncidents();
                }
            } catch (err) {
                console.error("Live dispatch error:", err);
            } finally {
                btnSimulateIradLive.textContent = "🚨 Simulate Incoming Crash Alert";
            }
        });
    }

    // Run initial What-If test comparison so Tab 2 is pre-populated
    setTimeout(() => {
        runWhatIfComparison();
        loadIradIncidents();
    }, 600);
});
