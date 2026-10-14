"""
RoadSense AI: Disaster Prevention & Emergency Mitigation Engine (Indian Roads)
==============================================================================
Designed for Indian traffic conditions, road infrastructure (NH, Expressways,
Ghats), high-risk weather (Monsoon, Winter Fog, Summer Heatwaves), and behavioral
factors (wrong-side driving, overloading, high-beam glare).

Provides:
  1. Catastrophic Disaster Risk Index (DRI) calculation (0 - 100%)
  2. Multi-Hazard Escalation Identification (Pileups, Valley Plunges, Fire/Tanker)
  3. Actionable In-Transit Mitigation Directives (Safe speed limits, gear, spacing)
  4. Post-Crash Secondary Collision Prevention (Scene shielding, barrier evacuation)
  5. Emergency "Golden Hour" Protocol & Helpline Integration (112, 1033)
"""

from typing import Dict, List, Any
import numpy as np


class DisasterPreventionEngine:
    """
    Expert engine for assessing systemic disaster potential and providing
    actionable prevention protocols for Indian road networks.
    """

    @staticmethod
    def assess_disaster_risk(scenario: Dict[str, Any], severity_probs: Dict[str, float]) -> Dict[str, Any]:
        """
        Computes Disaster Risk Index (DRI) and generates targeted disaster
        prevention directives based on the specific Indian road context.
        """
        road_cat = scenario.get("road_category", "National Highway (NH)")
        weather = scenario.get("weather_condition", "Clear / Sunny")
        surface = scenario.get("road_surface", "Dry Asphalt / Concrete")
        light = scenario.get("light_condition", "Daylight")
        junction = scenario.get("junction_type", "Straight Road (Not at Junction)")
        v1 = scenario.get("primary_vehicle_type", "Car / Hatchback / Sedan")
        v2 = scenario.get("secondary_vehicle_type", "None")
        speed_limit = scenario.get("speed_limit_kmh", 80)
        speed_delta = scenario.get("speed_delta_kmh", 0)
        actual_speed = speed_limit + speed_delta

        wrong_side = bool(scenario.get("is_wrong_side", 0))
        overloaded = bool(scenario.get("is_overloaded", 0))
        gear_violation = bool(scenario.get("safety_gear_violation", 0))
        high_beam = bool(scenario.get("high_beam_hazard", 0))
        stray_animal = bool(scenario.get("stray_animal_hazard", 0))
        drowsy_time = bool(scenario.get("drowsy_window", 0))
        pothole_hazard = bool(scenario.get("pothole_or_speedbreaker", 0))

        # Base disaster points from model's fatal + grievous prediction
        fatal_prob = severity_probs.get("Fatal", 0.05)
        grievous_prob = severity_probs.get("Grievous", 0.25)

        # Scale base DRI dynamically with operating kinetic velocity
        speed_factor = max(0.25, min(1.25, actual_speed / 80.0))
        base_dri = (fatal_prob * 1.4 + grievous_prob * 0.6) * 55.0 * speed_factor

        risk_multipliers: List[Dict[str, Any]] = []
        mitigation_directives: List[str] = []
        secondary_prevention_protocols: List[str] = []
        escalation_threats: List[Dict[str, str]] = []

        # ----------------------------------------------------------------------
        # Escalation Threat 1: Winter Fog / Smog Expressway Pileup
        # ----------------------------------------------------------------------
        if "Fog" in weather or "Smog" in weather:
            fog_boost = 25.0 if "Expressway" in road_cat or "Highway" in road_cat else 15.0
            base_dri += fog_boost
            risk_multipliers.append({
                "factor": "Dense Winter Fog / Smog",
                "impact": f"+{fog_boost:.0f}% Disaster Risk",
                "reason": "Severe visibility drop (<30m) at highway velocities induces massive 20+ vehicle chain-reaction collisions."
            })
            escalation_threats.append({
                "title": "Chain-Reaction Expressway Pileup Hazard",
                "severity": "CRITICAL",
                "description": "Initial bumper-to-bumper contact triggers multi-vehicle pileup with heavy commercial freight slamming into stopped passenger vehicles."
            })
            mitigation_directives.append(
                f"🚨 DYNAMIC SPEED RESTRICTION: Do NOT exceed 40 km/h on expressways/highways. Turn on yellow fog lamps and LOW beams (never high beams, which cause blinding white wall backscatter)."
            )
            mitigation_directives.append(
                "⚠️ HAZARD LIGHT DISCIPLINE: Do NOT drive with hazard flashers on while moving (this confuses following drivers into thinking you are stationary). Use hazards ONLY when completely pulled off onto the shoulder."
            )

        # ----------------------------------------------------------------------
        # Escalation Threat 2: Monsoon Aquaplaning & Waterlogging
        # ----------------------------------------------------------------------
        if "Monsoon" in weather or "Waterlogged" in surface or "Wet" in surface:
            monsoon_boost = 18.0 if actual_speed > 60 else 10.0
            base_dri += monsoon_boost
            risk_multipliers.append({
                "factor": "Monsoon Aquaplaning & Brake Fade",
                "impact": f"+{monsoon_boost:.0f}% Disaster Risk",
                "reason": "Standing water creates hydroplaning hydrodynamic lift; two-wheeler front wheel lockup causes instantaneous loss of control."
            })
            escalation_threats.append({
                "title": "High-Speed Hydroplaning & Secondary Run-Off",
                "severity": "HIGH",
                "description": "Tires lose road contact, leading to violent spinouts into oncoming traffic or deep roadside drainage ditches."
            })
            safe_rain_speed = min(actual_speed, max(40, speed_limit - 25))
            mitigation_directives.append(
                f"🌧️ AQUAPLANING ADVISORY: Cap travel speed at {safe_rain_speed} km/h. Maintain at least 5 to 6 vehicle lengths (4-second rule). Tap brakes gently to dry water from brake discs/drums."
            )

        # ----------------------------------------------------------------------
        # Escalation Threat 3: Wrong-Side Driving (Contraflow Catastrophe)
        # ----------------------------------------------------------------------
        if wrong_side:
            ws_boost = 30.0 if "Highway" in road_cat or "Expressway" in road_cat else 20.0
            base_dri += ws_boost
            risk_multipliers.append({
                "factor": "Wrong-Side (Contraflow) Driving",
                "impact": f"+{ws_boost:.0f}% Disaster Risk",
                "reason": "Relative closing speed doubles (e.g. 80 + 80 = 160 km/h kinetic impact), almost guaranteeing fatal outcomes."
            })
            escalation_threats.append({
                "title": "High-Energy Head-On Collision",
                "severity": "CRITICAL",
                "description": "Kinetic energy release E = 0.5 * m * (v1 + v2)^2 crushes cabin passenger cell beyond structural crumple zone limits."
            })
            mitigation_directives.append(
                "🛑 ZERO-TOLERANCE CONTRAFLOW WARNING: Never drive on the wrong side to avoid a U-turn or toll plaza. Proceed to the designated grade-separated interchange or authorized median opening."
            )

        # ----------------------------------------------------------------------
        # Escalation Threat 4: High Pavement Heat & Concrete Tire Blowout
        # ----------------------------------------------------------------------
        if "Heatwave" in weather and ("Expressway" in road_cat or actual_speed >= 90):
            heat_boost = 15.0
            base_dri += heat_boost
            risk_multipliers.append({
                "factor": "Extreme Pavement Thermal Load (>45°C)",
                "impact": f"+{heat_boost:.0f}% Disaster Risk",
                "reason": "High-friction cement concrete highways cause air pressure buildup and carcass delamination, leading to tire blowouts."
            })
            escalation_threats.append({
                "title": "High-Velocity Tire Delamination / Rollover",
                "severity": "HIGH",
                "description": "Rear or front tire blowout at >100 km/h leads to immediate rollover and ejection of unbelted passengers."
            })
            mitigation_directives.append(
                "🔥 TIRE INTEGRITY ADVISORY: Check cold tire pressure before highway run. On concrete expressways, take a mandatory 15-minute halt every 90 minutes to allow tires to dissipate heat."
            )

        # ----------------------------------------------------------------------
        # Escalation Threat 5: Ghat Section / Mountain Drop-Off
        # ----------------------------------------------------------------------
        if "Ghat" in road_cat or "Hairpin" in junction:
            ghat_boost = 22.0
            base_dri += ghat_boost
            risk_multipliers.append({
                "factor": "Ghat / Hill Terrain Dynamics",
                "impact": f"+{ghat_boost:.0f}% Disaster Risk",
                "reason": "Steep descent induces brake fluid boiling (brake fade); blind corners prevent sighting oncoming heavy trucks."
            })
            escalation_threats.append({
                "title": "Gorge Plunge / Blind Corner Head-On",
                "severity": "CRITICAL",
                "description": "Brake failure on downhill gradients or overtaking on blind hairpin curves pushes vehicle over unprotected valley precipice."
            })
            mitigation_directives.append(
                "🏔️ GHAT SECTION PROTOCOL: Use engine braking (lower gear like 2nd or 3rd) on descents. Honk audibly before entering blind curves. Strictly NO overtaking across solid center lines."
            )

        # ----------------------------------------------------------------------
        # Escalation Threat 6: High-Beam Glare & Undivided Highways
        # ----------------------------------------------------------------------
        if high_beam and "Undivided" in scenario.get("road_type", ""):
            beam_boost = 14.0
            base_dri += beam_boost
            risk_multipliers.append({
                "factor": "High-Beam Retinal Glare",
                "impact": f"+{beam_boost:.0f}% Disaster Risk",
                "reason": "Causes 3-5 seconds of physiological blindness at night; pedestrian, cyclist, or unlit tractor trolley becomes invisible."
            })
            escalation_threats.append({
                "title": "Night Blindness & Undivided Carriageway Impact",
                "severity": "HIGH",
                "description": "Blinded driver veers across the centerline into oncoming vehicles or strikes unlit obstacles on road shoulders."
            })
            mitigation_directives.append(
                "💡 NIGHT GLARE MITIGATION: Switch to low beam when approaching oncoming traffic within 200m. If blinded by oncoming high beams, look slightly down towards the left road shoulder white line to stay in your lane."
            )

        # ----------------------------------------------------------------------
        # Escalation Threat 7: Overloading & Safety Gear Violations
        # ----------------------------------------------------------------------
        if overloaded or gear_violation:
            gear_boost = 18.0
            base_dri += gear_boost
            risk_multipliers.append({
                "factor": "Overloading / Non-Usage of Helmet/Seatbelt",
                "impact": f"+{gear_boost:.0f}% Disaster Risk",
                "reason": "Lack of strapped helmet multiplies head trauma fatality risk by 4.2x; unbelted passengers become lethal human projectiles inside vehicle."
            })
            escalation_threats.append({
                "title": "Mass Vulnerable Occupant Ejection & Trauma",
                "severity": "HIGH",
                "description": "Overcrowded passenger compartments or unhelmeted pillion riders suffer catastrophic crush and blunt-force head trauma upon impact."
            })
            mitigation_directives.append(
                "🪖 PASSENGER PROTECTION MANDATE: Ensure full-face BIS/ISI certified helmets with chin strap firmly fastened for all two-wheeler riders. All occupants (front and rear) must wear 3-point seatbelts."
            )

        # ----------------------------------------------------------------------
        # Escalation Threat 8: Drowsy Window / Midnight Commercial Traffic
        # ----------------------------------------------------------------------
        if drowsy_time:
            drowsy_boost = 12.0
            base_dri += drowsy_boost
            risk_multipliers.append({
                "factor": "Circadian Low Drowsy Window (01:00 - 05:30 AM)",
                "impact": f"+{drowsy_boost:.0f}% Disaster Risk",
                "reason": "Micro-sleep episodes (3-4 seconds) cause vehicle to drift into opposing lanes or rear-end stationary roadside trucks without braking."
            })
            escalation_threats.append({
                "title": "Unbraked High-Speed Impact Due to Microsleep",
                "severity": "CRITICAL",
                "description": "Vehicle impacts stationary roadside hazard or rear of slow-moving freight vehicle at full speed without skid marks or evasive maneuver."
            })
            mitigation_directives.append(
                "☕ FATIGUE ALERT: Highway driving during 01:00 AM - 05:30 AM has the highest fatality-to-collision ratio in India. If eyelids feel heavy, immediately halt at a 24x7 highway dhabha or fuel station for a 20-minute power nap."
            )

        # ----------------------------------------------------------------------
        # Post-Crash Disaster Containment Protocol (Secondary Collision Prevention)
        # ----------------------------------------------------------------------
        secondary_prevention_protocols = [
            "1️⃣ IMMEDIATE OCCUPANT EVACUATION: Do NOT stay inside the vehicle or stand in live traffic lanes arguing or inspecting bumper damage! Immediately evacuate all passengers to the far side of the metal crash barrier (W-beam guard rail) or outer grassy slope.",
            "2️⃣ EARLY WARNING TRIANGLE DEPLOYMENT: Place a retro-reflective hazard triangle or blinking red beacon at least 75 to 100 METERS UPSTREAM (behind the stopped vehicle). On fast expressways, give speeding trucks adequate stopping distance.",
            "3️⃣ ACTIVATE HAZARD FLASHERS: Turn on emergency hazard flashers and parking lights on the disabled vehicle to pierce darkness or fog.",
            "4️⃣ CALL NATIONAL HIGHWAY PATROL (1033) & EMERGENCY (112): Immediately dial 1033 (NHAI 24x7 Incident Management & Free Ambulance/Tow Service) or 112 for local emergency response within the crucial 'Golden Hour'.",
            "5️⃣ GOOD SAMARITAN LEGAL PROTECTION: Under Supreme Court of India guidelines & Motor Vehicles Act Sec 134A, any citizen aiding accident victims is protected from police questioning, civil/criminal liability, or hospital detention."
        ]

        # Clamp DRI score between 5% and 98%
        dri_score = float(np.clip(base_dri, 5.0, 98.0))

        if dri_score >= 70.0:
            dri_band = "CATASTROPHIC DISASTER RISK"
            dri_color = "red"
            dri_emoji = "🚨"
            urgency = "EXTREME DANGER: Multiple high-order disaster catalysts present. High likelihood of mass casualties."
        elif dri_score >= 50.0:
            dri_band = "SEVERE HAZARD ALERT"
            dri_color = "orange"
            dri_emoji = "🟠"
            urgency = "HIGH DANGER: Potentially lethal conditions. Immediate preventive driving intervention required."
        elif dri_score >= 30.0:
            dri_band = "MODERATE RISK ZONE"
            dri_color = "yellow"
            dri_emoji = "🟡"
            urgency = "ELEVATED RISK: Complex road dynamics. Exercise heightened situational awareness."
        else:
            dri_band = "LOW / NORMAL MONITORING"
            dri_color = "green"
            dri_emoji = "🟢"
            urgency = "BASELINE RISK: Standard Indian defensive driving protocols apply."

        return {
            "disaster_risk_index": round(dri_score, 1),
            "risk_band": dri_band,
            "risk_emoji": dri_emoji,
            "risk_color": dri_color,
            "urgency_assessment": urgency,
            "risk_multipliers": risk_multipliers,
            "escalation_threats": escalation_threats,
            "mitigation_directives": mitigation_directives,
            "secondary_prevention_protocols": secondary_prevention_protocols,
        }
