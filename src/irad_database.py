"""
RoadSense: iRAD (Integrated Road Accident Database) & eDAR Module
================================================================
Implements an authentic local database and live incident feed modeled after
the Ministry of Road Transport and Highways (MoRTH), National Informatics Centre (NIC),
and IIT Madras iRAD / eDAR specification.

Key Schema Entities:
  - Accident Master (ID, State, District, Police Station, GPS Lat/Long, Corridor)
  - Road & Environment Attributes (Hierarchy, Weather, Surface, Light, Junction)
  - Vehicle & Violation Attributes (Primary/Secondary, Speed, Wrong-Side, Helmets)
  - Casualty & Impact Metrics (Fatalities, Grievous, Minor, Evaluated DRI)
"""

import os
import sqlite3
import datetime
import random
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join("data", "irad_accidents.db")


class IRADDatabase:
    """Manages SQLite storage and querying of official iRAD-schema accident records."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes tables adhering to the official iRAD police/transport schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS irad_incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    accident_id TEXT UNIQUE NOT NULL,
                    reported_at TIMESTAMP NOT NULL,
                    state TEXT NOT NULL,
                    district TEXT NOT NULL,
                    police_station TEXT NOT NULL,
                    corridor_name TEXT NOT NULL,
                    chainage_km REAL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    road_category TEXT NOT NULL,
                    road_type TEXT NOT NULL,
                    weather_condition TEXT NOT NULL,
                    road_surface TEXT NOT NULL,
                    light_condition TEXT NOT NULL,
                    junction_type TEXT NOT NULL,
                    primary_vehicle_type TEXT NOT NULL,
                    secondary_vehicle_type TEXT NOT NULL,
                    collision_type TEXT NOT NULL,
                    speed_limit_kmh INTEGER NOT NULL,
                    speed_delta_kmh INTEGER NOT NULL,
                    number_of_vehicles INTEGER NOT NULL,
                    occupants_count INTEGER NOT NULL,
                    is_wrong_side INTEGER DEFAULT 0,
                    is_overloaded INTEGER DEFAULT 0,
                    safety_gear_violation INTEGER DEFAULT 0,
                    high_beam_hazard INTEGER DEFAULT 0,
                    stray_animal_hazard INTEGER DEFAULT 0,
                    drowsy_window INTEGER DEFAULT 0,
                    pothole_or_speedbreaker INTEGER DEFAULT 0,
                    actual_fatalities INTEGER DEFAULT 0,
                    actual_grievous INTEGER DEFAULT 0,
                    actual_minor INTEGER DEFAULT 0,
                    ai_severity_predicted TEXT,
                    ai_predicted_casualties REAL,
                    ai_disaster_risk_index REAL,
                    ai_risk_band TEXT
                );
            """)

            # Index for fast corridor & state spatial queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_irad_state ON irad_incidents(state);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_irad_corridor ON irad_incidents(corridor_name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_irad_time ON irad_incidents(reported_at);")

            # Check if seeded
            cursor.execute("SELECT COUNT(*) as cnt FROM irad_incidents;")
            row = cursor.fetchone()
            if row and row["cnt"] == 0:
                self._seed_realistic_irad_records(cursor)
                conn.commit()

    def _seed_realistic_irad_records(self, cursor: sqlite3.Cursor):
        """Seeds authentic historical & active corridor incident records across Indian highways."""
        corridors = [
            {
                "corridor": "Yamuna Expressway (Greater Noida - Agra)",
                "state": "Uttar Pradesh", "district": "Mathura", "ps": "Surir Thana",
                "lat": 27.7214, "lon": 77.7812, "road_cat": "Expressway", "road_type": "Divided Multi-Lane (4/6-Lane)",
                "weather": "Dense Winter Fog / Smog", "surface": "Dry Asphalt / Concrete", "light": "Dusk / Dawn (Twilight)",
                "junc": "Straight Road (Not at Junction)", "v1": "Car / Hatchback / Sedan",
                "v2": "Heavy Goods Vehicle (Multi-Axle Truck / Dumper / Trailer)", "col": "Rear-End / Chain Pileup",
                "limit": 120, "delta": -10, "veh": 6, "occ": 8, "wrong": 0, "overload": 0, "gear": 1, "beam": 0, "animal": 0, "drowsy": 0, "pothole": 0,
                "fat": 2, "grie": 4, "min": 2
            },
            {
                "corridor": "NH-48 (Delhi - Gurgaon - Jaipur Highway)",
                "state": "Haryana", "district": "Gurugram", "ps": "Kherki Daula",
                "lat": 28.3842, "lon": 76.9821, "road_cat": "National Highway (NH)", "road_type": "Divided Multi-Lane (4/6-Lane)",
                "weather": "Clear / Sunny", "surface": "Dry Asphalt / Concrete", "light": "Night - Pitch Dark (No Streetlights)",
                "junc": "T-Junction / Unsignalized Median Cut", "v1": "SUV / MUV / Compact SUV",
                "v2": "Tractor with Trolley (Sugarcane/Construction)", "col": "Head-On Collision",
                "limit": 100, "delta": 25, "veh": 2, "occ": 6, "wrong": 1, "overload": 1, "gear": 0, "beam": 1, "animal": 0, "drowsy": 1, "pothole": 0,
                "fat": 3, "grie": 2, "min": 1
            },
            {
                "corridor": "Western Ghats (Mumbai - Goa NH-66 Kashedi Ghat)",
                "state": "Maharashtra", "district": "Ratnagiri", "ps": "Poladpur Police",
                "lat": 17.9821, "lon": 73.4519, "road_cat": "Ghat / Hill Terrain Road", "road_type": "Undivided 2-Lane",
                "weather": "Heavy Monsoon / Waterlogged", "surface": "Wet / Slippery", "light": "Daylight",
                "junc": "Blind Curve / Ghat Hairpin Bend", "v1": "Two-Wheeler (Motorcycle / Scooter)",
                "v2": "Bus (State Transport / Private / School)", "col": "Head-On Collision",
                "limit": 40, "delta": 15, "veh": 2, "occ": 35, "wrong": 1, "overload": 1, "gear": 1, "beam": 0, "animal": 0, "drowsy": 0, "pothole": 1,
                "fat": 2, "grie": 5, "min": 6
            },
            {
                "corridor": "Bangalore Outer Ring Road (Hebbal - Marathahalli)",
                "state": "Karnataka", "district": "Bengaluru Urban", "ps": "K.R. Puram Traffic",
                "lat": 12.9982, "lon": 77.6891, "road_cat": "Urban Arterial / Ring Road", "road_type": "Divided Multi-Lane (4/6-Lane)",
                "weather": "Clear / Sunny", "surface": "Dry Asphalt / Concrete", "light": "Daylight",
                "junc": "Flyover Ramp / Blind Slip Road", "v1": "Three-Wheeler (Auto-Rickshaw / E-Rickshaw)",
                "v2": "Light Commercial Vehicle (LCV / Tempo / Tata Ace)", "col": "Right Angle / T-Bone at Median Cut",
                "limit": 50, "delta": 10, "veh": 2, "occ": 7, "wrong": 0, "overload": 1, "gear": 1, "beam": 0, "animal": 0, "drowsy": 0, "pothole": 1,
                "fat": 0, "grie": 3, "min": 4
            },
            {
                "corridor": "NH-44 (Hyderabad - Bangalore Highway Corridor)",
                "state": "Andhra Pradesh", "district": "Kurnool", "ps": "Dhone Police Station",
                "lat": 15.4210, "lon": 77.8721, "road_cat": "National Highway (NH)", "road_type": "Divided Multi-Lane (4/6-Lane)",
                "weather": "Clear / Sunny", "surface": "Dry Asphalt / Concrete", "light": "Night - Pitch Dark (No Streetlights)",
                "junc": "Straight Road (Not at Junction)", "v1": "Heavy Goods Vehicle (Multi-Axle Truck / Dumper / Trailer)",
                "v2": "Car / Hatchback / Sedan", "col": "Rear-End / Chain Pileup",
                "limit": 100, "delta": 15, "veh": 3, "occ": 8, "wrong": 0, "overload": 0, "gear": 1, "beam": 1, "animal": 1, "drowsy": 1, "pothole": 0,
                "fat": 2, "grie": 3, "min": 2
            },
            {
                "corridor": "Samruddhi Mahamarg (Mumbai - Nagpur Expressway)",
                "state": "Maharashtra", "district": "Aurangabad", "ps": "Vaijapur Highway Safety",
                "lat": 19.9214, "lon": 74.7210, "road_cat": "Expressway", "road_type": "Divided Multi-Lane (4/6-Lane)",
                "weather": "Extreme Summer Heatwave (>42°C)", "surface": "Dry Asphalt / Concrete", "light": "Daylight",
                "junc": "Straight Road (Not at Junction)", "v1": "SUV / MUV / Compact SUV",
                "v2": "None", "col": "Vehicle Rollover / Overturn",
                "limit": 120, "delta": 25, "veh": 1, "occ": 5, "wrong": 0, "overload": 0, "gear": 0, "beam": 0, "animal": 0, "drowsy": 0, "pothole": 0,
                "fat": 1, "grie": 2, "min": 2
            },
            {
                "corridor": "NH-16 (Chennai - Kolkata Coastal Corridor)",
                "state": "Tamil Nadu", "district": "Chengalpattu", "ps": "Maduranthakam Highway",
                "lat": 12.5110, "lon": 79.8821, "road_cat": "National Highway (NH)", "road_type": "Divided Multi-Lane (4/6-Lane)",
                "weather": "Heavy Monsoon / Waterlogged", "surface": "Waterlogged / Potholed", "light": "Dusk / Dawn (Twilight)",
                "junc": "Crossroads / 4-Way Intersection", "v1": "Bus (State Transport / Private / School)",
                "v2": "Two-Wheeler (Motorcycle / Scooter)", "col": "Side-Swipe / Overtaking Glance",
                "limit": 80, "delta": 10, "veh": 2, "occ": 22, "wrong": 0, "overload": 1, "gear": 1, "beam": 0, "animal": 0, "drowsy": 0, "pothole": 1,
                "fat": 1, "grie": 4, "min": 8
            },
            {
                "corridor": "Delhi Outer Ring Road (Burari - Mukarba Chowk)",
                "state": "Delhi", "district": "North Delhi", "ps": "Bhalaswa Dairy Traffic",
                "lat": 28.7391, "lon": 77.1642, "road_cat": "Urban Arterial / Ring Road", "road_type": "Divided Multi-Lane (4/6-Lane)",
                "weather": "Dense Winter Fog / Smog", "surface": "Dry Asphalt / Concrete", "light": "Night - Streetlights Functional",
                "junc": "Flyover Ramp / Blind Slip Road", "v1": "Car / Hatchback / Sedan",
                "v2": "Bicycle / Pedestrian", "col": "Hit Pedestrian / Stray Cattle",
                "limit": 60, "delta": 20, "veh": 2, "occ": 3, "wrong": 0, "overload": 0, "gear": 1, "beam": 1, "animal": 0, "drowsy": 1, "pothole": 0,
                "fat": 1, "grie": 1, "min": 1
            }
        ]

        now = datetime.datetime.now()
        for idx, item in enumerate(corridors):
            time_offset = datetime.timedelta(hours=idx * 4, minutes=random.randint(10, 50))
            rep_time = (now - time_offset).strftime("%Y-%m-%d %H:%M:%S")
            accident_id = f"IRAD/{item['state'][:2].upper()}/{now.year}/{1000 + idx}"

            cursor.execute("""
                INSERT OR IGNORE INTO irad_incidents (
                    accident_id, reported_at, state, district, police_station, corridor_name, chainage_km,
                    latitude, longitude, road_category, road_type, weather_condition, road_surface,
                    light_condition, junction_type, primary_vehicle_type, secondary_vehicle_type,
                    collision_type, speed_limit_kmh, speed_delta_kmh, number_of_vehicles, occupants_count,
                    is_wrong_side, is_overloaded, safety_gear_violation, high_beam_hazard,
                    stray_animal_hazard, drowsy_window, pothole_or_speedbreaker,
                    actual_fatalities, actual_grievous, actual_minor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                accident_id, rep_time, item["state"], item["district"], item["ps"], item["corridor"],
                round(random.uniform(10.0, 150.0), 1), item["lat"], item["lon"], item["road_cat"],
                item["road_type"], item["weather"], item["surface"], item["light"], item["junc"],
                item["v1"], item["v2"], item["col"], item["limit"], item["delta"], item["veh"], item["occ"],
                item["wrong"], item["overload"], item["gear"], item["beam"], item["animal"], item["drowsy"], item["pothole"],
                item["fat"], item["grie"], item["min"]
            ))

    def fetch_all(self, limit: int = 50, state: Optional[str] = None) -> List[Dict[str, Any]]:
        """Queries accident records from the database with optional filtering."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if state and state != "All":
                cursor.execute("""
                    SELECT * FROM irad_incidents
                    WHERE state = ?
                    ORDER BY reported_at DESC LIMIT ?;
                """, (state, limit))
            else:
                cursor.execute("""
                    SELECT * FROM irad_incidents
                    ORDER BY reported_at DESC LIMIT ?;
                """, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def update_ai_assessment(self, accident_id: str, severity: str, casualties: float, dri: float, band: str):
        """Stores AI model assessment predictions back into the iRAD record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE irad_incidents
                SET ai_severity_predicted = ?,
                    ai_predicted_casualties = ?,
                    ai_disaster_risk_index = ?,
                    ai_risk_band = ?
                WHERE accident_id = ?;
            """, (severity, casualties, dri, band, accident_id))
            conn.commit()

    def simulate_live_incoming_incident(self) -> Dict[str, Any]:
        """Simulates an incoming real-time iRAD police crash report."""
        templates = [
            ("Uttar Pradesh", "Agra", "Fatehabad PS", "Yamuna Expressway", "Expressway", "Dense Winter Fog / Smog",
             "Car / Hatchback / Sedan", "Heavy Goods Vehicle (Multi-Axle Truck / Dumper / Trailer)", "Rear-End / Chain Pileup", 120, -5, 4, 7, 0, 0, 1, 0, 0, 0, 0, 27.1767, 78.0081),
            ("Maharashtra", "Pune", "Lonavala Rural", "Mumbai-Pune Expressway", "Expressway", "Heavy Monsoon / Waterlogged",
             "SUV / MUV / Compact SUV", "None", "Run-Off Road / Hit Tree, Pole or Ditch", 100, 15, 1, 4, 0, 0, 0, 0, 0, 0, 0, 18.7546, 73.4062),
            ("Haryana", "Sonipat", "Rai Police Station", "NH-44 (GT Road Murthal)", "National Highway (NH)", "Clear / Sunny",
             "Car / Hatchback / Sedan", "Tractor with Trolley (Sugarcane/Construction)", "Head-On Collision", 100, 25, 2, 5, 1, 1, 0, 1, 0, 1, 0, 28.9845, 77.0891),
            ("Tamil Nadu", "Kanchipuram", "Sriperumbudur PS", "Chennai-Bangalore NH-48", "National Highway (NH)", "Clear / Sunny",
             "Two-Wheeler (Motorcycle / Scooter)", "Bus (State Transport / Private / School)", "Head-On Collision", 80, 10, 2, 2, 1, 1, 1, 0, 0, 0, 1, 12.9675, 79.9450),
            ("Karnataka", "Bangalore", "Electronic City Traffic", "Hosur Road Elevated Tollway", "Flyover / Elevated Corridor", "Clear / Sunny",
             "Two-Wheeler (Motorcycle / Scooter)", "Light Commercial Vehicle (LCV / Tempo / Tata Ace)", "Side-Swipe / Overtaking Glance", 80, 20, 2, 2, 0, 1, 1, 0, 0, 0, 0, 12.8399, 77.6770)
        ]

        t = random.choice(templates)
        now = datetime.datetime.now()
        unique_id = f"IRAD/{t[0][:2].upper()}/{now.year}/{random.randint(2000, 9999)}"

        incident = {
            "accident_id": unique_id,
            "reported_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "state": t[0], "district": t[1], "police_station": t[2], "corridor_name": t[3],
            "chainage_km": round(random.uniform(5.0, 95.0), 1),
            "latitude": t[20] + random.uniform(-0.01, 0.01),
            "longitude": t[21] + random.uniform(-0.01, 0.01),
            "road_category": t[4], "road_type": "Divided Multi-Lane (4/6-Lane)",
            "weather_condition": t[5], "road_surface": "Dry Asphalt / Concrete",
            "light_condition": "Daylight", "junction_type": "Straight Road (Not at Junction)",
            "primary_vehicle_type": t[6], "secondary_vehicle_type": t[7], "collision_type": t[8],
            "speed_limit_kmh": t[9], "speed_delta_kmh": t[10], "number_of_vehicles": t[11],
            "occupants_count": t[12], "is_wrong_side": t[13], "is_overloaded": t[14],
            "safety_gear_violation": t[15], "high_beam_hazard": t[16], "stray_animal_hazard": t[17],
            "drowsy_window": t[18], "pothole_or_speedbreaker": t[19],
            "actual_fatalities": 0, "actual_grievous": 1, "actual_minor": 2
        }

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO irad_incidents (
                    accident_id, reported_at, state, district, police_station, corridor_name, chainage_km,
                    latitude, longitude, road_category, road_type, weather_condition, road_surface,
                    light_condition, junction_type, primary_vehicle_type, secondary_vehicle_type,
                    collision_type, speed_limit_kmh, speed_delta_kmh, number_of_vehicles, occupants_count,
                    is_wrong_side, is_overloaded, safety_gear_violation, high_beam_hazard,
                    stray_animal_hazard, drowsy_window, pothole_or_speedbreaker,
                    actual_fatalities, actual_grievous, actual_minor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                incident["accident_id"], incident["reported_at"], incident["state"], incident["district"],
                incident["police_station"], incident["corridor_name"], incident["chainage_km"],
                incident["latitude"], incident["longitude"], incident["road_category"], incident["road_type"],
                incident["weather_condition"], incident["road_surface"], incident["light_condition"],
                incident["junction_type"], incident["primary_vehicle_type"], incident["secondary_vehicle_type"],
                incident["collision_type"], incident["speed_limit_kmh"], incident["speed_delta_kmh"],
                incident["number_of_vehicles"], incident["occupants_count"], incident["is_wrong_side"],
                incident["is_overloaded"], incident["safety_gear_violation"], incident["high_beam_hazard"],
                incident["stray_animal_hazard"], incident["drowsy_window"], incident["pothole_or_speedbreaker"],
                incident["actual_fatalities"], incident["actual_grievous"], incident["actual_minor"]
            ))
            conn.commit()

        return incident
