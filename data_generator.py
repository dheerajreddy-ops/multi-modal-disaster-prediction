"""
data_generator.py
Generates synthetic multi-modal disaster data:
- Text reports (news/social media style)
- Sensor readings (weather, seismic, environmental)
- Labels (disaster type, severity)

Generates ~10,000 samples with balanced classes.
"""

import os
import random
import csv
import json

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

DISASTER_TYPES = ["earthquake", "flood", "hurricane", "wildfire", "tornado", "tsunami"]

SEVERITY_LEVELS = ["low", "moderate", "high", "critical"]

LOCATIONS = [
    "Coastal California", "Gulf Coast Texas", "Pacific Northwest", "Florida Peninsula",
    "Central Oklahoma", "Himalayan Region", "Southeast Asia", "Caribbean Islands",
    "Mediterranean Coast", "Sub-Saharan Africa", "South America", "Japan Archipelago",
    "New Zealand", "Philippines", "Bangladesh Delta", "Mozambique Channel",
    "Indonesia", "Haiti", "Nepal", "Chile Coast", "Turkey", "Iran Plateau",
    "Italy", "Greece", "Iceland", "Alaska", "Hawaii", "New York Coast",
    "Mississippi Valley", "Great Plains",
]

TEXT_TEMPLATES = {
    "earthquake": {
        "low": [
            "Minor tremor felt in {loc}, no damage reported. Residents reported轻微 shaking.",
            "Light earthquake of magnitude {mag} detected near {loc}. Buildings shook briefly.",
            "Seismic activity recorded in {loc} region. People felt mild vibrations.",
        ],
        "moderate": [
            "Magnitude {mag} earthquake strikes {loc}. Cracks appear in walls, minor structural damage.",
            "Moderate earthquake in {loc} disrupts power supply. Multiple aftershocks reported.",
            "5.{mag_r} magnitude earthquake hits {loc}. Roads damaged, emergency teams deployed.",
        ],
        "high": [
            "Powerful magnitude {mag} earthquake devastates {loc}. Buildings collapsed, casualties reported.",
            "Major earthquake of {mag} magnitude rocks {loc}. Tsunami warning issued for coastal areas.",
            "Severe earthquake in {loc} causes widespread destruction. Thousands displaced.",
        ],
        "critical": [
            "Catastrophic magnitude {mag} earthquake annihilates {loc}. Entire neighborhoods flattened. Mass casualties.",
            "Devastating {mag} magnitude earthquake hits {loc}. Infrastructure completely destroyed. State of emergency declared.",
            "Extreme seismic event in {loc} - magnitude {mag}. Massive destruction, rescue operations overwhelmed.",
        ],
    },
    "flood": {
        "low": [
            "Minor waterlogging in {loc} after heavy rainfall. Traffic disrupted in low-lying areas.",
            "Small flash flood in {loc} submerges streets. Residents advised to move to higher ground.",
            "River levels rising in {loc} due to continuous rain. Flood alert issued.",
        ],
        "moderate": [
            "Moderate flooding in {loc} submerges residential areas. Hundreds evacuated.",
            "Flash floods in {loc} damage homes and infrastructure. Relief teams mobilized.",
            "River breach in {loc} causes significant flooding. Emergency shelters opened.",
        ],
        "high": [
            "Severe flooding devastates {loc}. Thousands stranded, rooftops visible above water.",
            "Catastrophic floods in {loc} destroy homes, bridges, roads. Mass evacuation underway.",
            "Major river flooding in {loc} displaces tens of thousands. National Guard deployed.",
        ],
        "critical": [
            "Unprecedented flood catastrophe in {loc}. Entire towns submerged. Hundreds feared dead.",
            "Extreme flooding in {loc} causes dam breach. Massive destruction, international aid requested.",
            "Historic flood event in {loc} - unprecedented water levels. Complete infrastructure collapse.",
        ],
    },
    "hurricane": {
        "low": [
            "Tropical storm approaching {loc}. Winds at 60 mph, heavy rain expected.",
            "Category 1 hurricane making landfall near {loc}. Minor flooding and power outages.",
            "Weak tropical system affecting {loc}. Some tree damage and localized flooding.",
        ],
        "moderate": [
            "Category 2 hurricane battering {loc}. Winds reaching 100 mph, significant damage.",
            "Hurricane warnings escalated in {loc}. Storm surge threatening coastal communities.",
            "Moderate hurricane impacting {loc}. Power outages widespread, roofs damaged.",
        ],
        "high": [
            "Major Category 3 hurricane hammering {loc}. Catastrophic storm surge, massive destruction.",
            "Powerful hurricane in {loc} with 125 mph winds. Entire communities destroyed.",
            "Dangerous hurricane hitting {loc}. Complete power grid failure, buildings leveled.",
        ],
        "critical": [
            "Category 5 super hurricane annihilates {loc}. Winds exceed 160 mph. Total devastation.",
            "Unprecedented hurricane catastrophe in {loc}. Storm surge of 20+ feet. Mass casualties.",
            "Maximum intensity hurricane makes direct hit on {loc}. Complete destruction of infrastructure.",
        ],
    },
    "wildfire": {
        "low": [
            "Small brush fire reported near {loc}. Firefighters contain blaze quickly.",
            "Minor wildfire burning in {loc} forest area. Air quality slightly affected.",
            "Controlled fire activity in {loc}. No structures threatened at this time.",
        ],
        "moderate": [
            "Growing wildfire in {loc} threatens homes. Evacuations ordered for nearby communities.",
            "Multiple wildfires burning across {loc}. Firefighters struggling to contain blazes.",
            "Significant wildfire in {loc} destroys several structures. Smoke blankets region.",
        ],
        "high": [
            "Massive wildfire raging through {loc}. Hundreds of homes destroyed. Towns evacuated.",
            "Catastrophic firestorm in {loc} consuming everything in path. Thousands displaced.",
            "Extreme wildfire conditions in {loc}. Fire tornadoes reported, complete devastation.",
        ],
        "critical": [
            "Megafire engulfs {loc}. Entire communities incinerated. Firefighter casualties reported.",
            "Unprecedented wildfire catastrophe in {loc}. Fire crosses into urban areas. Total destruction.",
            "Extreme fire emergency in {loc}. Firestorm creates own weather system. Mass evacuation.",
        ],
    },
    "tornado": {
        "low": [
            "Weak tornado spotted near {loc}. Minor damage to trees and signage.",
            "EF0 tornado touches down in {loc}. Brief but no significant damage.",
            "Small funnel cloud observed in {loc}. Minimal impact on ground.",
        ],
        "moderate": [
            "EF2 tornado tears through {loc}. Roofs ripped off, cars overturned.",
            "Strong tornado hits {loc} causing significant structural damage. Injuries reported.",
            "Multiple tornadoes reported in {loc} area. Homes damaged, power lines downed.",
        ],
        "high": [
            "Violent EF4 tornado devastates {loc}. Buildings completely leveled. Mass casualties.",
            "Deadly tornado outbreak in {loc}. Entire neighborhoods wiped out.",
            "Major tornado strikes {loc} with 200 mph winds. Catastrophic damage and injuries.",
        ],
        "critical": [
            "EF5 monster tornado annihilates {loc}. Nothing left standing. Mass fatalities.",
            "Unprecedented tornado catastrophe in {loc}. Buildings swept clean from foundations.",
            "Maximum intensity tornado destroys {loc}. Complete annihilation of structures.",
        ],
    },
    "tsunami": {
        "low": [
            "Minor wave activity detected near {loc}. Coastal residents advised to stay alert.",
            "Small tsunami wave of 1 meter reaches {loc} coast. No significant damage.",
            "Tsunami advisory for {loc}. Minor coastal flooding reported.",
        ],
        "moderate": [
            "Tsunami waves of 3-5 meters hitting {loc} coast. Coastal areas flooded.",
            "Moderate tsunami impacting {loc}. Buildings near shore damaged. Evacuations underway.",
            "Significant tsunami in {loc}. Fishing boats destroyed, coastal roads flooded.",
        ],
        "high": [
            "Major tsunami with 10+ meter waves striking {loc}. Massive coastal destruction.",
            "Devastating tsunami in {loc} sweeps away buildings. Thousands displaced.",
            "Large tsunami inundates {loc} coast. Complete destruction of waterfront areas.",
        ],
        "critical": [
            "Mega-tsunami with 20+ meter waves obliterates {loc} coastline. Mass casualties.",
            "Catastrophic tsunami event in {loc}. Entire coastal cities swept away. International emergency.",
            "Extreme tsunami in {loc} reaches kilometers inland. Complete coastal annihilation.",
        ],
    },
}

SENSOR_PROFILES = {
    "earthquake": {
        "seismic_activity": (0.3, 0.95),
        "ground_vibration": (0.2, 0.9),
        "temperature_c": (15, 35),
        "humidity_pct": (30, 70),
        "wind_speed_kmh": (5, 40),
        "air_pressure_hpa": (990, 1025),
        "rainfall_mm": (0, 20),
        "water_level_m": (0, 5),
        "visibility_km": (5, 15),
    },
    "flood": {
        "seismic_activity": (0.0, 0.2),
        "ground_vibration": (0.0, 0.15),
        "temperature_c": (20, 38),
        "humidity_pct": (70, 100),
        "wind_speed_kmh": (10, 60),
        "air_pressure_hpa": (985, 1010),
        "rainfall_mm": (50, 300),
        "water_level_m": (2, 15),
        "visibility_km": (0.5, 5),
    },
    "hurricane": {
        "seismic_activity": (0.0, 0.1),
        "ground_vibration": (0.0, 0.1),
        "temperature_c": (22, 35),
        "humidity_pct": (80, 100),
        "wind_speed_kmh": (60, 350),
        "air_pressure_hpa": (920, 990),
        "rainfall_mm": (30, 250),
        "water_level_m": (1, 12),
        "visibility_km": (0.1, 5),
    },
    "wildfire": {
        "seismic_activity": (0.0, 0.05),
        "ground_vibration": (0.0, 0.05),
        "temperature_c": (30, 55),
        "humidity_pct": (5, 30),
        "wind_speed_kmh": (15, 100),
        "air_pressure_hpa": (995, 1020),
        "rainfall_mm": (0, 2),
        "water_level_m": (0, 1),
        "visibility_km": (0.2, 5),
    },
    "tornado": {
        "seismic_activity": (0.0, 0.15),
        "ground_vibration": (0.1, 0.6),
        "temperature_c": (18, 40),
        "humidity_pct": (50, 95),
        "wind_speed_kmh": (100, 500),
        "air_pressure_hpa": (950, 1005),
        "rainfall_mm": (10, 100),
        "water_level_m": (0, 3),
        "visibility_km": (0.1, 5),
    },
    "tsunami": {
        "seismic_activity": (0.5, 1.0),
        "ground_vibration": (0.3, 0.8),
        "temperature_c": (20, 32),
        "humidity_pct": (60, 95),
        "wind_speed_kmh": (20, 120),
        "air_pressure_hpa": (970, 1010),
        "rainfall_mm": (5, 60),
        "water_level_m": (3, 20),
        "visibility_km": (1, 8),
    },
}

SEVERITY_MULTIPLIERS = {
    "low": 0.25,
    "moderate": 0.50,
    "high": 0.75,
    "critical": 1.0,
}


def generate_sensor_reading(disaster_type, severity):
    profile = SENSOR_PROFILES[disaster_type]
    mult = SEVERITY_MULTIPLIERS[severity]
    reading = {}
    for sensor, (low, high) in profile.items():
        range_span = high - low
        base = low + range_span * mult
        noise = random.gauss(0, range_span * 0.1)
        value = max(low, min(high, base + noise))
        reading[sensor] = round(value, 2)
    return reading


def generate_text_report(disaster_type, severity, location):
    templates = TEXT_TEMPLATES[disaster_type][severity]
    template = random.choice(templates)
    mag = round(random.uniform(4.0, 9.5), 1)
    mag_r = random.randint(1, 9)
    text = template.format(loc=location, mag=mag, mag_r=mag_r)
    return text


def generate_sample(idx):
    disaster_type = random.choice(DISASTER_TYPES)
    severity = random.choices(
        SEVERITY_LEVELS, weights=[0.25, 0.30, 0.25, 0.20]
    )[0]
    location = random.choice(LOCATIONS)

    text = generate_text_report(disaster_type, severity, location)
    sensors = generate_sensor_reading(disaster_type, severity)

    return {
        "id": idx,
        "disaster_type": disaster_type,
        "severity": severity,
        "location": location,
        "text_report": text,
        **sensors,
    }


def generate_dataset(n_samples=10000):
    os.makedirs(DATA_DIR, exist_ok=True)

    samples = []
    for i in range(n_samples):
        sample = generate_sample(i)
        samples.append(sample)

    csv_path = os.path.join(DATA_DIR, "disaster_data.csv")
    fieldnames = [
        "id", "disaster_type", "severity", "location", "text_report",
        "seismic_activity", "ground_vibration", "temperature_c", "humidity_pct",
        "wind_speed_kmh", "air_pressure_hpa", "rainfall_mm", "water_level_m",
        "visibility_km",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)

    stats = {
        "total_samples": len(samples),
        "disaster_types": {},
        "severity_distribution": {},
        "sensor_columns": [k for k in samples[0].keys() if k not in ["id", "disaster_type", "severity", "location", "text_report"]],
    }

    for s in samples:
        dt = s["disaster_type"]
        sv = s["severity"]
        stats["disaster_types"][dt] = stats["disaster_types"].get(dt, 0) + 1
        stats["severity_distribution"][sv] = stats["severity_distribution"].get(sv, 0) + 1

    stats_path = os.path.join(DATA_DIR, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Generated {len(samples):,} samples")
    print(f"CSV: {csv_path}")
    print(f"Stats: {stats_path}")
    print(f"Disaster types: {stats['disaster_types']}")
    print(f"Severity: {stats['severity_distribution']}")
    print(f"Sensor columns: {stats['sensor_columns']}")

    return csv_path, stats


if __name__ == "__main__":
    print("=" * 60)
    print("  MULTI-MODAL DISASTER DATA GENERATOR")
    print("=" * 60)
    print()
    generate_dataset(10000)
    print()
    print("Done!")
