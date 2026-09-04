"""
Generate synthetic demo data for CYCLONE-AI.

Creates clearly-labeled SYNTHETIC DEMONSTRATION DATA including:
- Cyclone track records (inspired by real North Indian Ocean patterns)
- Synthetic satellite image tensors
- Environmental feature vectors
- Demo metadata

THIS DATA IS SYNTHETIC AND FOR DEMONSTRATION PURPOSES ONLY.
It does not represent real observations or official data.
"""
import json
import os
import sys
import math
import random
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Famous North Indian Ocean cyclones (names used as inspiration)
DEMO_CYCLONES = [
    {
        "id": "DEMO_NI_2023_001",
        "name": "DEMO_BIPARJOY",
        "basin": "NI",
        "sub_basin": "Arabian Sea",
        "season": 2023,
        "start_lat": 10.5,
        "start_lon": 68.0,
        "direction": "NNW",
        "max_wind_kt": 105,
        "min_pressure": 956,
        "peak_category": "Extremely Severe Cyclonic Storm",
        "num_points": 48,
    },
    {
        "id": "DEMO_NI_2020_002",
        "name": "DEMO_AMPHAN",
        "basin": "NI",
        "sub_basin": "Bay of Bengal",
        "season": 2020,
        "start_lat": 8.0,
        "start_lon": 86.0,
        "direction": "NNE",
        "max_wind_kt": 140,
        "min_pressure": 920,
        "peak_category": "Super Cyclonic Storm",
        "num_points": 42,
    },
    {
        "id": "DEMO_NI_2019_003",
        "name": "DEMO_FANI",
        "basin": "NI",
        "sub_basin": "Bay of Bengal",
        "season": 2019,
        "start_lat": 7.5,
        "start_lon": 88.5,
        "direction": "NNW",
        "max_wind_kt": 130,
        "min_pressure": 932,
        "peak_category": "Extremely Severe Cyclonic Storm",
        "num_points": 44,
    },
    {
        "id": "DEMO_NI_2021_004",
        "name": "DEMO_TAUKTAE",
        "basin": "NI",
        "sub_basin": "Arabian Sea",
        "season": 2021,
        "start_lat": 11.0,
        "start_lon": 72.0,
        "direction": "NNE",
        "max_wind_kt": 115,
        "min_pressure": 950,
        "peak_category": "Very Severe Cyclonic Storm",
        "num_points": 36,
    },
    {
        "id": "DEMO_NI_2018_005",
        "name": "DEMO_GAJA",
        "basin": "NI",
        "sub_basin": "Bay of Bengal",
        "season": 2018,
        "start_lat": 9.5,
        "start_lon": 85.0,
        "direction": "WNW",
        "max_wind_kt": 65,
        "min_pressure": 976,
        "peak_category": "Severe Cyclonic Storm",
        "num_points": 30,
    },
]


def wind_to_imd_category(wind_kt):
    """Convert wind speed (knots) to IMD category."""
    thresholds = [
        (120, "Super Cyclonic Storm"),
        (90, "Extremely Severe Cyclonic Storm"),
        (64, "Very Severe Cyclonic Storm"),
        (48, "Severe Cyclonic Storm"),
        (34, "Cyclonic Storm"),
        (28, "Deep Depression"),
        (22, "Depression"),
        (17, "Well Marked Low"),
        (0, "Low Pressure Area"),
    ]
    for threshold, category in thresholds:
        if wind_kt >= threshold:
            return category
    return "Low Pressure Area"


def generate_cyclone_track(cyclone_info):
    """Generate a realistic-looking synthetic cyclone track."""
    np.random.seed(hash(cyclone_info["id"]) % (2**31))
    
    points = []
    lat = cyclone_info["start_lat"]
    lon = cyclone_info["start_lon"]
    n = cyclone_info["num_points"]
    
    # Base time
    base_time = datetime(cyclone_info["season"], 6, 1, 0, 0, 0)
    
    # Direction vectors
    dir_map = {
        "NNW": (0.15, -0.08),
        "NNE": (0.18, 0.10),
        "WNW": (0.05, -0.15),
        "NW": (0.12, -0.12),
    }
    base_dlat, base_dlon = dir_map.get(cyclone_info["direction"], (0.15, 0.05))
    
    # Intensity lifecycle: genesis -> intensification -> peak -> weakening
    peak_idx = int(n * 0.6)
    max_wind = cyclone_info["max_wind_kt"]
    min_pres = cyclone_info["min_pressure"]
    
    for i in range(n):
        timestamp = base_time + timedelta(hours=i * 6)
        
        # Track movement with natural variability
        lat += base_dlat + np.random.normal(0, 0.05)
        lon += base_dlon + np.random.normal(0, 0.05)
        
        # Recurvature near end
        if i > n * 0.7:
            lon += 0.1 * (i - n * 0.7) / (n * 0.3)
            lat += 0.05
        
        # Intensity lifecycle
        if i <= peak_idx:
            # Intensification phase
            t = i / peak_idx
            wind_kt = 20 + (max_wind - 20) * (t ** 0.7)
            pressure = 1008 - (1008 - min_pres) * (t ** 0.7)
        else:
            # Weakening phase
            t = (i - peak_idx) / (n - peak_idx)
            wind_kt = max_wind - (max_wind - 25) * (t ** 1.2)
            pressure = min_pres + (1005 - min_pres) * (t ** 1.2)
        
        wind_kt = max(15, wind_kt + np.random.normal(0, 3))
        pressure = max(880, min(1015, pressure + np.random.normal(0, 2)))
        
        category = wind_to_imd_category(wind_kt)
        
        # Distance to coast (approximate)
        dist_coast = max(0, min(1000, 
            200 + 300 * math.sin(lat * 0.1) - 100 * (i / n)
        ))
        
        point = {
            "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "latitude": round(lat, 2),
            "longitude": round(lon, 2),
            "wind_knots": round(wind_kt, 1),
            "wind_kph": round(wind_kt * 1.852, 1),
            "pressure_hpa": round(pressure, 1),
            "category": category,
            "dist_to_coast_km": round(dist_coast, 0),
            "data_type": "SYNTHETIC_DEMO",
            "point_index": i,
        }
        points.append(point)
    
    return points


def generate_satellite_tensor(n_samples, channels=4, size=64):
    """
    Generate synthetic satellite image tensors.
    
    Creates cyclone-like patterns (bright eye, spiral arms)
    These are SYNTHETIC and do not represent real satellite data.
    """
    tensors = []
    
    for i in range(n_samples):
        np.random.seed(i * 42 + 7)
        img = np.zeros((channels, size, size), dtype=np.float32)
        
        cy, cx = size // 2 + np.random.randint(-5, 5), size // 2 + np.random.randint(-5, 5)
        
        for c in range(channels):
            # Background
            img[c] = np.random.normal(0.3, 0.05, (size, size)).astype(np.float32)
            
            # Cyclone structure - spiral pattern
            for y in range(size):
                for x in range(size):
                    dy, dx = y - cy, x - cx
                    r = math.sqrt(dy**2 + dx**2)
                    theta = math.atan2(dy, dx)
                    
                    # Eye
                    if r < 3:
                        img[c, y, x] = 0.1 + np.random.normal(0, 0.02)
                    # Eyewall
                    elif r < 8:
                        img[c, y, x] = 0.9 + np.random.normal(0, 0.05)
                    # Spiral arms
                    elif r < 25:
                        spiral = math.sin(theta * 2 + r * 0.3 + c * 0.5)
                        img[c, y, x] = 0.3 + 0.4 * max(0, spiral) * math.exp(-r / 20)
            
            # Channel-specific adjustments
            if c == 0:  # IR - brighter cloud tops
                img[c] *= 1.1
            elif c == 1:  # WV - moisture
                img[c] *= 0.9
            elif c == 2:  # VIS - if available
                img[c] *= 0.8
            elif c == 3:  # PMW
                img[c] *= 0.7
        
        # Clip and normalize
        img = np.clip(img, 0, 1)
        tensors.append(img)
    
    return np.array(tensors)


def generate_env_features(n_samples, dim=20):
    """
    Generate synthetic environmental feature vectors.
    
    Features represent: SST, wind shear, humidity, temperature,
    geopotential, etc.
    """
    np.random.seed(123)
    features = np.zeros((n_samples, dim), dtype=np.float32)
    
    feature_names = [
        "sst_celsius", "wind_shear_200_850_ms", "rh_700_pct",
        "temp_200_K", "temp_500_K", "temp_850_K",
        "u_wind_200_ms", "v_wind_200_ms", "u_wind_850_ms", "v_wind_850_ms",
        "geopotential_500_m2s2", "mslp_hpa", "vorticity_850",
        "divergence_200", "specific_humidity_700", "theta_e_850",
        "vertical_wind_shear_dir", "ocean_heat_content", 
        "mid_level_moisture", "upper_level_divergence",
    ]
    
    for i in range(n_samples):
        features[i, 0] = 28 + np.random.normal(0, 2)    # SST
        features[i, 1] = 8 + np.random.normal(0, 5)     # Wind shear
        features[i, 2] = 70 + np.random.normal(0, 10)   # RH
        features[i, 3] = 220 + np.random.normal(0, 5)   # T200
        features[i, 4] = 260 + np.random.normal(0, 3)   # T500
        features[i, 5] = 290 + np.random.normal(0, 2)   # T850
        features[i, 6] = np.random.normal(0, 10)        # u200
        features[i, 7] = np.random.normal(0, 10)        # v200
        features[i, 8] = np.random.normal(0, 5)         # u850
        features[i, 9] = np.random.normal(0, 5)         # v850
        features[i, 10] = 5500 + np.random.normal(0, 100)  # Z500
        features[i, 11] = 1000 + np.random.normal(0, 10)   # MSLP
        features[i, 12] = np.random.normal(0, 0.001)    # Vorticity
        features[i, 13] = np.random.normal(0, 0.0001)   # Divergence
        features[i, 14] = np.random.normal(0.01, 0.003) # SpHum
        features[i, 15] = 340 + np.random.normal(0, 5)  # Theta-e
        features[i, 16] = np.random.uniform(0, 360)     # Shear dir
        features[i, 17] = 60 + np.random.normal(0, 20)  # OHC
        features[i, 18] = np.random.uniform(0.3, 0.9)   # Mid moisture
        features[i, 19] = np.random.normal(0, 0.0001)   # Div200
    
    return features, feature_names


def main():
    """Generate all demo data."""
    sample_dir = PROJECT_ROOT / "data" / "sample"
    metadata_dir = PROJECT_ROOT / "data" / "metadata"
    sample_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CYCLONE-AI Demo Data Generator")
    print("=" * 60)
    print()
    print("WARNING: All generated data is SYNTHETIC DEMONSTRATION DATA.")
    print("It does not represent real observations or official data.")
    print()
    
    # 1. Generate cyclone tracks
    print("[1/5] Generating synthetic cyclone tracks...")
    all_cyclones = []
    all_tracks = {}
    
    for info in DEMO_CYCLONES:
        track = generate_cyclone_track(info)
        all_tracks[info["id"]] = track
        
        cyclone_record = {
            "id": info["id"],
            "name": info["name"],
            "basin": info["basin"],
            "sub_basin": info["sub_basin"],
            "season": info["season"],
            "start_time": track[0]["timestamp"],
            "end_time": track[-1]["timestamp"],
            "peak_wind_kt": round(max(p["wind_knots"] for p in track), 1),
            "peak_wind_kph": round(max(p["wind_kph"] for p in track), 1),
            "min_pressure_hpa": round(min(p["pressure_hpa"] for p in track), 1),
            "peak_category": info["peak_category"],
            "num_observations": len(track),
            "data_type": "SYNTHETIC_DEMO",
            "data_source": "Generated by CYCLONE-AI demo data generator",
        }
        all_cyclones.append(cyclone_record)
        print(f"  - {info['name']}: {len(track)} points, peak {info['peak_category']}")
    
    # Save tracks
    with open(sample_dir / "demo_cyclones.json", "w") as f:
        json.dump(all_cyclones, f, indent=2)
    
    with open(sample_dir / "demo_tracks.json", "w") as f:
        json.dump(all_tracks, f, indent=2)
    
    # 2. Generate satellite tensors
    print("\n[2/5] Generating synthetic satellite tensors...")
    n_sat = 50
    sat_tensors = generate_satellite_tensor(n_sat)
    np.save(sample_dir / "demo_satellite_tensors.npy", sat_tensors)
    print(f"  - Shape: {sat_tensors.shape} (samples, channels, H, W)")
    
    # Create metadata for satellite samples
    sat_metadata = []
    cyclone_ids = [c["id"] for c in DEMO_CYCLONES]
    for i in range(n_sat):
        cid = cyclone_ids[i % len(cyclone_ids)]
        track = all_tracks[cid]
        pt_idx = min(i * len(track) // (n_sat // len(cyclone_ids) + 1), len(track) - 1)
        pt = track[pt_idx]
        
        sat_metadata.append({
            "sample_index": i,
            "cyclone_id": cid,
            "timestamp": pt["timestamp"],
            "latitude": pt["latitude"],
            "longitude": pt["longitude"],
            "wind_knots": pt["wind_knots"],
            "pressure_hpa": pt["pressure_hpa"],
            "category": pt["category"],
            "channels": ["IR", "WV", "VIS", "PMW"],
            "image_size": 64,
            "data_type": "SYNTHETIC_DEMO",
        })
    
    with open(sample_dir / "demo_satellite_metadata.json", "w") as f:
        json.dump(sat_metadata, f, indent=2)
    
    # 3. Generate environmental features
    print("\n[3/5] Generating synthetic environmental features...")
    env_features, feature_names = generate_env_features(n_sat)
    np.save(sample_dir / "demo_env_features.npy", env_features)
    
    with open(sample_dir / "demo_env_feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"  - Shape: {env_features.shape} ({len(feature_names)} features)")
    
    # 4. Generate labels
    print("\n[4/5] Generating labels...")
    labels = {
        "detection": np.ones(n_sat, dtype=np.int64),  # all are cyclones in demo
        "classification": np.array([
            sat_metadata[i]["wind_knots"] for i in range(n_sat)
        ]),
        "wind_kph": np.array([m["wind_knots"] * 1.852 for m in sat_metadata]),
        "pressure_hpa": np.array([m["pressure_hpa"] for m in sat_metadata]),
    }
    np.savez(sample_dir / "demo_labels.npz", **labels)
    print(f"  - Detection, classification, wind, pressure labels")
    
    # 5. Dataset metadata
    print("\n[5/5] Generating metadata...")
    dataset_meta = {
        "dataset_name": "CYCLONE-AI Demo Dataset",
        "version": "1.0.0",
        "created": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_type": "SYNTHETIC_DEMO",
        "description": (
            "Synthetic demonstration dataset for CYCLONE-AI. "
            "This data is generated programmatically and does NOT represent "
            "real observations, official data, or actual satellite imagery. "
            "It is intended solely for system demonstration and testing."
        ),
        "disclaimer": (
            "SYNTHETIC DEMONSTRATION DATA - Not for operational use. "
            "Does not represent real meteorological observations."
        ),
        "cyclones": len(all_cyclones),
        "satellite_samples": n_sat,
        "image_channels": 4,
        "image_size": 64,
        "env_features": len(feature_names),
        "feature_names": feature_names,
        "basins": ["NI"],
        "source": "CYCLONE-AI Demo Generator (synthetic)",
        "license": "Part of CYCLONE-AI project - synthetic data",
    }
    
    with open(metadata_dir / "demo_dataset_metadata.json", "w") as f:
        json.dump(dataset_meta, f, indent=2)
    
    # Create a data_source_info
    data_sources = [
        {
            "name": "CYCLONE-AI Demo Dataset",
            "type": "SYNTHETIC_DEMO",
            "description": "Synthetically generated data for demonstration",
            "real_source_inspiration": "North Indian Ocean cyclone patterns",
            "is_real_data": False,
            "license": "Project license",
        },
        {
            "name": "IBTrACS",
            "type": "EXTERNAL_DATASET",
            "description": "International Best Track Archive for Climate Stewardship",
            "provider": "NOAA National Centers for Environmental Information",
            "url": "https://www.ncei.noaa.gov/products/international-best-track-archive",
            "license": "Public domain (US Government work)",
            "status": "Not downloaded - configure in data.yaml",
        },
        {
            "name": "TCIR",
            "type": "EXTERNAL_DATASET",
            "description": "Tropical Cyclone Intensity estimation from IR imagery",
            "status": "Not downloaded - requires separate setup",
        },
        {
            "name": "ERA5",
            "type": "EXTERNAL_DATASET",
            "description": "ECMWF Reanalysis v5",
            "provider": "Copernicus Climate Data Store",
            "url": "https://cds.climate.copernicus.eu/",
            "license": "Copernicus License",
            "status": "Not downloaded - requires CDS API credentials",
        },
    ]
    
    with open(metadata_dir / "data_sources.json", "w") as f:
        json.dump(data_sources, f, indent=2)
    
    print()
    print("=" * 60)
    print("Demo data generation complete!")
    print(f"  Output directory: {sample_dir}")
    print(f"  Cyclones: {len(all_cyclones)}")
    print(f"  Satellite samples: {n_sat}")
    print(f"  Total files: 8")
    print("=" * 60)


if __name__ == "__main__":
    main()
