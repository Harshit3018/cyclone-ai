"""
Database models for CYCLONE-AI.
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

from ..database.session import Base


class Cyclone(Base):
    __tablename__ = "cyclones"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    basin = Column(String, default="NI")
    sub_basin = Column(String, nullable=True)
    season = Column(Integer)
    start_time = Column(String)
    end_time = Column(String)
    peak_wind_kt = Column(Float)
    peak_wind_kph = Column(Float)
    min_pressure_hpa = Column(Float)
    peak_category = Column(String)
    num_observations = Column(Integer, default=0)
    data_type = Column(String, default="SYNTHETIC_DEMO")
    data_source = Column(String, default="demo")
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    track_points = relationship("TrackPoint", back_populates="cyclone", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="cyclone", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="cyclone", cascade="all, delete-orphan")


class TrackPoint(Base):
    __tablename__ = "track_points"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cyclone_id = Column(String, ForeignKey("cyclones.id"), nullable=False)
    timestamp = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    wind_knots = Column(Float)
    wind_kph = Column(Float)
    pressure_hpa = Column(Float)
    category = Column(String)
    dist_to_coast_km = Column(Float)
    data_type = Column(String, default="SYNTHETIC_DEMO")
    point_index = Column(Integer)

    cyclone = relationship("Cyclone", back_populates="track_points")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cyclone_id = Column(String, ForeignKey("cyclones.id"), nullable=False)
    prediction_type = Column(String)  # detection, classification, intensity, track
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_name = Column(String)
    model_version = Column(String)
    model_status = Column(String, default="untrained")
    result = Column(JSON)
    inference_time_ms = Column(Float)

    cyclone = relationship("Cyclone", back_populates="predictions")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cyclone_id = Column(String, ForeignKey("cyclones.id"), nullable=False)
    alert_level = Column(String)
    alert_type = Column(String)
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    data_type = Column(String, default="SYNTHETIC_DEMO")

    cyclone = relationship("Cyclone", back_populates="alerts")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    source = Column(String)
    source_url = Column(String)
    version = Column(String)
    data_type = Column(String)
    license = Column(String)
    spatial_coverage = Column(String)
    temporal_coverage = Column(String)
    variables = Column(JSON)
    status = Column(String, default="not_downloaded")
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    version = Column(String)
    task = Column(String)
    architecture = Column(String)
    dataset_id = Column(String)
    training_date = Column(String)
    metrics = Column(JSON)
    checkpoint_path = Column(String)
    status = Column(String, default="untrained")
    created_at = Column(DateTime, default=datetime.utcnow)
