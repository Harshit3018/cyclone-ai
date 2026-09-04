import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
});

// System
export const getHealth = () => api.get('/health');
export const getSystemStatus = () => api.get('/system/status');

// Cyclones
export const getCyclones = (params?: Record<string, any>) => api.get('/cyclones', { params });
export const getCyclone = (id: string) => api.get(`/cyclones/${id}`);
export const getCycloneTrack = (id: string) => api.get(`/cyclones/${id}/track`);
export const getCycloneForecast = (id: string) => api.get(`/cyclones/${id}/forecast`);
export const getCycloneRisk = (id: string) => api.get(`/cyclones/${id}/risk`);

// Predictions
export const predictDetect = (sampleIndex?: number) => api.post('/predict/detect', null, { params: { sample_index: sampleIndex || 0 } });
export const predictClassification = (sampleIndex?: number) => api.post('/predict/classification', null, { params: { sample_index: sampleIndex || 0 } });
export const predictIntensity = (sampleIndex?: number) => api.post('/predict/intensity', null, { params: { sample_index: sampleIndex || 0 } });
export const predictTrack = (sampleIndex?: number, cycloneId?: string) => api.post('/predict/track', null, { params: { sample_index: sampleIndex || 0, cyclone_id: cycloneId } });
export const predictFull = (sampleIndex?: number) => api.post('/predict', null, { params: { sample_index: sampleIndex || 0 } });

// Datasets
export const getDatasets = () => api.get('/datasets');
export const getDataset = (id: string) => api.get(`/datasets/${id}`);

// Models
export const getModels = () => api.get('/models');
export const getModel = (id: string) => api.get(`/models/${id}`);

// Metrics
export const getMetrics = () => api.get('/metrics');

// Alerts
export const getAlerts = (active?: boolean) => api.get('/alerts', { params: { active } });
export const evaluateAlerts = (cycloneId: string) => api.post('/alerts/evaluate', null, { params: { cyclone_id: cycloneId } });

// Observations
export const getObservations = (cycloneId?: string) => api.get('/observations', { params: { cyclone_id: cycloneId } });

export default api;
