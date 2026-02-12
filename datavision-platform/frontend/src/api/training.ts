import { api } from './client';
import type { TrainingJob, TrainedModel } from '../types';

export const trainingApi = {
  createJob: (data: {
    project_id: string;
    name: string;
    model_architecture: string;
    task_type: string;
    epochs: number;
    batch_size: number;
    img_size: number;
    learning_rate: number;
    patience: number;
    optimizer: string;
  }) => api.post<TrainingJob>('/training/jobs', data),

  listJobs: (params?: { project_id?: string; status?: string }) =>
    api.get<TrainingJob[]>('/training/jobs', { params }),

  getJob: (id: string) => api.get<TrainingJob>(`/training/jobs/${id}`),

  cancelJob: (id: string) => api.post(`/training/jobs/${id}/cancel`),

  exportModel: (data: { model_id: string; formats: string[]; img_size: number; quantize: boolean }) =>
    api.post('/training/export', data),

  // Models
  listModels: (params?: { project_id?: string; task_type?: string }) =>
    api.get<{ models: TrainedModel[]; total: number }>('/models', { params }),

  getModel: (id: string) => api.get<TrainedModel>(`/models/${id}`),

  deleteModel: (id: string) => api.delete(`/models/${id}`),
};

// WebSocket connection for real-time training metrics
export function connectTrainingWS(
  jobId: string,
  onMessage: (metrics: Record<string, number>) => void,
  onClose?: () => void
): WebSocket {
  const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/v1/training/ws/training/${jobId}`;
  const ws = new WebSocket(wsUrl);

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (e) {
      console.error('WS parse error:', e);
    }
  };

  ws.onclose = () => onClose?.();

  return ws;
}
