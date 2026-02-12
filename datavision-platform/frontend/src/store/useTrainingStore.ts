import { create } from 'zustand';
import type { TrainingJob, TrainingMetrics, TrainedModel } from '../types';
import { trainingApi, connectTrainingWS } from '../api/training';

interface TrainingStore {
  jobs: TrainingJob[];
  currentJob: TrainingJob | null;
  liveMetrics: TrainingMetrics[];
  models: TrainedModel[];
  loading: boolean;
  ws: WebSocket | null;

  fetchJobs: (projectId?: string) => Promise<void>;
  createJob: (data: Parameters<typeof trainingApi.createJob>[0]) => Promise<TrainingJob>;
  cancelJob: (id: string) => Promise<void>;
  setCurrentJob: (job: TrainingJob | null) => void;

  connectLiveMetrics: (jobId: string) => void;
  disconnectLiveMetrics: () => void;

  fetchModels: (projectId?: string) => Promise<void>;
  deleteModel: (id: string) => Promise<void>;
  exportModel: (modelId: string, formats: string[]) => Promise<void>;
}

export const useTrainingStore = create<TrainingStore>((set, get) => ({
  jobs: [],
  currentJob: null,
  liveMetrics: [],
  models: [],
  loading: false,
  ws: null,

  fetchJobs: async (projectId) => {
    set({ loading: true });
    try {
      const res = await trainingApi.listJobs(projectId ? { project_id: projectId } : undefined);
      set({ jobs: res.data });
    } finally {
      set({ loading: false });
    }
  },

  createJob: async (data) => {
    const res = await trainingApi.createJob(data);
    const job = res.data;
    set((s) => ({ jobs: [job, ...s.jobs] }));
    return job;
  },

  cancelJob: async (id) => {
    await trainingApi.cancelJob(id);
    set((s) => ({
      jobs: s.jobs.map((j) => (j.id === id ? { ...j, status: 'cancelled' as const } : j)),
    }));
  },

  setCurrentJob: (job) => set({ currentJob: job, liveMetrics: [] }),

  connectLiveMetrics: (jobId) => {
    get().disconnectLiveMetrics();
    const ws = connectTrainingWS(
      jobId,
      (metrics) => {
        set((s) => ({ liveMetrics: [...s.liveMetrics, metrics as TrainingMetrics] }));
      },
      () => set({ ws: null })
    );
    set({ ws });
  },

  disconnectLiveMetrics: () => {
    const { ws } = get();
    if (ws) {
      ws.close();
      set({ ws: null });
    }
  },

  fetchModels: async (projectId) => {
    const res = await trainingApi.listModels(projectId ? { project_id: projectId } : undefined);
    set({ models: res.data.models });
  },

  deleteModel: async (id) => {
    await trainingApi.deleteModel(id);
    set((s) => ({ models: s.models.filter((m) => m.id !== id) }));
  },

  exportModel: async (modelId, formats) => {
    await trainingApi.exportModel({ model_id: modelId, formats, img_size: 640, quantize: false });
  },
}));
