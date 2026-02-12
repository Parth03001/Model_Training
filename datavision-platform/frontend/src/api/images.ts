import { api } from './client';
import type { ImageRecord } from '../types';

export const imagesApi = {
  upload: (projectId: string, files: File[]) => {
    const formData = new FormData();
    formData.append('project_id', projectId);
    files.forEach((file) => formData.append('files', file));
    return api.post<{ uploaded: number; failed: number; images: ImageRecord[] }>(
      '/images/upload',
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } }
    );
  },
  listByProject: (projectId: string, params?: { skip?: number; limit?: number; status?: string }) =>
    api.get<{ images: ImageRecord[]; total: number }>(`/images/project/${projectId}`, { params }),
  get: (id: string) => api.get<ImageRecord>(`/images/${id}`),
  delete: (id: string) => api.delete(`/images/${id}`),
  setSplit: (id: string, split: string) => api.patch(`/images/${id}/split`, null, { params: { split } }),
};
