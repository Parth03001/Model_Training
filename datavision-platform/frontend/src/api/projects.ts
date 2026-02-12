import { api } from './client';
import type { Project } from '../types';

export const projectsApi = {
  list: () => api.get<{ projects: Project[]; total: number }>('/projects'),
  get: (id: string) => api.get<Project>(`/projects/${id}`),
  create: (data: { name: string; description?: string; task_type: string; classes: string[] }) =>
    api.post<Project>('/projects', data),
  update: (id: string, data: Partial<Project>) => api.patch<Project>(`/projects/${id}`, data),
  delete: (id: string) => api.delete(`/projects/${id}`),
};
