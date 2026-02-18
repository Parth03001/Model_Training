import { api } from './client';
import type { Annotation, BBox, TaskResponse } from '../types';

export const annotationsApi = {
  getByImage: (imageId: string) => api.get<Annotation[]>(`/annotations/image/${imageId}`),

  create: (data: {
    image_id: string;
    class_name: string;
    annotation_type: string;
    bbox?: BBox;
    polygon_points?: number[][];
    source?: string;
  }) => api.post<Annotation>('/annotations', data),

  update: (id: string, data: { class_name?: string; bbox?: BBox; is_verified?: boolean }) =>
    api.patch<Annotation>(`/annotations/${id}`, data),

  delete: (id: string) => api.delete(`/annotations/${id}`),

  batchDelete: (imageId: string) => api.post('/annotations/batch-delete', null, { params: { image_id: imageId } }),

  batchVerify: (ids: string[]) => api.post('/annotations/batch-verify', ids),

  export: (projectId: string, format: string) =>
    api.get(`/annotations/export/${projectId}`, { params: { format } }),

  // --- Auto-Annotation ---
  autoAnnotateGroundingDino: (data: {
    image_ids: string[];
    text_prompt: string;
    box_threshold: number;
    text_threshold: number;
  }) => api.post<TaskResponse>('/auto-annotate/grounding-dino', data),

  autoAnnotateSAM2: (data: {
    image_id: string;
    box_prompts?: BBox[];
    point_prompts?: { x: number; y: number; label: number }[];
  }) => api.post<TaskResponse>('/auto-annotate/sam2', data),

  clipSearch: (data: {
    image_id: string;
    crop_bbox: BBox;
    class_name?: string;
    top_k: number;
    threshold: number;
  }) => api.post<TaskResponse>('/auto-annotate/clip-search', data),

  groundedSAM: (data: {
    image_ids: string[];
    text_prompt: string;
    box_threshold: number;
    text_threshold: number;
    generate_masks: boolean;
  }) => api.post<TaskResponse>('/auto-annotate/grounded-sam', data),

  buildClipIndex: (projectId: string) =>
    api.post<TaskResponse>(`/auto-annotate/build-index/${projectId}`),

  autoAnnotateTrainedModel: (data: {
    project_id: string;
    confidence_threshold?: number;
    image_ids?: string[];
  }) => api.post<TaskResponse>('/auto-annotate/trained-model', data),

  getTaskStatus: (taskId: string) => api.get(`/auto-annotate/task/${taskId}`),
};
