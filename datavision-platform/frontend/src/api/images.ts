import { api } from './client';
import type { ImageRecord } from '../types';

const CHUNK_SIZE = 50; // files per request
const MAX_RETRIES = 3;

export interface UploadProgress {
  /** Total files to upload */
  total: number;
  /** Files sent so far (includes current chunk) */
  sent: number;
  /** Successfully uploaded */
  uploaded: number;
  /** Failed uploads */
  failed: number;
  /** Current chunk index (1-based) */
  chunk: number;
  /** Total chunks */
  totalChunks: number;
}

type ProgressCallback = (progress: UploadProgress) => void;

async function uploadChunkWithRetry(
  projectId: string,
  files: File[],
  retries = MAX_RETRIES,
): Promise<{ uploaded: number; failed: number; images: ImageRecord[] }> {
  const formData = new FormData();
  formData.append('project_id', projectId);
  files.forEach((file) => formData.append('files', file));

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const res = await api.post<{ uploaded: number; failed: number; images: ImageRecord[] }>(
        '/images/upload',
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 120_000 },
      );
      return res.data;
    } catch (err) {
      if (attempt === retries) throw err;
      // Exponential backoff: 2s, 4s, 8s
      await new Promise((r) => setTimeout(r, 2000 * Math.pow(2, attempt - 1)));
    }
  }
  throw new Error('Upload failed after retries');
}

export const imagesApi = {
  /**
   * Upload files in chunks with progress reporting and retry.
   * Handles 6,000-10,000+ images by splitting into batches of 50.
   */
  uploadBatch: async (
    projectId: string,
    files: File[],
    onProgress?: ProgressCallback,
  ): Promise<{ uploaded: number; failed: number }> => {
    const totalChunks = Math.ceil(files.length / CHUNK_SIZE);
    let uploaded = 0;
    let failed = 0;

    for (let i = 0; i < totalChunks; i++) {
      const chunk = files.slice(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE);

      onProgress?.({
        total: files.length,
        sent: i * CHUNK_SIZE + chunk.length,
        uploaded,
        failed,
        chunk: i + 1,
        totalChunks,
      });

      try {
        const result = await uploadChunkWithRetry(projectId, chunk);
        uploaded += result.uploaded;
        failed += result.failed;
      } catch {
        // Entire chunk failed after retries — count all as failed
        failed += chunk.length;
      }
    }

    // Final progress
    onProgress?.({
      total: files.length,
      sent: files.length,
      uploaded,
      failed,
      chunk: totalChunks,
      totalChunks,
    });

    return { uploaded, failed };
  },

  /** Single-request upload (kept for small batches / backward compat) */
  upload: (projectId: string, files: File[]) => {
    const formData = new FormData();
    formData.append('project_id', projectId);
    files.forEach((file) => formData.append('files', file));
    return api.post<{ uploaded: number; failed: number; images: ImageRecord[] }>(
      '/images/upload',
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } },
    );
  },

  listByProject: (projectId: string, params?: { skip?: number; limit?: number; status?: string }) =>
    api.get<{ images: ImageRecord[]; total: number }>(`/images/project/${projectId}`, { params }),
  get: (id: string) => api.get<ImageRecord>(`/images/${id}`),
  delete: (id: string) => api.delete(`/images/${id}`),
  setSplit: (id: string, split: string) => api.patch(`/images/${id}/split`, null, { params: { split } }),
};
