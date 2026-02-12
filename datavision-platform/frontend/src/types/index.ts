// =============================================================================
// DataVision Platform — TypeScript Type Definitions
// =============================================================================

// --- Project ---
export interface Project {
  id: string;
  name: string;
  description: string | null;
  task_type: 'classification' | 'detection' | 'segmentation';
  classes: string[];
  image_count: number;
  annotated_count: number;
  created_at: string;
  updated_at: string;
}

// --- Image ---
export interface ImageRecord {
  id: string;
  project_id: string;
  filename: string;
  filepath: string;
  width: number;
  height: number;
  file_size: number;
  status: 'pending' | 'annotated' | 'reviewed' | 'approved';
  split: 'train' | 'val' | 'test' | null;
  annotation_count: number;
  created_at: string;
}

// --- Annotation ---
export interface BBox {
  x: number; // center x, normalized [0-1]
  y: number; // center y, normalized [0-1]
  w: number; // width, normalized [0-1]
  h: number; // height, normalized [0-1]
}

export interface Annotation {
  id: string;
  image_id: string;
  class_name: string;
  annotation_type: 'bbox' | 'polygon' | 'mask' | 'classification';
  bbox: BBox | null;
  polygon_points: number[][] | null;
  mask_rle: Record<string, unknown> | null;
  confidence: number | null;
  source: 'manual' | 'grounding_dino' | 'sam2' | 'grounded_sam' | 'clip' | 'model';
  is_verified: boolean;
  created_at: string;
}

// --- Training ---
export interface AugmentationConfig {
  hsv_h: number;
  hsv_s: number;
  hsv_v: number;
  degrees: number;
  translate: number;
  scale: number;
  shear: number;
  flipud: number;
  fliplr: number;
  mosaic: number;
  mixup: number;
}

export interface TrainingJob {
  id: string;
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
  status: 'queued' | 'preparing' | 'training' | 'completed' | 'failed' | 'cancelled';
  current_epoch: number;
  metrics: TrainingMetrics | null;
  best_metrics: TrainingMetrics | null;
  model_path: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface TrainingMetrics {
  epoch?: number;
  train_loss?: number;
  val_loss?: number;
  map50?: number;
  map50_95?: number;
  precision?: number;
  recall?: number;
  accuracy?: number;
  learning_rate?: number;
}

// --- Trained Model ---
export interface TrainedModel {
  id: string;
  project_id: string;
  training_job_id: string | null;
  name: string;
  architecture: string;
  task_type: string;
  version: string;
  accuracy: number | null;
  map50: number | null;
  map50_95: number | null;
  precision_val: number | null;
  recall_val: number | null;
  f1_score: number | null;
  inference_time_ms: number | null;
  model_path: string;
  model_size_mb: number | null;
  export_format: string;
  num_classes: number;
  class_names: string[] | null;
  description: string | null;
  created_at: string;
}

// --- Auto-Annotation ---
export interface AutoAnnotateRequest {
  image_ids: string[];
  text_prompt: string;
  box_threshold: number;
  text_threshold: number;
}

export interface TaskResponse {
  task_id: string;
  status: string;
  message: string;
}

// --- Canvas Tool ---
export type AnnotationTool =
  | 'select'
  | 'bbox'
  | 'polygon'
  | 'point'
  | 'brush'
  | 'eraser'
  | 'pan'
  | 'zoom';

export type AutoAnnotateMode =
  | 'grounding_dino'
  | 'sam2_box'
  | 'sam2_point'
  | 'clip_search'
  | 'grounded_sam';
