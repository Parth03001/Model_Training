import { create } from 'zustand';
import type { Annotation, ImageRecord, AnnotationTool, BBox } from '../types';
import { annotationsApi } from '../api/annotations';
import { imagesApi } from '../api/images';

interface AnnotationStore {
  // Image navigation
  images: ImageRecord[];
  currentImageIndex: number;
  currentImage: ImageRecord | null;

  // Annotations for current image
  annotations: Annotation[];
  selectedAnnotationId: string | null;

  // Tool state
  activeTool: AnnotationTool;
  activeClass: string;
  classes: string[];

  // Canvas state
  canvasScale: number;
  canvasOffset: { x: number; y: number };

  // Drawing state
  isDrawing: boolean;
  drawingPoints: number[];

  // Actions
  loadImages: (projectId: string) => Promise<void>;
  setCurrentImageIndex: (index: number) => Promise<void>;
  nextImage: () => void;
  prevImage: () => void;

  loadAnnotations: (imageId: string) => Promise<void>;
  addAnnotation: (data: {
    class_name: string;
    annotation_type: string;
    bbox?: BBox;
    polygon_points?: number[][];
    source?: string;
  }) => Promise<void>;
  updateAnnotation: (id: string, data: Partial<Annotation>) => Promise<void>;
  deleteAnnotation: (id: string) => Promise<void>;
  selectAnnotation: (id: string | null) => void;

  setActiveTool: (tool: AnnotationTool) => void;
  setActiveClass: (cls: string) => void;
  setClasses: (classes: string[]) => void;
  setCanvasScale: (scale: number) => void;
  setCanvasOffset: (offset: { x: number; y: number }) => void;
  setIsDrawing: (drawing: boolean) => void;
  setDrawingPoints: (points: number[]) => void;
}

export const useAnnotationStore = create<AnnotationStore>((set, get) => ({
  images: [],
  currentImageIndex: 0,
  currentImage: null,
  annotations: [],
  selectedAnnotationId: null,
  activeTool: 'select',
  activeClass: '',
  classes: [],
  canvasScale: 1,
  canvasOffset: { x: 0, y: 0 },
  isDrawing: false,
  drawingPoints: [],

  loadImages: async (projectId) => {
    const res = await imagesApi.listByProject(projectId, { limit: 1000 });
    const images = res.data.images;
    set({ images, currentImageIndex: 0, currentImage: images[0] || null });
    if (images[0]) {
      get().loadAnnotations(images[0].id);
    }
  },

  setCurrentImageIndex: async (index) => {
    const { images } = get();
    if (index >= 0 && index < images.length) {
      set({ currentImageIndex: index, currentImage: images[index], selectedAnnotationId: null });
      await get().loadAnnotations(images[index].id);
    }
  },

  nextImage: () => {
    const { currentImageIndex, images } = get();
    if (currentImageIndex < images.length - 1) {
      get().setCurrentImageIndex(currentImageIndex + 1);
    }
  },

  prevImage: () => {
    const { currentImageIndex } = get();
    if (currentImageIndex > 0) {
      get().setCurrentImageIndex(currentImageIndex - 1);
    }
  },

  loadAnnotations: async (imageId) => {
    const res = await annotationsApi.getByImage(imageId);
    set({ annotations: res.data });
  },

  addAnnotation: async (data) => {
    const { currentImage } = get();
    if (!currentImage) return;

    const res = await annotationsApi.create({
      image_id: currentImage.id,
      ...data,
    });
    set((s) => ({ annotations: [...s.annotations, res.data] }));
  },

  updateAnnotation: async (id, data) => {
    await annotationsApi.update(id, data);
    set((s) => ({
      annotations: s.annotations.map((a) => (a.id === id ? { ...a, ...data } : a)),
    }));
  },

  deleteAnnotation: async (id) => {
    await annotationsApi.delete(id);
    set((s) => ({
      annotations: s.annotations.filter((a) => a.id !== id),
      selectedAnnotationId: s.selectedAnnotationId === id ? null : s.selectedAnnotationId,
    }));
  },

  selectAnnotation: (id) => set({ selectedAnnotationId: id }),
  setActiveTool: (tool) => set({ activeTool: tool, isDrawing: false, drawingPoints: [] }),
  setActiveClass: (cls) => set({ activeClass: cls }),
  setClasses: (classes) => set({ classes, activeClass: classes[0] || '' }),
  setCanvasScale: (scale) => set({ canvasScale: Math.max(0.1, Math.min(5, scale)) }),
  setCanvasOffset: (offset) => set({ canvasOffset: offset }),
  setIsDrawing: (drawing) => set({ isDrawing: drawing }),
  setDrawingPoints: (points) => set({ drawingPoints: points }),
}));
