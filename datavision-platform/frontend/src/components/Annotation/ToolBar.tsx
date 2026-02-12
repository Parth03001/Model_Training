/**
 * ToolBar — Annotation tool selection and controls.
 */

import {
  MousePointer2, Square, Pentagon, Circle, Paintbrush, Eraser, Move, ZoomIn,
  Wand2, Sparkles, Search, RotateCcw, ChevronLeft, ChevronRight,
} from 'lucide-react';
import clsx from 'clsx';
import { useAnnotationStore } from '../../store/useAnnotationStore';
import type { AnnotationTool } from '../../types';

const tools: { id: AnnotationTool; icon: typeof Square; label: string; shortcut: string }[] = [
  { id: 'select', icon: MousePointer2, label: 'Select', shortcut: 'V' },
  { id: 'bbox', icon: Square, label: 'Bounding Box', shortcut: 'B' },
  { id: 'polygon', icon: Pentagon, label: 'Polygon', shortcut: 'P' },
  { id: 'point', icon: Circle, label: 'Point Prompt', shortcut: 'O' },
  { id: 'brush', icon: Paintbrush, label: 'Brush', shortcut: 'W' },
  { id: 'eraser', icon: Eraser, label: 'Eraser', shortcut: 'E' },
  { id: 'pan', icon: Move, label: 'Pan', shortcut: 'H' },
  { id: 'zoom', icon: ZoomIn, label: 'Zoom', shortcut: 'Z' },
];

interface ToolBarProps {
  onAutoAnnotate: () => void;
}

export default function ToolBar({ onAutoAnnotate }: ToolBarProps) {
  const {
    activeTool, setActiveTool,
    currentImageIndex, images,
    prevImage, nextImage,
    canvasScale, setCanvasScale,
  } = useAnnotationStore();

  return (
    <div className="flex items-center gap-1 border-b border-surface-800 bg-surface-900 px-3 py-2">
      {/* Drawing tools */}
      <div className="flex items-center gap-0.5 rounded-lg bg-surface-800 p-0.5">
        {tools.map(({ id, icon: Icon, label, shortcut }) => (
          <button
            key={id}
            onClick={() => setActiveTool(id)}
            className={clsx(
              'flex items-center justify-center rounded-md p-1.5 transition-colors',
              activeTool === id
                ? 'bg-primary-600 text-white'
                : 'text-surface-400 hover:bg-surface-700 hover:text-white'
            )}
            title={`${label} (${shortcut})`}
          >
            <Icon size={18} />
          </button>
        ))}
      </div>

      <div className="mx-2 h-6 w-px bg-surface-700" />

      {/* Auto-annotation tools */}
      <div className="flex items-center gap-0.5 rounded-lg bg-surface-800 p-0.5">
        <button
          onClick={onAutoAnnotate}
          className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm text-amber-400 hover:bg-surface-700 transition-colors"
          title="Auto-Annotate (Grounding DINO + SAM)"
        >
          <Wand2 size={16} />
          <span className="hidden md:inline">Auto-Annotate</span>
        </button>
        <button
          className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm text-violet-400 hover:bg-surface-700 transition-colors"
          title="Smart Select (SAM 2)"
        >
          <Sparkles size={16} />
          <span className="hidden md:inline">Smart Select</span>
        </button>
        <button
          className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm text-cyan-400 hover:bg-surface-700 transition-colors"
          title="Find Similar (CLIP)"
        >
          <Search size={16} />
          <span className="hidden md:inline">Find Similar</span>
        </button>
      </div>

      <div className="flex-1" />

      {/* Zoom controls */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => setCanvasScale(canvasScale / 1.2)}
          className="text-surface-400 hover:text-white"
        >
          -
        </button>
        <span className="text-xs text-surface-400 w-12 text-center">
          {(canvasScale * 100).toFixed(0)}%
        </span>
        <button
          onClick={() => setCanvasScale(canvasScale * 1.2)}
          className="text-surface-400 hover:text-white"
        >
          +
        </button>
        <button
          onClick={() => setCanvasScale(1)}
          className="text-surface-400 hover:text-white ml-1"
          title="Reset Zoom"
        >
          <RotateCcw size={14} />
        </button>
      </div>

      <div className="mx-2 h-6 w-px bg-surface-700" />

      {/* Image navigation */}
      <div className="flex items-center gap-2">
        <button onClick={prevImage} disabled={currentImageIndex === 0} className="text-surface-400 hover:text-white disabled:opacity-30">
          <ChevronLeft size={18} />
        </button>
        <span className="text-xs text-surface-400 min-w-[60px] text-center">
          {images.length > 0 ? `${currentImageIndex + 1} / ${images.length}` : 'No images'}
        </span>
        <button onClick={nextImage} disabled={currentImageIndex >= images.length - 1} className="text-surface-400 hover:text-white disabled:opacity-30">
          <ChevronRight size={18} />
        </button>
      </div>
    </div>
  );
}
