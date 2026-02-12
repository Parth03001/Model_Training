/**
 * AnnotationList — Shows all annotations for the current image with edit/delete controls.
 */

import { Trash2, Check, Eye } from 'lucide-react';
import clsx from 'clsx';
import { useAnnotationStore } from '../../store/useAnnotationStore';
import { getClassColor } from '../Canvas/colors';

export default function AnnotationList() {
  const { annotations, selectedAnnotationId, selectAnnotation, deleteAnnotation, updateAnnotation } = useAnnotationStore();

  if (annotations.length === 0) {
    return (
      <div className="px-3 py-4 text-center text-sm text-surface-500">
        No annotations yet. Use tools to annotate.
      </div>
    );
  }

  return (
    <div className="flex flex-col">
      <div className="flex items-center justify-between px-3 py-2">
        <h3 className="text-xs font-semibold uppercase text-surface-400">Annotations</h3>
        <span className="text-xs text-surface-500">{annotations.length}</span>
      </div>
      <div className="flex flex-col gap-0.5 px-2 pb-2 max-h-60 overflow-y-auto">
        {annotations.map((ann) => (
          <div
            key={ann.id}
            className={clsx(
              'group flex items-center gap-2 rounded px-2 py-1.5 cursor-pointer transition-colors',
              selectedAnnotationId === ann.id ? 'bg-surface-700' : 'hover:bg-surface-800'
            )}
            onClick={() => selectAnnotation(ann.id)}
          >
            <div className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: getClassColor(ann.class_name) }} />
            <span className="flex-1 text-sm truncate">{ann.class_name}</span>
            <span className="text-xs text-surface-600">{ann.annotation_type}</span>
            {ann.confidence && (
              <span className="text-xs text-surface-500">{(ann.confidence * 100).toFixed(0)}%</span>
            )}
            {ann.source !== 'manual' && (
              <span className="text-xs text-amber-500/60">{ann.source.replace('_', ' ')}</span>
            )}
            {!ann.is_verified && (
              <button
                onClick={(e) => { e.stopPropagation(); updateAnnotation(ann.id, { is_verified: true }); }}
                className="opacity-0 group-hover:opacity-100 text-green-400 hover:text-green-300"
                title="Verify"
              >
                <Check size={12} />
              </button>
            )}
            <button
              onClick={(e) => { e.stopPropagation(); deleteAnnotation(ann.id); }}
              className="opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-300"
              title="Delete"
            >
              <Trash2 size={12} />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
