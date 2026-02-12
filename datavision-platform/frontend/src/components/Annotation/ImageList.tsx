/**
 * ImageList — Thumbnail strip for navigating between images.
 */

import clsx from 'clsx';
import { CheckCircle2, Circle, AlertCircle } from 'lucide-react';
import { useAnnotationStore } from '../../store/useAnnotationStore';

export default function ImageList() {
  const { images, currentImageIndex, setCurrentImageIndex } = useAnnotationStore();

  const statusIcon = (status: string) => {
    switch (status) {
      case 'annotated': return <CheckCircle2 size={12} className="text-green-400" />;
      case 'reviewed': return <CheckCircle2 size={12} className="text-blue-400" />;
      case 'approved': return <CheckCircle2 size={12} className="text-amber-400" />;
      default: return <Circle size={12} className="text-surface-500" />;
    }
  };

  return (
    <div className="flex flex-col border-b border-surface-800">
      <div className="flex items-center justify-between px-3 py-2">
        <h3 className="text-xs font-semibold uppercase text-surface-400">Images</h3>
        <span className="text-xs text-surface-500">{images.length}</span>
      </div>
      <div className="flex flex-col gap-0.5 px-2 pb-2 max-h-[300px] overflow-y-auto">
        {images.map((img, idx) => (
          <button
            key={img.id}
            onClick={() => setCurrentImageIndex(idx)}
            className={clsx(
              'flex items-center gap-2 rounded px-2 py-1.5 text-left transition-colors text-sm',
              idx === currentImageIndex ? 'bg-primary-600/20 text-primary-300' : 'text-surface-300 hover:bg-surface-800'
            )}
          >
            {statusIcon(img.status)}
            <span className="flex-1 truncate">{img.filename}</span>
            {img.annotation_count > 0 && (
              <span className="text-xs text-surface-500">{img.annotation_count}</span>
            )}
          </button>
        ))}

        {images.length === 0 && (
          <div className="flex items-center gap-2 px-2 py-4 text-surface-500 text-sm">
            <AlertCircle size={16} />
            No images uploaded
          </div>
        )}
      </div>
    </div>
  );
}
