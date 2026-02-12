/**
 * ClassPanel — Shows class list, allows selection, and displays annotation count per class.
 */

import { useState } from 'react';
import { Plus, X, Check, Eye, EyeOff } from 'lucide-react';
import clsx from 'clsx';
import { useAnnotationStore } from '../../store/useAnnotationStore';
import { getClassColor } from '../Canvas/colors';

export default function ClassPanel() {
  const { classes, activeClass, setActiveClass, setClasses, annotations } = useAnnotationStore();
  const [newClassName, setNewClassName] = useState('');
  const [hiddenClasses, setHiddenClasses] = useState<Set<string>>(new Set());

  const addClass = () => {
    const name = newClassName.trim();
    if (name && !classes.includes(name)) {
      setClasses([...classes, name]);
      setNewClassName('');
    }
  };

  const removeClass = (cls: string) => {
    setClasses(classes.filter((c) => c !== cls));
    if (activeClass === cls) {
      setActiveClass(classes[0] || '');
    }
  };

  const toggleVisibility = (cls: string) => {
    const next = new Set(hiddenClasses);
    if (next.has(cls)) next.delete(cls);
    else next.add(cls);
    setHiddenClasses(next);
  };

  const getCount = (cls: string) => annotations.filter((a) => a.class_name === cls).length;

  return (
    <div className="flex flex-col border-b border-surface-800">
      <div className="flex items-center justify-between px-3 py-2">
        <h3 className="text-xs font-semibold uppercase text-surface-400">Classes</h3>
        <span className="text-xs text-surface-500">{classes.length}</span>
      </div>

      <div className="flex flex-col gap-0.5 px-2 pb-2 max-h-48 overflow-y-auto">
        {classes.map((cls) => (
          <div
            key={cls}
            className={clsx(
              'group flex items-center gap-2 rounded px-2 py-1.5 cursor-pointer transition-colors',
              activeClass === cls ? 'bg-surface-700' : 'hover:bg-surface-800'
            )}
            onClick={() => setActiveClass(cls)}
          >
            <div className="h-3 w-3 rounded-sm" style={{ backgroundColor: getClassColor(cls) }} />
            <span className="flex-1 text-sm truncate">{cls}</span>
            <span className="text-xs text-surface-500">{getCount(cls)}</span>
            <button onClick={(e) => { e.stopPropagation(); toggleVisibility(cls); }} className="opacity-0 group-hover:opacity-100">
              {hiddenClasses.has(cls) ? <EyeOff size={12} /> : <Eye size={12} />}
            </button>
            <button onClick={(e) => { e.stopPropagation(); removeClass(cls); }} className="opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-300">
              <X size={12} />
            </button>
          </div>
        ))}
      </div>

      {/* Add class */}
      <div className="flex items-center gap-1 px-2 pb-2">
        <input
          value={newClassName}
          onChange={(e) => setNewClassName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && addClass()}
          placeholder="Add class..."
          className="flex-1 rounded bg-surface-800 px-2 py-1 text-sm text-white placeholder-surface-500 outline-none focus:ring-1 focus:ring-primary-500"
        />
        <button onClick={addClass} className="rounded bg-primary-600 p-1 hover:bg-primary-500">
          <Plus size={14} />
        </button>
      </div>
    </div>
  );
}
