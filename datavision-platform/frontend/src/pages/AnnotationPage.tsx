/**
 * AnnotationPage — Main annotation workspace with canvas, tools, and panels.
 */

import { useEffect, useState, useRef, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import { useProjectStore } from '../store/useProjectStore';
import { useAnnotationStore } from '../store/useAnnotationStore';
import AnnotationCanvas from '../components/Canvas/AnnotationCanvas';
import ToolBar from '../components/Annotation/ToolBar';
import ClassPanel from '../components/Annotation/ClassPanel';
import ImageList from '../components/Annotation/ImageList';
import AnnotationList from '../components/Annotation/AnnotationList';
import AutoAnnotatePanel from '../components/Annotation/AutoAnnotatePanel';

export default function AnnotationPage() {
  const { projectId } = useParams();
  const { currentProject, setCurrentProject, fetchProjects } = useProjectStore();
  const { loadImages, setClasses } = useAnnotationStore();
  const [showAutoAnnotate, setShowAutoAnnotate] = useState(false);
  const [autoAnnotateTab, setAutoAnnotateTab] = useState<'auto_label' | 'smart_select' | 'box_prompting'>('auto_label');

  // Canvas container ref for responsive sizing
  const canvasContainerRef = useRef<HTMLDivElement>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 600 });

  // Load project and images
  useEffect(() => {
    if (projectId) {
      console.log('AnnotationPage: Loading project', projectId);
      const load = async () => {
        try {
          // Always refresh images for the project
          await loadImages(projectId);
          
          // Ensure project data is available
          const project = await setCurrentProject(projectId);
          if (project) {
            console.log('AnnotationPage: Project set', project.name);
            setClasses(project.classes);
          } else {
            console.warn('AnnotationPage: Project not found, fetching list...');
            await fetchProjects();
            const p = await setCurrentProject(projectId);
            if (p) setClasses(p.classes);
          }
        } catch (err) {
          console.error('AnnotationPage: Load failed', err);
        }
      };
      load();
    }
  }, [projectId]); // Simplify dependencies to avoid race conditions

  // Responsive canvas sizing
  const updateCanvasSize = useCallback(() => {
    if (canvasContainerRef.current) {
      const rect = canvasContainerRef.current.getBoundingClientRect();
      setCanvasSize({ width: rect.width, height: rect.height });
    }
  }, []);

  useEffect(() => {
    updateCanvasSize();
    window.addEventListener('resize', updateCanvasSize);
    return () => window.removeEventListener('resize', updateCanvasSize);
  }, [updateCanvasSize]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      const store = useAnnotationStore.getState();
      switch (e.key.toLowerCase()) {
        case 'v': store.setActiveTool('select'); break;
        case 'b': store.setActiveTool('bbox'); break;
        case 'p': store.setActiveTool('polygon'); break;
        case 'o': store.setActiveTool('point'); break;
        case 'h': store.setActiveTool('pan'); break;
        case 'delete':
        case 'backspace':
          if (store.selectedAnnotationId) store.deleteAnnotation(store.selectedAnnotationId);
          break;
        case 'arrowright': store.nextImage(); break;
        case 'arrowleft': store.prevImage(); break;
      }
    };

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  if (!projectId) {
    return (
      <div className="flex h-full items-center justify-center text-surface-500">
        Select a project from the Dashboard to start annotating.
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col -m-4">
      {/* Toolbar */}
      <ToolBar
        onAutoAnnotate={() => { setAutoAnnotateTab('auto_label'); setShowAutoAnnotate(!showAutoAnnotate); }}
        onSmartSelect={() => { setAutoAnnotateTab('smart_select'); setShowAutoAnnotate(true); }}
        onFindSimilar={() => { setAutoAnnotateTab('box_prompting'); setShowAutoAnnotate(true); }}
      />

      <div className="flex flex-1 overflow-hidden relative">
        {/* Left sidebar: Classes + Images */}
        <div className="w-56 flex-shrink-0 overflow-y-auto border-r border-surface-800 bg-surface-900">
          <ClassPanel />
          <ImageList />
        </div>

        {/* Canvas */}
        <div ref={canvasContainerRef} className="flex-1 bg-surface-950 overflow-hidden">
          <AnnotationCanvas width={canvasSize.width} height={canvasSize.height} />
        </div>

        {/* Right sidebar: Annotations */}
        <div className="w-56 flex-shrink-0 overflow-y-auto border-l border-surface-800 bg-surface-900">
          <AnnotationList />
        </div>

        {/* Auto-annotate panel (overlay) */}
        <AutoAnnotatePanel visible={showAutoAnnotate} onClose={() => setShowAutoAnnotate(false)} initialTab={autoAnnotateTab} />
      </div>
    </div>
  );
}
