import { useProjectStore } from '../../store/useProjectStore';

export default function Header() {
  const { currentProject } = useProjectStore();

  return (
    <header className="flex h-14 items-center justify-between border-b border-surface-800 bg-surface-900 px-6">
      <div className="flex items-center gap-4">
        {currentProject ? (
          <>
            <h1 className="text-lg font-semibold">{currentProject.name}</h1>
            <span className="rounded-full bg-primary-600/20 px-2.5 py-0.5 text-xs text-primary-400">
              {currentProject.task_type}
            </span>
            <span className="text-sm text-surface-400">
              {currentProject.image_count} images / {currentProject.annotated_count} annotated
            </span>
          </>
        ) : (
          <h1 className="text-lg font-semibold">DataVision Platform</h1>
        )}
      </div>

      <div className="flex items-center gap-3">
        <span className="text-xs text-surface-500">v0.1.0</span>
      </div>
    </header>
  );
}
