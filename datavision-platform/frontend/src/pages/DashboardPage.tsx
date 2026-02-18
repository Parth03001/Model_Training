/**
 * DashboardPage — Project list and creation.
 */

import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, FolderOpen, Trash2, Image, Tag, Cpu } from 'lucide-react';
import { useProjectStore } from '../store/useProjectStore';
import toast from 'react-hot-toast';

export default function DashboardPage() {
  const navigate = useNavigate();
  const { projects, loading, fetchProjects, createProject, deleteProject, setCurrentProject } = useProjectStore();
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [newType, setNewType] = useState('detection');
  const [newClasses, setNewClasses] = useState('');

  useEffect(() => {
    fetchProjects();
  }, [fetchProjects]);

  const handleCreate = async () => {
    if (!newName.trim()) return;
    try {
      const project = await createProject({
        name: newName,
        description: newDesc || undefined,
        task_type: newType,
        classes: newClasses.split(',').map((c) => c.trim()).filter(Boolean),
      });
      setShowCreate(false);
      setNewName('');
      setNewDesc('');
      setNewClasses('');
      toast.success(`Project "${project.name}" created`);
    } catch (e: any) {
      toast.error(e.response?.data?.detail || 'Failed to create project');
    }
  };

  const openProject = async (project: typeof projects[0]) => {
    await setCurrentProject(project);
    navigate(`/annotate/${project.id}`);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Projects</h1>
        <button
          onClick={() => setShowCreate(true)}
          className="flex items-center gap-2 rounded-lg bg-primary-600 px-4 py-2 text-sm font-medium hover:bg-primary-500 transition-colors"
        >
          <Plus size={16} /> New Project
        </button>
      </div>

      {/* Create modal */}
      {showCreate && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="w-[480px] rounded-xl bg-surface-900 p-6 space-y-4">
            <h2 className="text-lg font-semibold">Create Project</h2>
            <input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="Project name"
              className="w-full rounded-lg bg-surface-800 px-4 py-2.5 text-sm outline-none focus:ring-1 focus:ring-primary-500"
            />
            <input
              value={newDesc}
              onChange={(e) => setNewDesc(e.target.value)}
              placeholder="Description (optional)"
              className="w-full rounded-lg bg-surface-800 px-4 py-2.5 text-sm outline-none focus:ring-1 focus:ring-primary-500"
            />
            <select
              value={newType}
              onChange={(e) => setNewType(e.target.value)}
              className="w-full rounded-lg bg-surface-800 px-4 py-2.5 text-sm outline-none focus:ring-1 focus:ring-primary-500"
            >
              <option value="detection">Object Detection</option>
              <option value="classification">Image Classification</option>
              <option value="segmentation">Instance Segmentation</option>
            </select>
            <input
              value={newClasses}
              onChange={(e) => setNewClasses(e.target.value)}
              placeholder="Classes (comma-separated): cat, dog, bird"
              className="w-full rounded-lg bg-surface-800 px-4 py-2.5 text-sm outline-none focus:ring-1 focus:ring-primary-500"
            />
            <div className="flex justify-end gap-3">
              <button onClick={() => setShowCreate(false)} className="px-4 py-2 text-sm text-surface-400 hover:text-white">
                Cancel
              </button>
              <button onClick={handleCreate} className="rounded-lg bg-primary-600 px-4 py-2 text-sm font-medium hover:bg-primary-500">
                Create
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Project grid */}
      {loading ? (
        <div className="text-surface-400">Loading projects...</div>
      ) : projects.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-surface-500">
          <FolderOpen size={48} className="mb-4" />
          <p>No projects yet. Create one to get started.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {projects.map((project) => (
            <div
              key={project.id}
              onClick={() => openProject(project)}
              className="group cursor-pointer rounded-xl border border-surface-800 bg-surface-900 p-5 hover:border-primary-600/50 transition-all"
            >
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="font-semibold text-white group-hover:text-primary-400 transition-colors">
                    {project.name}
                  </h3>
                  {project.description && (
                    <p className="mt-1 text-sm text-surface-400 line-clamp-2">{project.description}</p>
                  )}
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    if (confirm('Delete this project?')) deleteProject(project.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-300 transition-opacity"
                >
                  <Trash2 size={16} />
                </button>
              </div>

              <div className="mt-4 flex items-center gap-4 text-xs text-surface-400">
                <span className="rounded-full bg-primary-600/20 px-2 py-0.5 text-primary-400">
                  {project.task_type}
                </span>
                <span className="flex items-center gap-1"><Image size={12} /> {project.image_count}</span>
                <span className="flex items-center gap-1"><Tag size={12} /> {project.classes.length} classes</span>
              </div>

              {project.classes.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-1">
                  {project.classes.slice(0, 5).map((cls) => (
                    <span key={cls} className="rounded bg-surface-800 px-2 py-0.5 text-xs text-surface-300">
                      {cls}
                    </span>
                  ))}
                  {project.classes.length > 5 && (
                    <span className="text-xs text-surface-500">+{project.classes.length - 5} more</span>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
