import { create } from 'zustand';
import type { Project } from '../types';
import { projectsApi } from '../api/projects';

interface ProjectStore {
  projects: Project[];
  currentProject: Project | null;
  loading: boolean;

  fetchProjects: () => Promise<void>;
  setCurrentProject: (project: Project | string | null) => Promise<Project | null>;
  createProject: (data: { name: string; description?: string; task_type: string; classes: string[] }) => Promise<Project>;
  deleteProject: (id: string) => Promise<void>;
}

export const useProjectStore = create<ProjectStore>((set, get) => ({
  projects: [],
  currentProject: null,
  loading: false,

  fetchProjects: async () => {
    set({ loading: true });
    try {
      const res = await projectsApi.list();
      set({ projects: res.data.projects });
    } finally {
      set({ loading: false });
    }
  },

  setCurrentProject: async (projectOrId) => {
    if (typeof projectOrId === 'string') {
      const existing = get().projects.find(p => p.id === projectOrId);
      if (existing) {
        set({ currentProject: existing });
        return existing;
      }
      // If not in local list, fetch it
      try {
        const res = await projectsApi.get(projectOrId);
        set({ currentProject: res.data });
        return res.data;
      } catch {
        return null;
      }
    }
    set({ currentProject: projectOrId });
    return projectOrId;
  },

  createProject: async (data) => {
    const res = await projectsApi.create(data);
    const newProject = res.data;
    set((s) => ({ projects: [...s.projects, newProject] }));
    return newProject;
  },

  deleteProject: async (id) => {
    await projectsApi.delete(id);
    set((s) => ({
      projects: s.projects.filter((p) => p.id !== id),
      currentProject: s.currentProject?.id === id ? null : s.currentProject,
    }));
  },
}));
