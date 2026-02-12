import { NavLink } from 'react-router-dom';
import { LayoutDashboard, PenTool, Cpu, Database, Box, Settings } from 'lucide-react';
import clsx from 'clsx';
import { useProjectStore } from '../../store/useProjectStore';

const navItemDefs = [
  { base: '/', icon: LayoutDashboard, label: 'Dashboard', needsProject: false },
  { base: '/annotate', icon: PenTool, label: 'Annotate', needsProject: true },
  { base: '/training', icon: Cpu, label: 'Training', needsProject: true },
  { base: '/datasets', icon: Database, label: 'Datasets', needsProject: true },
  { base: '/models', icon: Box, label: 'Models', needsProject: true },
];

export default function Sidebar() {
  const { currentProject } = useProjectStore();
  const projectId = currentProject?.id;

  const navItems = navItemDefs.map((item) => ({
    ...item,
    to: item.needsProject && projectId ? `${item.base}/${projectId}` : item.base,
  }));

  return (
    <aside className="flex w-16 flex-col items-center border-r border-surface-800 bg-surface-900 py-4 lg:w-56">
      {/* Logo */}
      <div className="mb-8 flex items-center gap-2 px-4">
        <div className="h-8 w-8 rounded-lg bg-primary-600 flex items-center justify-center text-sm font-bold">
          DV
        </div>
        <span className="hidden text-lg font-semibold lg:block">DataVision</span>
      </div>

      {/* Navigation */}
      <nav className="flex flex-1 flex-col gap-1 px-2 w-full">
        {navItems.map(({ to, base, icon: Icon, label }) => (
          <NavLink
            key={base}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors',
                isActive
                  ? 'bg-primary-600/20 text-primary-400'
                  : 'text-surface-300 hover:bg-surface-800 hover:text-white'
              )
            }
          >
            <Icon size={20} />
            <span className="hidden lg:block">{label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Settings */}
      <div className="px-2 w-full">
        <button className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm text-surface-300 hover:bg-surface-800 hover:text-white transition-colors">
          <Settings size={20} />
          <span className="hidden lg:block">Settings</span>
        </button>
      </div>
    </aside>
  );
}
