import { NavLink } from 'react-router-dom';
import { LayoutDashboard, PenTool, Cpu, Database, Box, Settings } from 'lucide-react';
import clsx from 'clsx';

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/annotate', icon: PenTool, label: 'Annotate' },
  { to: '/training', icon: Cpu, label: 'Training' },
  { to: '/datasets', icon: Database, label: 'Datasets' },
  { to: '/models', icon: Box, label: 'Models' },
];

export default function Sidebar() {
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
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
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
