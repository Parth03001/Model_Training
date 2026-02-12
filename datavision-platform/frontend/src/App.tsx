import { Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Layout from './components/Layout/Layout';
import DashboardPage from './pages/DashboardPage';
import AnnotationPage from './pages/AnnotationPage';
import TrainingPage from './pages/TrainingPage';
import DatasetPage from './pages/DatasetPage';
import ModelsPage from './pages/ModelsPage';

export default function App() {
  return (
    <>
      <Toaster position="top-right" toastOptions={{ style: { background: '#1e293b', color: '#f1f5f9' } }} />
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/annotate/:projectId?" element={<AnnotationPage />} />
          <Route path="/training/:projectId?" element={<TrainingPage />} />
          <Route path="/datasets/:projectId?" element={<DatasetPage />} />
          <Route path="/models/:projectId?" element={<ModelsPage />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  );
}
