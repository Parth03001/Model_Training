/**
 * TrainingPage — Model training configuration, job management, and live metrics.
 */

import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { Play, Square, Loader2, CheckCircle2, XCircle, Clock } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useTrainingStore } from '../store/useTrainingStore';
import { useProjectStore } from '../store/useProjectStore';
import toast from 'react-hot-toast';

const MODEL_OPTIONS = {
  detection: [
    { value: 'yolov8n', label: 'YOLOv8 Nano (3.2M)' },
    { value: 'yolov8s', label: 'YOLOv8 Small (11.2M)' },
    { value: 'yolov8m', label: 'YOLOv8 Medium (25.9M)' },
    { value: 'yolov11m', label: 'YOLOv11 Medium (20M)' },
    { value: 'yolov11l', label: 'YOLOv11 Large (43M)' },
    { value: 'rt_detr_l', label: 'RT-DETR Large (32M)' },
  ],
  classification: [
    { value: 'vit_b_16', label: 'ViT-B/16 (86M)' },
    { value: 'vit_b_32', label: 'ViT-B/32 (86M)' },
    { value: 'vit_l_16', label: 'ViT-L/16 (304M)' },
    { value: 'efficientnet_v2_s', label: 'EfficientNetV2-S (21.5M)' },
    { value: 'efficientnet_v2_m', label: 'EfficientNetV2-M (54.1M)' },
  ],
  segmentation: [
    { value: 'yolov8m', label: 'YOLOv8m-seg (27.3M)' },
    { value: 'yolov11m', label: 'YOLOv11m-seg (22M)' },
  ],
};

export default function TrainingPage() {
  const { projectId } = useParams();
  const { currentProject } = useProjectStore();
  const { jobs, liveMetrics, fetchJobs, createJob, cancelJob, connectLiveMetrics, disconnectLiveMetrics } = useTrainingStore();

  // Form state
  const [name, setName] = useState('');
  const [architecture, setArchitecture] = useState('yolov11m');
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(16);
  const [imgSize, setImgSize] = useState(640);
  const [lr, setLr] = useState(0.01);
  const [patience, setPatience] = useState(20);
  const [optimizer, setOptimizer] = useState('AdamW');

  const taskType = currentProject?.task_type || 'detection';
  const modelOptions = MODEL_OPTIONS[taskType as keyof typeof MODEL_OPTIONS] || MODEL_OPTIONS.detection;

  useEffect(() => {
    if (projectId) fetchJobs(projectId);
    return () => disconnectLiveMetrics();
  }, [projectId]);

  const handleStartTraining = async () => {
    if (!projectId || !name.trim()) {
      toast.error('Enter a job name');
      return;
    }

    try {
      const job = await createJob({
        project_id: projectId,
        name,
        model_architecture: architecture,
        task_type: taskType,
        epochs,
        batch_size: batchSize,
        img_size: imgSize,
        learning_rate: lr,
        patience,
        optimizer,
      });
      toast.success('Training job queued');
      connectLiveMetrics(job.id);
    } catch (e: any) {
      toast.error(e.response?.data?.detail || 'Failed to start training');
    }
  };

  const statusBadge = (status: string) => {
    const map: Record<string, { icon: typeof Clock; color: string }> = {
      queued: { icon: Clock, color: 'text-surface-400' },
      preparing: { icon: Loader2, color: 'text-amber-400' },
      training: { icon: Loader2, color: 'text-primary-400' },
      completed: { icon: CheckCircle2, color: 'text-green-400' },
      failed: { icon: XCircle, color: 'text-red-400' },
      cancelled: { icon: XCircle, color: 'text-surface-500' },
    };
    const { icon: Icon, color } = map[status] || map.queued;
    return (
      <span className={`flex items-center gap-1 text-xs ${color}`}>
        <Icon size={12} className={status === 'training' || status === 'preparing' ? 'animate-spin' : ''} />
        {status}
      </span>
    );
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Training</h1>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Training Configuration */}
        <div className="rounded-xl border border-surface-800 bg-surface-900 p-5 space-y-4">
          <h2 className="font-semibold">New Training Job</h2>

          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Job name"
            className="w-full rounded-lg bg-surface-800 px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-primary-500" />

          <div>
            <label className="block text-xs text-surface-400 mb-1">Model Architecture</label>
            <select value={architecture} onChange={(e) => setArchitecture(e.target.value)}
              className="w-full rounded-lg bg-surface-800 px-3 py-2 text-sm outline-none">
              {modelOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-surface-400 mb-1">Epochs</label>
              <input type="number" value={epochs} onChange={(e) => setEpochs(+e.target.value)}
                className="w-full rounded-lg bg-surface-800 px-3 py-2 text-sm outline-none" />
            </div>
            <div>
              <label className="block text-xs text-surface-400 mb-1">Batch Size</label>
              <input type="number" value={batchSize} onChange={(e) => setBatchSize(+e.target.value)}
                className="w-full rounded-lg bg-surface-800 px-3 py-2 text-sm outline-none" />
            </div>
            <div>
              <label className="block text-xs text-surface-400 mb-1">Image Size</label>
              <input type="number" value={imgSize} onChange={(e) => setImgSize(+e.target.value)}
                className="w-full rounded-lg bg-surface-800 px-3 py-2 text-sm outline-none" />
            </div>
            <div>
              <label className="block text-xs text-surface-400 mb-1">Learning Rate</label>
              <input type="number" value={lr} onChange={(e) => setLr(+e.target.value)} step="0.001"
                className="w-full rounded-lg bg-surface-800 px-3 py-2 text-sm outline-none" />
            </div>
            <div>
              <label className="block text-xs text-surface-400 mb-1">Patience</label>
              <input type="number" value={patience} onChange={(e) => setPatience(+e.target.value)}
                className="w-full rounded-lg bg-surface-800 px-3 py-2 text-sm outline-none" />
            </div>
            <div>
              <label className="block text-xs text-surface-400 mb-1">Optimizer</label>
              <select value={optimizer} onChange={(e) => setOptimizer(e.target.value)}
                className="w-full rounded-lg bg-surface-800 px-3 py-2 text-sm outline-none">
                <option value="AdamW">AdamW</option>
                <option value="SGD">SGD</option>
                <option value="Adam">Adam</option>
              </select>
            </div>
          </div>

          <button onClick={handleStartTraining}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-primary-600 px-4 py-2.5 text-sm font-medium hover:bg-primary-500 transition-colors">
            <Play size={16} /> Start Training
          </button>
        </div>

        {/* Live Metrics Chart */}
        <div className="lg:col-span-2 rounded-xl border border-surface-800 bg-surface-900 p-5">
          <h2 className="font-semibold mb-4">Training Metrics</h2>
          {liveMetrics.length > 0 ? (
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={liveMetrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="epoch" stroke="#94a3b8" fontSize={12} />
                <YAxis stroke="#94a3b8" fontSize={12} />
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
                <Legend />
                <Line type="monotone" dataKey="train_loss" stroke="#ef4444" strokeWidth={2} dot={false} name="Train Loss" />
                <Line type="monotone" dataKey="val_loss" stroke="#3b82f6" strokeWidth={2} dot={false} name="Val Loss" />
                <Line type="monotone" dataKey="map50" stroke="#10b981" strokeWidth={2} dot={false} name="mAP50" />
                <Line type="monotone" dataKey="accuracy" stroke="#f59e0b" strokeWidth={2} dot={false} name="Accuracy" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-[320px] items-center justify-center text-surface-500">
              Start a training job to see live metrics
            </div>
          )}
        </div>
      </div>

      {/* Job History */}
      <div className="rounded-xl border border-surface-800 bg-surface-900 p-5">
        <h2 className="font-semibold mb-4">Training Jobs</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-surface-800 text-left text-xs text-surface-400">
                <th className="pb-2">Name</th>
                <th className="pb-2">Architecture</th>
                <th className="pb-2">Status</th>
                <th className="pb-2">Epochs</th>
                <th className="pb-2">Best Metric</th>
                <th className="pb-2">Created</th>
                <th className="pb-2">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-800">
              {jobs.map((job) => (
                <tr key={job.id} className="hover:bg-surface-800/50">
                  <td className="py-2.5">{job.name}</td>
                  <td className="py-2.5 text-surface-400">{job.model_architecture}</td>
                  <td className="py-2.5">{statusBadge(job.status)}</td>
                  <td className="py-2.5 text-surface-400">{job.current_epoch}/{job.epochs}</td>
                  <td className="py-2.5">
                    {job.best_metrics && (
                      <span className="text-green-400">
                        {job.best_metrics.map50
                          ? `mAP50: ${(job.best_metrics.map50 * 100).toFixed(1)}%`
                          : job.best_metrics.accuracy
                          ? `Acc: ${(job.best_metrics.accuracy * 100).toFixed(1)}%`
                          : '-'}
                      </span>
                    )}
                  </td>
                  <td className="py-2.5 text-surface-500">{new Date(job.created_at).toLocaleDateString()}</td>
                  <td className="py-2.5">
                    {(job.status === 'training' || job.status === 'queued') && (
                      <button onClick={() => cancelJob(job.id)} className="text-red-400 hover:text-red-300 text-xs">
                        Cancel
                      </button>
                    )}
                    {job.status === 'training' && (
                      <button onClick={() => connectLiveMetrics(job.id)} className="ml-2 text-primary-400 hover:text-primary-300 text-xs">
                        Watch
                      </button>
                    )}
                  </td>
                </tr>
              ))}
              {jobs.length === 0 && (
                <tr><td colSpan={7} className="py-8 text-center text-surface-500">No training jobs yet</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
