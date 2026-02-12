/**
 * ModelsPage — Trained model registry, comparison, and export.
 */

import { useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { Box, Download, Trash2, BarChart3, Zap } from 'lucide-react';
import { useTrainingStore } from '../store/useTrainingStore';
import toast from 'react-hot-toast';

export default function ModelsPage() {
  const { projectId } = useParams();
  const { models, fetchModels, deleteModel, exportModel } = useTrainingStore();

  useEffect(() => {
    fetchModels(projectId);
  }, [projectId, fetchModels]);

  const handleExport = async (modelId: string, format: string) => {
    try {
      await exportModel(modelId, [format]);
      toast.success(`Exporting to ${format.toUpperCase()}...`);
    } catch {
      toast.error('Export failed');
    }
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Model Hub</h1>

      {models.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-surface-500">
          <Box size={48} className="mb-4" />
          <p>No trained models yet. Train a model from the Training page.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {models.map((model) => (
            <div key={model.id} className="rounded-xl border border-surface-800 bg-surface-900 p-5 space-y-4">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="font-semibold">{model.name}</h3>
                  <p className="text-sm text-surface-400">{model.architecture} - v{model.version}</p>
                </div>
                <span className="rounded-full bg-primary-600/20 px-2 py-0.5 text-xs text-primary-400">
                  {model.task_type}
                </span>
              </div>

              {/* Metrics */}
              <div className="grid grid-cols-2 gap-2">
                {model.map50 != null && (
                  <div className="rounded-lg bg-surface-800 p-2">
                    <div className="text-lg font-bold text-green-400">{(model.map50 * 100).toFixed(1)}%</div>
                    <div className="text-[10px] text-surface-400">mAP50</div>
                  </div>
                )}
                {model.accuracy != null && (
                  <div className="rounded-lg bg-surface-800 p-2">
                    <div className="text-lg font-bold text-amber-400">{(model.accuracy * 100).toFixed(1)}%</div>
                    <div className="text-[10px] text-surface-400">Accuracy</div>
                  </div>
                )}
                {model.precision_val != null && (
                  <div className="rounded-lg bg-surface-800 p-2">
                    <div className="text-lg font-bold text-primary-400">{(model.precision_val * 100).toFixed(1)}%</div>
                    <div className="text-[10px] text-surface-400">Precision</div>
                  </div>
                )}
                {model.recall_val != null && (
                  <div className="rounded-lg bg-surface-800 p-2">
                    <div className="text-lg font-bold text-violet-400">{(model.recall_val * 100).toFixed(1)}%</div>
                    <div className="text-[10px] text-surface-400">Recall</div>
                  </div>
                )}
              </div>

              {/* Model info */}
              <div className="flex items-center gap-3 text-xs text-surface-400">
                <span>{model.num_classes} classes</span>
                {model.model_size_mb && <span>{model.model_size_mb.toFixed(1)} MB</span>}
                {model.inference_time_ms && (
                  <span className="flex items-center gap-0.5">
                    <Zap size={10} /> {model.inference_time_ms.toFixed(1)}ms
                  </span>
                )}
                <span>{model.export_format}</span>
              </div>

              {/* Actions */}
              <div className="flex gap-2">
                <button
                  onClick={() => handleExport(model.id, 'onnx')}
                  className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-surface-800 px-3 py-2 text-xs hover:bg-surface-700 transition-colors"
                >
                  <Download size={12} /> ONNX
                </button>
                <button
                  onClick={() => handleExport(model.id, 'tflite')}
                  className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-surface-800 px-3 py-2 text-xs hover:bg-surface-700 transition-colors"
                >
                  <Download size={12} /> TFLite
                </button>
                <button
                  onClick={() => handleExport(model.id, 'torchscript')}
                  className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-surface-800 px-3 py-2 text-xs hover:bg-surface-700 transition-colors"
                >
                  <Download size={12} /> TorchScript
                </button>
                <button
                  onClick={() => { if (confirm('Delete this model?')) deleteModel(model.id); }}
                  className="rounded-lg bg-red-900/30 px-2 py-2 text-red-400 hover:bg-red-900/50 transition-colors"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
