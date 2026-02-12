/**
 * AutoAnnotatePanel — Controls for Grounding DINO, SAM2, CLIP auto-annotation.
 */

import { useState } from 'react';
import { Wand2, Sparkles, Search, Loader2, CheckCircle2 } from 'lucide-react';
import { useAnnotationStore } from '../../store/useAnnotationStore';
import { annotationsApi } from '../../api/annotations';
import toast from 'react-hot-toast';

interface AutoAnnotatePanelProps {
  visible: boolean;
  onClose: () => void;
}

export default function AutoAnnotatePanel({ visible, onClose }: AutoAnnotatePanelProps) {
  const { images, currentImage, loadAnnotations } = useAnnotationStore();

  const [textPrompt, setTextPrompt] = useState('');
  const [boxThreshold, setBoxThreshold] = useState(0.3);
  const [textThreshold, setTextThreshold] = useState(0.25);
  const [annotateAll, setAnnotateAll] = useState(false);
  const [useMasks, setUseMasks] = useState(true);
  const [loading, setLoading] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);

  if (!visible) return null;

  const handleGroundingDino = async () => {
    if (!textPrompt.trim()) {
      toast.error('Enter class names separated by periods');
      return;
    }

    setLoading(true);
    try {
      const imageIds = annotateAll ? images.map((i) => i.id) : currentImage ? [currentImage.id] : [];

      const res = await annotationsApi.autoAnnotateGroundingDino({
        image_ids: imageIds,
        text_prompt: textPrompt,
        box_threshold: boxThreshold,
        text_threshold: textThreshold,
      });

      setTaskId(res.data.task_id);
      toast.success(res.data.message);
      pollTask(res.data.task_id);
    } catch (e: any) {
      toast.error(e.response?.data?.detail || 'Auto-annotation failed');
    } finally {
      setLoading(false);
    }
  };

  const handleGroundedSAM = async () => {
    if (!textPrompt.trim()) {
      toast.error('Enter class names separated by periods');
      return;
    }

    setLoading(true);
    try {
      const imageIds = annotateAll ? images.map((i) => i.id) : currentImage ? [currentImage.id] : [];

      const res = await annotationsApi.groundedSAM({
        image_ids: imageIds,
        text_prompt: textPrompt,
        box_threshold: boxThreshold,
        text_threshold: textThreshold,
        generate_masks: useMasks,
      });

      setTaskId(res.data.task_id);
      toast.success(res.data.message);
      pollTask(res.data.task_id);
    } catch (e: any) {
      toast.error(e.response?.data?.detail || 'Auto-annotation failed');
    } finally {
      setLoading(false);
    }
  };

  const pollTask = async (id: string) => {
    const interval = setInterval(async () => {
      try {
        const res = await annotationsApi.getTaskStatus(id);
        if (res.data.status === 'SUCCESS') {
          clearInterval(interval);
          setTaskId(null);
          toast.success('Auto-annotation complete!');
          if (currentImage) {
            loadAnnotations(currentImage.id);
          }
        } else if (res.data.status === 'FAILURE') {
          clearInterval(interval);
          setTaskId(null);
          toast.error('Auto-annotation failed');
        }
      } catch {
        clearInterval(interval);
      }
    }, 2000);
  };

  return (
    <div className="absolute right-0 top-0 z-50 w-80 bg-surface-900 border-l border-surface-800 h-full overflow-y-auto">
      <div className="flex items-center justify-between px-4 py-3 border-b border-surface-800">
        <h3 className="font-semibold">Auto-Annotate</h3>
        <button onClick={onClose} className="text-surface-400 hover:text-white text-sm">Close</button>
      </div>

      <div className="p-4 space-y-4">
        {/* Text prompt */}
        <div>
          <label className="block text-xs text-surface-400 mb-1">
            Text Prompt (separate classes with periods)
          </label>
          <textarea
            value={textPrompt}
            onChange={(e) => setTextPrompt(e.target.value)}
            placeholder="hard hat . person . safety vest ."
            className="w-full rounded bg-surface-800 px-3 py-2 text-sm text-white placeholder-surface-500 outline-none focus:ring-1 focus:ring-primary-500 resize-none"
            rows={3}
          />
          <p className="text-xs text-surface-500 mt-1">
            Periods separate class names. Grounding DINO's BERT encoder uses them as boundaries.
          </p>
        </div>

        {/* Thresholds */}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs text-surface-400 mb-1">Box Threshold</label>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={boxThreshold}
              onChange={(e) => setBoxThreshold(parseFloat(e.target.value))}
              className="w-full"
            />
            <span className="text-xs text-surface-400">{boxThreshold.toFixed(2)}</span>
          </div>
          <div>
            <label className="block text-xs text-surface-400 mb-1">Text Threshold</label>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={textThreshold}
              onChange={(e) => setTextThreshold(parseFloat(e.target.value))}
              className="w-full"
            />
            <span className="text-xs text-surface-400">{textThreshold.toFixed(2)}</span>
          </div>
        </div>

        {/* Options */}
        <div className="space-y-2">
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={annotateAll}
              onChange={(e) => setAnnotateAll(e.target.checked)}
              className="rounded"
            />
            Annotate all images ({images.length})
          </label>
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={useMasks}
              onChange={(e) => setUseMasks(e.target.checked)}
              className="rounded"
            />
            Generate masks (SAM 2)
          </label>
        </div>

        {/* Action buttons */}
        <div className="space-y-2">
          <button
            onClick={handleGroundingDino}
            disabled={loading}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-amber-600 px-4 py-2.5 text-sm font-medium hover:bg-amber-500 disabled:opacity-50 transition-colors"
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : <Wand2 size={16} />}
            Grounding DINO (Boxes Only)
          </button>

          <button
            onClick={handleGroundedSAM}
            disabled={loading}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-violet-600 px-4 py-2.5 text-sm font-medium hover:bg-violet-500 disabled:opacity-50 transition-colors"
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
            Grounded SAM (Boxes + Masks)
          </button>
        </div>

        {/* Task status */}
        {taskId && (
          <div className="flex items-center gap-2 rounded bg-surface-800 px-3 py-2 text-sm">
            <Loader2 size={14} className="animate-spin text-primary-400" />
            Processing... Task: {taskId.slice(0, 8)}
          </div>
        )}

        {/* How it works */}
        <div className="rounded-lg bg-surface-800/50 p-3 space-y-2">
          <h4 className="text-xs font-semibold text-surface-300">How it works</h4>
          <div className="space-y-1.5 text-xs text-surface-500">
            <p><strong className="text-amber-400">Grounding DINO:</strong> Text prompt → BERT encoding → cross-attention with image features → bounding boxes</p>
            <p><strong className="text-violet-400">SAM 2:</strong> Box prompts → Hiera ViT image encoding (cached) → mask decoder → pixel-precise masks in ~8ms</p>
            <p><strong className="text-cyan-400">CLIP Search:</strong> Crop region → CLIP ViT-L/14 → 768-dim vector → FAISS nearest neighbors → similar objects</p>
          </div>
        </div>
      </div>
    </div>
  );
}
