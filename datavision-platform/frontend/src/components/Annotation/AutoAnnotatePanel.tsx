/**
 * AutoAnnotatePanel — Controls for Grounding DINO, SAM2, CLIP auto-annotation.
 */

import { useState } from 'react';
import { Wand2, Sparkles, Search, Loader2, ExternalLink } from 'lucide-react';
import { useAnnotationStore } from '../../store/useAnnotationStore';
import { annotationsApi } from '../../api/annotations';
import toast from 'react-hot-toast';

type PanelTab = 'auto_annotate' | 'smart_select' | 'find_similar';

interface ClipSearchResult {
  image_id: string;
  bbox: [number, number, number, number]; // [cx, cy, w, h]
  similarity: number;
  type: 'full' | 'grid';
}

interface AutoAnnotatePanelProps {
  visible: boolean;
  onClose: () => void;
  initialTab?: PanelTab;
}

export default function AutoAnnotatePanel({ visible, onClose, initialTab = 'auto_annotate' }: AutoAnnotatePanelProps) {
  const { images, currentImage, annotations, loadAnnotations, setCurrentImageIndex } = useAnnotationStore();

  const [activeTab, setActiveTab] = useState<PanelTab>(initialTab);

  // Grounding DINO / Grounded SAM state
  const [textPrompt, setTextPrompt] = useState('');
  const [boxThreshold, setBoxThreshold] = useState(0.3);
  const [textThreshold, setTextThreshold] = useState(0.25);
  const [annotateAll, setAnnotateAll] = useState(false);
  const [useMasks, setUseMasks] = useState(true);

  // CLIP search state
  const [clipTopK, setClipTopK] = useState(20);
  const [clipThreshold, setClipThreshold] = useState(0.75);
  const [clipResults, setClipResults] = useState<ClipSearchResult[]>([]);

  // Shared
  const [loading, setLoading] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);

  // Update tab when initialTab prop changes
  if (visible && initialTab !== activeTab && initialTab !== 'auto_annotate') {
    setActiveTab(initialTab);
  }

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

  const handleSmartSelect = async () => {
    if (!currentImage) {
      toast.error('No image selected');
      return;
    }

    // Use existing bbox annotations as prompts for SAM2
    const bboxAnnotations = annotations.filter((a) => a.bbox);
    if (bboxAnnotations.length === 0) {
      toast.error('Draw a bounding box first, then click Smart Select to generate a mask');
      return;
    }

    setLoading(true);
    try {
      const boxPrompts = bboxAnnotations.map((a) => ({
        x: a.bbox!.x,
        y: a.bbox!.y,
        w: a.bbox!.w,
        h: a.bbox!.h,
      }));

      const res = await annotationsApi.autoAnnotateSAM2({
        image_id: currentImage.id,
        box_prompts: boxPrompts,
      });

      setTaskId(res.data.task_id);
      toast.success(res.data.message);
      pollTask(res.data.task_id);
    } catch (e: any) {
      toast.error(e.response?.data?.detail || 'Smart Select failed');
    } finally {
      setLoading(false);
    }
  };

  const handleClipSearch = async () => {
    if (!currentImage) {
      toast.error('No image selected');
      return;
    }

    // Use selected annotation's bbox or most recent bbox
    const bboxAnnotation = annotations.find((a) => a.bbox);
    if (!bboxAnnotation?.bbox) {
      toast.error('Draw a bounding box around an object to search for similar ones');
      return;
    }

    setClipResults([]);
    setLoading(true);
    try {
      const res = await annotationsApi.clipSearch({
        image_id: currentImage.id,
        crop_bbox: bboxAnnotation.bbox,
        top_k: clipTopK,
        threshold: clipThreshold,
      });

      setTaskId(res.data.task_id);
      toast.success(res.data.message + ' (building index if first time — may take a moment)');
      pollClipTask(res.data.task_id);
    } catch (e: any) {
      toast.error(e.response?.data?.detail || 'CLIP search failed');
    } finally {
      setLoading(false);
    }
  };

  const handleBuildIndex = async () => {
    if (!currentImage) return;
    setLoading(true);
    try {
      const res = await annotationsApi.buildClipIndex(currentImage.project_id);
      setTaskId(res.data.task_id);
      toast.success('Building CLIP index...');
      pollTask(res.data.task_id);
    } catch (e: any) {
      toast.error(e.response?.data?.detail || 'Index build failed');
    } finally {
      setLoading(false);
    }
  };

  // Generic poller — reloads annotations on success
  const pollTask = (id: string) => {
    const interval = setInterval(async () => {
      try {
        const res = await annotationsApi.getTaskStatus(id);
        if (res.data.status === 'SUCCESS') {
          clearInterval(interval);
          setTaskId(null);
          toast.success('Task complete!');
          if (currentImage) loadAnnotations(currentImage.id);
        } else if (res.data.status === 'FAILURE') {
          clearInterval(interval);
          setTaskId(null);
          const errMsg = res.data.error || 'Task failed';
          toast.error(errMsg);
          console.error('Task error:', errMsg);
        }
      } catch {
        clearInterval(interval);
      }
    }, 2000);
  };

  // CLIP-specific poller — captures and displays search results
  const pollClipTask = (id: string) => {
    const interval = setInterval(async () => {
      try {
        const res = await annotationsApi.getTaskStatus(id);
        if (res.data.status === 'SUCCESS') {
          clearInterval(interval);
          setTaskId(null);
          const result = res.data.result as { results: ClipSearchResult[]; total: number } | undefined;
          if (result?.results?.length) {
            setClipResults(result.results);
            toast.success(`Found ${result.total} similar regions`);
          } else {
            toast.success('Search complete — no matches above threshold');
          }
        } else if (res.data.status === 'FAILURE') {
          clearInterval(interval);
          setTaskId(null);
          const errMsg = res.data.error || 'CLIP search failed';
          toast.error(errMsg);
          console.error('CLIP task error:', errMsg);
        }
      } catch {
        clearInterval(interval);
      }
    }, 2000);
  };

  const tabs: { id: PanelTab; label: string; icon: typeof Wand2; color: string }[] = [
    { id: 'auto_annotate', label: 'Auto-Annotate', icon: Wand2, color: 'text-amber-400' },
    { id: 'smart_select', label: 'Smart Select', icon: Sparkles, color: 'text-violet-400' },
    { id: 'find_similar', label: 'Find Similar', icon: Search, color: 'text-cyan-400' },
  ];

  return (
    <div className="absolute right-0 top-0 z-50 w-80 bg-surface-900 border-l border-surface-800 h-full overflow-y-auto">
      <div className="flex items-center justify-between px-4 py-3 border-b border-surface-800">
        <h3 className="font-semibold">AI Tools</h3>
        <button onClick={onClose} className="text-surface-400 hover:text-white text-sm">Close</button>
      </div>

      {/* Tab bar */}
      <div className="flex border-b border-surface-800">
        {tabs.map(({ id, label, icon: Icon, color }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={`flex-1 flex items-center justify-center gap-1.5 px-2 py-2.5 text-xs font-medium transition-colors ${
              activeTab === id
                ? `${color} border-b-2 border-current`
                : 'text-surface-500 hover:text-surface-300'
            }`}
          >
            <Icon size={14} />
            {label}
          </button>
        ))}
      </div>

      <div className="p-4 space-y-4">
        {/* ===== Auto-Annotate Tab ===== */}
        {activeTab === 'auto_annotate' && (
          <>
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

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-surface-400 mb-1">Box Threshold</label>
                <input type="range" min="0.1" max="0.9" step="0.05" value={boxThreshold}
                  onChange={(e) => setBoxThreshold(parseFloat(e.target.value))} className="w-full" />
                <span className="text-xs text-surface-400">{boxThreshold.toFixed(2)}</span>
              </div>
              <div>
                <label className="block text-xs text-surface-400 mb-1">Text Threshold</label>
                <input type="range" min="0.1" max="0.9" step="0.05" value={textThreshold}
                  onChange={(e) => setTextThreshold(parseFloat(e.target.value))} className="w-full" />
                <span className="text-xs text-surface-400">{textThreshold.toFixed(2)}</span>
              </div>
            </div>

            <div className="space-y-2">
              <label className="flex items-center gap-2 text-sm cursor-pointer">
                <input type="checkbox" checked={annotateAll}
                  onChange={(e) => setAnnotateAll(e.target.checked)} className="rounded" />
                Annotate all images ({images.length})
              </label>
              <label className="flex items-center gap-2 text-sm cursor-pointer">
                <input type="checkbox" checked={useMasks}
                  onChange={(e) => setUseMasks(e.target.checked)} className="rounded" />
                Generate masks (SAM 2)
              </label>
            </div>

            <div className="space-y-2">
              <button onClick={handleGroundingDino} disabled={loading}
                className="flex w-full items-center justify-center gap-2 rounded-lg bg-amber-600 px-4 py-2.5 text-sm font-medium hover:bg-amber-500 disabled:opacity-50 transition-colors">
                {loading ? <Loader2 size={16} className="animate-spin" /> : <Wand2 size={16} />}
                Grounding DINO (Boxes Only)
              </button>
              <button onClick={handleGroundedSAM} disabled={loading}
                className="flex w-full items-center justify-center gap-2 rounded-lg bg-violet-600 px-4 py-2.5 text-sm font-medium hover:bg-violet-500 disabled:opacity-50 transition-colors">
                {loading ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
                Grounded SAM (Boxes + Masks)
              </button>
            </div>
          </>
        )}

        {/* ===== Smart Select Tab (SAM 2) ===== */}
        {activeTab === 'smart_select' && (
          <>
            <div className="rounded-lg bg-surface-800/50 p-3 space-y-2">
              <h4 className="text-xs font-semibold text-violet-400">How to use Smart Select</h4>
              <ol className="space-y-1 text-xs text-surface-400 list-decimal list-inside">
                <li>Draw a <strong className="text-white">bounding box</strong> around the object (B key)</li>
                <li>Click <strong className="text-white">Generate Masks</strong> below</li>
                <li>SAM 2 generates pixel-precise masks from your boxes</li>
              </ol>
            </div>

            <div className="text-xs text-surface-500">
              Current image has <strong className="text-white">{annotations.filter((a) => a.bbox).length}</strong> box annotations
              that will be used as SAM 2 prompts.
            </div>

            <button onClick={handleSmartSelect} disabled={loading}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-violet-600 px-4 py-2.5 text-sm font-medium hover:bg-violet-500 disabled:opacity-50 transition-colors">
              {loading ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
              Generate Masks (SAM 2)
            </button>
          </>
        )}

        {/* ===== Find Similar Tab (CLIP) ===== */}
        {activeTab === 'find_similar' && (
          <>
            <div className="rounded-lg bg-surface-800/50 p-3 space-y-2">
              <h4 className="text-xs font-semibold text-cyan-400">How to use Find Similar</h4>
              <ol className="space-y-1 text-xs text-surface-400 list-decimal list-inside">
                <li>Draw a <strong className="text-white">bounding box</strong> around an object</li>
                <li>Click <strong className="text-white">Search Similar</strong></li>
                <li>Index is built automatically on first run</li>
              </ol>
            </div>

            <div className="space-y-3">
              <div>
                <label className="block text-xs text-surface-400 mb-1">Top K results</label>
                <input type="number" min={1} max={100} value={clipTopK}
                  onChange={(e) => setClipTopK(parseInt(e.target.value) || 20)}
                  className="w-full rounded bg-surface-800 px-3 py-2 text-sm text-white outline-none focus:ring-1 focus:ring-primary-500" />
              </div>
              <div>
                <label className="block text-xs text-surface-400 mb-1">
                  Similarity Threshold ({clipThreshold.toFixed(2)})
                </label>
                <input type="range" min="0.5" max="0.99" step="0.01" value={clipThreshold}
                  onChange={(e) => setClipThreshold(parseFloat(e.target.value))} className="w-full" />
                <div className="flex justify-between text-[10px] text-surface-500">
                  <span>More results</span>
                  <span>More precise</span>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <button onClick={handleClipSearch} disabled={loading || !!taskId}
                className="flex w-full items-center justify-center gap-2 rounded-lg bg-cyan-600 px-4 py-2.5 text-sm font-medium hover:bg-cyan-500 disabled:opacity-50 transition-colors">
                {(loading || taskId) ? <Loader2 size={16} className="animate-spin" /> : <Search size={16} />}
                {taskId ? 'Searching...' : 'Search Similar (CLIP)'}
              </button>
              <button onClick={handleBuildIndex} disabled={!!taskId}
                className="flex w-full items-center justify-center gap-2 rounded-lg bg-surface-700 px-4 py-2 text-xs text-surface-300 hover:bg-surface-600 disabled:opacity-50 transition-colors">
                {taskId ? <Loader2 size={14} className="animate-spin" /> : null}
                Rebuild FAISS Index
              </button>
            </div>

            {/* Results list */}
            {clipResults.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs font-semibold text-surface-300">
                  {clipResults.length} similar regions found
                </p>
                <div className="max-h-64 overflow-y-auto space-y-1">
                  {clipResults.map((r, i) => {
                    const matchedImage = images.find((img) => img.id === r.image_id);
                    const matchedIndex = matchedImage ? images.indexOf(matchedImage) : -1;
                    return (
                      <div
                        key={i}
                        className="flex items-center justify-between rounded bg-surface-800 px-3 py-2 text-xs"
                      >
                        <div className="flex flex-col min-w-0">
                          <span className="truncate text-white font-medium">
                            {matchedImage?.filename ?? r.image_id.slice(0, 8) + '…'}
                          </span>
                          <span className="text-surface-400">
                            {(r.similarity * 100).toFixed(1)}% match · {r.type}
                          </span>
                        </div>
                        {matchedIndex >= 0 && (
                          <button
                            onClick={() => setCurrentImageIndex(matchedIndex)}
                            title="Go to image"
                            className="ml-2 shrink-0 text-cyan-400 hover:text-cyan-300"
                          >
                            <ExternalLink size={14} />
                          </button>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </>
        )}

        {/* Task status */}
        {taskId && (
          <div className="flex items-center gap-2 rounded bg-surface-800 px-3 py-2 text-sm">
            <Loader2 size={14} className="animate-spin text-primary-400" />
            Processing... Task: {taskId.slice(0, 8)}
          </div>
        )}

        {/* How it works */}
        <div className="rounded-lg bg-surface-800/50 p-3 space-y-2">
          <h4 className="text-xs font-semibold text-surface-300">Pipeline overview</h4>
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
