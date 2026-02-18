/**
 * AutoAnnotatePanel — Controls for Grounding DINO, SAM2, CLIP auto-annotation.
 */

import { useState, useMemo } from 'react';
import { Wand2, Sparkles, Search, Loader2, ExternalLink, Check, X, GalleryHorizontal, Play } from 'lucide-react';
import { useAnnotationStore } from '../../store/useAnnotationStore';
import { useTrainingStore } from '../../store/useTrainingStore';
import { annotationsApi } from '../../api/annotations';
import toast from 'react-hot-toast';

type PanelTab = 'auto_label' | 'smart_select' | 'box_prompting';

interface ClipSearchResult {
  image_id: string;
  bbox: [number, number, number, number]; // [cx, cy, w, h]
  similarity: number;
  type: string;
}

interface AutoAnnotatePanelProps {
  visible: boolean;
  onClose: () => void;
  initialTab?: PanelTab;
}

export default function AutoAnnotatePanel({ visible, onClose, initialTab = 'auto_label' }: AutoAnnotatePanelProps) {
  const { 
    images, currentImage, annotations, 
    loadAnnotations, setCurrentImageIndex,
    batchVerify, batchDelete 
  } = useAnnotationStore();

  const { createJob, jobs, fetchJobs } = useTrainingStore();

  const [activeTab, setActiveTab] = useState<PanelTab>(initialTab);

  // Guided Pipeline State
  const [pipelineStep, setPipelineStep] = useState(1);
  const [manualCount, setManualCount] = useState(10);

  // Poll for jobs periodically to update pipeline status
  useEffect(() => {
    const interval = setInterval(() => {
      if (activeTab === 'auto_label') {
        const projId = currentImage?.project_id;
        if (projId) fetchJobs(projId);
      }
    }, 5000);
    return () => clearInterval(interval);
  }, [activeTab, currentImage?.project_id]);

  // Find most recent active job for this project
  const activeJob = useMemo(() => {
    return jobs.find(j => j.status === 'training' || j.status === 'preparing' || j.status === 'queued');
  }, [jobs]);

  const handleTrainSeedModel = async () => {
    if (!currentImage) return;
    setLoading(true);
    try {
      const jobName = `Seed_Model_${currentImage.project_id.slice(0, 4)}_${Date.now()}`;
      await createJob({
        project_id: currentImage.project_id,
        name: jobName,
        model_architecture: 'yolov11m',
        task_type: 'detection',
        epochs: 50, // Faster for seed
        batch_size: 8,
        img_size: 640,
        learning_rate: 0.01,
        patience: 10,
        optimizer: 'AdamW',
      });
      toast.success('Seed training started! Moving to next step...');
      setPipelineStep(3);
    } catch (e: any) {
      toast.error(e.response?.data?.detail || 'Failed to start seed training');
    } finally {
      setLoading(false);
    }
  };

  const handleTrainMainModel = async () => {
    if (!currentImage) return;
    setLoading(true);
    try {
      const jobName = `Main_Model_${currentImage.project_id.slice(0, 4)}_${Date.now()}`;
      await createJob({
        project_id: currentImage.project_id,
        name: jobName,
        model_architecture: 'yolov11m',
        task_type: 'detection',
        epochs: 100,
        batch_size: 16,
        img_size: 640,
        learning_rate: 0.01,
        patience: 20,
        optimizer: 'AdamW',
      });
      toast.success('Main training started!');
    } catch (e: any) {
      toast.error(e.response?.data?.detail || 'Failed to start main training');
    } finally {
      setLoading(false);
    }
  };

  const handleAutoLabelRemaining = async () => {
    if (!currentImage) return;
    setLoading(true);
    try {
      const res = await annotationsApi.autoAnnotateTrainedModel({
        project_id: currentImage.project_id,
        confidence_threshold: 0.3,
      });
      setTaskId(res.data.task_id);
      toast.success('Auto-labeling started using latest trained model!');
      pollTask(res.data.task_id);
      setPipelineStep(4);
    } catch (e: any) {
      toast.error(e.response?.data?.detail || 'Auto-labeling failed');
    } finally {
      setLoading(false);
    }
  };

  // Grounding DINO / Grounded SAM state
  const [textPrompt, setTextPrompt] = useState('');
  const [boxThreshold, setBoxThreshold] = useState(0.3);
  const [textThreshold, setTextThreshold] = useState(0.25);
  const [annotateAll, setAnnotateAll] = useState(false);
  const [useMasks, setUseMasks] = useState(true);

  // CLIP search state
  const [clipTopK, setClipTopK] = useState(40); // Increased for better gallery
  const [clipThreshold, setClipThreshold] = useState(0.70);
  const [clipResults, setClipResults] = useState<ClipSearchResult[]>([]);

  // Shared
  const [loading, setLoading] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [taskProgress, setTaskProgress] = useState<number | null>(null);

  // Memoize unverified annotations for the current image that came from clip_search
  const pendingClips = useMemo(() => {
    return annotations.filter(a => !a.is_verified && a.source === 'clip_search' && (a.confidence || 0) >= clipThreshold);
  }, [annotations, clipThreshold]);

  const filteredResults = useMemo(() => {
    return clipResults.filter(r => r.similarity >= clipThreshold);
  }, [clipResults, clipThreshold]);

  const handleAcceptAll = async () => {
    if (pendingClips.length === 0) return;
    setLoading(true);
    try {
      await batchVerify(pendingClips.map(a => a.id));
      toast.success(`Accepted ${pendingClips.length} annotations`);
    } catch (e) {
      toast.error('Failed to accept annotations');
    } finally {
      setLoading(false);
    }
  };

  const handleRejectAll = async () => {
    if (pendingClips.length === 0) return;
    setLoading(true);
    try {
      await batchDelete(pendingClips.map(a => a.id));
      toast.success(`Rejected ${pendingClips.length} annotations`);
    } catch (e) {
      toast.error('Failed to reject annotations');
    } finally {
      setLoading(false);
    }
  };

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
        class_name: bboxAnnotation.class_name,
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
    setTaskProgress(0);
    const interval = setInterval(async () => {
      try {
        const res = await annotationsApi.getTaskStatus(id);
        if (res.data.status === 'SUCCESS') {
          clearInterval(interval);
          setTaskId(null);
          setTaskProgress(null);
          toast.success('Task complete!');
          if (currentImage) loadAnnotations(currentImage.id);
        } else if (res.data.status === 'PROGRESS') {
          setTaskProgress(res.data.result?.progress || 0);
        } else if (res.data.status === 'FAILURE') {
          clearInterval(interval);
          setTaskId(null);
          setTaskProgress(null);
          const errMsg = res.data.error || 'Task failed';
          toast.error(errMsg);
          console.error('Task error:', errMsg);
        }
      } catch {
        clearInterval(interval);
      }
    }, 1000); // Polling every 1s for better progress resolution
  };

  // CLIP-specific poller — captures and displays search results
  const pollClipTask = (id: string) => {
    setTaskProgress(0);
    const interval = setInterval(async () => {
      try {
        const res = await annotationsApi.getTaskStatus(id);
        if (res.data.status === 'SUCCESS') {
          clearInterval(interval);
          setTaskId(null);
          setTaskProgress(null);
          const result = res.data.result as { results: ClipSearchResult[]; total: number } | undefined;
          if (result?.results?.length) {
            setClipResults(result.results);
            toast.success(`Found ${result.total} similar regions`);
            // Reload annotations to show them on canvas
            if (currentImage) loadAnnotations(currentImage.id);
          } else {
            toast.success('Search complete — no matches above threshold');
          }
        } else if (res.data.status === 'PROGRESS') {
          setTaskProgress(res.data.result?.progress || 0);
        } else if (res.data.status === 'FAILURE') {
          clearInterval(interval);
          setTaskId(null);
          setTaskProgress(null);
          const errMsg = res.data.error || 'CLIP search failed';
          toast.error(errMsg);
          console.error('CLIP task error:', errMsg);
        }
      } catch {
        clearInterval(interval);
      }
    }, 1000);
  };

  const tabs: { id: PanelTab; label: string; icon: typeof Wand2; color: string }[] = [
    { id: 'auto_label', label: 'Auto Label', icon: Wand2, color: 'text-amber-400' },
    { id: 'smart_select', label: 'Smart Select', icon: Sparkles, color: 'text-violet-400' },
    { id: 'box_prompting', label: 'Box Prompt', icon: Search, color: 'text-cyan-400' },
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
        {/* ===== Auto Label Tab (Guided Pipeline) ===== */}
        {activeTab === 'auto_label' && (
          <div className="space-y-4">
            <div className="rounded-lg bg-surface-800/50 p-3 border border-amber-500/20">
              <h4 className="text-xs font-bold text-amber-400 uppercase mb-2">Guided Auto-Label Pipeline</h4>
              
              <div className="space-y-3">
                {/* Step 1: Manual Seed */}
                <div className={`flex gap-3 ${pipelineStep > 1 ? 'opacity-50' : ''}`}>
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${pipelineStep === 1 ? 'bg-amber-500 text-black' : 'bg-surface-700 text-surface-400'}`}>1</div>
                  <div className="flex-1">
                    <p className="text-xs font-semibold text-white">Annotate Seed Set</p>
                    {pipelineStep === 1 && (
                      <div className="mt-2 space-y-2">
                        <p className="text-[10px] text-surface-400">Label some images manually to train the auto-labeler.</p>
                        <div className="flex items-center gap-2">
                          <input 
                            type="number" 
                            value={manualCount} 
                            onChange={e => setManualCount(parseInt(e.target.value))}
                            className="w-16 rounded bg-surface-800 px-2 py-1 text-xs"
                          />
                          <span className="text-[10px] text-surface-500">images target</span>
                        </div>
                        <button 
                          onClick={() => setPipelineStep(2)}
                          className="w-full py-1.5 bg-amber-600 hover:bg-amber-500 rounded text-[10px] font-bold"
                        >
                          I'VE LABELED {manualCount} IMAGES
                        </button>
                      </div>
                    )}
                  </div>
                </div>

                {/* Step 2: Intermediate Training */}
                <div className={`flex gap-3 ${pipelineStep !== 2 && !activeJob ? 'opacity-50' : ''}`}>
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${pipelineStep === 2 || (activeJob && activeJob.name.includes('Seed')) ? 'bg-amber-500 text-black' : 'bg-surface-700 text-surface-400'}`}>2</div>
                  <div className="flex-1">
                    <p className="text-xs font-semibold text-white">Train Auto-Labeler</p>
                    {(pipelineStep === 2 || (activeJob && activeJob.name.includes('Seed'))) && (
                      <div className="mt-2 space-y-2">
                        {activeJob && activeJob.name.includes('Seed') ? (
                          <div className="rounded bg-black/30 p-2 space-y-1">
                            <div className="flex justify-between text-[9px] text-amber-400 font-bold uppercase">
                              <span>{activeJob.status}...</span>
                              <span>{activeJob.current_epoch}/{activeJob.epochs}</span>
                            </div>
                            <div className="h-1 w-full bg-surface-700 rounded-full overflow-hidden">
                              <div 
                                className="h-full bg-amber-500 transition-all duration-500"
                                style={{ width: `${(activeJob.current_epoch / activeJob.epochs) * 100}%` }}
                              />
                            </div>
                          </div>
                        ) : (
                          <button 
                            onClick={handleTrainSeedModel}
                            disabled={loading}
                            className="w-full py-1.5 bg-amber-600 hover:bg-amber-500 rounded text-[10px] font-bold flex items-center justify-center gap-2"
                          >
                            {loading ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
                            TRAIN SEED MODEL
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                </div>

                {/* Step 3: Propagation */}
                <div className={`flex gap-3 ${pipelineStep !== 3 ? 'opacity-50' : ''}`}>
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${pipelineStep === 3 ? 'bg-amber-500 text-black' : 'bg-surface-700 text-surface-400'}`}>3</div>
                  <div className="flex-1">
                    <p className="text-xs font-semibold text-white">Label Remaining</p>
                    {pipelineStep === 3 && (
                      <div className="mt-2 space-y-2">
                        <button 
                          onClick={handleAutoLabelRemaining}
                          disabled={loading}
                          className="w-full py-1.5 bg-amber-600 hover:bg-amber-500 rounded text-[10px] font-bold flex items-center justify-center gap-2"
                        >
                          {loading ? <Loader2 size={12} className="animate-spin" /> : <Wand2 size={12} />}
                          AUTO-LABEL ALL
                        </button>
                      </div>
                    )}
                  </div>
                </div>

                {/* Step 4: Final Model */}
                <div className={`flex gap-3 ${pipelineStep !== 4 && !activeJob ? 'opacity-50' : ''}`}>
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${pipelineStep === 4 || (activeJob && activeJob.name.includes('Main')) ? 'bg-green-500 text-black' : 'bg-surface-700 text-surface-400'}`}>4</div>
                  <div className="flex-1">
                    <p className="text-xs font-semibold text-white">Train Main Model</p>
                    {(pipelineStep === 4 || (activeJob && activeJob.name.includes('Main'))) && (
                      <div className="mt-2 space-y-2">
                        {activeJob && activeJob.name.includes('Main') ? (
                          <div className="rounded bg-black/30 p-2 space-y-1">
                            <div className="flex justify-between text-[9px] text-green-400 font-bold uppercase">
                              <span>{activeJob.status}...</span>
                              <span>{activeJob.current_epoch}/{activeJob.epochs}</span>
                            </div>
                            <div className="h-1 w-full bg-surface-700 rounded-full overflow-hidden">
                              <div 
                                className="h-full bg-green-500 transition-all duration-500"
                                style={{ width: `${(activeJob.current_epoch / activeJob.epochs) * 100}%` }}
                              />
                            </div>
                          </div>
                        ) : (
                          <>
                            <button 
                              onClick={handleTrainMainModel}
                              disabled={loading}
                              className="w-full py-1.5 bg-green-600 hover:bg-green-500 rounded text-[10px] font-bold flex items-center justify-center gap-2"
                            >
                              {loading ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
                              START FINAL TRAINING
                            </button>
                            <button onClick={() => setPipelineStep(1)} className="text-[10px] text-surface-500 underline">Reset Pipeline</button>
                          </>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
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

        {/* ===== Box Prompt Tab (CLIP) ===== */}
        {activeTab === 'box_prompting' && (
          <>
            <div className="rounded-lg bg-surface-800/50 p-3 space-y-2">
              <h4 className="text-xs font-semibold text-cyan-400">How to use Box Prompt</h4>
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
                {taskId ? 'Searching...' : 'Predict (Box Prompt)'}
              </button>
              <button onClick={handleBuildIndex} disabled={!!taskId}
                className="flex w-full items-center justify-center gap-2 rounded-lg bg-surface-700 px-4 py-2 text-xs text-surface-300 hover:bg-surface-600 disabled:opacity-50 transition-colors">
                {taskId ? <Loader2 size={14} className="animate-spin" /> : null}
                Rebuild FAISS Index
              </button>
            </div>

            {/* Results Gallery (Modern UI) */}
            {clipResults.length > 0 && (
              <div className="space-y-3 pt-2">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-semibold text-surface-300 uppercase flex items-center gap-1.5">
                    <GalleryHorizontal size={14} className="text-cyan-400" />
                    Matches ({filteredResults.length})
                  </h4>
                  {pendingClips.length > 0 && (
                    <div className="flex gap-1">
                      <button onClick={handleAcceptAll} className="p-1 text-green-400 hover:bg-green-400/10 rounded" title="Accept All visible">
                        <Check size={14} />
                      </button>
                      <button onClick={handleRejectAll} className="p-1 text-red-400 hover:bg-red-400/10 rounded" title="Reject All visible">
                        <X size={14} />
                      </button>
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-2">
                  {filteredResults.map((r, i) => {
                    const matchedImage = images.find((img) => img.id === r.image_id);
                    if (!matchedImage) return null;
                    
                    const [cx, cy, cw, ch] = r.bbox;
                    // Calculate object-centered crop for preview
                    const zoom = 1.2; // Show a bit of context
                    const width = (cw * 100 * zoom);
                    const height = (ch * 100 * zoom);
                    const left = (cx - (cw * zoom / 2)) * 100;
                    const top = (cy - (ch * zoom / 2)) * 100;

                    return (
                      <div key={i} className="group relative aspect-square rounded bg-black overflow-hidden border border-surface-800 hover:border-cyan-500/50 transition-colors">
                        <img 
                          src={matchedImage.filepath} 
                          alt="match"
                          className="absolute max-w-none"
                          style={{
                            width: `${100 / (cw * zoom)}%`,
                            left: `${-left * (1 / (cw * zoom))}%`,
                            top: `${-top * (1 / (ch * zoom))}%`,
                          }}
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                        <div className="absolute bottom-1 left-1 right-1 flex items-center justify-between opacity-0 group-hover:opacity-100 transition-opacity">
                          <span className="text-[10px] font-medium text-white bg-black/50 px-1 rounded">
                            {(r.similarity * 100).toFixed(0)}%
                          </span>
                          <button 
                            onClick={() => setCurrentImageIndex(images.indexOf(matchedImage))}
                            className="p-1 bg-cyan-600 rounded text-white hover:bg-cyan-500"
                          >
                            <ExternalLink size={10} />
                          </button>
                        </div>
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
          <div className="space-y-2 rounded bg-surface-800 px-3 py-2">
            <div className="flex items-center gap-2 text-sm">
              <Loader2 size={14} className="animate-spin text-cyan-400" />
              <span>Processing Task...</span>
            </div>
            {taskProgress !== null && (
              <div className="space-y-1">
                <div className="flex justify-between text-[10px] text-surface-400">
                  <span>Progress</span>
                  <span>{taskProgress}%</span>
                </div>
                <div className="h-1.5 w-full bg-surface-700 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-cyan-500 transition-all duration-300"
                    style={{ width: `${taskProgress}%` }}
                  />
                </div>
              </div>
            )}
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
