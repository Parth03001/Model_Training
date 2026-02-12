/**
 * DatasetPage — Image upload, dataset browser, and split management.
 */

import { useEffect, useState, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import { Upload, FolderUp, Image as ImageIcon, SplitSquareVertical, Download, Trash2 } from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import { useRef } from 'react';
import { imagesApi } from '../api/images';
import { annotationsApi } from '../api/annotations';
import type { ImageRecord } from '../types';
import toast from 'react-hot-toast';

export default function DatasetPage() {
  const { projectId } = useParams();
  const [images, setImages] = useState<ImageRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);

  const loadImages = useCallback(async () => {
    if (!projectId) return;
    setLoading(true);
    try {
      const res = await imagesApi.listByProject(projectId, { limit: 200 });
      setImages(res.data.images);
      setTotal(res.data.total);
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    loadImages();
  }, [loadImages]);

  const onDrop = useCallback(async (files: File[]) => {
    if (!projectId || files.length === 0) return;
    setUploading(true);
    try {
      const res = await imagesApi.upload(projectId, files);
      toast.success(`Uploaded ${res.data.uploaded} images${res.data.failed > 0 ? `, ${res.data.failed} failed` : ''}`);
      loadImages();
    } catch (e: any) {
      toast.error('Upload failed');
    } finally {
      setUploading(false);
    }
  }, [projectId, loadImages]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'] },
    multiple: true,
  });

  const folderInputRef = useRef<HTMLInputElement>(null);

  const onFolderSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const fileList = e.target.files;
    if (!fileList || fileList.length === 0 || !projectId) return;
    const imageExts = new Set(['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']);
    const imageFiles = Array.from(fileList).filter((f) => {
      const ext = f.name.slice(f.name.lastIndexOf('.')).toLowerCase();
      return imageExts.has(ext);
    });
    if (imageFiles.length === 0) {
      toast.error('No supported image files found in folder');
      return;
    }
    await onDrop(imageFiles);
    // Reset so the same folder can be re-selected
    e.target.value = '';
  }, [projectId, onDrop]);

  const handleAutoSplit = async () => {
    // Random 80/15/5 split
    const shuffled = [...images].sort(() => Math.random() - 0.5);
    const nTrain = Math.floor(shuffled.length * 0.8);
    const nVal = Math.floor(shuffled.length * 0.15);

    for (let i = 0; i < shuffled.length; i++) {
      const split = i < nTrain ? 'train' : i < nTrain + nVal ? 'val' : 'test';
      await imagesApi.setSplit(shuffled[i].id, split);
    }
    toast.success('Dataset split applied');
    loadImages();
  };

  const handleExport = async (format: string) => {
    if (!projectId) return;
    try {
      const res = await annotationsApi.export(projectId, format);
      const blob = new Blob([JSON.stringify(res.data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `annotations_${format}.json`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success(`Exported as ${format.toUpperCase()}`);
    } catch {
      toast.error('Export failed');
    }
  };

  const trainCount = images.filter((i) => i.split === 'train').length;
  const valCount = images.filter((i) => i.split === 'val').length;
  const testCount = images.filter((i) => i.split === 'test').length;
  const annotatedCount = images.filter((i) => i.status === 'annotated' || i.annotation_count > 0).length;

  if (!projectId) {
    return <div className="flex h-full items-center justify-center text-surface-500">Select a project first.</div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Dataset</h1>
        <div className="flex gap-2">
          <button onClick={handleAutoSplit}
            className="flex items-center gap-2 rounded-lg bg-surface-800 px-3 py-2 text-sm hover:bg-surface-700 transition-colors">
            <SplitSquareVertical size={16} /> Auto Split (80/15/5)
          </button>
          <button onClick={() => handleExport('yolo')}
            className="flex items-center gap-2 rounded-lg bg-surface-800 px-3 py-2 text-sm hover:bg-surface-700 transition-colors">
            <Download size={16} /> Export YOLO
          </button>
          <button onClick={() => handleExport('coco')}
            className="flex items-center gap-2 rounded-lg bg-surface-800 px-3 py-2 text-sm hover:bg-surface-700 transition-colors">
            <Download size={16} /> Export COCO
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-5 gap-4">
        {[
          { label: 'Total', value: total, color: 'text-white' },
          { label: 'Annotated', value: annotatedCount, color: 'text-green-400' },
          { label: 'Train', value: trainCount, color: 'text-primary-400' },
          { label: 'Val', value: valCount, color: 'text-amber-400' },
          { label: 'Test', value: testCount, color: 'text-violet-400' },
        ].map((stat) => (
          <div key={stat.label} className="rounded-xl border border-surface-800 bg-surface-900 p-4">
            <div className={`text-2xl font-bold ${stat.color}`}>{stat.value}</div>
            <div className="text-xs text-surface-400">{stat.label}</div>
          </div>
        ))}
      </div>

      {/* Upload zone */}
      <div
        {...getRootProps()}
        className={`rounded-xl border-2 border-dashed p-8 text-center cursor-pointer transition-colors ${
          isDragActive ? 'border-primary-500 bg-primary-600/10' : 'border-surface-700 hover:border-surface-500'
        }`}
      >
        <input {...getInputProps()} />
        {/* Hidden folder input */}
        <input
          ref={folderInputRef}
          type="file"
          className="hidden"
          onChange={onFolderSelect}
          {...({ webkitdirectory: '', directory: '', mozdirectory: '' } as any)}
        />
        <Upload size={32} className="mx-auto mb-3 text-surface-400" />
        {uploading ? (
          <p className="text-surface-400">Uploading...</p>
        ) : isDragActive ? (
          <p className="text-primary-400">Drop images here</p>
        ) : (
          <p className="text-surface-400">Drag & drop images, or click to browse</p>
        )}
        <p className="mt-1 text-xs text-surface-500">JPG, PNG, BMP, TIFF, WebP supported</p>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            folderInputRef.current?.click();
          }}
          className="mt-3 inline-flex items-center gap-2 rounded-lg bg-surface-800 px-4 py-2 text-sm text-surface-300 hover:bg-surface-700 transition-colors"
        >
          <FolderUp size={16} /> Upload Folder
        </button>
      </div>

      {/* Image grid */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8">
        {images.map((img) => (
          <div key={img.id} className="group relative rounded-lg border border-surface-800 overflow-hidden hover:border-primary-600/50 transition-all">
            <div className="aspect-square bg-surface-800 flex items-center justify-center">
              <img src={img.filepath} alt={img.filename} className="h-full w-full object-cover" loading="lazy" />
            </div>
            <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
            <div className="absolute bottom-0 left-0 right-0 p-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
              <p className="text-xs text-white truncate">{img.filename}</p>
              <div className="flex items-center gap-1 mt-0.5">
                {img.split && (
                  <span className={`rounded px-1 text-[10px] ${
                    img.split === 'train' ? 'bg-primary-600/40 text-primary-300' :
                    img.split === 'val' ? 'bg-amber-600/40 text-amber-300' : 'bg-violet-600/40 text-violet-300'
                  }`}>{img.split}</span>
                )}
                {img.annotation_count > 0 && (
                  <span className="text-[10px] text-green-400">{img.annotation_count} ann</span>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
