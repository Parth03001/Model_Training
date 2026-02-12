/**
 * DrawingLayer — Handles active drawing interactions (bbox, polygon, points).
 *
 * Drawing Flow:
 * - bbox: mouseDown sets start point, mouseMove shows preview, mouseUp creates annotation
 * - polygon: click adds vertices, double-click closes polygon
 * - point: click adds a point prompt (for SAM2)
 */

import { useState, useCallback } from 'react';
import { Rect, Line, Circle, Group } from 'react-konva';
import { useAnnotationStore } from '../../store/useAnnotationStore';

interface DrawingLayerProps {
  imageWidth: number;
  imageHeight: number;
}

export default function DrawingLayer({ imageWidth, imageHeight }: DrawingLayerProps) {
  const {
    activeTool,
    activeClass,
    isDrawing,
    setIsDrawing,
    addAnnotation,
  } = useAnnotationStore();

  // Bbox drawing state
  const [bboxStart, setBboxStart] = useState<{ x: number; y: number } | null>(null);
  const [bboxEnd, setBboxEnd] = useState<{ x: number; y: number } | null>(null);

  // Polygon drawing state
  const [polygonPoints, setPolygonPoints] = useState<number[][]>([]);

  // Point prompts state
  const [pointPrompts, setPointPrompts] = useState<{ x: number; y: number; label: number }[]>([]);

  // --- BBOX Drawing ---
  const handleMouseDown = useCallback(
    (e: any) => {
      if (activeTool !== 'bbox') return;
      const pos = e.target.getStage()?.getRelativePointerPosition();
      if (!pos) return;
      setBboxStart(pos);
      setBboxEnd(pos);
      setIsDrawing(true);
    },
    [activeTool, setIsDrawing]
  );

  const handleMouseMove = useCallback(
    (e: any) => {
      if (!isDrawing || activeTool !== 'bbox' || !bboxStart) return;
      const pos = e.target.getStage()?.getRelativePointerPosition();
      if (pos) setBboxEnd(pos);
    },
    [isDrawing, activeTool, bboxStart]
  );

  const handleMouseUp = useCallback(() => {
    if (activeTool === 'bbox' && bboxStart && bboxEnd) {
      const x1 = Math.min(bboxStart.x, bboxEnd.x);
      const y1 = Math.min(bboxStart.y, bboxEnd.y);
      const x2 = Math.max(bboxStart.x, bboxEnd.x);
      const y2 = Math.max(bboxStart.y, bboxEnd.y);

      const w = x2 - x1;
      const h = y2 - y1;

      // Minimum size check
      if (w > 5 && h > 5 && activeClass) {
        addAnnotation({
          class_name: activeClass,
          annotation_type: 'bbox',
          bbox: {
            x: (x1 + w / 2) / imageWidth,
            y: (y1 + h / 2) / imageHeight,
            w: w / imageWidth,
            h: h / imageHeight,
          },
        });
      }

      setBboxStart(null);
      setBboxEnd(null);
      setIsDrawing(false);
    }
  }, [activeTool, bboxStart, bboxEnd, activeClass, imageWidth, imageHeight, addAnnotation, setIsDrawing]);

  // --- POLYGON Drawing ---
  const handleClick = useCallback(
    (e: any) => {
      if (activeTool === 'polygon') {
        const pos = e.target.getStage()?.getRelativePointerPosition();
        if (!pos) return;
        setPolygonPoints((prev) => [...prev, [pos.x, pos.y]]);
      }

      if (activeTool === 'point') {
        const pos = e.target.getStage()?.getRelativePointerPosition();
        if (!pos) return;
        setPointPrompts((prev) => [...prev, { x: pos.x / imageWidth, y: pos.y / imageHeight, label: 1 }]);
      }
    },
    [activeTool, imageWidth, imageHeight]
  );

  const handleDoubleClick = useCallback(() => {
    if (activeTool === 'polygon' && polygonPoints.length >= 3 && activeClass) {
      addAnnotation({
        class_name: activeClass,
        annotation_type: 'polygon',
        polygon_points: polygonPoints.map(([px, py]) => [px / imageWidth, py / imageHeight]),
      });
      setPolygonPoints([]);
    }
  }, [activeTool, polygonPoints, activeClass, imageWidth, imageHeight, addAnnotation]);

  return (
    <Group
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onClick={handleClick}
      onDblClick={handleDoubleClick}
    >
      {/* Invisible hit area */}
      <Rect width={imageWidth} height={imageHeight} fill="transparent" />

      {/* Bbox preview */}
      {activeTool === 'bbox' && bboxStart && bboxEnd && isDrawing && (
        <Rect
          x={Math.min(bboxStart.x, bboxEnd.x)}
          y={Math.min(bboxStart.y, bboxEnd.y)}
          width={Math.abs(bboxEnd.x - bboxStart.x)}
          height={Math.abs(bboxEnd.y - bboxStart.y)}
          stroke="#3b82f6"
          strokeWidth={2}
          dash={[6, 3]}
          fill="rgba(59, 130, 246, 0.1)"
        />
      )}

      {/* Polygon preview */}
      {activeTool === 'polygon' && polygonPoints.length > 0 && (
        <>
          <Line
            points={polygonPoints.flat()}
            stroke="#10b981"
            strokeWidth={2}
            dash={[6, 3]}
          />
          {polygonPoints.map(([px, py], i) => (
            <Circle key={i} x={px} y={py} radius={5} fill="#10b981" stroke="white" strokeWidth={1} />
          ))}
        </>
      )}

      {/* Point prompts preview */}
      {activeTool === 'point' &&
        pointPrompts.map((pt, i) => (
          <Circle
            key={i}
            x={pt.x * imageWidth}
            y={pt.y * imageHeight}
            radius={6}
            fill={pt.label === 1 ? '#10b981' : '#ef4444'}
            stroke="white"
            strokeWidth={2}
          />
        ))}
    </Group>
  );
}
