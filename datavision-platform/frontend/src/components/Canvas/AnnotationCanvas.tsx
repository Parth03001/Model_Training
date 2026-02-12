/**
 * AnnotationCanvas — Main canvas component using React Konva.
 *
 * Architecture:
 * - Stage (outermost) handles pan/zoom via wheel events
 * - Layer 1: Background image (separate layer = no redraw when annotations change)
 * - Layer 2: Annotations (bounding boxes, polygons, masks)
 * - Layer 3: Active drawing (current shape being drawn)
 *
 * Performance:
 * - Image layer is separate, so annotation updates don't redraw the image
 * - Uses batchDraw() for mouse move events (limits to browser refresh rate)
 * - Shapes use caching for complex polygons
 */

import { useRef, useCallback, useEffect } from 'react';
import { Stage, Layer, Image as KonvaImage } from 'react-konva';
import useImage from 'use-image';
import { useAnnotationStore } from '../../store/useAnnotationStore';
import BoundingBox from './BoundingBox';
import PolygonShape from './PolygonShape';
import DrawingLayer from './DrawingLayer';

interface AnnotationCanvasProps {
  width: number;
  height: number;
}

export default function AnnotationCanvas({ width, height }: AnnotationCanvasProps) {
  const stageRef = useRef<any>(null);
  const {
    currentImage,
    annotations,
    selectedAnnotationId,
    activeTool,
    canvasScale,
    canvasOffset,
    setCanvasScale,
    setCanvasOffset,
    selectAnnotation,
  } = useAnnotationStore();

  const imageUrl = currentImage ? currentImage.filepath : '';
  const [image] = useImage(imageUrl, 'anonymous');

  // Calculate image fit
  const imgWidth = currentImage?.width || 1;
  const imgHeight = currentImage?.height || 1;
  const scaleX = width / imgWidth;
  const scaleY = height / imgHeight;
  const fitScale = Math.min(scaleX, scaleY) * 0.9;

  // Wheel zoom
  const handleWheel = useCallback(
    (e: any) => {
      e.evt.preventDefault();
      const stage = stageRef.current;
      if (!stage) return;

      const oldScale = canvasScale;
      const pointer = stage.getPointerPosition();
      const mousePointTo = {
        x: (pointer.x - canvasOffset.x) / oldScale,
        y: (pointer.y - canvasOffset.y) / oldScale,
      };

      const direction = e.evt.deltaY > 0 ? -1 : 1;
      const newScale = direction > 0 ? oldScale * 1.1 : oldScale / 1.1;

      setCanvasScale(newScale);
      setCanvasOffset({
        x: pointer.x - mousePointTo.x * newScale,
        y: pointer.y - mousePointTo.y * newScale,
      });
    },
    [canvasScale, canvasOffset, setCanvasScale, setCanvasOffset]
  );

  // Click on empty space deselects
  const handleStageClick = useCallback(
    (e: any) => {
      if (e.target === e.target.getStage()) {
        selectAnnotation(null);
      }
    },
    [selectAnnotation]
  );

  // Reset view when image changes
  useEffect(() => {
    setCanvasScale(fitScale);
    setCanvasOffset({
      x: (width - imgWidth * fitScale) / 2,
      y: (height - imgHeight * fitScale) / 2,
    });
  }, [currentImage?.id]);

  const bboxAnnotations = annotations.filter((a) => a.annotation_type === 'bbox' && a.bbox);
  const polygonAnnotations = annotations.filter((a) => a.annotation_type === 'polygon' && a.polygon_points);

  return (
    <Stage
      ref={stageRef}
      width={width}
      height={height}
      scaleX={canvasScale}
      scaleY={canvasScale}
      x={canvasOffset.x}
      y={canvasOffset.y}
      onWheel={handleWheel}
      onClick={handleStageClick}
      className={
        activeTool === 'pan'
          ? 'cursor-grab'
          : activeTool === 'select'
          ? 'cursor-default'
          : 'cursor-crosshair'
      }
    >
      {/* Layer 1: Background Image */}
      <Layer listening={false}>
        {image && (
          <KonvaImage image={image} width={imgWidth} height={imgHeight} />
        )}
      </Layer>

      {/* Layer 2: Existing Annotations */}
      <Layer>
        {bboxAnnotations.map((ann) => (
          <BoundingBox
            key={ann.id}
            annotation={ann}
            imageWidth={imgWidth}
            imageHeight={imgHeight}
            isSelected={ann.id === selectedAnnotationId}
            onSelect={() => selectAnnotation(ann.id)}
          />
        ))}
        {polygonAnnotations.map((ann) => (
          <PolygonShape
            key={ann.id}
            annotation={ann}
            imageWidth={imgWidth}
            imageHeight={imgHeight}
            isSelected={ann.id === selectedAnnotationId}
            onSelect={() => selectAnnotation(ann.id)}
          />
        ))}
      </Layer>

      {/* Layer 3: Active Drawing */}
      <Layer>
        <DrawingLayer imageWidth={imgWidth} imageHeight={imgHeight} />
      </Layer>
    </Stage>
  );
}
