/**
 * BoundingBox — Renders and handles interaction for a bbox annotation.
 *
 * - Displays as a rectangle with class label
 * - Draggable when selected
 * - Resizable via Transformer handles
 * - Color-coded by class name
 */

import { useRef, useEffect } from 'react';
import { Rect, Text, Group, Transformer } from 'react-konva';
import type { Annotation } from '../../types';
import { getClassColor } from './colors';
import { useAnnotationStore } from '../../store/useAnnotationStore';

interface BoundingBoxProps {
  annotation: Annotation;
  imageWidth: number;
  imageHeight: number;
  isSelected: boolean;
  onSelect: () => void;
}

export default function BoundingBox({
  annotation,
  imageWidth,
  imageHeight,
  isSelected,
  onSelect,
}: BoundingBoxProps) {
  const shapeRef = useRef<any>(null);
  const transformerRef = useRef<any>(null);
  const { verifyAnnotation, deleteAnnotation, updateAnnotation } = useAnnotationStore();

  const bbox = annotation.bbox!;
  const color = getClassColor(annotation.class_name);

  // Convert normalized center format to absolute top-left format
  const x = (bbox.x - bbox.w / 2) * imageWidth;
  const y = (bbox.y - bbox.h / 2) * imageHeight;
  const w = bbox.w * imageWidth;
  const h = bbox.h * imageHeight;

  useEffect(() => {
    if (isSelected && transformerRef.current && shapeRef.current) {
      transformerRef.current.nodes([shapeRef.current]);
      transformerRef.current.getLayer()?.batchDraw();
    }
  }, [isSelected]);

  const handleDragEnd = (e: any) => {
    const node = e.target;
    const newX = node.x();
    const newY = node.y();

    // Convert absolute top-left back to normalized center
    const normalizedX = (newX + (w / 2)) / imageWidth;
    const normalizedY = (newY + (h / 2)) / imageHeight;

    updateAnnotation(annotation.id, {
      bbox: {
        ...bbox,
        x: normalizedX,
        y: normalizedY,
      },
    });
  };

  const handleTransformEnd = () => {
    const node = shapeRef.current;
    const scaleX = node.scaleX();
    const scaleY = node.scaleY();

    // Reset scale and apply to width/height
    node.scaleX(1);
    node.scaleY(1);

    const newW = Math.max(5, node.width() * scaleX);
    const newH = Math.max(5, node.height() * scaleY);
    const newX = node.x();
    const newY = node.y();

    // Convert back to normalized center format
    const normalizedW = newW / imageWidth;
    const normalizedH = newH / imageHeight;
    const normalizedX = (newX + (newW / 2)) / imageWidth;
    const normalizedY = (newY + (newH / 2)) / imageHeight;

    updateAnnotation(annotation.id, {
      bbox: {
        x: normalizedX,
        y: normalizedY,
        w: normalizedW,
        h: normalizedH,
      },
    });
  };

  const handleTransform = () => {
    // Force redraw of children (labels) while transforming
    shapeRef.current.getLayer()?.batchDraw();
  };

  const sourceLabel = annotation.source !== 'manual' ? ` [${annotation.source}]` : '';
  const confidenceLabel = annotation.confidence ? ` ${(annotation.confidence * 100).toFixed(0)}%` : '';

  return (
    <Group>
      {/* Bounding box rectangle */}
      <Rect
        ref={shapeRef}
        x={x}
        y={y}
        width={w}
        height={h}
        stroke={annotation.is_verified ? color : '#94a3b8'} // Grayish if unverified
        strokeWidth={2}
        dash={annotation.is_verified ? [] : [5, 5]} // Dashed if unverified
        fill={annotation.is_verified ? `${color}15` : 'rgba(255, 255, 255, 0.01)'} // Ensure hit detection
        onClick={onSelect}
        onTap={onSelect}
        draggable={isSelected}
        onDragEnd={handleDragEnd}
        onTransform={handleTransform}
        onTransformEnd={handleTransformEnd}
        onDragMove={handleTransform}
      />

      {/* Class label */}
      <Rect
        x={x}
        y={y - 20}
        width={annotation.class_name.length * 8 + (confidenceLabel.length * 7) + 16}
        height={20}
        fill={annotation.is_verified ? color : '#64748b'}
        cornerRadius={[4, 4, 0, 0]}
      />
      <Text
        x={x + 4}
        y={y - 17}
        text={`${annotation.class_name}${confidenceLabel}${sourceLabel}`}
        fontSize={12}
        fill="white"
        fontStyle="bold"
      />

      {/* Accept/Reject actions for unverified AI annotations */}
      {!annotation.is_verified && (
        <Group x={x} y={y + h + 4}>
          {/* Accept Button */}
          <Group onClick={() => verifyAnnotation(annotation.id)}>
            <Rect width={50} height={20} fill="#22c55e" cornerRadius={4} />
            <Text x={5} y={5} text="Accept" fontSize={11} fill="white" fontStyle="bold" />
          </Group>
          {/* Reject Button */}
          <Group x={55} onClick={() => deleteAnnotation(annotation.id)}>
            <Rect width={50} height={20} fill="#ef4444" cornerRadius={4} />
            <Text x={5} y={5} text="Reject" fontSize={11} fill="white" fontStyle="bold" />
          </Group>
        </Group>
      )}

      {/* Transformer for resize handles */}
      {isSelected && (
        <Transformer
          ref={transformerRef}
          rotateEnabled={false}
          borderStroke={color}
          anchorStroke={color}
          anchorFill="white"
          anchorSize={8}
          borderStrokeWidth={2}
        />
      )}
    </Group>
  );
}
