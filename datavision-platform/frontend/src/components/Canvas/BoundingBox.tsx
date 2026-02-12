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
        stroke={color}
        strokeWidth={2}
        fill={`${color}15`}
        onClick={onSelect}
        onTap={onSelect}
        draggable={isSelected}
      />

      {/* Class label */}
      <Rect
        x={x}
        y={y - 20}
        width={annotation.class_name.length * 8 + (confidenceLabel.length * 7) + 16}
        height={20}
        fill={color}
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
