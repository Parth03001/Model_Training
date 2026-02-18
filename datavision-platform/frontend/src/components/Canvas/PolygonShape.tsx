/**
 * PolygonShape — Renders polygon annotations on the canvas.
 */

import { Line, Circle, Group, Text, Rect } from 'react-konva';
import type { Annotation } from '../../types';
import { getClassColor } from './colors';
import { useAnnotationStore } from '../../store/useAnnotationStore';

interface PolygonShapeProps {
  annotation: Annotation;
  imageWidth: number;
  imageHeight: number;
  isSelected: boolean;
  onSelect: () => void;
}

export default function PolygonShape({
  annotation,
  imageWidth,
  imageHeight,
  isSelected,
  onSelect,
}: PolygonShapeProps) {
  const { verifyAnnotation, deleteAnnotation } = useAnnotationStore();
  const points = annotation.polygon_points!;
  const color = getClassColor(annotation.class_name);

  // Convert normalized points to absolute
  const flatPoints = points.flatMap(([px, py]) => [px * imageWidth, py * imageHeight]);

  // Calculate centroid for label
  const cx = points.reduce((s, [px]) => s + px, 0) / points.length * imageWidth;
  const cy = points.reduce((s, [, py]) => s + py, 0) / points.length * imageHeight;

  return (
    <Group onClick={onSelect} onTap={onSelect}>
      {/* Polygon fill */}
      <Line
        points={flatPoints}
        closed
        fill={annotation.is_verified ? `${color}20` : 'transparent'}
        stroke={annotation.is_verified ? color : '#94a3b8'}
        strokeWidth={isSelected ? 3 : 2}
        dash={annotation.is_verified ? [] : [5, 5]}
      />

      {/* Vertex circles (when selected) */}
      {isSelected &&
        points.map(([px, py], i) => (
          <Circle
            key={i}
            x={px * imageWidth}
            y={py * imageHeight}
            radius={4}
            fill="white"
            stroke={color}
            strokeWidth={2}
            draggable
          />
        ))}

      {/* Label */}
      <Rect
        x={cx - 20}
        y={cy - 10}
        width={annotation.class_name.length * 8 + 12}
        height={18}
        fill={annotation.is_verified ? color : '#64748b'}
        cornerRadius={4}
        opacity={0.9}
      />
      <Text
        x={cx - 14}
        y={cy - 7}
        text={annotation.class_name}
        fontSize={11}
        fill="white"
        fontStyle="bold"
      />

      {/* Accept/Reject actions for unverified AI annotations */}
      {!annotation.is_verified && (
        <Group x={cx - 50} y={cy + 12}>
          <Group onClick={() => verifyAnnotation(annotation.id)}>
            <Rect width={50} height={20} fill="#22c55e" cornerRadius={4} />
            <Text x={5} y={5} text="Accept" fontSize={11} fill="white" fontStyle="bold" />
          </Group>
          <Group x={55} onClick={() => deleteAnnotation(annotation.id)}>
            <Rect width={50} height={20} fill="#ef4444" cornerRadius={4} />
            <Text x={5} y={5} text="Reject" fontSize={11} fill="white" fontStyle="bold" />
          </Group>
        </Group>
      )}
    </Group>
  );
}
