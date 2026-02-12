/**
 * PolygonShape — Renders polygon annotations on the canvas.
 */

import { Line, Circle, Group, Text, Rect } from 'react-konva';
import type { Annotation } from '../../types';
import { getClassColor } from './colors';

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
        fill={`${color}20`}
        stroke={color}
        strokeWidth={isSelected ? 3 : 2}
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
        fill={color}
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
    </Group>
  );
}
