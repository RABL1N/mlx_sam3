"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import type { SegmentationResult, RLEMask } from "@/lib/api";

interface Point {
  x: number;
  y: number;
  label: boolean; // true for positive, false for negative
}

interface Props {
  imageUrl: string | null;
  imageWidth: number;
  imageHeight: number;
  result: SegmentationResult | null;
  boxMode: "positive" | "negative" | null; // null means box mode is off
  pointMode: "positive" | "negative" | null; // null means point mode is off
  onBoxDrawn: (box: number[]) => void;
  onPointClicked?: (point: number[], label: boolean) => void; // [x, y] normalized
  isLoading: boolean;
}

// Color palette for masks
const COLORS = [
  [59, 235, 161], // Emerald
  [96, 165, 250], // Blue
  [251, 191, 36], // Amber
  [248, 113, 113], // Red
  [167, 139, 250], // Violet
  [52, 211, 153], // Green
  [251, 146, 60], // Orange
  [147, 197, 253], // Light Blue
];

/**
 * Decode RLE mask to ImageData for canvas rendering.
 * Returns an ImageData object with the mask color applied.
 */
function decodeRLEToImageData(rle: RLEMask, color: number[]): ImageData | null {
  const [height, width] = rle.size;
  if (height === 0 || width === 0) return null;

  const imageData = new ImageData(width, height);
  const data = imageData.data;
  const { counts } = rle;

  let pixelIdx = 0;
  let isForeground = false; // RLE starts with background count

  for (const count of counts) {
    if (isForeground) {
      // Fill foreground pixels with color
      for (let j = 0; j < count && pixelIdx < width * height; j++) {
        const idx = pixelIdx * 4;
        data[idx] = color[0];
        data[idx + 1] = color[1];
        data[idx + 2] = color[2];
        data[idx + 3] = 255; // Opaque
        pixelIdx++;
      }
    } else {
      // Skip background pixels (already transparent)
      pixelIdx += count;
    }
    isForeground = !isForeground;
  }

  return imageData;
}

export function SegmentationCanvas({
  imageUrl,
  imageWidth,
  imageHeight,
  result,
  boxMode,
  pointMode,
  onBoxDrawn,
  onPointClicked,
  isLoading,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);

  const [isDrawing, setIsDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState<{ x: number; y: number } | null>(
    null
  );
  const [currentPoint, setCurrentPoint] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [displayScale, setDisplayScale] = useState(1);
  const [points, setPoints] = useState<Point[]>([]);

  // Calculate display scale to fit image in container
  useEffect(() => {
    if (!containerRef.current || !imageWidth || !imageHeight) return;

    const containerWidth = containerRef.current.clientWidth;
    const maxHeight = window.innerHeight * 0.7;

    const scaleX = containerWidth / imageWidth;
    const scaleY = maxHeight / imageHeight;
    const scale = Math.min(scaleX, scaleY, 1);

    setDisplayScale(scale);
  }, [imageWidth, imageHeight]);

  // Load and draw image
  useEffect(() => {
    if (!imageUrl || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      imageRef.current = img;
      drawCanvas();
    };
    img.src = imageUrl;
  }, [imageUrl]);

  // Clear points when point mode is disabled or image changes
  useEffect(() => {
    if (pointMode === null) {
      setPoints([]);
    }
  }, [pointMode, imageUrl]);

  // Sync local points with backend prompted_points when result changes
  useEffect(() => {
    if (result?.prompted_points && imageWidth > 0 && imageHeight > 0) {
      const syncedPoints: Point[] = result.prompted_points.map((pp) => ({
        x: pp.point[0],
        y: pp.point[1],
        label: pp.label,
      }));
      setPoints(syncedPoints);
    } else if (result && (!result.prompted_points || result.prompted_points.length === 0)) {
      // If no prompted_points in result, clear local points
      setPoints([]);
    }
  }, [result?.prompted_points, imageWidth, imageHeight]);

  // Redraw when result changes
  const drawCanvas = useCallback(() => {
    if (!canvasRef.current || !imageRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Use integer dimensions for canvas
    const displayWidth = Math.floor(imageWidth * displayScale);
    const displayHeight = Math.floor(imageHeight * displayScale);

    canvas.width = displayWidth;
    canvas.height = displayHeight;

    // Clear canvas
    ctx.clearRect(0, 0, displayWidth, displayHeight);

    // Draw image
    ctx.drawImage(imageRef.current, 0, 0, displayWidth, displayHeight);

    // Draw masks with semi-transparency using RLE decoding + canvas compositing
    if (result?.masks && result.masks.length > 0) {
      // Category name mapping
      const categoryNames: Record<number, string> = {
        1: "aspergillus",
        2: "penicillium",
        3: "rhizopus",
        4: "mucor",
        5: "other_fungus",
      };
      
      for (let i = 0; i < result.masks.length; i++) {
        const mask = result.masks[i];
        const box = result.boxes?.[i];
        const categoryId = result.category_ids?.[i] ?? null;
        const categoryName = categoryId ? categoryNames[categoryId] || `Cat ${categoryId}` : null;
        const color = COLORS[i % COLORS.length];

        // Decode RLE mask and draw
        if (mask && mask.counts && mask.size) {
          const maskImageData = decodeRLEToImageData(mask, color);
          if (!maskImageData) continue;

          const [maskH, maskW] = mask.size;

          // Create offscreen canvas at mask resolution
          const offscreen = document.createElement("canvas");
          offscreen.width = maskW;
          offscreen.height = maskH;
          const offCtx = offscreen.getContext("2d");
          if (!offCtx) continue;

          // Put decoded mask to offscreen canvas
          offCtx.putImageData(maskImageData, 0, 0);

          // Composite onto main canvas with transparency (GPU-accelerated scaling & blending)
          ctx.globalAlpha = 0.5;
          ctx.drawImage(offscreen, 0, 0, displayWidth, displayHeight);
          ctx.globalAlpha = 1.0;
        }

        // Draw bounding box
        if (box) {
          const [x0, y0, x1, y1] = box;
          ctx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
          ctx.lineWidth = 2;
          ctx.strokeRect(
            x0 * displayScale,
            y0 * displayScale,
            (x1 - x0) * displayScale,
            (y1 - y0) * displayScale
          );

          // Draw category label (or "None" if no category)
          // Set font before measuring text to ensure accurate width calculation
          ctx.font = "bold 12px JetBrains Mono, monospace";
          const labelText = categoryName || "None";
          const textWidth = ctx.measureText(labelText).width;
          const labelWidth = Math.max(textWidth + 8, 50);
          
          ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
          ctx.fillRect(x0 * displayScale, y0 * displayScale - 24, labelWidth, 20);
          ctx.fillStyle = "#000";
          ctx.fillText(
            labelText,
            x0 * displayScale + 4,
            y0 * displayScale - 8
          );
        }
      }
    }

    // Draw points (positive = green, negative = red)
    points.forEach((point) => {
      const x = point.x * displayScale;
      const y = point.y * displayScale;
      const radius = 8;
      
      // Draw outer circle
      ctx.fillStyle = point.label ? "rgba(34, 197, 94, 0.3)" : "rgba(239, 68, 68, 0.3)";
      ctx.beginPath();
      ctx.arc(x, y, radius + 4, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw inner circle
      ctx.fillStyle = point.label ? "#22c55e" : "#ef4444";
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw border
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.stroke();
    });

    // Draw prompted boxes
    if (result?.prompted_boxes) {
      for (const promptedBox of result.prompted_boxes) {
        const [x0, y0, x1, y1] = promptedBox.box;
        ctx.strokeStyle = promptedBox.label ? "#3beba1" : "#f87171";
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(
          x0 * displayScale,
          y0 * displayScale,
          (x1 - x0) * displayScale,
          (y1 - y0) * displayScale
        );
        ctx.setLineDash([]);
      }
    }

    // Draw prompted points from backend (these are synced with local points via useEffect)
    // The local points state is already being drawn above, so this ensures consistency

    // Draw current drawing box
    if (isDrawing && startPoint && currentPoint && boxMode !== null) {
      const x = Math.min(startPoint.x, currentPoint.x);
      const y = Math.min(startPoint.y, currentPoint.y);
      const width = Math.abs(currentPoint.x - startPoint.x);
      const height = Math.abs(currentPoint.y - startPoint.y);

      ctx.strokeStyle = boxMode === "positive" ? "#3beba1" : "#f87171";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(x, y, width, height);
      ctx.setLineDash([]);
    }
  }, [
    imageWidth,
    imageHeight,
    displayScale,
    result,
    isDrawing,
    startPoint,
    currentPoint,
    boxMode,
    points,
  ]);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  const getCanvasCoordinates = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;

    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isLoading) return;
    
    // Handle point clicks
    if (pointMode !== null && onPointClicked) {
      const coords = getCanvasCoordinates(e);
      if (coords) {
        // Check if clicking on an existing point (to remove it)
        const clickRadius = 12;
        const clickedPointIndex = points.findIndex((p) => {
          const px = p.x * displayScale;
          const py = p.y * displayScale;
          const dist = Math.sqrt(
            Math.pow(coords.x - px, 2) + Math.pow(coords.y - py, 2)
          );
          return dist < clickRadius;
        });

        if (clickedPointIndex !== -1) {
          // Remove the clicked point
          setPoints((prev) => prev.filter((_, i) => i !== clickedPointIndex));
          return;
        }

        // Add new point
        const normalizedX = coords.x / displayScale / imageWidth;
        const normalizedY = coords.y / displayScale / imageHeight;
        const newPoint: Point = {
          x: normalizedX * imageWidth,
          y: normalizedY * imageHeight,
          label: pointMode === "positive",
        };
        setPoints((prev) => [...prev, newPoint]);
        onPointClicked([normalizedX, normalizedY], pointMode === "positive");
      }
      return;
    }

    // Handle box drawing (only if box mode is enabled and point mode is disabled)
    if (boxMode !== null && pointMode === null) {
      const coords = getCanvasCoordinates(e);
      if (coords) {
        setIsDrawing(true);
        setStartPoint(coords);
        setCurrentPoint(coords);
      }
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (pointMode !== null || boxMode === null) return; // Don't draw boxes if point mode is active or box mode is disabled
    if (!isDrawing) return;
    const coords = getCanvasCoordinates(e);
    if (coords) {
      setCurrentPoint(coords);
    }
  };

  const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (pointMode !== null || boxMode === null) return; // Don't handle box drawing if point mode is active or box mode is disabled
    if (!isDrawing || !startPoint) {
      setIsDrawing(false);
      return;
    }

    const coords = getCanvasCoordinates(e);
    if (!coords) {
      setIsDrawing(false);
      return;
    }

    // Calculate box in original image coordinates
    const x0 = Math.min(startPoint.x, coords.x) / displayScale;
    const y0 = Math.min(startPoint.y, coords.y) / displayScale;
    const x1 = Math.max(startPoint.x, coords.x) / displayScale;
    const y1 = Math.max(startPoint.y, coords.y) / displayScale;

    // Minimum box size check
    if (Math.abs(x1 - x0) < 10 || Math.abs(y1 - y0) < 10) {
      setIsDrawing(false);
      setStartPoint(null);
      setCurrentPoint(null);
      return;
    }

    // Convert to normalized center x, center y, width, height format
    const centerX = (x0 + x1) / 2 / imageWidth;
    const centerY = (y0 + y1) / 2 / imageHeight;
    const width = (x1 - x0) / imageWidth;
    const height = (y1 - y0) / imageHeight;

    onBoxDrawn([centerX, centerY, width, height]);

    setIsDrawing(false);
    setStartPoint(null);
    setCurrentPoint(null);
  };

  const handleMouseLeave = () => {
    if (isDrawing) {
      setIsDrawing(false);
      setStartPoint(null);
      setCurrentPoint(null);
    }
  };

  if (!imageUrl) {
    return (
      <div
        ref={containerRef}
        className="flex items-center justify-center h-96 border-2 border-dashed border-border rounded-xl bg-card/50"
      >
        <p className="text-muted-foreground text-sm">
          Upload an image to begin segmentation
        </p>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="relative">
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        className={`rounded-lg shadow-xl ${
          isLoading ? "opacity-50 pointer-events-none" : ""
        }`}
        style={{
          cursor: isLoading
            ? "wait"
            : pointMode !== null
            ? "crosshair"
            : "crosshair",
        }}
      />
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="flex items-center gap-3 bg-card/90 backdrop-blur-sm px-4 py-2 rounded-lg border border-border">
            <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
            <span className="text-sm">Processing...</span>
          </div>
        </div>
      )}
    </div>
  );
}
