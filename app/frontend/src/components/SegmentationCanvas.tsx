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
  onInstanceClick?: (index: number) => void; // Called when an instance is clicked
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

/**
 * Check if a normalized point [x, y] (0-1 range) is within an RLE mask.
 * The point is in normalized image coordinates [0, 1], and we need to convert
 * it to mask pixel coordinates based on the mask's size.
 */
function isPointInRLEMask(
  point: [number, number],
  rle: RLEMask
): boolean {
  const [maskHeight, maskWidth] = rle.size;
  if (maskHeight === 0 || maskWidth === 0) return false;

  // Convert normalized point [0, 1] to mask pixel coordinates
  // The point is already normalized to image coordinates, so we can directly
  // scale it by the mask dimensions
  const maskX = Math.floor(point[0] * maskWidth);
  const maskY = Math.floor(point[1] * maskHeight);

  // Clamp to mask bounds
  if (maskX < 0 || maskX >= maskWidth || maskY < 0 || maskY >= maskHeight) {
    return false;
  }

  // Decode RLE to check the specific pixel
  // Calculate the target pixel index once
  const targetIdx = maskY * maskWidth + maskX;
  let counts: number[];
  
  // Handle both array and string formats for counts
  if (typeof rle.counts === 'string') {
    counts = rle.counts.split(' ').map(Number).filter(n => !isNaN(n));
  } else {
    counts = rle.counts;
  }

  let pixelIdx = 0;
  let isForeground = false; // RLE starts with background count

  for (const count of counts) {
    const endIdx = pixelIdx + count;

    // Check if target pixel is within this run
    if (targetIdx >= pixelIdx && targetIdx < endIdx) {
      return isForeground;
    }

    // Move to next run
    pixelIdx = endIdx;
    isForeground = !isForeground;
  }

  return false;
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
  onInstanceClick,
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
  const [hoveredInstanceIndex, setHoveredInstanceIndex] = useState<number | null>(null);

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
  // Backend sends normalized coordinates [0, 1], convert to pixel coordinates
  useEffect(() => {
    if (result?.prompted_points && imageWidth > 0 && imageHeight > 0) {
      const syncedPoints: Point[] = result.prompted_points.map((pp) => ({
        x: pp.point[0] * imageWidth,  // Convert normalized x [0,1] to pixel x
        y: pp.point[1] * imageHeight, // Convert normalized y [0,1] to pixel y
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
        const isHovered = hoveredInstanceIndex === i;

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
          // Add pulse animation when hovered
          if (isHovered) {
            // Pulse effect: oscillate opacity between 0.5 and 0.7 (subtle)
            const time = Date.now() / 1000; // Current time in seconds
            const pulseSpeed = 0.8; // Speed of pulse (cycles per second) - slower for subtlety
            const pulseRange = 0.2; // Range of opacity change (smaller for subtlety)
            const baseOpacity = 0.5;
            const pulseOpacity = baseOpacity + (Math.sin(time * pulseSpeed * Math.PI * 2) * pulseRange + pulseRange) / 2;
            ctx.globalAlpha = pulseOpacity;
          } else {
            ctx.globalAlpha = 0.5;
          }
          ctx.drawImage(offscreen, 0, 0, displayWidth, displayHeight);
          ctx.globalAlpha = 1.0;
        }

        // Draw bounding box (no hover effect on border, pulse is on mask)
        // Boxes are in pixel coordinates, need to normalize then scale
        if (box) {
          const [x0_px, y0_px, x1_px, y1_px] = box;
          // Normalize from pixel coordinates to [0, 1]
          const originalWidth = result.original_width || imageWidth;
          const originalHeight = result.original_height || imageHeight;
          const x0 = x0_px / originalWidth;
          const y0 = y0_px / originalHeight;
          const x1 = x1_px / originalWidth;
          const y1 = y1_px / originalHeight;
          
          // Standard border (no hover effect)
          ctx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
          ctx.lineWidth = 2;
          ctx.strokeRect(
            x0 * displayWidth,
            y0 * displayHeight,
            (x1 - x0) * displayWidth,
            (y1 - y0) * displayHeight
          );

          // Draw category label (or "None" if no category)
          // Set font before measuring text to ensure accurate width calculation
          ctx.font = "bold 12px JetBrains Mono, monospace";
          const labelText = categoryName || "None";
          const textWidth = ctx.measureText(labelText).width;
          const labelWidth = Math.max(textWidth + 8, 50);
          
          ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
          ctx.fillRect(x0 * displayWidth, y0 * displayHeight - 24, labelWidth, 20);
          ctx.fillStyle = "#000";
          ctx.fillText(
            labelText,
            x0 * displayWidth + 4,
            y0 * displayHeight - 8
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
    hoveredInstanceIndex,
  ]);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  // Animation loop for pulse effect when hovering
  useEffect(() => {
    if (hoveredInstanceIndex === null) return;
    
    let animationFrameId: number;
    const animate = () => {
      drawCanvas();
      animationFrameId = requestAnimationFrame(animate);
    };
    animate();
    
    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [hoveredInstanceIndex, drawCanvas]);

  const getCanvasCoordinates = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;

    const rect = canvas.getBoundingClientRect();
    // Account for CSS scaling if canvas display size differs from internal size
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const getInstanceAtPoint = (coords: { x: number; y: number }): number | null => {
    if (!result || !result.masks || result.masks.length === 0) {
      return null;
    }
    
    const canvas = canvasRef.current;
    if (!canvas) {
      return null;
    }
    
    const displayWidth = canvas.width;
    const displayHeight = canvas.height;
    const normalizedX = coords.x / displayWidth;
    const normalizedY = coords.y / displayHeight;
    
    // Get original image dimensions for normalizing bounding boxes
    const originalWidth = result.original_width || imageWidth;
    const originalHeight = result.original_height || imageHeight;
    
    // Check instances in reverse order (last drawn = top layer)
    for (let i = result.masks.length - 1; i >= 0; i--) {
      const box = result.boxes?.[i];
      if (box) {
        const [x0_px, y0_px, x1_px, y1_px] = box;
        // Normalize bounding box coordinates from pixels to [0, 1]
        const x0 = x0_px / originalWidth;
        const y0 = y0_px / originalHeight;
        const x1 = x1_px / originalWidth;
        const y1 = y1_px / originalHeight;
        
        if (normalizedX >= x0 && normalizedX <= x1 && normalizedY >= y0 && normalizedY <= y1) {
          return i;
        }
      }
    }
    return null;
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isLoading) return;
    
    const coords = getCanvasCoordinates(e);
    if (!coords) return;

    // Handle point clicks
    if (pointMode !== null && onPointClicked) {
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
      return;
    }

    // Handle box drawing (only if box mode is enabled and point mode is disabled)
    if (boxMode !== null && pointMode === null) {
      setIsDrawing(true);
      setStartPoint(coords);
      setCurrentPoint(coords);
      return;
    }

    // Instance selection is handled by onClick handler
    // Don't handle it here to avoid conflicts with box/point modes
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    // Handle box drawing
    if (boxMode !== null && pointMode === null && isDrawing) {
      const coords = getCanvasCoordinates(e);
      if (coords) {
        setCurrentPoint(coords);
      }
      return;
    }
    
    // Handle hover for instance selection (when no modes are active)
    if (boxMode === null && pointMode === null && result && result.masks && result.masks.length > 0) {
      const coords = getCanvasCoordinates(e);
      if (coords) {
        const instanceIndex = getInstanceAtPoint(coords);
        setHoveredInstanceIndex(instanceIndex);
      } else {
        setHoveredInstanceIndex(null);
      }
    } else {
      setHoveredInstanceIndex(null);
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
    setHoveredInstanceIndex(null);
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


  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    console.log("=== handleCanvasClick ===", {
      boxMode,
      pointMode,
      isLoading,
      hasOnInstanceClick: !!onInstanceClick,
      hasResult: !!result,
      numMasks: result?.masks?.length || 0,
    });
    
    // Don't prevent default - let mouseDown handle box/point modes
    // Only handle instance selection here when no modes are active
    if (boxMode !== null || pointMode !== null || isLoading) {
      console.log("  -> Blocked: mode active or loading");
      return; // Let mouseDown handle it
    }
    
    if (!onInstanceClick || !result || !result.masks || result.masks.length === 0) {
      console.log("  -> Cannot select instance:", {
        hasOnInstanceClick: !!onInstanceClick,
        hasResult: !!result,
        hasMasks: !!(result && result.masks),
        numMasks: result?.masks?.length || 0,
      });
      return;
    }
    
    const coords = getCanvasCoordinates(e);
    if (!coords) {
      console.log("No coords from click");
      return;
    }
    
    const canvas = canvasRef.current;
    if (!canvas) {
      console.log("No canvas ref");
      return;
    }
    
    const displayWidth = canvas.width;
    const displayHeight = canvas.height;
    const normalizedX = coords.x / displayWidth;
    const normalizedY = coords.y / displayHeight;
    
    console.log("Checking for instance at:", { 
      normalizedX, 
      normalizedY, 
      coords,
      displayWidth,
      displayHeight,
      numMasks: result.masks.length 
    });
    
    // Check instances in reverse order (last drawn = top layer)
    for (let i = result.masks.length - 1; i >= 0; i--) {
      const mask = result.masks[i];
      const box = result.boxes?.[i];
      
      // Use bounding box for selection (simpler and more reliable)
      if (box) {
        const [x0_px, y0_px, x1_px, y1_px] = box;
        // Normalize bounding box coordinates from pixels to [0, 1]
        const originalWidth = result.original_width || imageWidth;
        const originalHeight = result.original_height || imageHeight;
        const x0 = x0_px / originalWidth;
        const y0 = y0_px / originalHeight;
        const x1 = x1_px / originalWidth;
        const y1 = y1_px / originalHeight;
        
        console.log(`Checking instance ${i}: bbox pixels [${x0_px}, ${y0_px}, ${x1_px}, ${y1_px}], normalized [${x0.toFixed(3)}, ${y0.toFixed(3)}, ${x1.toFixed(3)}, ${y1.toFixed(3)}], click [${normalizedX.toFixed(3)}, ${normalizedY.toFixed(3)}]`);
        
        // Check if click is inside bounding box
        if (normalizedX >= x0 && normalizedX <= x1 && normalizedY >= y0 && normalizedY <= y1) {
          console.log(`  -> ✓ INSIDE bbox! Selecting instance ${i}`);
          try {
            console.log(`  -> Calling onInstanceClick(${i})...`);
            onInstanceClick(i);
            console.log(`  -> ✓ Callback called successfully`);
            e.stopPropagation();
            e.preventDefault();
            return;
          } catch (error) {
            console.error(`  -> ✗ Error calling onInstanceClick:`, error);
          }
        } else {
          console.log(`  -> Outside bbox`);
        }
      } else {
        console.log(`Instance ${i} has no bbox`);
      }
    }
    console.log("No instance found at click location");
  };

  return (
    <div ref={containerRef} className="relative">
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onClick={handleCanvasClick}
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
            : boxMode !== null
            ? "crosshair"
            : hoveredInstanceIndex !== null
            ? "pointer"
            : "default",
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
