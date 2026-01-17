"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import { Upload, Download, Loader2 } from "lucide-react";
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
  showMasksAndPrompts?: boolean; // Whether to show masks and prompts (default: true)
  selectedInstanceIndex?: number | null; // Currently selected instance index from sidebar
  onFileSelect?: (file: File) => void; // Called when a file is selected
  onDrop?: (e: React.DragEvent) => void; // Called when a file is dropped
  fileInputRef?: React.RefObject<HTMLInputElement | null>; // Ref to the file input
  // Session loading props
  availableSessions?: Array<{ session_folder: string; timestamp: string; image_filename: string; num_instances: number }>;
  selectedSessionFolder?: string;
  onSessionFolderChange?: (folder: string) => void;
  onLoadSession?: () => void;
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
  showMasksAndPrompts = true,
  selectedInstanceIndex = null,
  onFileSelect,
  onDrop,
  fileInputRef,
  availableSessions = [],
  selectedSessionFolder = "",
  onSessionFolderChange,
  onLoadSession,
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
  const [pulsingInstanceIndex, setPulsingInstanceIndex] = useState<number | null>(null);
  const [pulseStartTime, setPulseStartTime] = useState<number | null>(null);
  const justDrewBoxRef = useRef(false);

  // Calculate display scale to fit image in container responsively
  // Canvas must always fit within viewport, no minimum size constraint
  useEffect(() => {
    if (!containerRef.current || !imageWidth || !imageHeight) return;

    const updateScale = () => {
      if (!containerRef.current) return;
      
      const containerWidth = containerRef.current.clientWidth;
      // Use available viewport height minus header, padding, and other UI elements
      const availableHeight = window.innerHeight - 200; // Account for header, padding, etc.
      const containerHeight = Math.max(availableHeight, 400); // Minimum 400px but prefer available space
      
      // Calculate scale to fit container - always fit, no minimum size
      const scaleX = containerWidth / imageWidth;
      const scaleY = containerHeight / imageHeight;
      
      // Use the smaller scale to ensure it fits both dimensions
      let scale = Math.min(scaleX, scaleY, 1);
      
      // Never upscale beyond original size
      scale = Math.min(scale, 1);

      setDisplayScale(scale);
    };

    updateScale();
    
    // Listen for window resize to update scale
    window.addEventListener('resize', updateScale);
    
    // Also use ResizeObserver for container size changes
    const resizeObserver = new ResizeObserver(updateScale);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }
    
    return () => {
      window.removeEventListener('resize', updateScale);
      resizeObserver.disconnect();
    };
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
    if (showMasksAndPrompts && result?.masks && result.masks.length > 0) {
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
        const isPulsing = pulsingInstanceIndex === i;

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
          // Add pulse animation when hovered or selected
          if (isHovered || isPulsing) {
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

    // Draw points (positive = green, negative = red) - only if masks are visible
    if (showMasksAndPrompts) {
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
    }

    // Draw prompted boxes - only if masks are visible
    if (showMasksAndPrompts && result?.prompted_boxes) {
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
    pulsingInstanceIndex,
    pulseStartTime,
    showMasksAndPrompts,
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

  // Trigger pulse animation when instance is selected in sidebar
  useEffect(() => {
    if (selectedInstanceIndex !== null && selectedInstanceIndex !== undefined) {
      setPulsingInstanceIndex(selectedInstanceIndex);
      setPulseStartTime(Date.now());
    } else {
      setPulsingInstanceIndex(null);
      setPulseStartTime(null);
    }
  }, [selectedInstanceIndex]);

  // Animation loop for selection pulse (one cycle)
  useEffect(() => {
    if (pulsingInstanceIndex === null || pulseStartTime === null) return;
    
    const pulseSpeed = 0.8; // Same speed as hover pulse
    const pulseDuration = 1000 / pulseSpeed; // Duration for one cycle (in ms)
    
    let animationFrameId: number;
    const animate = () => {
      const elapsed = Date.now() - pulseStartTime;
      if (elapsed >= pulseDuration) {
        // One cycle complete, stop animation
        setPulsingInstanceIndex(null);
        setPulseStartTime(null);
        drawCanvas(); // Final draw without pulse
        return;
      }
      drawCanvas();
      animationFrameId = requestAnimationFrame(animate);
    };
    animate();
    
    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [pulsingInstanceIndex, pulseStartTime, drawCanvas]);

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
      // Mark that we just added a point to prevent click event from interfering
      justDrewBoxRef.current = true;
      onPointClicked([normalizedX, normalizedY], pointMode === "positive");
      // Reset the flag after a short delay
      setTimeout(() => {
        justDrewBoxRef.current = false;
      }, 100);
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

    // Mark that we just drew a box to prevent click event from interfering
    justDrewBoxRef.current = true;
    onBoxDrawn([centerX, centerY, width, height]);

    setIsDrawing(false);
    setStartPoint(null);
    setCurrentPoint(null);
    
    // Reset the flag after a short delay to allow click event to be ignored
    setTimeout(() => {
      justDrewBoxRef.current = false;
    }, 100);
    
    // Prevent click event from firing
    e.stopPropagation();
    e.preventDefault();
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
      <>
        <div
          ref={containerRef}
          onClick={() => fileInputRef?.current?.click()}
          onDrop={onDrop}
          onDragOver={(e) => {
            e.preventDefault();
            e.stopPropagation();
          }}
          className="flex items-center justify-center min-h-[500px] border-2 border-dashed border-border rounded-xl bg-card/50 relative cursor-pointer hover:bg-primary/5 transition-all"
        >
          {isLoading && (
            <div className="absolute inset-0 bg-background/80 backdrop-blur-sm rounded-xl flex items-center justify-center z-50">
              <div className="flex flex-col items-center gap-4">
                <Loader2 className="w-12 h-12 text-primary animate-spin" />
                <p className="text-sm font-medium text-foreground">Uploading image...</p>
              </div>
            </div>
          )}
          <div
            className="flex flex-col items-center justify-center p-12 text-center w-full h-full pointer-events-none"
          >
            <Upload className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
            <p className="text-lg font-medium mb-2">Upload an image to begin segmentation</p>
            <p className="text-sm text-muted-foreground mb-4">
              Click or drag and drop an image here
            </p>
            {fileInputRef && (
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file && onFileSelect) {
                    onFileSelect(file);
                  }
                }}
                className="hidden"
              />
            )}
          </div>
        </div>
        
        {/* Load Session below canvas */}
        <div className="mt-4 p-4 border border-border rounded-lg bg-card">
          <div className="flex items-center gap-2 mb-3">
            <Download className="w-4 h-4 text-muted-foreground" />
            <p className="text-sm font-medium">Load Previous Session</p>
          </div>
          <div className="flex gap-2">
            <select
              value={selectedSessionFolder}
              onChange={(e) => onSessionFolderChange?.(e.target.value)}
              disabled={isLoading || availableSessions.length === 0}
              className="flex-1 text-sm px-3 py-2 bg-background border border-border rounded focus:outline-none focus:ring-2 focus:ring-primary"
            >
              <option value="">Select a session...</option>
              {availableSessions.map((session) => (
                <option key={session.session_folder} value={session.session_folder}>
                  {session.timestamp} - {session.image_filename} ({session.num_instances} instances)
                </option>
              ))}
            </select>
            <button
              onClick={onLoadSession}
              disabled={isLoading || !selectedSessionFolder}
              className="px-6 py-2 bg-primary text-primary-foreground rounded hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 text-sm font-medium transition-colors"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Loading...</span>
                </>
              ) : (
                <>
                  <Download className="w-4 h-4" />
                  <span>Import</span>
                </>
              )}
            </button>
          </div>
        </div>
      </>
    );
  }


  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    // Ignore click if we just finished drawing a box or adding a point
    if (justDrewBoxRef.current) {
      return;
    }
    
    // Don't prevent default - let mouseDown handle box/point modes
    // Only handle instance selection here when no modes are active
    if (boxMode !== null || pointMode !== null || isLoading) {
      return; // Let mouseDown handle it
    }
    
    if (!onInstanceClick || !result || !result.masks || result.masks.length === 0) {
      return;
    }
    
    const coords = getCanvasCoordinates(e);
    if (!coords) {
      return;
    }
    
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    
    const displayWidth = canvas.width;
    const displayHeight = canvas.height;
    const normalizedX = coords.x / displayWidth;
    const normalizedY = coords.y / displayHeight;
    
    // Check instances in reverse order (last drawn = top layer)
    for (let i = result.masks.length - 1; i >= 0; i--) {
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
        
        // Check if click is inside bounding box
        if (normalizedX >= x0 && normalizedX <= x1 && normalizedY >= y0 && normalizedY <= y1) {
          try {
            onInstanceClick(i);
            e.stopPropagation();
            e.preventDefault();
            return;
          } catch (error) {
            console.error("Error calling onInstanceClick:", error);
          }
        }
      }
    }
  };

  return (
    <div ref={containerRef} className={`relative w-full max-w-full overflow-hidden ${isLoading && (!imageWidth || !imageHeight) ? 'min-h-[500px]' : ''}`}>
      {isLoading && (!imageWidth || !imageHeight) && (
        <div className="absolute inset-0 bg-card/80 backdrop-blur-sm rounded-lg flex items-center justify-center z-50">
          <div className="flex flex-col items-center gap-4">
            <Loader2 className="w-12 h-12 text-primary animate-spin" />
            <p className="text-sm font-medium text-foreground">Uploading image...</p>
          </div>
        </div>
      )}
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onClick={handleCanvasClick}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        className={`rounded-lg shadow-xl max-w-full h-auto ${
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
        <div className="absolute inset-0 bg-card/80 backdrop-blur-sm rounded-lg flex items-center justify-center z-50">
          <div className="flex flex-col items-center gap-4">
            <Loader2 className="w-12 h-12 text-primary animate-spin" />
            <p className="text-sm font-medium text-foreground">Uploading image...</p>
          </div>
        </div>
      )}
    </div>
  );
}
