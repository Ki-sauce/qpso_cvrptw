"use client";

import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import { GripVertical } from "lucide-react";
import { cn } from "cn";

interface DraggablePanelProps {
  children: ReactNode;
  label: string;
  initialPosition?: { x: number; y: number };
  className?: string;
}

interface DragState {
  pointerId: number;
  startX: number;
  startY: number;
  originX: number;
  originY: number;
}

/** A bounded, pointer-friendly floating panel for map/dashboard controls. */
export default function DraggablePanel({
  children,
  label,
  initialPosition = { x: 16, y: 16 },
  className,
}: DraggablePanelProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const dragStateRef = useRef<DragState | null>(null);
  const parentLeftRef = useRef<number | null>(null);
  const [position, setPosition] = useState(initialPosition);

  const constrainPosition = (candidate: { x: number; y: number }) => {
    const panel = panelRef.current;
    const offsetParent = panel?.offsetParent as HTMLElement | null;
    if (!panel || !offsetParent) return candidate;

    const maxX = Math.max(8, offsetParent.clientWidth - panel.offsetWidth - 8);
    const maxY = Math.max(8, offsetParent.clientHeight - panel.offsetHeight - 8);
    return {
      x: Math.max(8, Math.min(maxX, candidate.x)),
      y: Math.max(8, Math.min(maxY, candidate.y)),
    };
  };

  useEffect(() => {
    const panel = panelRef.current;
    const offsetParent = panel?.offsetParent as HTMLElement | null;
    if (!panel || !offsetParent) return;

    const keepPanelVisible = () => {
      const parentLeft = offsetParent.getBoundingClientRect().left;
      const previousParentLeft = parentLeftRef.current;
      parentLeftRef.current = parentLeft;

      // Panel coordinates are relative to the map container. When the
      // sidebar collapses, that container moves left; compensate by moving
      // the panel's local x position right so its screen position stays put.
      // When the sidebar expands, leave x unchanged so the panel naturally
      // follows the map to the right.
      const leftwardContainerShift = previousParentLeft !== null && parentLeft < previousParentLeft
        ? previousParentLeft - parentLeft
        : 0;

      setPosition((current) => {
        const anchored = leftwardContainerShift > 0
          ? { ...current, x: current.x + leftwardContainerShift }
          : current;
        const constrained = constrainPosition(anchored);
        if (constrained.x === current.x && constrained.y === current.y) return current;
        return constrained;
      });
    };

    const resizeObserver = new ResizeObserver(keepPanelVisible);
    resizeObserver.observe(offsetParent);
    resizeObserver.observe(panel);
    window.addEventListener("resize", keepPanelVisible);
    keepPanelVisible();

    return () => {
      resizeObserver.disconnect();
      window.removeEventListener("resize", keepPanelVisible);
    };
  }, []);

  const handlePointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || !panelRef.current) return;
    const panel = panelRef.current;
    const offsetParent = panel.offsetParent as HTMLElement | null;
    const parentRect = offsetParent?.getBoundingClientRect();
    const panelRect = panel.getBoundingClientRect();

    dragStateRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      originX: panelRect.left - (parentRect?.left ?? 0),
      originY: panelRect.top - (parentRect?.top ?? 0),
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragStateRef.current;
    const panel = panelRef.current;
    if (!drag || !panel || drag.pointerId !== event.pointerId) return;

    setPosition(constrainPosition({
      x: drag.originX + event.clientX - drag.startX,
      y: drag.originY + event.clientY - drag.startY,
    }));
  };

  const stopDragging = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (dragStateRef.current?.pointerId === event.pointerId) {
      dragStateRef.current = null;
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
    }
  };

  return (
    <div
      ref={panelRef}
      className={cn("absolute z-30 max-w-[calc(100vw-1rem)]", className)}
      style={{ left: `${position.x}px`, top: `${position.y}px` }}
    >
      <div
        role="toolbar"
        aria-label={`Move ${label}`}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={stopDragging}
        onPointerCancel={stopDragging}
        className="flex cursor-grab touch-none select-none items-center gap-1.5 border-b border-border/70 px-2.5 py-1.5 text-[10px] font-medium uppercase tracking-wider text-muted-foreground active:cursor-grabbing"
      >
        <GripVertical className="size-3.5 shrink-0" />
        <span>{label}</span>
        <span className="ml-auto normal-case tracking-normal text-muted-foreground/60">drag to move</span>
      </div>
      {children}
    </div>
  );
}
