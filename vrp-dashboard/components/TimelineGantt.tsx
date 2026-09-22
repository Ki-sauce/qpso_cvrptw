"use client";

import { Clock, Truck, AlertTriangle } from "lucide-react";

interface CustomerData {
  id: string;
  lat: number;
  lng: number;
  demand: number;
  ready_time?: number;
  due_time?: number;
}

interface ScheduleItem {
  vehicle: number;
  color: string;
  stops: string[];
  totalTime_min: number;
  timeline: Array<{ stopId: string; arrivalTime_s: number; isDepot: boolean }>;
}

export default function TimelineGantt({ 
  schedules, 
  customers 
}: { 
  schedules: ScheduleItem[], 
  customers?: CustomerData[] 
}) {
  if (!schedules || schedules.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-xs text-muted-foreground font-mono">
        No active dispatch sequences. Execute optimization to generate timelines.
      </div>
    );
  }

  // Convert raw seconds from midnight into readable 12-hour HH:MM format
  const formatTime = (seconds: number) => {
    if (!seconds) return "--:--";
    const h = Math.floor(seconds / 3600) % 24;
    const m = Math.floor((seconds % 3600) / 60);
    const ampm = h >= 12 ? 'PM' : 'AM';
    const h12 = h % 12 || 12;
    return `${h12}:${m.toString().padStart(2, '0')} ${ampm}`;
  };

  return (
    <div className="space-y-4 p-4">
      <div className="flex items-center justify-between pb-2 border-b border-border">
        <h3 className="text-xs uppercase tracking-wider font-mono font-medium flex items-center gap-2">
          <Clock className="h-4 w-4 text-primary" /> SLA & Time Window Allocations
        </h3>
        <span className="text-[11px] text-muted-foreground font-mono bg-background/50 px-2 py-1 rounded">
          Dispatch: 9:00 AM | Svc Window: 5m
        </span>
      </div>

      <div className="space-y-4">
        {schedules.map((item) => (
          <div key={item.vehicle} className="space-y-2 bg-background/40 p-3 rounded-lg border border-border/50">
            <div className="flex justify-between items-center text-xs font-mono border-b border-border/50 pb-2">
              <span className="flex items-center gap-2 font-medium" style={{ color: item.color }}>
                <Truck className="h-3.5 w-3.5" /> Vehicle #{item.vehicle}
              </span>
              <span className="text-muted-foreground">Est. Total: {item.totalTime_min} min</span>
            </div>

            <div className="flex items-start gap-2 overflow-x-auto py-2 custom-scrollbar">
              {item.timeline.map((stopEvent, idx) => {
                const customer = customers?.find(c => c.id === stopEvent.stopId);
                const isDepot = stopEvent.isDepot;
                
                let isLate = false;
                if (!isDepot && customer?.due_time) {
                   isLate = stopEvent.arrivalTime_s > customer.due_time;
                }

                return (
                  <div key={idx} className="flex items-center shrink-0">
                    <div className="flex flex-col items-center gap-1.5 min-w-[100px]">
                      
                      {/* Top Node */}
                      <div
                        className={`px-3 py-1 rounded text-[11px] font-mono font-medium border text-white shadow-xs w-full text-center flex justify-center items-center gap-1
                          ${isLate ? 'animate-pulse bg-destructive border-destructive' : ''}
                        `}
                        style={!isLate ? { backgroundColor: `${item.color}cc`, borderColor: item.color } : {}}
                      >
                        {isLate && <AlertTriangle className="h-3 w-3" />}
                        {stopEvent.stopId}
                      </div>
                      
                      {/* Time Window Bounding Box */}
                      {!isDepot && customer?.ready_time && customer?.due_time ? (
                        <div className="text-[9px] font-mono flex flex-col w-full border border-border bg-black/40 rounded overflow-hidden">
                          <div className={`px-1.5 py-0.5 text-center font-semibold ${isLate ? 'bg-destructive/20 text-red-400' : 'bg-emerald-500/10 text-emerald-400'}`}>
                            Arr: {formatTime(stopEvent.arrivalTime_s)}
                          </div>
                          <div className="px-1.5 py-1 text-muted-foreground/80 text-center leading-tight">
                            Req: {formatTime(customer.ready_time)}<br/>
                            Due: {formatTime(customer.due_time)}
                          </div>
                        </div>
                      ) : (
                        <div className="text-[9px] font-mono text-muted-foreground/60 border border-transparent py-1">
                          Depot Core
                        </div>
                      )}
                    </div>
                    
                    {/* Directional Arrow */}
                    {idx < item.timeline.length - 1 && (
                      <span className="text-muted-foreground/40 text-xs mx-2 pt-2 self-start">→</span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}