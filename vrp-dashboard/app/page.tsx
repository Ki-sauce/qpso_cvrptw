"use client";

import { useState, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { 
  Truck, Play, RefreshCw, AlertCircle, Layers, CalendarDays, Table as TableIcon,
  PanelLeftClose, PanelLeftOpen,
} from "lucide-react";
import TimelineGantt from "@/components/TimelineGantt";
import DraggablePanel from "@/components/DraggablePanel";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
type AlgorithmKey = "QPSO" | "GA" | "A*";
type RouteCollection = GeoJSON.FeatureCollection<GeoJSON.LineString, Record<string, unknown>>;

interface AlgorithmSummary {
  score: number;
  distance_km: number;
  travel_time_min: number;
  total_duration_min: number;
  tw_penalty: number;
  total_lateness_s: number;
  history: number[];
}

interface ScheduleData {
  vehicle: number;
  color: string;
  stops: string[];
  totalTime_min: number;
  timeline: Array<{ stopId: string; arrivalTime_s: number; isDepot: boolean }>;
}

interface DashboardResults {
  business_impact: {
    rupees_saved: number;
    liters_saved: number;
    distance_saved_km: number;
    co2_saved_kg: number;
  };
  algorithms: Record<AlgorithmKey, AlgorithmSummary>;
  routes?: RouteCollection;
  routes_by_algorithm?: Partial<Record<AlgorithmKey, RouteCollection>>;
  schedules?: ScheduleData[];
  schedules_by_algorithm?: Partial<Record<AlgorithmKey, ScheduleData[]>>;
  locations: {
    depot: { id: string; lat: number; lng: number };
    customers: Array<{ id: string; lat: number; lng: number; demand: number; ready_time?: number; due_time?: number }>;
  };
}

const MapViewport = dynamic(
  () => import("@/components/MapViewport").then((mod) => mod.default),
  { ssr: false }
);

export default function VRPDashboard() {
  // Sidebar Resize State
  const [sidebarWidth, setSidebarWidth] = useState(480);
  const [isResizing, setIsResizing] = useState(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isCompactViewport, setIsCompactViewport] = useState(false);

  const [activeCity, setActiveCity] = useState<"salt-lake" | "manhattan">("salt-lake");
  const [customersN, setCustomersN] = useState<number[]>([10]);
  const [vehicles, setVehicles] = useState<number[]>([4]);
  const [capacity, setCapacity] = useState<number[]>([25]);

  const [trafficMode, setTrafficMode] = useState("simulated");
  const [trafficSeed, setTrafficSeed] = useState(42);

  const [distanceWeight, setDistanceWeight] = useState<number[]>([0.2]);
  const [particles, setParticles] = useState<number[]>([40]);
  const [iterations, setIterations] = useState<number[]>([100]);
  const [slaStrictness, setSlaStrictness] = useState<number[]>([4]); // NEW
  const [inspectAlgo, setInspectAlgo] = useState<AlgorithmKey>("QPSO");

  const [isSolving, setIsSolving] = useState(false);
  const [results, setResults] = useState<DashboardResults | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Resize Handlers
  const startResizing = useCallback(() => setIsResizing(true), []);
  const stopResizing = useCallback(() => setIsResizing(false), []);
  const resize = useCallback((e: MouseEvent) => {
    if (isResizing) {
      const newWidth = e.clientX;
      if (newWidth >= 320 && newWidth <= 800) setSidebarWidth(newWidth);
    }
  }, [isResizing]);

  useEffect(() => {
    if (isResizing) {
      window.addEventListener("mousemove", resize);
      window.addEventListener("mouseup", stopResizing);
    }
    return () => {
      window.removeEventListener("mousemove", resize);
      window.removeEventListener("mouseup", stopResizing);
    };
  }, [isResizing, resize, stopResizing]);

  useEffect(() => {
    const updateViewportMode = () => setIsCompactViewport(window.innerWidth <= 720);
    updateViewportMode();
    window.addEventListener("resize", updateViewportMode);
    return () => window.removeEventListener("resize", updateViewportMode);
  }, []);

  const safeSetArray = (
    val: number | readonly number[],
    setter: (v: number[]) => void,
  ) => {
    setter(Array.isArray(val) ? [...val] : [val]);
  };

  const runSolver = async () => {
    setIsSolving(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/optimize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          place_name: activeCity === "salt-lake" ? "Salt Lake, Kolkata, India" : "Manhattan, New York, USA",
          customers_n: customersN[0] ?? 10,
          num_vehicles: vehicles[0] ?? 4,
          vehicle_capacity: capacity[0] ?? 25,
          distance_weight: distanceWeight[0] ?? 0.2,
          particles: particles[0] ?? 40,
          iterations: iterations[0] ?? 100,
          traffic_mode: trafficMode,
          traffic_seed: trafficSeed,
          sla_strictness_hours: slaStrictness[0] ?? 4, // NEW
        }),
      });

      // NEW: Accurately parse the FastAPI guardrail message
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Optimization failed: HTTP ${response.status}`);
      }
      const data = await response.json();
      setResults(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to reach FastAPI optimization engine.");
    } finally {
      setIsSolving(false);
    }
  };

  const chartData = results?.algorithms?.QPSO?.history
    ? Array.from({
        length: Math.max(
          results.algorithms.QPSO.history.length,
          results.algorithms.GA.history.length,
        ),
      }, (_, idx) => ({
        iteration: idx + 1,
        QPSO: results.algorithms.QPSO.history[idx] == null
          ? null
          : Number(results.algorithms.QPSO.history[idx].toFixed(1)),
        GA: results.algorithms.GA.history[idx] == null
          ? null
          : Number(results.algorithms.GA.history[idx].toFixed(1)),
        "A* Baseline": Number(results.algorithms["A*"].score.toFixed(1)),
      }))
    : null;

  return (
    <div className={`dashboard-shell flex h-dvh min-h-0 w-full max-w-full overflow-hidden bg-background text-foreground font-sans ${isSidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      
      {/* Resizable Sidebar */}
      <aside
        aria-hidden={isSidebarCollapsed}
        style={{ width: isSidebarCollapsed ? 0 : `${sidebarWidth}px` }}
        className={`relative z-10 flex h-full min-h-0 min-w-0 shrink-0 flex-col overflow-hidden border-border bg-card/60 backdrop-blur-md transition-[width] duration-200 ${isSidebarCollapsed ? "border-r-0" : "border-r"}`}
      >
        <header className="px-6 py-5 border-b border-border bg-background/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary/10 text-primary border border-primary/20">
                <Truck className="h-5 w-5" />
              </div>
              <div>
                <h1 className="scroll-m-20 text-xl font-semibold tracking-tight">QPSO Dispatch Platform</h1>
                <p className="text-sm text-muted-foreground leading-none mt-1">Enterprise Route Optimizer</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="font-mono text-[10px]">v2.0.1</Badge>
              <Button
                type="button"
                size="icon-sm"
                variant="ghost"
                onClick={() => setIsSidebarCollapsed(true)}
                aria-label="Collapse sidebar"
                title="Collapse sidebar"
              >
                <PanelLeftClose className="size-4" />
              </Button>
            </div>
          </div>
        </header>

        {/* Native overflow-y-auto to fix ScrollArea clipping */}
        <div className="sidebar-scroll min-h-0 min-w-0 flex-1 overflow-x-hidden overflow-y-auto p-6">
          <div className="space-y-8 pb-12">
            
            <section className="space-y-4">
              <h2 className="scroll-m-20 border-b border-border/50 pb-2 text-sm font-semibold tracking-tight uppercase text-muted-foreground">
                1. Environment
              </h2>
              <div className="space-y-3">
  <Label className="text-xs font-medium leading-none">Operating Region</Label>
  <div className="grid grid-cols-2 gap-2">
    <Button 
      variant={activeCity === "salt-lake" ? "default" : "outline"} 
      onClick={() => setActiveCity("salt-lake")}
      className="text-xs h-9 font-medium shadow-sm"
    >
      Salt Lake Sector V
    </Button>
    <Button 
      variant={activeCity === "manhattan" ? "default" : "outline"} 
      onClick={() => setActiveCity("manhattan")}
      className="text-xs h-9 font-medium shadow-sm"
    >
      Manhattan, NY
    </Button>
  </div>
</div>

              <div className="space-y-3 pt-2">
                <div className="flex justify-between items-center">
                  <Label className="text-xs font-medium leading-none">SLA Strictness</Label>
                  <span className="text-sm font-bold font-mono text-primary">{slaStrictness[0] ?? 4} hrs</span>
                </div>
                <Slider 
                  value={slaStrictness} 
                  onValueChange={(v) => safeSetArray(v, setSlaStrictness)} 
                  min={1} max={8} step={0.5} 
                  className="cursor-grab active:cursor-grabbing"
                />
              </div>
              <div className="space-y-3 pt-2">
                <div className="flex justify-between items-center">
                  <Label className="text-xs font-medium leading-none">Delivery Stops</Label>
                  <span className="text-sm font-bold font-mono text-primary">{customersN[0] ?? 10} stops</span>
                </div>
                <Slider 
                  value={customersN} 
                  onValueChange={(v) => safeSetArray(v, setCustomersN)} 
                  min={4} max={40} step={1} 
                  className="cursor-grab active:cursor-grabbing"
                />
              </div>

              <div className="grid grid-cols-2 gap-4 pt-2">
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <Label className="text-xs font-medium leading-none">Fleet Size</Label>
                    <span className="text-sm font-bold font-mono text-primary">{vehicles[0] ?? 4} units</span>
                  </div>
                  <Slider 
                    value={vehicles} 
                    onValueChange={(v) => safeSetArray(v, setVehicles)} 
                    min={1} max={10} step={1} 
                    className="cursor-grab active:cursor-grabbing"
                  />
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <Label className="text-xs font-medium leading-none">Capacity</Label>
                    <span className="text-sm font-bold font-mono text-primary">{capacity[0] ?? 25} items</span>
                  </div>
                  <Slider 
                    value={capacity} 
                    onValueChange={(v) => safeSetArray(v, setCapacity)} 
                    min={5} max={50} step={1} 
                    className="cursor-grab active:cursor-grabbing"
                  />
                </div>
              </div>
            </section>

            <section className="space-y-4">
              <h2 className="scroll-m-20 border-b border-border/50 pb-2 text-sm font-semibold tracking-tight uppercase text-muted-foreground">
                2. Traffic Engine
              </h2>
              <RadioGroup value={trafficMode} onValueChange={setTrafficMode} className="flex flex-col gap-2.5">
                <div className="flex items-center space-x-2 border border-border/50 p-2.5 rounded-md bg-background/50">
                  <RadioGroupItem value="simulated" id="simulated" />
                  <Label htmlFor="simulated" className="text-sm font-medium leading-none cursor-pointer">Simulated (Seeded)</Label>
                </div>
                <div className="flex items-center space-x-2 border border-border/50 p-2.5 rounded-md bg-background/50">
                  <RadioGroupItem value="live" id="live" />
                  <Label htmlFor="live" className="text-sm font-medium leading-none cursor-pointer text-muted-foreground">Live Traffic API (Mock)</Label>
                </div>
              </RadioGroup>
              {trafficMode === "simulated" && (
                <div className="space-y-2 pt-1">
                  <Label htmlFor="seed" className="text-xs font-medium leading-none">Stochastic Seed</Label>
                  <Input 
                    id="seed" 
                    type="number" 
                    value={trafficSeed} 
                    onChange={(e) => setTrafficSeed(Number(e.target.value))} 
                    className="h-9 font-mono text-sm"
                  />
                </div>
              )}
            </section>

            <section className="space-y-4">
              <h2 className="scroll-m-20 border-b border-border/50 pb-2 text-sm font-semibold tracking-tight uppercase text-muted-foreground">
                3. Optimization Engine
              </h2>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <Label className="text-xs font-medium leading-none">Distance Penalty (β)</Label>
                  <span className="text-sm font-bold font-mono text-primary">{(distanceWeight[0] ?? 0.2).toFixed(2)}</span>
                </div>
                <Slider 
                  value={distanceWeight} 
                  onValueChange={(v) => safeSetArray(v, setDistanceWeight)} 
                  min={0} max={1} step={0.05} 
                  className="cursor-grab active:cursor-grabbing"
                />
              </div>

              <div className="grid grid-cols-2 gap-4 pt-2">
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <Label className="text-xs font-medium leading-none text-muted-foreground">Swarm Size</Label>
                    <span className="text-xs font-bold font-mono">{particles[0] ?? 40}</span>
                  </div>
                  <Slider 
                    value={particles} 
                    onValueChange={(v) => safeSetArray(v, setParticles)} 
                    min={10} max={100} step={10} 
                    className="cursor-grab active:cursor-grabbing"
                  />
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <Label className="text-xs font-medium leading-none text-muted-foreground">Iterations</Label>
                    <span className="text-xs font-bold font-mono">{iterations[0] ?? 100}</span>
                  </div>
                  <Slider 
                    value={iterations} 
                    onValueChange={(v) => safeSetArray(v, setIterations)} 
                    min={20} max={300} step={20} 
                    className="cursor-grab active:cursor-grabbing"
                  />
                </div>
              </div>
            </section>

            {/* Restored UI Feature: Inspect Algorithm Toggle */}
            {results && (
              <section className="space-y-4 animate-in fade-in slide-in-from-bottom-2">
                <h2 className="scroll-m-20 border-b border-border/50 pb-2 text-sm font-semibold tracking-tight uppercase text-muted-foreground">
                  4. Inspect Algorithm
                </h2>
                <RadioGroup value={inspectAlgo} onValueChange={(value) => setInspectAlgo(value as AlgorithmKey)} className="grid grid-cols-3 gap-2">
                  {['QPSO', 'GA', 'A*'].map((algo) => (
                    <div key={algo} className={`flex items-center space-x-2 border p-2 rounded-md justify-center transition-colors ${inspectAlgo === algo ? 'border-primary/50 bg-primary/10' : 'border-border/50 bg-background/50'}`}>
                      <RadioGroupItem value={algo} id={algo} className="hidden" />
                      <Label htmlFor={algo} className={`text-xs font-bold cursor-pointer ${inspectAlgo === algo ? 'text-primary' : 'text-muted-foreground'}`}>{algo}</Label>
                    </div>
                  ))}
                </RadioGroup>
              </section>
            )}

            <Button 
              onClick={runSolver} 
              disabled={isSolving}
              className="w-full h-11 text-sm font-medium tracking-wide shadow-sm transition-all"
            >
              {isSolving ? (
                <><RefreshCw className="mr-2 h-4 w-4 animate-spin" /> Ingesting GIS & Solving...</>
              ) : (
                <><Play className="mr-2 h-4 w-4 fill-current" /> Initialize Dispatch Sequence</>
              )}
            </Button>

            {error && (
              <div className="flex items-center gap-2 p-3 text-sm bg-destructive/10 border border-destructive/20 text-destructive rounded-lg">
                <AlertCircle className="h-4 w-4 shrink-0" />
                <span className="leading-tight">{error}</span>
              </div>
            )}
          </div>
        </div>

        {/* Dynamic Edge Resizer Handle */}
        <div
          className="absolute top-0 right-0 z-50 h-full w-2 cursor-col-resize bg-transparent transition-colors hover:bg-primary/50"
          onMouseDown={startResizing}
        />
      </aside>

      <main className="relative flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-background">
        {isSidebarCollapsed && (
          <Button
            type="button"
            size="icon-sm"
            variant="secondary"
            className="absolute left-4 top-4 z-30 shadow-lg"
            onClick={() => setIsSidebarCollapsed(false)}
            aria-label="Expand sidebar"
            title="Expand sidebar"
          >
            <PanelLeftOpen className="size-4" />
          </Button>
        )}
        
        {results && (
          <div className="pointer-events-none absolute inset-0 z-20">
            <DraggablePanel
              label="Impact summary"
              initialPosition={{ x: 16, y: 24 }}
              className="pointer-events-auto min-h-[180px] w-[220px] min-w-[180px] max-h-[80vh] max-w-[min(90vw,420px)] resize overflow-auto rounded-xl border border-border/80 bg-card/90 shadow-2xl shadow-black/30 backdrop-blur-xl"
            >
              <div className="grid divide-y divide-border/70 px-3">
                <div className="py-3">
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-emerald-500">Op savings</p>
                  <p className="mt-1 text-xl font-bold font-mono leading-none tracking-tight">₹{results.business_impact.rupees_saved}</p>
                </div>
                <div className="py-3">
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-blue-400">Fuel cut</p>
                  <p className="mt-1 text-xl font-bold font-mono leading-none tracking-tight">{results.business_impact.liters_saved}<span className="ml-0.5 text-sm font-normal text-muted-foreground">L</span></p>
                </div>
                <div className="py-3">
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-primary">Path saved</p>
                  <p className="mt-1 text-xl font-bold font-mono leading-none tracking-tight">{results.business_impact.distance_saved_km}<span className="ml-0.5 text-sm font-normal text-muted-foreground">km</span></p>
                </div>
                <div className="py-3">
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-purple-400">SLA compliance</p>
                  <p className="mt-1 text-xl font-bold font-mono leading-none tracking-tight">
                    {results.algorithms.QPSO.tw_penalty > 0 ? <span className="text-destructive">Failed</span> : <span className="text-emerald-400">100% On-Time</span>}
                  </p>
                </div>
              </div>
            </DraggablePanel>
          </div>
        )}

        <Tabs defaultValue="spatial" className="relative flex h-full min-h-0 min-w-0 flex-col">
          <DraggablePanel
            label="Workspace views"
            initialPosition={isCompactViewport ? { x: 8, y: 120 } : { x: 320, y: 16 }}
            className="pointer-events-auto max-w-[calc(100vw-1rem)] rounded-lg border border-border/80 bg-card/90 shadow-xl backdrop-blur-md"
          >
            <TabsList className="max-w-[calc(100vw-1rem)] overflow-x-auto bg-transparent shadow-none">
              <TabsTrigger value="spatial" className="text-xs font-medium gap-1.5"><Layers className="h-3.5 w-3.5" /> Spatial GIS</TabsTrigger>
              <TabsTrigger value="temporal" className="text-xs font-medium gap-1.5"><CalendarDays className="h-3.5 w-3.5" /> Gantt Schedule</TabsTrigger>
              <TabsTrigger value="benchmarks" className="text-xs font-medium gap-1.5"><TableIcon className="h-3.5 w-3.5" /> Benchmarks</TabsTrigger>
            </TabsList>
          </DraggablePanel>

          <TabsContent value="spatial" className="relative m-0 h-full min-h-0 min-w-0 w-full flex-1">
            <div className="absolute inset-0">
                <MapViewport
                  routesGeoJSON={results?.routes_by_algorithm?.[inspectAlgo] ?? results?.routes}
                  customers={results?.locations?.customers || []}
                  depot={results?.locations?.depot || { id: "D", lat: 0, lng: 0 }}
                  activeCity={activeCity}
                />
            </div>
          </TabsContent>

          <TabsContent value="temporal" className="m-0 h-full min-h-0 w-full flex-1 overflow-y-auto bg-background p-6 pt-20">
            <div className="max-w-4xl mx-auto">
              <TimelineGantt 
                schedules={results?.schedules_by_algorithm?.[inspectAlgo] ?? (results?.schedules || [])}
                customers={results?.locations?.customers || []}
              />
            </div>
          </TabsContent>

          <TabsContent value="benchmarks" className="m-0 h-full min-h-0 w-full flex-1 overflow-y-auto bg-background p-6 pt-20">
            <div className="max-w-4xl mx-auto space-y-6">
              <Card className="border-border/50 bg-card/30">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg font-semibold tracking-tight">Academic / Technical Proof</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="rounded-md border border-border/50">
                    <Table>
                      <TableHeader className="bg-muted/50">
                        <TableRow>
                          <TableHead className="font-semibold text-foreground">Methodology</TableHead>
                        <TableHead className="text-right font-semibold text-foreground">Total Cost</TableHead>
                        <TableHead className="text-right font-semibold text-foreground">Distance</TableHead>
                        <TableHead className="text-right font-semibold text-foreground">Travel Time</TableHead>
                        <TableHead className="text-right font-semibold text-foreground">SLA Penalty</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {results ? (
                          <>
                            <TableRow>
                              <TableCell className="font-medium">QPSO (Quantum Swarm)</TableCell>
                              <TableCell className="text-right font-mono text-emerald-400">{results.algorithms.QPSO.score.toFixed(2)}</TableCell>
                              <TableCell className="text-right font-mono">{results.algorithms.QPSO.distance_km.toFixed(2)} km</TableCell>
                              <TableCell className="text-right font-mono">{results.algorithms.QPSO.travel_time_min.toFixed(1)} min</TableCell>
                              <TableCell className={`text-right font-mono font-bold ${results.algorithms.QPSO.tw_penalty > 0 ? "text-destructive" : "text-emerald-400"}`}>
                                {results.algorithms.QPSO.tw_penalty > 0 ? "Violated" : "Zero Penalties"}
                              </TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell className="font-medium">Genetic Algorithm</TableCell>
                              <TableCell className="text-right font-mono">{results.algorithms.GA.score.toFixed(2)}</TableCell>
                              <TableCell className="text-right font-mono">{results.algorithms.GA.distance_km.toFixed(2)} km</TableCell>
                              <TableCell className="text-right font-mono">{results.algorithms.GA.travel_time_min.toFixed(1)} min</TableCell>
                              <TableCell className={`text-right font-mono font-bold ${results.algorithms.GA.tw_penalty > 0 ? "text-destructive" : "text-emerald-400"}`}>
                                {results.algorithms.GA.tw_penalty > 0 ? "Massive Delay (Late)" : "Zero Penalties"}
                              </TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell className="font-medium text-muted-foreground">A* Constructive Baseline</TableCell>
                              <TableCell className="text-right font-mono text-muted-foreground">{results.algorithms["A*"].score.toFixed(2)}</TableCell>
                              <TableCell className="text-right font-mono text-muted-foreground">{results.algorithms["A*"].distance_km.toFixed(2)} km</TableCell>
                              <TableCell className="text-right font-mono text-muted-foreground">{results.algorithms["A*"].travel_time_min.toFixed(1)} min</TableCell>
                              <TableCell className={`text-right font-mono font-bold ${results.algorithms["A*"].tw_penalty > 0 ? "text-destructive" : "text-muted-foreground"}`}>
                                {results.algorithms["A*"].tw_penalty > 0 ? "Massive Delay (Late)" : "Zero Penalties"}
                              </TableCell>
                            </TableRow>
                          </>
                        ) : (
                          <TableRow>
                            <TableCell colSpan={5} className="h-24 text-center text-sm text-muted-foreground">
                              Initialize dispatch sequence to generate metrics.
                            </TableCell>
                          </TableRow>
                        )}
                      </TableBody>
                    </Table>
                  </div>
                </CardContent>
              </Card>

              <Card className="border-border/50 bg-card/30 h-[400px] flex flex-col">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg font-semibold tracking-tight">Engine Convergence</CardTitle>
                </CardHeader>
                <CardContent className="flex-1 w-full p-4 pt-0">
                  {chartData ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData} margin={{ top: 20, right: 20, left: 0, bottom: 0 }}>
                        <XAxis dataKey="iteration" stroke="#71717a" fontSize={12} tickLine={false} />
                        <YAxis stroke="#71717a" fontSize={12} tickLine={false} domain={["auto", "auto"]} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: "#09090b", borderColor: "#27272a", borderRadius: "8px" }} 
                          labelStyle={{ color: "#a1a1aa", marginBottom: "4px", fontSize: "12px" }}
                          itemStyle={{ fontSize: "13px", fontFamily: "monospace", fontWeight: 600 }}
                        />
                        <Legend wrapperStyle={{ paddingTop: "20px", fontSize: "13px" }} />
                        <Line name="QPSO" type="monotone" dataKey="QPSO" stroke="#D90429" strokeWidth={3} dot={false} />
                        <Line name="Genetic Algo" type="monotone" dataKey="GA" stroke="#F4A261" strokeWidth={2} dot={false} />
                        <Line name="A* Baseline" type="stepAfter" dataKey="A* Baseline" stroke="#71717a" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-full flex items-center justify-center text-sm text-muted-foreground">
                      Awaiting sequence execution
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
