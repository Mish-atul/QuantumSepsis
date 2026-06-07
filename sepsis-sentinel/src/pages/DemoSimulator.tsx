import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/hooks/useAuth";
import { Shield, Activity, Play, AlertCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

interface VitalInputs {
  heart_rate: number;
  map: number;
  temperature: number;
  resp_rate: number;
  spo2: number;
  gcs_total: number;
  lactate: number;
  wbc: number;
  creatinine: number;
  platelets: number;
  age: number;
  gender: string;
}

interface PredictionResult {
  risk_score: number;
  lstm_score: number;
  xgb_score?: number;
  confidence: number;
  conformal_interval: [number, number];
  alert_level: "WATCH" | "AMBER" | "CRITICAL" | "FAST-TRACK";
  fast_tracked: boolean;
  tripwires: Array<{
    name: string;
    triggered: boolean;
    value: number;
    threshold: string;
    reason: string;
  }>;
  n_active_tripwires: number;
  reasoning: string;
  actions: string[];
  backend: string;
}

export default function DemoSimulator() {
  const { toast } = useToast();
  const { profile } = useAuth();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);

  const [vitals, setVitals] = useState<VitalInputs>({
    heart_rate: 75,
    map: 85,
    temperature: 37.0,
    resp_rate: 14,
    spo2: 98,
    gcs_total: 15,
    lactate: 1.0,
    wbc: 8.0,
    creatinine: 0.9,
    platelets: 220,
    age: 55,
    gender: "M",
  });

  const handlePredict = async () => {
    setLoading(true);
    setResult(null);

    try {
      // Call AWS EC2 backend via proxy
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(vitals),
      });

      if (!response.ok) {
        throw new Error(`API returned ${response.status}: ${response.statusText}`);
      }

      const prediction: PredictionResult = await response.json();
      setResult(prediction);

      // Save to Supabase (frontend-first approach)
      if (profile?.hospital_id) {
        const { error } = await supabase
          .from("risk_assessments")
          .insert({
            patient_id: `DEMO_${Date.now()}`,
            stay_id: `DEMO_ICU_${Date.now()}`,
            quantum_risk_score: prediction.risk_score,
            lstm_risk_score: prediction.lstm_score,
            xgboost_risk_score: prediction.xgb_score || null,
            tier: prediction.alert_level,
            confidence_score: prediction.confidence,
            confidence_interval_lower: prediction.conformal_interval[0],
            confidence_interval_upper: prediction.conformal_interval[1],
            fast_tracked: prediction.fast_tracked,
            reasoning: prediction.reasoning,
            hospital_id: profile.hospital_id,
            assessed_at: new Date().toISOString(),
          });

        if (error) {
          console.error("Failed to save risk assessment:", error);
        }

        // Save tripwire alerts
        if (prediction.tripwires && prediction.tripwires.length > 0) {
          const tripwireInserts = prediction.tripwires
            .filter(t => t.triggered)
            .map(t => ({
              patient_id: `DEMO_${Date.now()}`,
              tripwire_code: t.name,
              severity: prediction.alert_level,
              value: t.value,
              threshold: t.threshold,
              reason: t.reason,
              hospital_id: profile.hospital_id,
              triggered_at: new Date().toISOString(),
            }));

          if (tripwireInserts.length > 0) {
            await supabase.from("tripwire_alerts").insert(tripwireInserts);
          }
        }
      }

      toast({
        title: `Alert: ${prediction.alert_level}`,
        description: `Risk Score: ${(prediction.risk_score * 100).toFixed(1)}% | Confidence: ${(prediction.confidence * 100).toFixed(1)}%`,
      });
    } catch (error: any) {
      console.error("Prediction error:", error);
      toast({
        title: "Prediction Failed",
        description: error.message || "Failed to connect to ML backend",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const loadPreset = (preset: "normal" | "early_warning" | "critical") => {
    const presets = {
      normal: {
        heart_rate: 72,
        map: 87,
        temperature: 36.8,
        resp_rate: 14,
        spo2: 98,
        gcs_total: 15,
        lactate: 0.9,
        wbc: 7.5,
        creatinine: 0.9,
        platelets: 225,
        age: 55,
        gender: "M",
      },
      early_warning: {
        heart_rate: 95,
        map: 82,
        temperature: 38.2,
        resp_rate: 22,
        spo2: 94,
        gcs_total: 14,
        lactate: 2.3,
        wbc: 15.2,
        creatinine: 1.4,
        platelets: 180,
        age: 67,
        gender: "M",
      },
      critical: {
        heart_rate: 115,
        map: 65,
        temperature: 38.9,
        resp_rate: 28,
        spo2: 88,
        gcs_total: 12,
        lactate: 4.1,
        wbc: 18.5,
        creatinine: 2.1,
        platelets: 95,
        age: 72,
        gender: "F",
      },
    };

    setVitals(presets[preset]);
    toast({ title: "Preset loaded", description: `${preset.replace('_', ' ')} scenario` });
  };

  const getAlertColor = (level: string) => {
    switch (level) {
      case "WATCH": return "bg-green-500/20 text-green-400 border-green-500/50";
      case "AMBER": return "bg-yellow-500/20 text-yellow-400 border-yellow-500/50";
      case "CRITICAL": return "bg-red-500/20 text-red-400 border-red-500/50";
      case "FAST-TRACK": return "bg-red-600/30 text-red-300 border-red-600/70";
      default: return "bg-gray-500/20 text-gray-400 border-gray-500/50";
    }
  };

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="flex items-center gap-3 mb-6">
        <Shield className="h-8 w-8 text-primary" />
        <div>
          <h1 className="text-3xl font-bold">QuantumSepsis Demo Simulator</h1>
          <p className="text-muted-foreground">Connect to AWS EC2 ML backend for real-time predictions</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Panel */}
        <Card>
          <CardHeader>
            <CardTitle>Patient Vitals Input</CardTitle>
            <CardDescription>Enter current vital signs and lab values</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Quick Presets */}
            <div className="flex gap-2">
              <Button size="sm" variant="outline" onClick={() => loadPreset("normal")}>
                Normal Patient
              </Button>
              <Button size="sm" variant="outline" onClick={() => loadPreset("early_warning")}>
                Early Warning
              </Button>
              <Button size="sm" variant="outline" onClick={() => loadPreset("critical")}>
                Critical
              </Button>
            </div>

            {/* Vital Inputs */}
            <div className="grid grid-cols-2 gap-4">
              {Object.entries(vitals).map(([key, value]) => (
                <div key={key} className="space-y-2">
                  <Label className="text-xs">{key.replace(/_/g, ' ').toUpperCase()}</Label>
                  <Input
                    type={key === "gender" ? "text" : "number"}
                    value={value}
                    onChange={(e) => setVitals({ ...vitals, [key]: key === "gender" ? e.target.value : parseFloat(e.target.value) })}
                    className="text-sm"
                  />
                </div>
              ))}
            </div>

            <Button onClick={handlePredict} disabled={loading} className="w-full">
              {loading ? (
                <>
                  <Activity className="h-4 w-4 mr-2 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Run Prediction
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Results Panel */}
        <Card>
          <CardHeader>
            <CardTitle>Prediction Results</CardTitle>
            <CardDescription>Real-time sepsis risk assessment</CardDescription>
          </CardHeader>
          <CardContent>
            {result ? (
              <div className="space-y-4">
                {/* Alert Level Badge */}
                <div className={`p-4 rounded-lg border-2 ${getAlertColor(result.alert_level)}`}>
                  <div className="flex items-center justify-between">
                    <h3 className="text-xl font-bold">{result.alert_level}</h3>
                    {result.fast_tracked && (
                      <Badge variant="destructive" className="animate-pulse">FAST-TRACKED</Badge>
                    )}
                  </div>
                </div>

                {/* Risk Score */}
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Risk Score</span>
                    <span className="font-mono">{(result.risk_score * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={result.risk_score * 100} className="h-3" />
                </div>

                {/* Confidence */}
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Confidence</span>
                    <span className="font-mono">{(result.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={result.confidence * 100} className="h-2" />
                </div>

                {/* Conformal Interval */}
                <div className="text-sm space-y-1">
                  <div className="font-medium">90% Prediction Interval</div>
                  <div className="font-mono text-muted-foreground">
                    [{(result.conformal_interval[0] * 100).toFixed(1)}%, {(result.conformal_interval[1] * 100).toFixed(1)}%]
                  </div>
                </div>

                {/* Active Tripwires */}
                {result.n_active_tripwires > 0 && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <AlertCircle className="h-4 w-4 text-destructive" />
                      <span className="font-medium">{result.n_active_tripwires} Active Tripwire{result.n_active_tripwires > 1 ? 's' : ''}</span>
                    </div>
                    {result.tripwires.filter(t => t.triggered).map((tw, idx) => (
                      <div key={idx} className="text-xs bg-destructive/10 p-2 rounded border border-destructive/30">
                        <div className="font-mono">{tw.name}</div>
                        <div className="text-muted-foreground">{tw.reason}</div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Reasoning */}
                <div className="text-sm">
                  <div className="font-medium mb-1">Clinical Reasoning</div>
                  <div className="text-muted-foreground">{result.reasoning}</div>
                </div>

                {/* Recommended Actions */}
                <div className="text-sm">
                  <div className="font-medium mb-1">Recommended Actions</div>
                  <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                    {result.actions.map((action, idx) => (
                      <li key={idx}>{action}</li>
                    ))}
                  </ul>
                </div>

                {/* Backend Info */}
                <div className="text-xs text-muted-foreground font-mono border-t pt-2">
                  Backend: {result.backend} | LSTM: {(result.lstm_score * 100).toFixed(1)}% 
                  {result.xgb_score && ` | XGB: ${(result.xgb_score * 100).toFixed(1)}%`}
                </div>
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-12">
                <Activity className="h-12 w-12 mx-auto mb-3 opacity-30" />
                <p>Enter patient vitals and click "Run Prediction"</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
