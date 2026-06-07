import { useNavigate } from "react-router-dom";
import { useAuth } from "@/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/shared/ThemeToggle";
import { Shield, Activity, Brain, Zap, Clock, ArrowRight, ChevronDown } from "lucide-react";

export default function Landing() {
  const navigate = useNavigate();
  const { user } = useAuth();

  return (
    <div className="min-h-screen bg-background">
      {/* Nav */}
      <nav className="glass-nav px-6 py-4 sticky top-0 z-50">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            <span className="font-semibold text-sm text-foreground tracking-tight">
              QUANTUM SEPSIS SHIELD
            </span>
          </div>
          <div className="flex items-center gap-3">
            <ThemeToggle />
            {user ? (
              <Button
                size="sm"
                className="text-xs"
                onClick={() => navigate("/dashboard")}
              >
                Open Dashboard
                <ArrowRight className="h-3.5 w-3.5 ml-1" />
              </Button>
            ) : (
              <>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-xs"
                  onClick={() => navigate("/login")}
                >
                  Sign In
                </Button>
                <Button
                  size="sm"
                  className="text-xs"
                  onClick={() => navigate("/login")}
                >
                  Get Started
                  <ArrowRight className="h-3.5 w-3.5 ml-1" />
                </Button>
              </>
            )}
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="px-6 py-20 lg:py-32">
        <div className="max-w-4xl mx-auto text-center space-y-8">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-primary/20 bg-primary/5 backdrop-blur-sm">
            <Activity className="h-3.5 w-3.5 text-primary animate-pulse" />
            <span className="text-xs text-primary font-medium">Hybrid Quantum-Classical ML Pipeline</span>
          </div>

          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black text-foreground leading-tight tracking-tight">
            Detect Sepsis
            <br />
            <span className="text-primary">3–4 Hours Early</span>
          </h1>

          <p className="text-base sm:text-lg lg:text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
            A real-time early warning system for resource-constrained Indian ICUs. 
            Continuous 15-minute monitoring cycles powered by quantum computing.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Button
              size="lg"
              className="text-sm px-8 w-full sm:w-auto"
              onClick={() => navigate(user ? "/dashboard" : "/login")}
            >
              {user ? "Open Dashboard" : "Access Dashboard"}
              <ArrowRight className="h-4 w-4 ml-2" />
            </Button>
            <Button
              variant="outline"
              size="lg"
              className="text-sm px-8 w-full sm:w-auto"
              onClick={() => document.getElementById("features")?.scrollIntoView({ behavior: "smooth" })}
            >
              Learn More
              <ChevronDown className="h-4 w-4 ml-2" />
            </Button>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-4 sm:gap-8 pt-12 border-t border-border max-w-lg mx-auto">
            <div>
              <p className="text-2xl sm:text-3xl font-black text-foreground">3-4h</p>
              <p className="text-[10px] sm:text-xs text-muted-foreground mt-1">Lead Time</p>
            </div>
            <div>
              <p className="text-2xl sm:text-3xl font-black text-foreground">15min</p>
              <p className="text-[10px] sm:text-xs text-muted-foreground mt-1">Cycle Time</p>
            </div>
            <div>
              <p className="text-2xl sm:text-3xl font-black text-foreground">5</p>
              <p className="text-[10px] sm:text-xs text-muted-foreground mt-1">Pipeline Layers</p>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="px-6 py-20 border-t border-border">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-2xl lg:text-3xl font-bold text-foreground text-center mb-16">
            5-Layer Detection Pipeline
          </h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <FeatureCard
              icon={<Activity className="h-5 w-5" />}
              title="Layer 1 — Vitals Ingestion"
              description="Continuous 15-min capture of HR, BP, MAP, SpO₂, Temp, RR, and nurse-assessed mental status from bedside monitors."
              color="text-tier-watch"
            />
            <FeatureCard
              icon={<Brain className="h-5 w-5" />}
              title="Layer 2 — LSTM Features"
              description="Deep learning temporal feature extraction captures subtle trajectory shifts invisible to traditional scoring systems."
              color="text-primary"
            />
            <FeatureCard
              icon={<Zap className="h-5 w-5" />}
              title="Layer 3 — Quantum Scoring"
              description="Variational quantum circuit evaluates multi-dimensional risk on a 0.0–1.0 scale using quantum superposition and entanglement."
              color="text-tier-amber"
            />
            <FeatureCard
              icon={<Shield className="h-5 w-5" />}
              title="Layer 4a — Conformal Prediction"
              description="Statistical confidence intervals with uncertainty flagging prevent overconfident predictions on edge cases."
              color="text-ci-normal"
            />
            <FeatureCard
              icon={<Zap className="h-5 w-5" />}
              title="Layer 4b — Red Team Tripwires"
              description="Hardcoded safety-net thresholds that override ML predictions when 2+ clinical indicators breach simultaneously."
              color="text-tier-critical"
            />
            <FeatureCard
              icon={<Clock className="h-5 w-5" />}
              title="Layer 5 — Orchestrator"
              description="Automated clinical actions: WATCH monitoring, AMBER concurrent lab orders, CRITICAL sepsis bundle initiation."
              color="text-foreground"
            />
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="px-6 py-20 border-t border-border">
        <div className="max-w-2xl mx-auto text-center space-y-6">
          <h2 className="text-2xl font-bold text-foreground">
            Ready to protect your ICU patients?
          </h2>
          <p className="text-sm text-muted-foreground">
            Sign in with your hospital credentials to access the real-time monitoring dashboard.
          </p>
          <Button
            size="lg"
            className="text-sm px-8 w-full sm:w-auto"
            onClick={() => navigate(user ? "/dashboard" : "/login")}
          >
            {user ? "Go to Dashboard" : "Sign In Now"}
            <ArrowRight className="h-4 w-4 ml-2" />
          </Button>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border px-6 py-8">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Shield className="h-4 w-4 text-muted-foreground" />
            <span className="text-xs text-muted-foreground">
              Quantum Sepsis Shield © {new Date().getFullYear()}
            </span>
          </div>
          <span className="text-[10px] text-muted-foreground">
            For authorized medical personnel only
          </span>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  description,
  color,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  color: string;
}) {
  return (
    <div className="glass-card p-5 space-y-3">
      <div className={color}>{icon}</div>
      <h3 className="text-sm font-semibold text-foreground">{title}</h3>
      <p className="text-xs text-muted-foreground leading-relaxed">{description}</p>
    </div>
  );
}
