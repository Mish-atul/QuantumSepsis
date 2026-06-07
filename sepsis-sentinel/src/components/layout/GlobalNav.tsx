import { useAuth, type StaffRole } from "@/hooks/useAuth";
import { useNavigate, useLocation } from "react-router-dom";
import { Shield, LogOut, Activity, Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ThemeToggle } from "@/components/shared/ThemeToggle";
import { ConnectionStatusIndicator } from "@/components/observability/ConnectionStatusIndicator";
import { PipelineActivityDrawer } from "@/components/observability/PipelineActivityFeed";

const ROLE_STYLES: Record<StaffRole, { label: string; className: string }> = {
  attending: { label: "Attending", className: "bg-primary/20 text-primary border-primary/30" },
  nurse: { label: "Nurse", className: "bg-tier-watch/20 text-tier-watch border-tier-watch/30" },
  admin: { label: "Admin", className: "bg-purple-500/20 text-purple-400 border-purple-500/30" },
};

export function GlobalNav() {
  const { profile, signOut } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const roleStyle = ROLE_STYLES[profile?.role ?? "nurse"];

  return (
    <header className="glass-nav px-3 sm:px-6 py-3 sticky top-0 z-50">
      <div className="flex items-center justify-between max-w-[1600px] mx-auto gap-2">
        {/* Left: Logo + Nav */}
        <div className="flex items-center gap-2 sm:gap-6 shrink-0">
          <button
            onClick={() => navigate("/dashboard")}
            className="flex items-center gap-2 hover:opacity-80 transition-opacity shrink-0"
          >
            <Shield className="h-5 w-5 text-primary" />
            <span className="font-semibold text-sm text-foreground tracking-tight hidden md:inline">
              QUANTUM SEPSIS SHIELD
            </span>
          </button>

          <nav className="flex items-center gap-1">
            <NavTab to="/dashboard" active={location.pathname === "/dashboard"}>
              <Activity className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Ward</span>
            </NavTab>
            {profile?.role === "admin" && (
              <NavTab to="/admin" active={location.pathname === "/admin"}>
                <Settings className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Staff</span>
              </NavTab>
            )}
          </nav>
        </div>

        {/* Right: Status + Controls */}
        <div className="flex items-center gap-1.5 sm:gap-3 overflow-hidden">
          {/* Bidirectional: Live connection indicator */}
          <ConnectionStatusIndicator />
          {/* Observability: Pipeline activity drawer */}
          <PipelineActivityDrawer />
          {profile?.hospital_name && (
            <span className="text-[10px] text-muted-foreground hidden lg:inline">
              {profile.hospital_name}
            </span>
          )}
          <Badge variant="outline" className={`text-[10px] px-2 py-0.5 hidden sm:inline-flex ${roleStyle.className}`}>
            {roleStyle.label}
          </Badge>
          <span className="text-xs text-foreground truncate max-w-[100px] hidden sm:inline">
            {profile?.full_name || "Staff"}
          </span>
          <ThemeToggle />
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 text-muted-foreground hover:text-foreground shrink-0"
            onClick={async () => { await signOut(); navigate("/"); }}
            title="Sign out"
          >
            <LogOut className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>
    </header>
  );
}

function NavTab({ to, active, children }: { to: string; active: boolean; children: React.ReactNode }) {
  const navigate = useNavigate();
  return (
    <button
      onClick={() => navigate(to)}
      className={`flex items-center gap-1.5 px-2 sm:px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${
        active
          ? "bg-primary/10 text-primary border border-primary/20"
          : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
      }`}
    >
      {children}
    </button>
  );
}
