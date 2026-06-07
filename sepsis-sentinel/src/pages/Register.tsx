import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { Shield, Building2, User, Mail, Lock, Hash, BedDouble } from "lucide-react";
import { ThemeToggle } from "@/components/shared/ThemeToggle";
import { useAuth } from "@/hooks/useAuth";
import { Navigate } from "react-router-dom";

export default function Register() {
  const [hospitalName, setHospitalName] = useState("");
  const [tier, setTier] = useState("Tier-2");
  const [beds, setBeds] = useState("20");
  const [adminName, setAdminName] = useState("");
  const [adminEmail, setAdminEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();
  const navigate = useNavigate();
  const { user, loading: authLoading } = useAuth();

  if (authLoading) return null;
  if (user) return <Navigate to="/dashboard" replace />;

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!hospitalName.trim() || !adminName.trim()) {
      toast({ title: "Missing fields", description: "Please fill all required fields.", variant: "destructive" });
      return;
    }
    setLoading(true);
    try {
      // 1. Create the auth user with admin role metadata
      const { data: authData, error: authError } = await supabase.auth.signUp({
        email: adminEmail,
        password,
        options: {
          data: { full_name: adminName, role: "admin", department: "ICU" },
          emailRedirectTo: window.location.origin,
        },
      });
      if (authError) throw authError;

      if (!authData.user) throw new Error("Failed to create user account");

      // 2. Create the hospital record
      const { data: hospital, error: hospError } = await supabase
        .from("hospitals")
        .insert({ name: hospitalName, tier, total_icu_beds: parseInt(beds) || 20 })
        .select("id")
        .single();
      if (hospError) throw hospError;

      // 3. Link the admin profile to the hospital
      const { error: profileError } = await supabase
        .from("profiles")
        .update({ hospital_id: hospital.id, role: "admin" })
        .eq("user_id", authData.user.id);
      if (profileError) throw profileError;

      toast({
        title: "Hospital registered",
        description: "Check your email to verify your account, then sign in.",
      });
      navigate("/login");
    } catch (error: any) {
      toast({ title: "Registration failed", description: error.message, variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="absolute top-4 right-4">
        <ThemeToggle />
      </div>
      <div className="w-full max-w-lg space-y-6">
        <div className="flex flex-col items-center gap-3">
          <div className="p-3 rounded-lg bg-primary/10 border border-primary/20">
            <Shield className="h-8 w-8 text-primary" />
          </div>
          <div className="text-center">
            <h1 className="text-xl font-mono font-bold text-foreground tracking-tight">
              REGISTER HOSPITAL
            </h1>
            <p className="text-xs font-mono text-muted-foreground mt-1">
              Set up your ICU monitoring instance
            </p>
          </div>
        </div>

        <Card>
          <CardHeader className="pb-4">
            <CardTitle className="text-base font-mono">Hospital & Admin Setup</CardTitle>
            <CardDescription className="text-xs font-mono">
              Create your hospital and initial administrator account
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleRegister} className="space-y-4">
              {/* Hospital Section */}
              <div className="space-y-3 pb-3 border-b border-border">
                <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider">Hospital Details</p>
                <div className="space-y-2">
                  <Label className="text-xs font-mono">Hospital Name *</Label>
                  <div className="relative">
                    <Building2 className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      placeholder="AIIMS New Delhi"
                      value={hospitalName}
                      onChange={(e) => setHospitalName(e.target.value)}
                      className="pl-10 font-mono text-sm"
                      required
                    />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <Label className="text-xs font-mono">Tier</Label>
                    <Select value={tier} onValueChange={setTier}>
                      <SelectTrigger className="font-mono text-sm">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Tier-1">Tier 1</SelectItem>
                        <SelectItem value="Tier-2">Tier 2</SelectItem>
                        <SelectItem value="Tier-3">Tier 3</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label className="text-xs font-mono">ICU Beds</Label>
                    <div className="relative">
                      <BedDouble className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        type="number"
                        min="1"
                        max="500"
                        value={beds}
                        onChange={(e) => setBeds(e.target.value)}
                        className="pl-10 font-mono text-sm"
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Admin Section */}
              <div className="space-y-3">
                <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider">Administrator Account</p>
                <div className="space-y-2">
                  <Label className="text-xs font-mono">Full Name *</Label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      placeholder="Dr. Sharma"
                      value={adminName}
                      onChange={(e) => setAdminName(e.target.value)}
                      className="pl-10 font-mono text-sm"
                      required
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label className="text-xs font-mono">Email *</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      type="email"
                      placeholder="admin@hospital.org"
                      value={adminEmail}
                      onChange={(e) => setAdminEmail(e.target.value)}
                      className="pl-10 font-mono text-sm"
                      required
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label className="text-xs font-mono">Password *</Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      type="password"
                      placeholder="••••••••"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className="pl-10 font-mono text-sm"
                      minLength={6}
                      required
                    />
                  </div>
                </div>
              </div>

              <Button type="submit" className="w-full font-mono text-sm" disabled={loading}>
                {loading ? "Registering..." : "Register Hospital & Admin"}
              </Button>
            </form>

            <div className="text-center mt-4">
              <button
                type="button"
                onClick={() => navigate("/login")}
                className="text-xs font-mono text-muted-foreground hover:text-foreground transition-colors"
              >
                Already registered? Sign in
              </button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
